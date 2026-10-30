"""Shared feature matrix for the movie embedding (R4 project).

Input (Kaggle "The Movies Dataset", folder passed as argument):
    movies_metadata.csv, credits.csv, keywords.csv, links_small.csv, ratings_small.csv
Output (folder passed as second argument):
    catalogue.csv        one row per film: movieId, tmdbId, title, year, genres, split
    movies_clean.csv     THE single readable dataset: one row per film, every covariate we use
                         (lists joined with "|"), same row order as catalogue.csv
    features.csv         THE single numeric matrix: movieId + one named column per feature (also in features.npz)
    features.npz         X (dense, float32, films x columns), movieId, column names
    blocks.json          which columns belong to which block, and the block weights
    raw_<block>.npz      sparse multi-hot / TF-IDF blocks before compression (for Jaccard / cosine kernels)
    builder.joblib       the fitted FeatureBuilder: builder.transform(new_metadata) places new films
    report.txt           coverage and size of every block

Usage:  python build_features.py data/raw data/processed
Everything is fitted on the TRAINING films only; held-out films are only transformed.
"""

import ast
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

# values filtered out as too rare are expected to be "unknown": no need to warn about each of them
warnings.filterwarnings("ignore", message="unknown class", category=UserWarning)

SEED = 2026
HELDOUT_FRACTION = 0.10
SETTINGS = {  # block -> (min training films per value, SVD dimension or None)
    "keywords": (5, 50),
    "cast": (5, 50),
    "directors": (3, 30),
    "writers": (3, 30),
    "companies": (5, 30),
    "collection": (
        2,
        None,
    ),  # a saga needs >= 2 films; kept as 0/1 columns, no compression
    "text": (5, 100),
}
SET_BLOCKS = ["keywords", "cast", "directors", "writers", "companies", "collection"]
WRITER_JOBS = {"Screenplay", "Writer", "Story"}
TOP_ACTORS = 5
TOP_LANGUAGES = 10
WEIGHTS = {
    "text": 1.0,
    "genres": 1.0,
    "language": 0.5,
    "keywords": 1.0,
    "cast": 1.0,
    "directors": 1.0,
    "writers": 1.0,
    "companies": 1.0,
    "collection": 1.0,
    "numeric": 1.0,
}
BLOCK_ORDER = [
    "text",
    "genres",
    "language",
    "keywords",
    "cast",
    "directors",
    "writers",
    "companies",
    "collection",
    "numeric",
]


# ----------------------------------------------------------------------------- loading
def parse_dicts(s):
    """ "[{'id': 1, 'name': 'Drama'}, ...]" -> list of dicts; anything malformed or missing -> []."""
    try:
        out = ast.literal_eval(s) if isinstance(s, str) else []
        return [x for x in out if isinstance(x, dict)] if isinstance(out, list) else []
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return []


def parse_list(s, key="name"):
    return [d[key] for d in parse_dicts(s) if key in d]


def parse_collection(s):
    """ "{'id': 10194, 'name': 'Toy Story Collection', ...}" -> ['Toy Story Collection']; else []."""
    try:
        x = ast.literal_eval(s) if isinstance(s, str) else None
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        x = None
    return [x["name"]] if isinstance(x, dict) and "name" in x else []


def load(raw):
    raw = Path(raw)
    meta = pd.read_csv(raw / "movies_metadata.csv", low_memory=False)
    meta["id"] = pd.to_numeric(meta["id"], errors="coerce")  # 3 rows hold dates here
    meta = meta.dropna(subset=["id"]).astype({"id": int}).drop_duplicates("id")
    credits = pd.read_csv(raw / "credits.csv").drop_duplicates("id")
    keywords = pd.read_csv(raw / "keywords.csv").drop_duplicates("id")
    links = (
        pd.read_csv(raw / "links_small.csv")
        .dropna(subset=["tmdbId"])
        .astype({"tmdbId": int})
    )
    ratings = pd.read_csv(raw / "ratings_small.csv")

    rated = pd.DataFrame({"movieId": np.sort(ratings.movieId.unique())})
    df = (
        rated.merge(links[["movieId", "tmdbId"]], on="movieId", how="left")
        .merge(meta, left_on="tmdbId", right_on="id", how="left")
        .merge(
            credits, left_on="tmdbId", right_on="id", how="left", suffixes=("", "_c")
        )
        .merge(
            keywords, left_on="tmdbId", right_on="id", how="left", suffixes=("", "_k")
        )
    )
    found = df["title"].notna()
    report = [
        f"films in ratings_small: {len(rated)}",
        f"  with a tmdbId in links_small: {int(df.tmdbId.notna().sum())}",
        f"  with metadata (kept in the catalogue): {int(found.sum())}",
        f"  dropped (no metadata): {int((~found).sum())} -> listed in dropped_films.csv",
    ]
    dropped = df.loc[~found, ["movieId", "tmdbId"]]
    df = df[found].reset_index(drop=True)
    n_ratings_kept = ratings.movieId.isin(df.movieId).sum()
    report.append(f"ratings on kept films: {n_ratings_kept} of {len(ratings)}")
    return df, dropped, report


def tidy(df):
    """Turn the raw columns into the lists / numbers the builder uses."""
    out = pd.DataFrame(
        {"movieId": df.movieId, "tmdbId": df.tmdbId.astype(int), "title": df.title}
    )
    out["genres"] = df.genres.map(parse_list)
    out["keywords"] = df.keywords.map(parse_list)
    out["cast"] = df.cast.map(
        lambda s: [
            d["name"]
            for d in sorted(parse_dicts(s), key=lambda d: d.get("order", 99))[
                :TOP_ACTORS
            ]
            if "name" in d
        ]
    )
    out["directors"] = df.crew.map(
        lambda s: [
            d["name"]
            for d in parse_dicts(s)
            if d.get("job") == "Director" and "name" in d
        ]
    )
    out["writers"] = df.crew.map(
        lambda s: sorted(
            {
                d["name"]
                for d in parse_dicts(s)
                if d.get("job") in WRITER_JOBS and "name" in d
            }
        )
    )
    out["companies"] = df.production_companies.map(parse_list)
    out["collection"] = df.belongs_to_collection.map(parse_collection)
    out["text"] = (df.overview.fillna("") + " " + df.tagline.fillna("")).str.strip()
    out["language"] = df.original_language.fillna("unknown")
    out["year"] = pd.to_datetime(df.release_date, errors="coerce").dt.year
    out["runtime"] = pd.to_numeric(df.runtime, errors="coerce").replace(0, np.nan)
    out["budget"] = pd.to_numeric(df.budget, errors="coerce").replace(
        0, np.nan
    )  # 0 = unknown
    # popularity, vote_average, vote_count, revenue are NOT used: they measure success / users' behaviour
    return out


# ----------------------------------------------------------------------------- builder
class FeatureBuilder:
    """fit() on training films, transform() on any films (also new ones)."""

    def fit(self, d):
        self.mlb = {}
        for b in SET_BLOCKS:
            min_df = SETTINGS[b][0]
            counts = pd.Series([v for lst in d[b] for v in set(lst)]).value_counts()
            vocab = sorted(counts[counts >= min_df].index)
            self.mlb[b] = MultiLabelBinarizer(classes=vocab, sparse_output=True).fit(
                [vocab]
            )
        self.mlb["genres"] = MultiLabelBinarizer(sparse_output=True).fit(d.genres)
        self.languages = list(d.language.value_counts().index[:TOP_LANGUAGES])
        self.tfidf = TfidfVectorizer(
            min_df=SETTINGS["text"][0],
            max_df=0.5,
            stop_words="english",
            sublinear_tf=True,
            max_features=20000,
        ).fit(d.text)
        raw = self._raw(d)
        self.svd = {}
        for b in SET_BLOCKS + ["text"]:
            k = SETTINGS[b][1]
            if k is not None and raw[b].shape[1] > k:
                self.svd[b] = TruncatedSVD(k, random_state=SEED).fit(raw[b])
        num = self._numeric_raw(d)
        self.num_median = num.median()
        self.num_mean = self._impute(num).mean()
        self.num_std = self._impute(num).std().replace(0, 1)
        blocks = self._dense_blocks(d, raw)
        # each block scaled to total variance 1 on the training films, then multiplied by its weight
        self.scale = {b: np.sqrt(B.var(axis=0).sum()) or 1.0 for b, B in blocks.items()}
        return self

    def _raw(self, d):
        raw = {
            b: self.mlb[b].transform(d[b]).astype(np.float32).tocsr()
            for b in ["genres"] + SET_BLOCKS
        }
        raw["text"] = self.tfidf.transform(d.text).astype(np.float32)
        return raw

    def _numeric_raw(self, d):
        return pd.DataFrame(
            {
                "year": d.year,
                "log_runtime": np.log1p(d.runtime),
                "log_budget": np.log1p(d.budget),
            }
        )

    def _impute(self, num):
        out = num.fillna(self.num_median)
        out["budget_missing"] = num.log_budget.isna().astype(float)
        return out

    def _dense_blocks(self, d, raw):
        blocks = {}
        for b in ["text"] + SET_BLOCKS:
            M = self.svd[b].transform(raw[b]) if b in self.svd else raw[b].toarray()
            blocks[b] = M
        blocks["genres"] = raw["genres"].toarray()
        lang = d.language.where(d.language.isin(self.languages), "other")
        blocks["language"] = np.column_stack(
            [(lang == l).to_numpy() for l in self.languages + ["other"]]
        ).astype(float)
        blocks["numeric"] = (
            (self._impute(self._numeric_raw(d)) - self.num_mean) / self.num_std
        ).to_numpy()
        return blocks

    def column_names(self, slices):
        names = []
        for b, (a, z) in slices.items():
            if b == "genres":
                names += [f"genre_{g}" for g in self.mlb["genres"].classes_]
            elif b == "language":
                names += [f"lang_{l}" for l in self.languages + ["other"]]
            elif b == "numeric":
                names += ["year", "log_runtime", "log_budget", "budget_missing"]
            elif b in self.svd:  # compressed blocks: SVD components
                names += [f"{b}_{i}" for i in range(z - a)]
            else:  # uncompressed 0/1 blocks: one column per value
                names += [f"{b}_{v}" for v in self.mlb[b].classes_]
        return names

    def transform(self, d):
        raw = self._raw(d)
        blocks = self._dense_blocks(d, raw)
        parts, slices, start = [], {}, 0
        for b in BLOCK_ORDER:
            B = blocks[b] / self.scale[b] * WEIGHTS[b]
            parts.append(B)
            slices[b] = [start, start + B.shape[1]]
            start += B.shape[1]
        return np.hstack(parts).astype(np.float32), slices, raw


# ----------------------------------------------------------------------------- main
def main(raw_dir, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df, dropped, report = load(raw_dir)
    d = tidy(df)
    rng = np.random.default_rng(SEED)
    d["split"] = np.where(rng.random(len(d)) < HELDOUT_FRACTION, "heldout", "train")

    builder = FeatureBuilder().fit(d[d.split == "train"])
    X, slices, raw = builder.transform(d)

    cat = d[["movieId", "tmdbId", "title", "year", "split"]].copy()
    cat["genres"] = d.genres.map("|".join)
    cat.to_csv(out / "catalogue.csv", index=False)
    dropped.to_csv(out / "dropped_films.csv", index=False)
    names = builder.column_names(slices)
    np.savez_compressed(
        out / "features.npz", X=X, movieId=d.movieId.to_numpy(), columns=np.array(names)
    )
    pd.concat(
        [d[["movieId"]].reset_index(drop=True), pd.DataFrame(X, columns=names)], axis=1
    ).to_csv(out / "features.csv", index=False, float_format="%.6g")
    clean = d.copy()
    for c in ["genres"] + SET_BLOCKS:
        clean[c] = clean[c].map("|".join)
    clean.rename(columns={"text": "overview_tagline"}).to_csv(
        out / "movies_clean.csv", index=False
    )
    sparse.save_npz(out / "raw_genres.npz", raw["genres"])
    for b in SET_BLOCKS + ["text"]:
        sparse.save_npz(out / f"raw_{b}.npz", raw[b].tocsr())
    json.dump(
        {"slices": slices, "weights": WEIGHTS, "settings": SETTINGS, "seed": SEED},
        open(out / "blocks.json", "w"),
        indent=2,
    )
    joblib.dump(builder, out / "builder.joblib")

    report += [
        f"train films: {(d.split == 'train').sum()}, held-out films: {(d.split == 'heldout').sum()}",
        f"feature matrix: {X.shape[0]} films x {X.shape[1]} columns",
    ]
    for b, (a, z) in slices.items():
        report.append(f"  {b:<11} columns {a:>4}-{z - 1:<4} ({z - a})")
    for b in ["genres"] + SET_BLOCKS:
        empty = int((raw[b].sum(axis=1) == 0).sum())
        report.append(f"  films with no {b} after the vocabulary filter: {empty}")
    report.append(f"  films with no overview/tagline: {int((d.text == '').sum())}")
    report.append(
        f"  missing year / runtime / budget: {int(d.year.isna().sum())} / "
        f"{int(d.runtime.isna().sum())} / {int(d.budget.isna().sum())}"
    )
    (out / "report.txt").write_text("\n".join(report) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    # import ourselves so the pickled builder refers to build_features.FeatureBuilder,
    # which other scripts can load with:  import build_features; joblib.load(...)
    import build_features

    # default folders: <repo>/data/raw and <repo>/data/processed, with this file in <repo>/scripts/
    repo = Path(__file__).resolve().parent.parent
    raw_dir = sys.argv[1] if len(sys.argv) > 1 else repo / "data" / "raw"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else repo / "data" / "processed"
    if not (Path(raw_dir) / "movies_metadata.csv").exists():
        sys.exit(
            f"movies_metadata.csv not found in {raw_dir}.\n"
            "Usage: python scripts/build_features.py data/raw data/processed"
        )
    print(f"reading {raw_dir}\nwriting {out_dir}\n")
    build_features.main(raw_dir, out_dir)

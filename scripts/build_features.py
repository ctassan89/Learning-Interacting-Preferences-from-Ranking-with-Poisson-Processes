"""Shared feature matrix for the movie embedding (R4 project). See data/processed/README.md.

Input (Kaggle "The Movies Dataset"):
    movies_metadata.csv, credits.csv, keywords.csv, links_small.csv, ratings_small.csv   (data/raw/)
Output (data/processed/):
    catalogue.csv        one row per film: movieId, tmdbId, title, year, genres, split
    movies_clean.csv     the readable dataset: one row per film, every covariate we use
                         (lists joined with "|"), same row order as catalogue.csv
    features.npz         the numeric matrix: X (dense, float32, films x columns), movieId, column names
    blocks.json          which columns belong to which block, how each block is encoded, the block weights
    builder.joblib       the fitted FeatureBuilder: builder.transform(new_metadata) places new films
    report.txt           coverage and size of every block
  only with --extras:
    features.csv         the same matrix as a table (large, not committed)
    raw_<block>.npz      sparse multi-hot / TF-IDF blocks before encoding (for Jaccard / cosine kernels)

Usage (repository root):
    python scripts/build_features.py                       # data/raw -> data/processed
    python scripts/build_features.py data/raw data/processed --extras
Everything is fitted on the TRAINING films only; held-out films are only transformed.

Version 2 (31 October 2026), changes from version 1 (30 October, still in the git history):
  - writers: every job of TMDb's "Writing" department except Storyboard (v1: only Screenplay, Writer, Story),
    so authors of the source novel (Author, Novel, ...) are included;
  - every list block (keywords, cast, directors, writers, companies, sagas) is encoded with random codes
    (see ENCODING) instead of SVD / 0-1 columns. With SVD only the most frequent names survived (the ~30
    most prolific directors, the most frequent actors such as Tom Hanks) and their films dominated the
    distances; the rescaled 0-1 saga columns made "islands". In an ablation study (31 Oct) this
    encoding beat v1 on the taste test (AUC 0.574 vs 0.525), on genre agreement and on the neighbours;
  - block weights chosen with a weight search on the users' tastes (see WEIGHTS);
  - a missing budget (half of the films) is filled by a regression on year, runtime, genres, language and
    production companies, instead of median + a 0/1 "budget missing" column: that column split the map in
    two halves (budget known / unknown in TMDb), i.e. the main structure came from a gap in the data.
"""
import ast
import hashlib
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
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

# values filtered out as too rare are expected to be "unknown": no need to warn about each of them
warnings.filterwarnings("ignore", message="unknown class", category=UserWarning)

SEED = 2026
HELDOUT_FRACTION = 0.10
CODE_DIM = 128
SET_BLOCKS = ["keywords", "cast", "directors", "writers", "companies", "collection"]

# How each list block (several names per film) becomes numbers:
#     block -> (encoding, min training films per name, number of columns)
#   "code"    every name gets a fixed random code of CODE_DIM numbers, generated from the name itself;
#             a film's block is the sum of the codes of its names, rescaled to length 1.
#             Films sharing a name are close; films sharing nothing are all at the same "standard"
#             distance. A name seen in one film only, or an empty list, gets a code of its own, so it is
#             neutral (no vocabulary filter needed, no group of "empty" films). New names work too.
#             Cost: a little random noise (two unrelated codes look about +-9% similar).
#   "svd"     one 0/1 column per name seen in >= min training films, compressed with truncated SVD.
#             Good when names co-occur (actors who often work together, related keywords).
#   "onehot"  one 0/1 column per name seen in >= min training films, no compression.
# For "code" blocks the min count only sets the vocabulary of the raw_<block>.npz files.
ENCODING = {b: ("code", 2, CODE_DIM) for b in SET_BLOCKS}
ENCODING_V2A = {                    # first draft of v2 (30 Oct), kept for comparison
    "keywords":   ("svd", 5, 50),
    "cast":       ("svd", 5, 50),
    "directors":  ("code", 2, CODE_DIM),
    "writers":    ("code", 2, CODE_DIM),
    "companies":  ("svd", 5, 30),
    "collection": ("code", 2, CODE_DIM),
}
ENCODING_V1 = {                     # version 1 of the matrix, kept for comparison
    "keywords":   ("svd", 5, 50),
    "cast":       ("svd", 5, 50),
    "directors":  ("svd", 3, 30),
    "writers":    ("svd", 3, 30),
    "companies":  ("svd", 5, 30),
    "collection": ("onehot", 2, None),
}
TEXT_SETTINGS = (5, 100)            # overview + tagline: min training films per word, SVD dimension
BUDGET = "regression"               # missing budget: "regression" (predicted from year, runtime, genres, language,
                                    # companies) or "flag" (median + 0/1 budget_missing column, as in v1)
NOT_WRITERS = {"Storyboard"}        # jobs of the Writing department that we do not count as writers
TOP_ACTORS = 5
TOP_LANGUAGES = 10
# Block weights, chosen with a weight search (31 Oct): grid of 108 combinations, choice on half of the users
# with the time-matched taste test, check on the other half, where they beat the old weights in all three
# taste tests (random +0.009, time +0.007, top-vs-disliked +0.017). Genres and runtime / budget are the most
# robust taste signals; text and language add nothing (language even hurts top-vs-disliked).
WEIGHTS = {"text": 0.5, "genres": 1.5, "language": 0.25, "keywords": 1.0, "cast": 1.0,
           "directors": 1.0, "writers": 1.0, "companies": 1.0, "collection": 1.0, "numeric": 1.5}
WEIGHTS_V1 = {"text": 1.0, "genres": 1.0, "language": 0.5, "keywords": 1.0, "cast": 1.0,   # weights of v1
              "directors": 1.0, "writers": 1.0, "companies": 1.0, "collection": 1.0, "numeric": 1.0}
BLOCK_ORDER = ["text", "genres", "language", "keywords", "cast", "directors", "writers",
               "companies", "collection", "numeric"]


# ----------------------------------------------------------------------------- loading
def parse_dicts(s):
    """"[{'id': 1, 'name': 'Drama'}, ...]" -> list of dicts; anything malformed or missing -> []."""
    try:
        out = ast.literal_eval(s) if isinstance(s, str) else []
        return [x for x in out if isinstance(x, dict)] if isinstance(out, list) else []
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return []


def parse_list(s, key="name"):
    return [d[key] for d in parse_dicts(s) if key in d]


def parse_collection(s):
    """"{'id': 10194, 'name': 'Toy Story Collection', ...}" -> ['Toy Story Collection']; else []."""
    try:
        x = ast.literal_eval(s) if isinstance(s, str) else None
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        x = None
    return [x["name"]] if isinstance(x, dict) and "name" in x else []


def is_writer(c):
    if "department" in c:
        return c["department"] == "Writing" and c.get("job") not in NOT_WRITERS
    return c.get("job") in {"Screenplay", "Writer", "Story"}


def load(raw):
    raw = Path(raw)
    meta = pd.read_csv(raw / "movies_metadata.csv", low_memory=False)
    meta["id"] = pd.to_numeric(meta["id"], errors="coerce")          # 3 rows hold dates here
    meta = meta.dropna(subset=["id"]).astype({"id": int}).drop_duplicates("id")
    credits = pd.read_csv(raw / "credits.csv").drop_duplicates("id")
    keywords = pd.read_csv(raw / "keywords.csv").drop_duplicates("id")
    links = pd.read_csv(raw / "links_small.csv").dropna(subset=["tmdbId"]).astype({"tmdbId": int})
    ratings = pd.read_csv(raw / "ratings_small.csv")

    rated = pd.DataFrame({"movieId": np.sort(ratings.movieId.unique())})
    df = (rated.merge(links[["movieId", "tmdbId"]], on="movieId", how="left")
               .merge(meta, left_on="tmdbId", right_on="id", how="left")
               .merge(credits, left_on="tmdbId", right_on="id", how="left", suffixes=("", "_c"))
               .merge(keywords, left_on="tmdbId", right_on="id", how="left", suffixes=("", "_k")))
    found = df["title"].notna()
    report = [f"films in ratings_small: {len(rated)}",
              f"  with a tmdbId in links_small: {int(df.tmdbId.notna().sum())}",
              f"  with metadata (kept in the catalogue): {int(found.sum())}",
              f"  dropped (no metadata): {int((~found).sum())} -> listed in dropped_films.csv"]
    dropped = df.loc[~found, ["movieId", "tmdbId"]]
    df = df[found].reset_index(drop=True)
    n_ratings_kept = ratings.movieId.isin(df.movieId).sum()
    report.append(f"ratings on kept films: {n_ratings_kept} of {len(ratings)}")
    return df, dropped, report


def tidy(df):
    """Turn the raw columns into the lists / numbers the builder uses."""
    out = pd.DataFrame({"movieId": df.movieId, "tmdbId": df.tmdbId.astype(int), "title": df.title})
    out["genres"] = df.genres.map(parse_list)
    out["keywords"] = df.keywords.map(parse_list)
    out["cast"] = df.cast.map(lambda s: [d["name"] for d in sorted(
        parse_dicts(s), key=lambda d: d.get("order", 99))[:TOP_ACTORS] if "name" in d])
    out["directors"] = df.crew.map(lambda s: [d["name"] for d in parse_dicts(s)
                                              if d.get("job") == "Director" and "name" in d])
    out["writers"] = df.crew.map(lambda s: sorted({d["name"] for d in parse_dicts(s)
                                                   if is_writer(d) and "name" in d}))
    out["companies"] = df.production_companies.map(parse_list)
    out["collection"] = df.belongs_to_collection.map(parse_collection)
    out["text"] = (df.overview.fillna("") + " " + df.tagline.fillna("")).str.strip()
    out["language"] = df.original_language.fillna("unknown")
    out["year"] = pd.to_datetime(df.release_date, errors="coerce").dt.year
    out["runtime"] = pd.to_numeric(df.runtime, errors="coerce").replace(0, np.nan)
    out["budget"] = pd.to_numeric(df.budget, errors="coerce").replace(0, np.nan)   # 0 = unknown
    # popularity, vote_average, vote_count, revenue are NOT used: they measure success / users' behaviour
    return out


def prepare(raw_dir):
    """Catalogue films with tidy covariates and the train / held-out split (same seed every time)."""
    df, dropped, report = load(raw_dir)
    d = tidy(df)
    rng = np.random.default_rng(SEED)
    d["split"] = np.where(rng.random(len(d)) < HELDOUT_FRACTION, "heldout", "train")
    return d, dropped, report


# ----------------------------------------------------------------------------- random codes
def name_code(block, name, dim):
    """Fixed random code of a name: the same name always gets the same code, on any computer."""
    h = hashlib.blake2b(f"{block}|{name}".encode("utf-8"), digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(h, "little")).standard_normal(dim)


def code_block(names_per_film, film_keys, block, dim):
    """Sum of the codes of each film's names, rescaled to length 1; an empty list gets a code of its own."""
    cache, M = {}, np.empty((len(names_per_film), dim))
    for i, (names, key) in enumerate(zip(names_per_film, film_keys)):
        names = list(dict.fromkeys(names)) or [f"__empty_list_of_film_{key}"]
        v = np.zeros(dim)
        for n in names:
            if n not in cache:
                cache[n] = name_code(block, n, dim)
            v += cache[n]
        M[i] = v / np.linalg.norm(v)
    return M


# ----------------------------------------------------------------------------- builder
class FeatureBuilder:
    """fit() on training films, transform() on any films (also new ones)."""

    def __init__(self, encoding=None, weights=None, budget=BUDGET):
        self.encoding = dict(encoding or ENCODING)
        self.weights = dict(weights or WEIGHTS)
        self.budget = budget

    def _budget_mode(self):
        return self.__dict__.get("budget", "flag")          # builders saved before 31 Oct used the flag

    def _enc(self, b):
        # builders saved by version 1 have no .encoding attribute
        return self.__dict__.get("encoding", ENCODING_V1)[b]

    def fit(self, d):
        self.mlb = {}
        for b in SET_BLOCKS:
            min_df = self._enc(b)[1]
            counts = pd.Series([v for lst in d[b] for v in set(lst)], dtype=object).value_counts()
            vocab = sorted(counts[counts >= min_df].index)
            self.mlb[b] = MultiLabelBinarizer(classes=vocab, sparse_output=True).fit([vocab])
        self.mlb["genres"] = MultiLabelBinarizer(sparse_output=True).fit(d.genres)
        self.languages = list(d.language.value_counts().index[:TOP_LANGUAGES])
        self.tfidf = TfidfVectorizer(min_df=TEXT_SETTINGS[0], max_df=0.5, stop_words="english",
                                     sublinear_tf=True, max_features=20000).fit(d.text)
        raw = self._raw(d)
        self.svd = {}
        for b in SET_BLOCKS + ["text"]:
            kind, _, k = ("svd", None, TEXT_SETTINGS[1]) if b == "text" else self._enc(b)
            if kind == "svd" and raw[b].shape[1] > k:
                self.svd[b] = TruncatedSVD(k, random_state=SEED).fit(raw[b])
        num = self._numeric_raw(d)
        self.num_median = num.median()
        if self._budget_mode() == "regression":
            known = num.log_budget.notna().to_numpy()
            self.budget_model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 11))).fit(
                self._budget_predictors(d.loc[known], num.loc[known]), num.log_budget[known])
        imputed = self._impute(num, d)
        self.num_mean = imputed.mean()
        self.num_std = imputed.std().replace(0, 1)
        blocks = self._dense_blocks(d, raw)
        # each block scaled to total variance 1 on the training films (the weight is applied in transform)
        self.scale = {b: np.sqrt(B.var(axis=0).sum()) or 1.0 for b, B in blocks.items()}
        return self

    def _raw(self, d):
        raw = {b: self.mlb[b].transform(d[b]).astype(np.float32).tocsr()
               for b in ["genres"] + SET_BLOCKS}
        raw["text"] = self.tfidf.transform(d.text).astype(np.float32)
        return raw

    def _numeric_raw(self, d):
        return pd.DataFrame({"year": d.year, "log_runtime": np.log1p(d.runtime),
                             "log_budget": np.log1p(d.budget)})

    def _budget_predictors(self, d, num):
        """Year, log runtime (median-filled), genres, language, production companies (random codes)."""
        lang = d.language.where(d.language.isin(self.languages), "other")
        keys = (d.movieId if "movieId" in d else d.title).astype(str)
        return np.column_stack([
            num.year.fillna(self.num_median["year"]).to_numpy(),
            num.log_runtime.fillna(self.num_median["log_runtime"]).to_numpy(),
            self.mlb["genres"].transform(d.genres).toarray(),
            np.column_stack([(lang == l).to_numpy() for l in self.languages + ["other"]]).astype(float),
            code_block(d.companies, keys, "companies", CODE_DIM)])

    def _impute(self, num, d):
        out = num.copy()
        for c in ("year", "log_runtime"):
            out[c] = out[c].fillna(self.num_median[c])
        missing = out.log_budget.isna().to_numpy()
        if self._budget_mode() == "regression":
            if missing.any():
                out.loc[missing, "log_budget"] = self.budget_model.predict(
                    self._budget_predictors(d.loc[missing], num.loc[missing]))
        else:
            out["log_budget"] = out.log_budget.fillna(self.num_median["log_budget"])
            out["budget_missing"] = missing.astype(float)
        return out

    def numeric_names(self):
        return ["year", "log_runtime", "log_budget"] + (["budget_missing"] if self._budget_mode() == "flag" else [])

    def _dense_blocks(self, d, raw):
        blocks = {}
        keys = (d.movieId if "movieId" in d else d.title).astype(str)
        for b in ["text"] + SET_BLOCKS:
            if b != "text" and self._enc(b)[0] == "code":
                blocks[b] = code_block(d[b], keys, b, self._enc(b)[2])
            elif b in self.svd:
                blocks[b] = self.svd[b].transform(raw[b])
            else:
                blocks[b] = raw[b].toarray()
        blocks["genres"] = raw["genres"].toarray()
        lang = d.language.where(d.language.isin(self.languages), "other")
        blocks["language"] = np.column_stack([(lang == l).to_numpy() for l in self.languages + ["other"]]).astype(float)
        blocks["numeric"] = ((self._impute(self._numeric_raw(d), d) - self.num_mean) / self.num_std).to_numpy()
        return blocks

    def column_names(self, slices):
        names = []
        for b, (a, z) in slices.items():
            if b == "genres":
                names += [f"genre_{g}" for g in self.mlb["genres"].classes_]
            elif b == "language":
                names += [f"lang_{l}" for l in self.languages + ["other"]]
            elif b == "numeric":
                names += self.numeric_names()
            elif b != "text" and self._enc(b)[0] == "code":
                names += [f"{b}_code{i}" for i in range(z - a)]
            elif b in self.svd:                     # compressed blocks: SVD components
                names += [f"{b}_{i}" for i in range(z - a)]
            else:                                   # uncompressed 0/1 blocks: one column per value
                names += [f"{b}_{v}" for v in self.mlb[b].classes_]
        return names

    def transform_blocks(self, d):
        """Every block scaled to total variance 1 (no weights): {block: matrix}, plus the raw sparse blocks."""
        raw = self._raw(d)
        blocks = self._dense_blocks(d, raw)
        return {b: blocks[b] / self.scale[b] for b in BLOCK_ORDER}, raw

    def transform(self, d):
        blocks, raw = self.transform_blocks(d)
        weights = self.__dict__.get("weights", WEIGHTS)
        parts, slices, start = [], {}, 0
        for b in BLOCK_ORDER:
            B = blocks[b] * weights[b]
            parts.append(B)
            slices[b] = [start, start + B.shape[1]]
            start += B.shape[1]
        return np.hstack(parts).astype(np.float32), slices, raw


# ----------------------------------------------------------------------------- main
def main(raw_dir, out_dir, extras=False):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    d, dropped, report = prepare(raw_dir)

    builder = FeatureBuilder().fit(d[d.split == "train"])
    X, slices, raw = builder.transform(d)

    cat = d[["movieId", "tmdbId", "title", "year", "split"]].copy()
    cat["genres"] = d.genres.map("|".join)
    cat.to_csv(out / "catalogue.csv", index=False)
    dropped.to_csv(out / "dropped_films.csv", index=False)
    names = builder.column_names(slices)
    np.savez_compressed(out / "features.npz", X=X, movieId=d.movieId.to_numpy(), columns=np.array(names))
    if extras:
        pd.concat([d[["movieId"]].reset_index(drop=True), pd.DataFrame(X, columns=names)], axis=1) \
          .to_csv(out / "features.csv", index=False, float_format="%.6g")
    clean = d.copy()
    for c in ["genres"] + SET_BLOCKS:
        clean[c] = clean[c].map("|".join)
    clean.rename(columns={"text": "overview_tagline"}).to_csv(out / "movies_clean.csv", index=False)
    if extras:
        sparse.save_npz(out / "raw_genres.npz", raw["genres"])
        for b in SET_BLOCKS + ["text"]:
            sparse.save_npz(out / f"raw_{b}.npz", raw[b].tocsr())
    json.dump({"version": 2, "slices": slices, "weights": builder.weights, "encoding": builder.encoding,
               "budget": builder.budget,
               "text_settings": TEXT_SETTINGS, "code_dim": CODE_DIM, "seed": SEED},
              open(out / "blocks.json", "w"), indent=2)
    joblib.dump(builder, out / "builder.joblib")

    report += [f"train films: {(d.split == 'train').sum()}, held-out films: {(d.split == 'heldout').sum()}",
               f"feature matrix: {X.shape[0]} films x {X.shape[1]} columns"]
    for b, (a, z) in slices.items():
        enc = "" if b in ("genres", "language", "numeric") else \
              "  svd" if b == "text" else f"  {builder.encoding[b][0]}"
        report.append(f"  {b:<11} columns {a:>4}-{z - 1:<4} ({z - a}){enc}")
    for b in ["genres"] + SET_BLOCKS:
        line = f"  {b:<11} films with an empty list: {int((d[b].map(len) == 0).sum())}"
        if b == "genres" or builder.encoding[b][0] != "code":
            line += f"; with no value left after the vocabulary filter: {int((raw[b].sum(axis=1) == 0).sum())}"
        report.append(line)
    report.append(f"  films with no overview/tagline: {int((d.text == '').sum())}")
    if builder.budget == "regression":
        ho = d[(d.split == "heldout") & d.budget.notna()]
        nh = builder._numeric_raw(ho)
        pred = builder.budget_model.predict(builder._budget_predictors(ho, nh))
        r2 = 1 - ((nh.log_budget - pred) ** 2).sum() / ((nh.log_budget - nh.log_budget.mean()) ** 2).sum()
        report.append(f"  missing budgets ({int(d.budget.isna().sum())} films) filled by a ridge regression on year, "
                      f"runtime, genres, language, companies; R^2 on held-out films with a known budget: {r2:.2f}")
    report.append(f"  missing year / runtime / budget: {int(d.year.isna().sum())} / "
                  f"{int(d.runtime.isna().sum())} / {int(d.budget.isna().sum())}")
    (out / "report.txt").write_text("\n".join(report) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    import argparse
    # import ourselves so the pickled builder refers to build_features.FeatureBuilder,
    # which other scripts can load with:  import build_features; joblib.load(...)
    import build_features
    repo = Path(__file__).resolve().parent.parent          # this file is in <repo>/scripts/
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw_dir", nargs="?", default=repo / "data" / "raw")
    ap.add_argument("out_dir", nargs="?", default=repo / "data" / "processed")
    ap.add_argument("--extras", action="store_true", help="also write features.csv and raw_<block>.npz")
    a = ap.parse_args()
    if not (Path(a.raw_dir) / "movies_metadata.csv").exists():
        sys.exit(f"movies_metadata.csv not found in {a.raw_dir}.\n"
                 "Usage: python scripts/build_features.py [data/raw data/processed] [--extras]")
    print(f"reading {a.raw_dir}\nwriting {a.out_dir}\n")
    build_features.main(a.raw_dir, a.out_dir, a.extras)

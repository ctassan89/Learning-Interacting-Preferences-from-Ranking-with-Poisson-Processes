"""Shared evaluation of a movie embedding (R4 project).

Every embedding method saves its result in the same format and runs this script, so that all
methods are compared with exactly the same metrics.

EMBEDDING FILE FORMAT
    .npz with two arrays:  Z (n_movies x d coordinates)  and  movieId (n_movies)
    (or a .csv with a movieId column and one column per coordinate).
    It must contain every movie of data/processed/catalogue.csv; the order does not matter.
    Use save_embedding(path, Z, movie_ids) from this file to write it.

USAGE
    python3 scripts/evaluate_embedding.py --embedding results/embeddings/pca_2d.npz --name pca_2d
    python3 scripts/evaluate_embedding.py --embedding ... --name ... --umap      # also UMAP 2D and 3D
    From a notebook:  from evaluate_embedding import evaluate;  evaluate(Z, movie_ids, "pca_2d")

WHAT IT MEASURES (on each "space": the embedding as given, plus UMAP 2D/3D if requested)
  1. Cluster separation  - KMeans with k clusters fitted on TRAIN movies:
                           silhouette (higher = better), Davies-Bouldin (lower = better),
                           Calinski-Harabasz (higher = better).
  2. Meaning             - genre purity of the KMeans clusters (first listed genre);
                           kNN genre agreement: mean Jaccard between the genres of a movie and of its
                           10 nearest neighbours, compared with random pairs.
  3. Taste tests         - movies in the same user's top-10 list should be closer than other pairs.
                           AUC = P(other pair farther than same-list pair): 0.5 = no signal, 1 = perfect,
                           with a 95% bootstrap interval over users (_lo, _hi). Three kinds of "other pairs":
                           taste_auc       random pairs drawn from all users' lists (same popularity);
                           taste_auc_time  random pairs drawn from the lists of the ~20 users active at
                                           about the same time (a user active in 1997 cannot have 2010
                                           films in the list: this removes that advantage of the year);
                           taste_auc_user  a top-10 movie and a movie the SAME user disliked (rating <= 2.5).
                           Ratings are used ONLY here, to evaluate, never to build the embedding.
  4. New movies          - the same metrics for HELD-OUT movies (neighbours searched among train movies,
                           clusters from train): they should not be much worse than for train movies.
  5. Look                - nearest neighbours of a few known movies, and a 2D map coloured by genre.

OUTPUT (folder results/ by default)
    metrics.csv                one row per (method, space); re-running a method replaces its rows
    <name>_<space>_map.png     2D map (only for 2D spaces)
    <name>_neighbors.txt       nearest neighbours of the reference movies
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)
from sklearn.neighbors import NearestNeighbors

SEED = 2026
K_CLUSTERS = [10, 20]          # number of KMeans clusters (same for every method)
N_NEIGHBORS = 10               # neighbours for the genre-agreement test
TOP_M = 10                     # length of the users' top lists for the taste tests
LOW_RATING = 2.5               # "disliked" = rating <= 2.5 (within-user taste test)
TIME_GROUP = 20                # users per time group (time-matched taste test)
N_BOOT = 1000                  # bootstrap resamples of the users
REFERENCE_TITLES = ["Toy Story", "The Godfather", "Pulp Fiction", "The Matrix",
                    "Titanic", "The Shining", "Finding Nemo", "Casablanca"]

REPO = Path(__file__).resolve().parent.parent
DEFAULTS = {"catalogue": REPO / "data" / "processed" / "catalogue.csv",
            "ratings": REPO / "data" / "raw" / "ratings_small.csv",
            "out": REPO / "results"}


# ----------------------------------------------------------------------------- input / output
def save_embedding(path, Z, movie_ids):
    """Write an embedding in the shared format."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, Z=np.asarray(Z, dtype=np.float32), movieId=np.asarray(movie_ids))


def load_embedding(path):
    path = Path(path)
    if path.suffix == ".npz":
        z = np.load(path)
        return np.asarray(z["Z"], dtype=float), np.asarray(z["movieId"])
    df = pd.read_csv(path)
    return df.drop(columns="movieId").to_numpy(float), df["movieId"].to_numpy()


def align(Z, movie_ids, catalogue):
    """Reorder the embedding rows to the catalogue order; fail loudly if movies are missing."""
    pos = pd.Series(np.arange(len(movie_ids)), index=movie_ids)
    if pos.index.duplicated().any():
        sys.exit("The embedding contains duplicated movieIds.")
    missing = ~catalogue.movieId.isin(pos.index)
    if missing.any():
        sys.exit(f"{missing.sum()} catalogue movies are missing from the embedding "
                 f"(e.g. movieId {catalogue.movieId[missing].iloc[0]}).")
    Z = Z[pos.loc[catalogue.movieId].to_numpy()]
    if not np.isfinite(Z).all():
        sys.exit("The embedding contains NaN or infinite values.")
    return Z


def user_lists(ratings, catalogue, m=TOP_M, seed=SEED):
    """For every user (catalogue movies only, users with < 2 movies dropped): 'top' = top-m movies, ties
    broken at random with a fixed seed; 'low' = up to m disliked movies (rating <= LOW_RATING, lowest first);
    'time' = median time of the user's ratings. Movies are catalogue row positions."""
    rng = np.random.default_rng(seed)
    row = pd.Series(np.arange(len(catalogue)), index=catalogue.movieId)
    r = ratings[ratings.movieId.isin(row.index)].copy()
    r["row"] = row.loc[r.movieId].to_numpy()
    r["u"] = rng.random(len(r))
    r = r.sort_values(["userId", "rating", "u"], ascending=[True, False, True])
    if "timestamp" not in r:
        r["timestamp"] = 0
    users = []
    for uid, g in r.groupby("userId", sort=True):
        top = g.row.to_numpy()[:m]
        if len(top) < 2:
            continue
        rest = g.iloc[len(top):]
        low = rest[rest.rating <= LOW_RATING].row.to_numpy()[::-1][:m]
        users.append({"userId": uid, "top": top, "low": low, "time": float(g.timestamp.median())})
    return users


def top_lists(ratings, catalogue, m=TOP_M, seed=SEED):
    """Each user's top-m movies (list of arrays of catalogue row positions), as in user_lists."""
    return [u["top"] for u in user_lists(ratings, catalogue, m, seed)]


# ----------------------------------------------------------------------------- metrics
def genre_sets(catalogue):
    return [frozenset(g.split("|")) if isinstance(g, str) and g else frozenset()
            for g in catalogue.genres]


def jaccard(a, b):
    return len(a & b) / len(a | b) if (a or b) else 0.0


def cluster_metrics(Z, train, heldout, first_genre, k):
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Z[train])
    lab_tr, lab_ho = km.labels_, km.predict(Z[heldout])
    out = {
        f"k{k}_silhouette": silhouette_score(Z[train], lab_tr, sample_size=min(5000, train.sum()),
                                             random_state=SEED),
        f"k{k}_davies_bouldin": davies_bouldin_score(Z[train], lab_tr),
        f"k{k}_calinski_harabasz": calinski_harabasz_score(Z[train], lab_tr),
        f"k{k}_genre_purity": purity(lab_tr, first_genre[train]),
    }
    if len(np.unique(lab_ho)) > 1:
        out[f"k{k}_silhouette_heldout"] = silhouette_score(Z[heldout], lab_ho)
    return out


def purity(labels, classes):
    t = pd.crosstab(labels, classes)
    return t.max(axis=1).sum() / t.to_numpy().sum()


def genre_agreement(Z, train, heldout, genres, rng):
    idx_tr = np.flatnonzero(train)
    nn = NearestNeighbors(n_neighbors=N_NEIGHBORS + 1).fit(Z[idx_tr])
    _, nb_tr = nn.kneighbors(Z[idx_tr])                      # first neighbour = the movie itself
    _, nb_ho = nn.kneighbors(Z[heldout], n_neighbors=N_NEIGHBORS)
    def mean_j(rows, nbs):
        return np.mean([np.mean([jaccard(genres[r], genres[idx_tr[j]]) for j in nb]) for r, nb in zip(rows, nbs)])
    a, b = rng.integers(0, len(genres), (2, 20000))
    return {"genre_knn_train": mean_j(idx_tr, nb_tr[:, 1:]),
            "genre_knn_heldout": mean_j(np.flatnonzero(heldout), nb_ho),
            "genre_random_pairs": np.mean([jaccard(genres[i], genres[j]) for i, j in zip(a, b) if i != j])}


def pair_dist(Z, pairs):
    return np.linalg.norm(Z[pairs[:, 0]] - Z[pairs[:, 1]], axis=1)


class PairTest:
    """Positive pairs (two movies of the same user's top list) against negative pairs. A positive pair of
    user u is compared only with the negative pairs of u's group: one group for everybody (random pairs),
    the users active at about the same time, or u alone (u's disliked movies).
    AUC = share of (positive, negative) comparisons in which the negative pair is farther apart (ties 1/2);
    users are resampled with replacement for the bootstrap (the same resamples for every embedding)."""

    def __init__(self, pos, pos_user, neg, neg_group, user_group, heldout, n_users, n_boot, rng):
        self.pos, self.pos_user, self.pos_group = pos, pos_user, user_group[pos_user]
        self.neg, self.neg_group = neg, neg_group
        self.n_users = n_users
        self.pos_ho, self.neg_ho = heldout[pos].any(axis=1), heldout[neg].any(axis=1)
        draws = rng.integers(0, n_users, (n_boot, n_users))
        self.boot = np.stack([np.bincount(d, minlength=n_users) for d in draws]).astype(float)

    def _scores(self, Z, pm, nm):
        dp, gp = pair_dist(Z, self.pos[pm]), self.pos_group[pm]
        dn, gn = pair_dist(Z, self.neg[nm]), self.neg_group[nm]
        order = np.lexsort((dn, gn))                      # negatives sorted by group, then distance
        dn, gn = dn[order], gn[order]
        start, end = np.searchsorted(gn, gp, "left"), np.searchsorted(gn, gp, "right")
        n = end - start
        # one sorted key "group, then distance": a binary search counts the negatives of the same group
        # that are closer (lo) or not farther (hi) than each positive pair
        span = max(dn.max(initial=0), dp.max(initial=0)) + 1.0
        kn, kp = gn * span + dn, gp * span + dp
        lo = np.searchsorted(kn, kp, "left") - start
        hi = np.searchsorted(kn, kp, "right") - start
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(n > 0, 1 - (lo + hi) / (2 * n), np.nan)

    def run(self, Z, heldout=True):
        """{'all': (auc, bootstrap aucs), 'heldout': (...)} - heldout = pairs with a held-out movie."""
        out = {}
        parts = [("all", np.ones(len(self.pos), bool), np.ones(len(self.neg), bool))]
        if heldout:
            parts.append(("heldout", self.pos_ho, self.neg_ho))
        for key, pm, nm in parts:
            s = self._scores(Z, pm, nm)
            ok = ~np.isnan(s)
            users = self.pos_user[pm][ok]
            S = np.bincount(users, weights=s[ok], minlength=self.n_users)
            n = np.bincount(users, minlength=self.n_users).astype(float)
            out[key] = (S.sum() / max(n.sum(), 1), (self.boot @ S) / np.maximum(self.boot @ n, 1))
        return out

    def distance_ratio(self, Z):
        return float(np.median(pair_dist(Z, self.pos)) / np.median(pair_dist(Z, self.neg)))


def taste_tests(users, heldout, n_boot=N_BOOT, seed=SEED):
    """The three taste tests: {'random': PairTest, 'time': PairTest, 'user': PairTest}."""
    rng = np.random.default_rng(seed)
    n = len(users)
    pos, pos_user = [], []
    for k, u in enumerate(users):
        t = u["top"]
        pos += [(t[i], t[j]) for i in range(len(t)) for j in range(i + 1, len(t))]
        pos_user += [k] * (len(t) * (len(t) - 1) // 2)
    pos, pos_user = np.array(pos), np.array(pos_user)
    n_pos = np.bincount(pos_user, minlength=n)

    def random_negatives(group):
        """3 random pairs per positive pair, drawn from the pooled top lists of the same group."""
        neg, neg_group = [], []
        for g in np.unique(group):
            members = np.flatnonzero(group == g)
            pool = np.concatenate([users[k]["top"] for k in members])
            p = rng.choice(pool, size=(3 * n_pos[members].sum(), 2))
            p = p[p[:, 0] != p[:, 1]]
            neg.append(p)
            neg_group.append(np.full(len(p), g))
        return np.concatenate(neg), np.concatenate(neg_group)

    tests = {}
    everybody = np.zeros(n, dtype=int)
    tests["random"] = PairTest(pos, pos_user, *random_negatives(everybody), everybody, heldout, n, n_boot, rng)

    order = np.argsort([u["time"] for u in users], kind="stable")
    time_group = np.empty(n, dtype=int)
    time_group[order] = np.arange(n) // TIME_GROUP
    last = time_group.max()
    if last > 0 and (time_group == last).sum() < TIME_GROUP // 2:      # merge a small last group
        time_group[time_group == last] = last - 1
    tests["time"] = PairTest(pos, pos_user, *random_negatives(time_group), time_group, heldout, n, n_boot, rng)
    times = np.array([u["time"] for u in users])
    spans = [np.ptp(times[time_group == g]) / (365.25 * 86400) for g in np.unique(time_group)]
    tests["time"].median_span_years = float(np.median(spans))

    neg = [(a, b) for u in users for a in u["top"] for b in u["low"]]
    neg_user = [k for k, u in enumerate(users) for _ in range(len(u["top"]) * len(u["low"]))]
    tests["user"] = PairTest(pos, pos_user, np.array(neg, dtype=int).reshape(-1, 2), np.array(neg_user, dtype=int),
                             np.arange(n), heldout, n, n_boot, rng)
    tests["user"].n_users_with_low = int(sum(len(u["low"]) > 0 for u in users))
    return tests


def taste_metrics(Z, tests):
    """Columns of metrics.csv for the three taste tests."""
    out = {}
    for name, t in tests.items():
        key = "taste_auc" if name == "random" else f"taste_auc_{name}"
        r = t.run(Z)
        (auc, b), (aho, _) = r["all"], r["heldout"]
        out.update({key: auc, f"{key}_lo": float(np.percentile(b, 2.5)),
                    f"{key}_hi": float(np.percentile(b, 97.5)), f"{key}_heldout": aho})
    out["taste_distance_ratio"] = tests["random"].distance_ratio(Z)
    out["taste_pairs"] = len(tests["random"].pos)
    return out


def neighbours_text(Z, catalogue, k=5):
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Z)
    lines = []
    for t in REFERENCE_TITLES:
        hit = np.flatnonzero(catalogue.title.to_numpy() == t)
        if len(hit) == 0:
            lines.append(f"{t}: not in the catalogue")
            continue
        i = hit[0]
        _, nb = nn.kneighbors(Z[i:i + 1])
        names = [f"{catalogue.title.iloc[j]} ({catalogue.year.iloc[j]:.0f})" if pd.notna(catalogue.year.iloc[j])
                 else catalogue.title.iloc[j] for j in nb[0] if j != i][:k]
        lines.append(f"{t} -> " + "; ".join(names))
    return lines


def plot_map(Z2, catalogue, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    first = catalogue.genres.fillna("").str.split("|").str[0].replace("", "none")
    top = first.value_counts().index[:8]
    fig, ax = plt.subplots(figsize=(7, 6))
    other = ~first.isin(top).to_numpy()
    ax.scatter(Z2[other, 0], Z2[other, 1], s=3, c="lightgrey", label="other", linewidths=0)
    cmap = plt.get_cmap("tab10")
    for c, g in enumerate(top):
        m = (first == g).to_numpy()
        ax.scatter(Z2[m, 0], Z2[m, 1], s=3, color=cmap(c), label=g, linewidths=0)
    ax.set_title(title)
    ax.legend(markerscale=4, fontsize=8, loc="best", frameon=False)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# ----------------------------------------------------------------------------- main entry
def evaluate(Z, movie_ids, name, catalogue_path=DEFAULTS["catalogue"], ratings_path=DEFAULTS["ratings"],
             out_dir=DEFAULTS["out"], use_umap=False):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    catalogue = pd.read_csv(catalogue_path)
    ratings = pd.read_csv(ratings_path)
    Z = align(np.asarray(Z, dtype=float), np.asarray(movie_ids), catalogue)
    train = (catalogue.split == "train").to_numpy()
    heldout = ~train
    genres = genre_sets(catalogue)
    first_genre = catalogue.genres.fillna("").str.split("|").str[0].replace("", "none").to_numpy()
    tests = taste_tests(user_lists(ratings, catalogue), heldout)

    spaces = {"native": Z}
    if use_umap:
        try:
            import umap
        except ImportError:
            sys.exit("--umap needs the umap-learn package:  pip install umap-learn")
        for c in (2, 3):
            red = umap.UMAP(n_components=c, random_state=SEED).fit(Z[train])
            spaces[f"umap{c}d"] = red.transform(Z)

    rows = []
    for space, S in spaces.items():
        rng = np.random.default_rng(SEED)
        row = {"method": name, "space": space, "dim": S.shape[1]}
        for k in K_CLUSTERS:
            row.update(cluster_metrics(S, train, heldout, first_genre, k))
        row.update(genre_agreement(S, train, heldout, genres, rng))
        row.update(taste_metrics(S, tests))
        rows.append(row)
        if S.shape[1] == 2:
            plot_map(S, catalogue, out / f"{name}_{space}_map.png", f"{name} ({space})")
        print(f"\n{name} [{space}, dim {S.shape[1]}]")
        for key, v in row.items():
            if key not in ("method", "space"):
                print(f"  {key:<26} {v:.4f}" if isinstance(v, float) else f"  {key:<26} {v}")

    (out / f"{name}_neighbors.txt").write_text("\n".join(neighbours_text(Z, catalogue)) + "\n")
    path = out / "metrics.csv"
    new = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old[old.method != name], new], ignore_index=True)
    new.to_csv(path, index=False, float_format="%.4f")
    print(f"\nsaved: {path}, {out / (name + '_neighbors.txt')}")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--embedding", required=True, help=".npz with Z and movieId, or .csv with a movieId column")
    p.add_argument("--name", required=True, help="method name, e.g. pca_2d (used in metrics.csv and file names)")
    p.add_argument("--umap", action="store_true", help="also evaluate UMAP 2D and 3D fitted on train movies")
    p.add_argument("--catalogue", default=DEFAULTS["catalogue"])
    p.add_argument("--ratings", default=DEFAULTS["ratings"])
    p.add_argument("--out", default=DEFAULTS["out"])
    a = p.parse_args()
    Z, ids = load_embedding(a.embedding)
    evaluate(Z, ids, a.name, a.catalogue, a.ratings, a.out, a.umap)

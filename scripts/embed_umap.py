"""UMAP embedding of the movies (R4 project): the method chosen for the cluster-focused track.

UMAP of the shared feature matrix with n_neighbors 15, min_dist 0 (UMAP_SETTINGS in evaluate_embedding.py),
fitted on the TRAIN movies; every movie is then placed with transform (held-out movies included).

Why UMAP with these settings (3 Nov 2026, full table in results/metrics.csv):
  - it keeps the similarities of the feature matrix best: in 3D, 26% of each movie's 10 nearest neighbours
    stay among its 10 nearest (23% for held-out movies), trustworthiness 0.97; parametric UMAP 3D kept 19%,
    VaDE 2D 9%, PCA 3D 6%;
  - min_dist 0 separates the clusters most at no cost in similarity (silhouette 0.53 vs 0.47 with the
    default 0.1; HDBSCAN finds 36 dense clusters); 15 neighbours kept more neighbours and sagas than 30 or 50;
  - 3D keeps more than 2D (26% vs 21% of the neighbours, genre agreement 0.80 vs 0.77).

Output, for each dimension d:
    results/embeddings/umap_<d>d.npz       coordinates in the shared format (Z, movieId)
    results/models/umap_<d>d.joblib        the fitted UMAP, to place new movies (see place_new_movies)
    results/umap_<d>d_clusters.csv/.txt    HDBSCAN clusters of the movies and their description
    results/metrics.csv etc.               the shared evaluation (evaluate_embedding.py)

Usage (repository root):  python scripts/embed_umap.py            # 3D and 2D
                          python scripts/embed_umap.py --dims 3
"""
import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_embedding import HDBSCAN_MIN, SEED, UMAP_SETTINGS, evaluate, save_embedding  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


def clusters(Z, train, catalogue, movies, n_ratings):
    """HDBSCAN on the train movies; movies between clusters and held-out movies go to the nearest cluster
    centre. Clusters are numbered by size; the text names the genres shared by at least half of a cluster."""
    lab = np.full(len(Z), -1)
    lab[train] = HDBSCAN(min_cluster_size=HDBSCAN_MIN, copy=True).fit_predict(Z[train])
    core = lab >= 0
    centres = np.array([Z[lab == c].mean(0) for c in range(lab.max() + 1)])
    rest = np.flatnonzero(lab < 0)
    lab[rest] = np.argmin(((Z[rest, None, :] - centres[None]) ** 2).sum(-1), axis=1)
    number = {c: k for k, c in enumerate(pd.Series(lab).value_counts().index, 1)}
    lab = np.array([number[c] for c in lab])
    genres = catalogue.genres.fillna("").str.split("|")
    lines = [f"{lab.max()} clusters (HDBSCAN, at least {HDBSCAN_MIN} movies); {np.mean(~core[train]):.0%} of the "
             "train movies lie between clusters and, like held-out movies, go to the nearest one", ""]
    names = {}
    for k in range(1, lab.max() + 1):
        m = lab == k
        g = pd.Series([x for gs in genres[m] for x in gs if x]).value_counts() / m.sum()
        top = [x for x, v in g.items() if v >= 0.5][:3] or [g.index[0]]
        names[k] = f"{' + '.join(top)} · {int(np.nanmedian(catalogue.year[m]) // 10 * 10)}s"
        best = np.argsort(-n_ratings[m])[:5]
        examples = "; ".join(catalogue.title.to_numpy()[m][best])
        lines += [f"{k:>2}. {names[k]} ({m.sum()} movies; {movies.language[m].value_counts(normalize=True).iloc[0]:.0%} "
                  f"{movies.language[m].value_counts().index[0]})", f"    e.g. {examples}"]
    table = pd.DataFrame({"movieId": catalogue.movieId, "cluster": lab, "cluster_name": [names[k] for k in lab],
                          "between_clusters": ~core})
    return table, "\n".join(lines) + "\n"


def place_new_movies(feature_rows, model_path):
    """Coordinates of new movies: feature_rows come from builder.transform(...) of build_features.py."""
    return joblib.load(model_path).transform(np.asarray(feature_rows, dtype=float))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dims", type=int, nargs="+", default=[3, 2])
    a = p.parse_args()
    import umap

    z = np.load(REPO / "data" / "processed" / "features.npz")
    X, ids = z["X"].astype(float), z["movieId"]
    catalogue = pd.read_csv(REPO / "data" / "processed" / "catalogue.csv")
    if len(catalogue) != len(ids) or (catalogue.movieId.to_numpy() != ids).any():
        sys.exit("features.npz and catalogue.csv do not match: regenerate both with build_features.py.")
    movies = pd.read_csv(REPO / "data" / "processed" / "movies_clean.csv").set_index("movieId").loc[ids].reset_index()
    n_ratings = (pd.read_csv(REPO / "data" / "raw" / "ratings_small.csv").movieId.value_counts()
                 .reindex(ids).fillna(0).to_numpy())
    train = (catalogue.split == "train").to_numpy()

    for d in a.dims:
        name = f"umap_{d}d"
        print(f"{name}: fitting on {train.sum()} train movies ...", flush=True)
        reducer = umap.UMAP(n_components=d, random_state=SEED, **UMAP_SETTINGS).fit(X[train])
        Z = reducer.transform(X)
        save_embedding(REPO / "results" / "embeddings" / f"{name}.npz", Z, ids)
        (REPO / "results" / "models").mkdir(parents=True, exist_ok=True)
        joblib.dump(reducer, REPO / "results" / "models" / f"{name}.joblib")
        table, text = clusters(Z, train, catalogue, movies, n_ratings)
        table.to_csv(REPO / "results" / f"{name}_clusters.csv", index=False)
        (REPO / "results" / f"{name}_clusters.txt").write_text(text)
        evaluate(Z, ids, name)


if __name__ == "__main__":
    main()

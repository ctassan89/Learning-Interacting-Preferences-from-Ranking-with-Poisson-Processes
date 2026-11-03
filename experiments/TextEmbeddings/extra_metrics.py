"""Metrics that complement the shared ``scripts/evaluate_embedding.py``.

The shared script is the primary, team-wide comparison. This module adds:

From ``metrics/metrics.py`` (reused, not reimplemented); everything that applies to
an embedding without a reconstruction:
    gmm_*        Gaussian mixture clustering fitted on train movies (diagonal
                 covariances in high dimension, full ones in 2D/3D), held-out movies
                 assigned with ``assign_heldout_clusters``;
    hdbscan_*    density-based clusters (2D/3D spaces only: in hundreds of
                 dimensions HDBSCAN is slow and labels most movies as noise);
    *_purity_ml  multi-label genre purity (a movie counts for every genre it has;
                 the shared script uses only the first listed genre);
    heldout_genre_jaccard_cosine   held-out to train neighbour genre similarity with
                 the cosine metric.

New, for requirements 2 (clusters visible after projection) and 3 (new movies):
    trustworthiness       how well UMAP 2D/3D keeps the 10-nearest-neighbour
                          structure of the native space (1 = perfectly);
    placement_gap         median distance of a held-out movie to its nearest train
                          movie, divided by the same for train movies (leave-one-out);
                          about 1 means new movies land "in distribution", >1 means
                          they land in empty regions of the space;
    genre_f1_*            genres of a held-out movie predicted by the vote of its 10
                          nearest train movies (micro/macro F1), and the same for
                          train movies (leave-one-out) for comparison;
    same_collection_nn,   share of movies whose nearest neighbour belongs to the
    same_director_nn      same saga / has the same director (only movies whose saga or
                          director has at least 2 movies). Trivially high for inputs that
                          contain these names; informative mainly for overview-only;
    anisotropy            mean cosine similarity of random pairs (high = the vectors
                          fill a narrow cone, which hurts kNN and clustering).
"""

from collections.abc import Sequence

import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import trustworthiness
from sklearn.metrics import f1_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer

from metrics.metrics import (
    assign_heldout_clusters,
    calinski_harabasz_index,
    davies_bouldin_index,
    fit_hdbscan,
    genre_purity,
    heldout_neighbor_genre_similarity,
    silhouette_score,
)

SEED = 2026
K_CLUSTERS = 10
K_NEIGHBORS = 10
TRUST_SAMPLE = 3000  # trustworthiness is quadratic in memory: use a sample
HDBSCAN_MIN_CLUSTER_SIZE = 25


def genre_matrix(genres: Sequence) -> np.ndarray:
    """Binary multi-hot matrix from pipe-joined strings or lists of genres."""
    sets = []
    for g in genres:
        if isinstance(g, str):
            sets.append([x for x in g.split("|") if x])
        elif isinstance(g, (list, tuple, set, frozenset)):
            sets.append(list(g))
        else:  # missing value
            sets.append([])
    return MultiLabelBinarizer().fit_transform(sets).astype(int)


def _safe(fn, *args, **kwargs):
    try:
        return float(fn(*args, **kwargs))
    except ValueError:
        return float("nan")


def cluster_scores(
    prefix: str, Z: np.ndarray, labels: np.ndarray, G: np.ndarray
) -> dict[str, float]:
    """Carlo's clustering scores (noise label -1 excluded) for one labelling."""
    return {
        f"{prefix}_silhouette": _safe(silhouette_score, Z, labels),
        f"{prefix}_davies_bouldin": _safe(davies_bouldin_index, Z, labels),
        f"{prefix}_calinski_harabasz": _safe(calinski_harabasz_index, Z, labels),
        f"{prefix}_purity_ml": _safe(genre_purity, labels, G),
    }


GMM_REG_COVARS = (1e-6, 1e-3)  # scikit-learn default first, then a fallback


def fit_gmm_model(Ztr: np.ndarray) -> tuple[GaussianMixture | None, float]:
    """Fit the GMM on train movies with scikit-learn's default regularization; only
    if that fails (columns with zero variance inside a component, e.g. the 0/1
    blocks of the shared feature matrix) retry with a larger ``reg_covar``.
    Returns (model or None, reg_covar used or NaN)."""
    covariance = "full" if Ztr.shape[1] <= 3 else "diag"
    for reg_covar in GMM_REG_COVARS:
        try:
            model = GaussianMixture(
                n_components=K_CLUSTERS,
                covariance_type=covariance,
                reg_covar=reg_covar,
                random_state=SEED,
            ).fit(np.asarray(Ztr, dtype=np.float64))
            return model, reg_covar
        except ValueError:
            continue
    return None, float("nan")


def clustering_metrics(Z: np.ndarray, train: np.ndarray, G: np.ndarray) -> dict:
    """KMeans (multi-label purity only), GMM and, in 2D/3D, HDBSCAN."""
    Ztr, Zho = Z[train], Z[~train]
    Gtr, Gho = G[train], G[~train]
    out = {}

    km = KMeans(n_clusters=K_CLUSTERS, n_init=10, random_state=SEED).fit(Ztr)
    out["kmeans_purity_ml"] = _safe(genre_purity, km.labels_, Gtr)
    out["kmeans_purity_ml_heldout"] = _safe(
        genre_purity, assign_heldout_clusters(km, Zho), Gho
    )

    gmm, reg_covar = fit_gmm_model(Ztr)
    out["gmm_reg_covar"] = reg_covar
    if gmm is not None:
        out.update(cluster_scores("gmm", Ztr, gmm.predict(Ztr), Gtr))
        out["gmm_purity_ml_heldout"] = _safe(
            genre_purity, assign_heldout_clusters(gmm, Zho), Gho
        )

    if Z.shape[1] <= 3:
        labels = fit_hdbscan(Ztr, min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE, copy=True)
        out.update(cluster_scores("hdbscan", Ztr, labels, Gtr))
        out["hdbscan_n_clusters"] = int(len(set(labels) - {-1}))
        out["hdbscan_noise_fraction"] = float(np.mean(labels == -1))
    return out


def placement_gap(Z: np.ndarray, train: np.ndarray) -> float:
    """Median nearest-train distance of held-out movies / of train movies (LOO)."""
    nn = NearestNeighbors(n_neighbors=2).fit(Z[train])
    d_train = nn.kneighbors(Z[train])[0][:, 1]  # column 0 is the movie itself
    d_held = nn.kneighbors(Z[~train], n_neighbors=1)[0][:, 0]
    return float(np.median(d_held) / max(np.median(d_train), 1e-12))


def knn_genre_f1(
    Z: np.ndarray, train: np.ndarray, G: np.ndarray, k: int = K_NEIGHBORS
) -> dict[str, float]:
    """Predict genres by majority vote (share >= 0.5) of the k nearest train movies."""
    idx_tr = np.flatnonzero(train)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Z[idx_tr])
    nb_tr = nn.kneighbors(Z[idx_tr])[1][:, 1:]
    nb_ho = nn.kneighbors(Z[~train], n_neighbors=k)[1]
    Gtr = G[idx_tr]
    out = {}
    for name, nb, truth in (("train", nb_tr, Gtr), ("heldout", nb_ho, G[~train])):
        pred = (Gtr[nb].mean(axis=1) >= 0.5).astype(int)
        keep = truth.sum(axis=1) > 0  # movies without genres cannot be scored
        out[f"genre_f1_micro_{name}"] = float(
            f1_score(truth[keep], pred[keep], average="micro", zero_division=0)
        )
        out[f"genre_f1_macro_{name}"] = float(
            f1_score(truth[keep], pred[keep], average="macro", zero_division=0)
        )
    return out


def same_entity_nn(Z: np.ndarray, entities: Sequence) -> float:
    """Share of movies (with an entity shared by >= 2 movies) whose nearest
    neighbour has at least one entity in common with them."""
    sets = [
        frozenset(e) if isinstance(e, (list, tuple, set, frozenset)) else frozenset()
        for e in entities
    ]
    counts: dict[str, int] = {}
    for s in sets:
        for e in s:
            counts[e] = counts.get(e, 0) + 1
    query = np.array([any(counts[e] >= 2 for e in s) for s in sets])
    if not query.any():
        return float("nan")
    nn = NearestNeighbors(n_neighbors=2).fit(Z)
    nb = nn.kneighbors(Z[query])[1]
    rows = np.flatnonzero(query)
    hits = [bool(sets[r] & sets[n[1] if n[0] == r else n[0]]) for r, n in zip(rows, nb)]
    return float(np.mean(hits))


def anisotropy(Z: np.ndarray, n_pairs: int = 20000, seed: int = SEED) -> float:
    rng = np.random.default_rng(seed)
    a, b = rng.integers(0, len(Z), (2, n_pairs))
    keep = a != b
    U = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
    return float(np.mean(np.sum(U[a[keep]] * U[b[keep]], axis=1)))


def projection_trustworthiness(
    Z_native: np.ndarray, Z_low: np.ndarray, k: int = K_NEIGHBORS, seed: int = SEED
) -> float:
    rng = np.random.default_rng(seed)
    n = min(TRUST_SAMPLE, len(Z_native))
    rows = rng.choice(len(Z_native), size=n, replace=False)
    return float(trustworthiness(Z_native[rows], Z_low[rows], n_neighbors=k))


def extra_metrics(
    Z: np.ndarray,
    train: np.ndarray,
    G: np.ndarray,
    collections: Sequence,
    directors: Sequence,
    Z_native: np.ndarray | None = None,
) -> dict:
    """All extra metrics for one space (``Z_native`` given for projected spaces)."""
    out = clustering_metrics(Z, train, G)
    out["heldout_genre_jaccard_cosine"] = _safe(
        heldout_neighbor_genre_similarity,
        Z[train],
        Z[~train],
        G[train],
        G[~train],
        k=K_NEIGHBORS,
        metric="cosine",
    )
    out["placement_gap"] = placement_gap(Z, train)
    out.update(knn_genre_f1(Z, train, G))
    out["same_collection_nn"] = same_entity_nn(Z, collections)
    out["same_director_nn"] = same_entity_nn(Z, directors)
    if Z_native is None:
        out["anisotropy"] = anisotropy(Z)
    else:
        out["trustworthiness"] = projection_trustworthiness(Z_native, Z)
    return out

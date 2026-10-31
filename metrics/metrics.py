"""Reusable evaluation utilities for movie embedding experiments.

Clustering scores ignore HDBSCAN noise points (label ``-1``) by default. A
score is returned only when the remaining points form at least two clusters.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score as _calinski_harabasz_score,
    davies_bouldin_score as _davies_bouldin_score,
    silhouette_score as _silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE


def _as_numpy(values: Any, name: str) -> np.ndarray:
    """Convert NumPy arrays or PyTorch tensors to a finite NumPy array."""
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    array = np.asarray(values)
    if not (
        np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_)
    ):
        raise ValueError(f"{name} must contain numeric values.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _validate_embeddings(embeddings: Any) -> np.ndarray:
    embeddings = _as_numpy(embeddings, "embeddings")
    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape (n_movies, latent_dim).")
    if embeddings.shape[0] < 2 or embeddings.shape[1] < 1:
        raise ValueError("embeddings must contain at least two movies and one feature.")
    return embeddings


def _validate_labels(labels: Any, n_movies: int) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.shape[0] != n_movies:
        raise ValueError("labels must have shape (n_movies,) matching embeddings.")
    return labels


def _validate_query_embeddings(embeddings: Any, n_features: int) -> np.ndarray:
    """Validate held-out/query embeddings against a fitted training feature space."""
    embeddings = _as_numpy(embeddings, "heldout_embeddings")
    if embeddings.ndim != 2 or embeddings.shape[0] < 1:
        raise ValueError("heldout_embeddings must have shape (n_movies, latent_dim).")
    if embeddings.shape[1] != n_features:
        raise ValueError(
            "Training and held-out embeddings must have the same dimension."
        )
    return embeddings


def _scorable_clusters(
    embeddings: Any, labels: Any, ignore_noise: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Validate inputs and select points eligible for standard cluster scores."""
    embeddings = _validate_embeddings(embeddings)
    labels = _validate_labels(labels, len(embeddings))
    mask = labels != -1 if ignore_noise else np.ones(len(labels), dtype=bool)
    scored_embeddings = embeddings[mask]
    scored_labels = labels[mask]
    n_clusters = len(np.unique(scored_labels))
    if n_clusters < 2:
        raise ValueError(
            "At least two non-noise clusters are required for this metric."
        )
    if len(scored_embeddings) <= n_clusters:
        raise ValueError(
            "The number of scored samples must exceed the number of clusters."
        )
    return scored_embeddings, scored_labels


def silhouette_score(
    embeddings: Any, labels: Any, *, ignore_noise: bool = True
) -> float:
    """Return silhouette score; HDBSCAN noise (``-1``) is excluded by default."""
    X, y = _scorable_clusters(embeddings, labels, ignore_noise)
    return float(_silhouette_score(X, y))


def davies_bouldin_index(
    embeddings: Any, labels: Any, *, ignore_noise: bool = True
) -> float:
    """Return Davies-Bouldin index; HDBSCAN noise (``-1``) is excluded by default."""
    X, y = _scorable_clusters(embeddings, labels, ignore_noise)
    return float(_davies_bouldin_score(X, y))


def calinski_harabasz_index(
    embeddings: Any, labels: Any, *, ignore_noise: bool = True
) -> float:
    """Return Calinski-Harabasz index; HDBSCAN noise (``-1``) is excluded by default."""
    X, y = _scorable_clusters(embeddings, labels, ignore_noise)
    return float(_calinski_harabasz_score(X, y))


def fit_kmeans(
    embeddings: Any, n_clusters: int, *, random_state: int = 0, **kwargs: Any
) -> np.ndarray:
    """Fit K-Means and return one cluster label per movie."""
    X = _validate_embeddings(embeddings)
    if not 2 <= n_clusters <= len(X):
        raise ValueError("n_clusters must be between 2 and the number of movies.")
    return KMeans(
        n_clusters=n_clusters, random_state=random_state, n_init=10, **kwargs
    ).fit_predict(X)


def fit_gmm(
    embeddings: Any, n_clusters: int, *, random_state: int = 0, **kwargs: Any
) -> np.ndarray:
    """Fit a Gaussian mixture model and return its most likely component labels."""
    X = _validate_embeddings(embeddings)
    if not 2 <= n_clusters <= len(X):
        raise ValueError("n_clusters must be between 2 and the number of movies.")
    return GaussianMixture(
        n_components=n_clusters, random_state=random_state, **kwargs
    ).fit_predict(X)


def fit_hdbscan(
    embeddings: Any, min_cluster_size: int = 5, **kwargs: Any
) -> np.ndarray:
    """Fit scikit-learn HDBSCAN and return labels, with noise labeled ``-1``.

    Raises an informative error when the installed scikit-learn version does not
    provide HDBSCAN. Noise points are retained in the returned labels and are
    excluded by default from the standard clustering metrics in this module.
    """
    X = _validate_embeddings(embeddings)
    if not 2 <= min_cluster_size <= len(X):
        raise ValueError("min_cluster_size must be between 2 and the number of movies.")
    try:
        from sklearn.cluster import HDBSCAN
    except ImportError as exc:
        raise ImportError(
            "HDBSCAN requires scikit-learn 1.3 or later. "
            "Upgrade scikit-learn to use fit_hdbscan."
        ) from exc
    return HDBSCAN(min_cluster_size=min_cluster_size, **kwargs).fit_predict(X)


def _validate_genres(genres: Any, n_movies: int) -> np.ndarray:
    genres = _as_numpy(genres, "genres")
    if genres.ndim != 2 or genres.shape[0] != n_movies:
        raise ValueError("genres must have shape (n_movies, n_genres).")
    if genres.shape[1] < 1:
        raise ValueError("genres must contain at least one genre column.")
    if np.any((genres != 0) & (genres != 1)):
        raise ValueError("genres must be a binary multi-hot matrix.")
    return genres.astype(bool)


def genre_purity(
    cluster_labels: Any, genres: Any, *, ignore_noise: bool = True
) -> float:
    """Return size-weighted dominant-genre purity for multi-label movies.

    For each non-empty cluster ``C``, purity is ``max_g mean_i(G[i, g])`` for
    movies ``i`` in ``C``. The result is the cluster-size-weighted mean of
    those values. Thus, a movie contributes to every genre it has; genres are
    not collapsed into a single label. Noise label ``-1`` is excluded by
    default rather than treated as a cluster.
    """
    labels = np.asarray(cluster_labels)
    if labels.ndim != 1:
        raise ValueError("cluster_labels must have shape (n_movies,).")
    genre_matrix = _validate_genres(genres, len(labels))
    mask = labels != -1 if ignore_noise else np.ones(len(labels), dtype=bool)
    if not np.any(mask):
        raise ValueError("No non-noise movies are available for genre purity.")

    weighted_purity = 0.0
    scored_movies = 0
    for label in np.unique(labels[mask]):
        members = genre_matrix[mask][labels[mask] == label]
        if len(members) == 0:
            continue
        weighted_purity += len(members) * float(members.mean(axis=0).max())
        scored_movies += len(members)
    if scored_movies == 0:
        raise ValueError("No non-empty clusters are available for genre purity.")
    return weighted_purity / scored_movies


def nearest_neighbors(
    embeddings: Any, k: int = 10, *, metric: str = "euclidean"
) -> tuple[np.ndarray, np.ndarray]:
    """Return each movie's ``k`` nearest other movies' indices and distances."""
    X = _validate_embeddings(embeddings)
    if metric not in {"euclidean", "cosine"}:
        raise ValueError("metric must be either 'euclidean' or 'cosine'.")
    if not 1 <= k < len(X):
        raise ValueError("k must be at least 1 and smaller than the number of movies.")

    candidates = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(X)
    distances, indices = candidates.kneighbors(X)
    neighbor_indices = np.empty((len(X), k), dtype=int)
    neighbor_distances = np.empty((len(X), k), dtype=float)
    for row, movie_index in enumerate(range(len(X))):
        keep = indices[row] != movie_index
        neighbor_indices[row] = indices[row][keep][:k]
        neighbor_distances[row] = distances[row][keep][:k]
    return neighbor_indices, neighbor_distances


def average_neighbor_genre_similarity(
    embeddings: Any, genres: Any, k: int = 10, *, metric: str = "euclidean"
) -> float:
    """Return mean Jaccard genre similarity to each movie's ``k`` neighbors.

    Two empty genre sets have Jaccard similarity 1.0 because they are equal;
    an empty set compared with a non-empty set has similarity 0.0.
    """
    X = _validate_embeddings(embeddings)
    genre_matrix = _validate_genres(genres, len(X))
    indices, _ = nearest_neighbors(X, k=k, metric=metric)
    query_genres = genre_matrix[:, np.newaxis, :]
    neighbor_genres = genre_matrix[indices]
    intersection = np.logical_and(query_genres, neighbor_genres).sum(axis=2)
    union = np.logical_or(query_genres, neighbor_genres).sum(axis=2)
    similarities = np.divide(
        intersection,
        union,
        out=np.ones_like(intersection, dtype=float),
        where=union != 0,
    )
    return float(similarities.mean())


def reconstruction_mse(target: Any, reconstruction: Any) -> float:
    """Return global mean squared reconstruction error."""
    target, reconstruction = _validate_reconstruction_inputs(target, reconstruction)
    return float(np.mean((target - reconstruction) ** 2))


def reconstruction_mae(target: Any, reconstruction: Any) -> float:
    """Return global mean absolute reconstruction error."""
    target, reconstruction = _validate_reconstruction_inputs(target, reconstruction)
    return float(np.mean(np.abs(target - reconstruction)))


def _validate_reconstruction_inputs(
    target: Any, reconstruction: Any
) -> tuple[np.ndarray, np.ndarray]:
    target = _as_numpy(target, "target")
    reconstruction = _as_numpy(reconstruction, "reconstruction")
    if target.shape != reconstruction.shape or target.ndim < 1:
        raise ValueError(
            "target and reconstruction must have the same non-empty shape."
        )
    if target.size == 0:
        raise ValueError("target and reconstruction must not be empty.")
    return target, reconstruction


def reconstruction_metrics(
    target: Any,
    reconstruction: Any,
    feature_blocks: Mapping[str, slice | Sequence[int]] | None = None,
) -> dict[str, Any]:
    """Return global MSE/MAE and optional per-feature-block MSE/MAE.

    ``feature_blocks`` maps a block name to a slice or feature indices along
    the final axis. It permits comparisons across models with different input
    dimensions without assuming a common feature layout.
    """
    target, reconstruction = _validate_reconstruction_inputs(target, reconstruction)
    result: dict[str, Any] = {
        "mse": reconstruction_mse(target, reconstruction),
        "mae": reconstruction_mae(target, reconstruction),
    }
    if feature_blocks is None:
        return result

    block_metrics: dict[str, dict[str, float]] = {}
    for name, selector in feature_blocks.items():
        selected_target = target[..., selector]
        selected_reconstruction = reconstruction[..., selector]
        if selected_target.size == 0:
            raise ValueError(f"Feature block '{name}' is empty.")
        block_metrics[name] = {
            "mse": reconstruction_mse(selected_target, selected_reconstruction),
            "mae": reconstruction_mae(selected_target, selected_reconstruction),
        }
    result["feature_blocks"] = block_metrics
    return result


def evaluate_embeddings(
    embeddings: Any,
    cluster_labels: Any | None = None,
    genres: Any | None = None,
    k_neighbors: int = 10,
    *,
    neighbor_metric: str = "euclidean",
) -> dict[str, Any]:
    """Return available JSON-serializable embedding evaluation metrics.

    Clustering scores omit HDBSCAN noise points. Metrics needing labels or
    genres are omitted when that input is not supplied.
    """
    X = _validate_embeddings(embeddings)
    result: dict[str, Any] = {}
    if cluster_labels is not None:
        labels = _validate_labels(cluster_labels, len(X))
        non_noise = labels != -1
        result["clustering"] = {
            "silhouette_score": silhouette_score(X, labels),
            "davies_bouldin_index": davies_bouldin_index(X, labels),
            "calinski_harabasz_index": calinski_harabasz_index(X, labels),
            "n_noise": int((~non_noise).sum()),
            "n_scored": int(non_noise.sum()),
        }
    if genres is not None:
        genre_matrix = _validate_genres(genres, len(X))
        result["average_neighbor_genre_similarity"] = average_neighbor_genre_similarity(
            X, genre_matrix, k=k_neighbors, metric=neighbor_metric
        )
        if cluster_labels is not None:
            result["genre_purity"] = genre_purity(cluster_labels, genre_matrix)
    return result


def fit_pca_projection(
    embeddings: Any, n_components: int = 2, *, random_state: int = 0, **kwargs: Any
) -> tuple[PCA, np.ndarray]:
    """Fit PCA and return the fitted transformer and 2D/3D coordinates.

    The returned PCA instance can transform held-out embeddings with the same
    feature dimension without fitting on held-out data.
    """
    X = _validate_embeddings(embeddings)
    _validate_projection_components(n_components, X)
    if n_components > min(X.shape):
        raise ValueError("PCA n_components cannot exceed samples or input features.")
    model = PCA(n_components=n_components, random_state=random_state, **kwargs)
    return model, model.fit_transform(X)


def project_embeddings(
    embeddings: Any,
    method: str = "pca",
    n_components: int = 2,
    *,
    random_state: int = 0,
    **kwargs: Any,
) -> np.ndarray:
    """Fit a projection on ``embeddings`` and return 2D or 3D coordinates.

    ``method`` supports ``"pca"``, ``"tsne"``, and optional ``"umap"``.
    PCA is the only built-in method here with an out-of-sample transform; use
    :func:`fit_pca_projection` and its returned model for held-out movies.
    Standard scikit-learn t-SNE has no ``transform`` method, so it must not be
    used to project held-out movies independently for direct comparison.
    """
    X = _validate_embeddings(embeddings)
    _validate_projection_components(n_components, X)
    normalized_method = method.lower()
    if normalized_method == "pca":
        _, projected = fit_pca_projection(
            X, n_components=n_components, random_state=random_state, **kwargs
        )
        return projected
    if normalized_method == "tsne":
        perplexity = kwargs.get("perplexity", 30.0)
        if not 0 < perplexity < len(X):
            raise ValueError(
                "t-SNE perplexity must be positive and smaller than n_movies."
            )
        return TSNE(
            n_components=n_components, random_state=random_state, **kwargs
        ).fit_transform(X)
    if normalized_method == "umap":
        try:
            import umap
        except ImportError as exc:
            raise ImportError(
                "UMAP projection requires the optional 'umap-learn' package."
            ) from exc
        return umap.UMAP(
            n_components=n_components, random_state=random_state, **kwargs
        ).fit_transform(X)
    raise ValueError("method must be one of: 'pca', 'tsne', or 'umap'.")


def _validate_projection_components(n_components: int, embeddings: np.ndarray) -> None:
    if n_components not in {2, 3}:
        raise ValueError("n_components must be 2 or 3.")
    if len(embeddings) <= n_components:
        raise ValueError("Projection requires more movies than n_components.")


def nearest_training_neighbors(
    train_embeddings: Any,
    heldout_embeddings: Any,
    k: int = 10,
    *,
    metric: str = "euclidean",
) -> tuple[np.ndarray, np.ndarray]:
    """Return each held-out movie's nearest training indices and distances."""
    train = _validate_embeddings(train_embeddings)
    heldout = _validate_query_embeddings(heldout_embeddings, train.shape[1])
    if metric not in {"euclidean", "cosine"}:
        raise ValueError("metric must be either 'euclidean' or 'cosine'.")
    if not 1 <= k <= len(train):
        raise ValueError("k must be at least 1 and no greater than training movies.")
    neighbors = NearestNeighbors(n_neighbors=k, metric=metric).fit(train)
    distances, indices = neighbors.kneighbors(heldout)
    return indices, distances


def heldout_neighbor_genre_similarity(
    train_embeddings: Any,
    heldout_embeddings: Any,
    train_genres: Any,
    heldout_genres: Any,
    k: int = 10,
    *,
    metric: str = "euclidean",
) -> float:
    """Return held-out-to-training nearest-neighbor mean Jaccard similarity."""
    train = _validate_embeddings(train_embeddings)
    heldout = _validate_query_embeddings(heldout_embeddings, train.shape[1])
    training_genres = _validate_genres(train_genres, len(train))
    query_genres = _validate_genres(heldout_genres, len(heldout))
    indices, _ = nearest_training_neighbors(train, heldout, k=k, metric=metric)
    neighbor_genres = training_genres[indices]
    intersection = np.logical_and(query_genres[:, np.newaxis, :], neighbor_genres).sum(
        axis=2
    )
    union = np.logical_or(query_genres[:, np.newaxis, :], neighbor_genres).sum(axis=2)
    similarities = np.divide(
        intersection,
        union,
        out=np.ones_like(intersection, dtype=float),
        where=union != 0,
    )
    return float(similarities.mean())


def assign_heldout_clusters(model: Any, heldout_embeddings: Any) -> np.ndarray:
    """Assign held-out embeddings using a K-Means or GMM fitted on training data.

    HDBSCAN is supported only when the supplied implementation provides a
    compatible ``predict`` method. This function never refits a model on
    held-out embeddings.
    """
    if model.__class__.__name__ == "HDBSCAN":
        if not hasattr(model, "predict"):
            raise NotImplementedError(
                "This HDBSCAN implementation does not support out-of-sample prediction."
            )
    elif not isinstance(model, (KMeans, GaussianMixture)):
        raise NotImplementedError(
            "Out-of-sample cluster assignment requires a fitted K-Means or GMM model."
        )
    n_features = getattr(model, "n_features_in_", None)
    if n_features is None:
        raise ValueError("The clustering model must be fitted on training embeddings.")
    heldout = _validate_query_embeddings(heldout_embeddings, n_features)
    return np.asarray(model.predict(heldout))


def heldout_placement(
    train_embeddings: Any,
    heldout_embeddings: Any,
    k_neighbors: int = 10,
    *,
    clustering_model: Any | None = None,
    train_genres: Any | None = None,
    heldout_genres: Any | None = None,
    neighbor_metric: str = "euclidean",
) -> dict[str, Any]:
    """Describe held-out placement relative to training embeddings and clusters."""
    train = _validate_embeddings(train_embeddings)
    heldout = _validate_query_embeddings(heldout_embeddings, train.shape[1])
    indices, distances = nearest_training_neighbors(
        train, heldout, k=k_neighbors, metric=neighbor_metric
    )
    result: dict[str, Any] = {
        "nearest_training_indices": indices.tolist(),
        "nearest_training_distances": distances.tolist(),
    }
    if clustering_model is not None:
        labels = assign_heldout_clusters(clustering_model, heldout)
        unique, counts = np.unique(labels, return_counts=True)
        result["assigned_clusters"] = labels.tolist()
        result["cluster_assignment_fractions"] = {
            str(label): float(count / len(labels))
            for label, count in zip(unique, counts, strict=True)
        }
        noise_count = int((labels == -1).sum())
        result["n_noise"] = noise_count
        result["noise_fraction"] = float(noise_count / len(labels))
    if (train_genres is None) != (heldout_genres is None):
        raise ValueError("Provide both train_genres and heldout_genres, or neither.")
    if train_genres is not None:
        result["average_neighbor_genre_similarity"] = heldout_neighbor_genre_similarity(
            train,
            heldout,
            train_genres,
            heldout_genres,
            k=k_neighbors,
            metric=neighbor_metric,
        )
    return result


def _clustering_summary(embeddings: Any, labels: Any) -> dict[str, Any]:
    """Return clustering scores, using None for mathematically unavailable values."""
    X = _validate_embeddings(embeddings)
    y = _validate_labels(labels, len(X))
    result: dict[str, Any] = {
        "n_noise": int((y == -1).sum()),
        "n_scored": int((y != -1).sum()),
    }
    for name, metric in (
        ("silhouette_score", silhouette_score),
        ("davies_bouldin_index", davies_bouldin_index),
        ("calinski_harabasz_index", calinski_harabasz_index),
    ):
        try:
            result[name] = metric(X, y)
        except ValueError:
            result[name] = None
    return result


def evaluate_generalization(
    train_embeddings: Any,
    heldout_embeddings: Any,
    train_cluster_labels: Any,
    heldout_cluster_labels: Any,
    train_genres: Any | None = None,
    heldout_genres: Any | None = None,
    k_neighbors: int = 10,
    *,
    neighbor_metric: str = "euclidean",
) -> dict[str, Any]:
    """Compare fixed train/held-out cluster labels without fitting on held-out data.

    Scores are computed independently per split. Invalid configurations are
    represented by ``None`` rather than zero; a silhouette difference is
    returned only if both split scores are mathematically defined.
    """
    train = _validate_embeddings(train_embeddings)
    heldout = _validate_query_embeddings(heldout_embeddings, train.shape[1])
    train_labels = _validate_labels(train_cluster_labels, len(train))
    heldout_labels = _validate_labels(heldout_cluster_labels, len(heldout))
    train_summary = _clustering_summary(train, train_labels)
    heldout_summary = _clustering_summary(heldout, heldout_labels)
    train_silhouette = train_summary["silhouette_score"]
    heldout_silhouette = heldout_summary["silhouette_score"]
    result: dict[str, Any] = {
        "train": train_summary,
        "heldout": heldout_summary,
        "silhouette_difference": (
            None
            if train_silhouette is None or heldout_silhouette is None
            else float(heldout_silhouette - train_silhouette)
        ),
        "heldout_placement": heldout_placement(
            train, heldout, k_neighbors, neighbor_metric=neighbor_metric
        ),
    }
    if (train_genres is None) != (heldout_genres is None):
        raise ValueError("Provide both train_genres and heldout_genres, or neither.")
    if train_genres is not None:
        result["heldout_placement"]["average_neighbor_genre_similarity"] = (
            heldout_neighbor_genre_similarity(
                train,
                heldout,
                train_genres,
                heldout_genres,
                k=k_neighbors,
                metric=neighbor_metric,
            )
        )
    return result

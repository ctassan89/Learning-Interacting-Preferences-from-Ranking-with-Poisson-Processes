"""Train-fitted UMAP visualizations and neighborhood-preservation evaluation.

UMAP is fitted only on selected-run training embeddings. Validation and held-out rows
are transformed with the same fitted mapper, never used during fitting.
"""

import argparse
import csv
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import trustworthiness
from sklearn.mixture import GaussianMixture

from experiments.dimensionality_reduction.postprocess import genre_matrix
from experiments.dimensionality_reduction.visualize_pca import (
    MOVIES_CLEAN_PATH,
    SPLITS,
    EmbeddingRun,
    genre_color_map,
    load_experiment,
    metadata_for_movie_ids,
)
from metrics.metrics import (
    average_neighbor_genre_similarity,
    calinski_harabasz_index,
    davies_bouldin_index,
    genre_purity,
    heldout_neighbor_genre_similarity,
    nearest_neighbors,
    nearest_training_neighbors,
    silhouette_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AE_A_PATH = PROJECT_ROOT / "results" / "dimensionality_reduction" / "AE_A"
RANDOM_STATE = 42
METRIC = "euclidean"
NEIGHBOR_K = 10
TRUSTWORTHINESS_SAMPLE_SIZE = 2_000
CLUSTER_COUNT = 10
CLUSTER_SILHOUETTE_SAMPLE_SIZE = 2_000


@dataclass(frozen=True)
class UmapConfiguration:
    """One reproducible UMAP hyperparameter setting."""

    n_components: int
    n_neighbors: int
    min_dist: float

    @property
    def label(self) -> str:
        return (
            f"umap_{self.n_components}d_neighbors_{self.n_neighbors}_"
            f"min_dist_{self.min_dist:g}"
        )


HYPERPARAMETERS = ((15, 0.1), (30, 0.1), (100, 0.3))


def configurations(n_components: tuple[int, ...] = (2, 3)) -> list[UmapConfiguration]:
    """Return each requested dimensionality paired with all fixed settings."""
    return [
        UmapConfiguration(dimension, neighbors, min_dist)
        for dimension in n_components
        for neighbors, min_dist in HYPERPARAMETERS
    ]


def fit_and_transform_umap(
    embeddings: dict[str, np.ndarray], configuration: UmapConfiguration
) -> tuple[Any, dict[str, np.ndarray]]:
    """Fit UMAP on training rows only, then transform validation and held-out rows."""
    try:
        import umap
    except ImportError as exc:
        raise ImportError("UMAP requires the umap-learn dependency.") from exc
    mapper = umap.UMAP(
        n_components=configuration.n_components,
        n_neighbors=configuration.n_neighbors,
        min_dist=configuration.min_dist,
        metric=METRIC,
        random_state=RANDOM_STATE,
    )
    coordinates = {"train": mapper.fit_transform(embeddings["train"])}
    for split in ("val", "heldout"):
        coordinates[split] = mapper.transform(embeddings[split])
    return mapper, coordinates


def _neighbor_overlap(reference: np.ndarray, projected: np.ndarray, k: int) -> float:
    """Return mean set overlap between k-neighbor lists in two spaces."""
    reference_indices, _ = nearest_neighbors(reference, k=k, metric=METRIC)
    projected_indices, _ = nearest_neighbors(projected, k=k, metric=METRIC)
    return float(
        np.mean(
            [
                len(set(left).intersection(right)) / k
                for left, right in zip(
                    reference_indices, projected_indices, strict=True
                )
            ]
        )
    )


def _heldout_neighbor_overlap(
    reference_train: np.ndarray,
    reference_heldout: np.ndarray,
    projected_train: np.ndarray,
    projected_heldout: np.ndarray,
    k: int,
) -> float:
    """Compare each held-out item's k training neighbours before and after UMAP."""
    reference_indices, _ = nearest_training_neighbors(
        reference_train, reference_heldout, k=k, metric=METRIC
    )
    projected_indices, _ = nearest_training_neighbors(
        projected_train, projected_heldout, k=k, metric=METRIC
    )
    return float(
        np.mean(
            [
                len(set(left).intersection(right)) / k
                for left, right in zip(
                    reference_indices, projected_indices, strict=True
                )
            ]
        )
    )


def evaluate_umap(
    run: EmbeddingRun,
    coordinates: dict[str, np.ndarray],
    *,
    sample_size: int = TRUSTWORTHINESS_SAMPLE_SIZE,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    """Evaluate structural preservation and genre-neighbor similarity.

    Trustworthiness and train neighbor overlap operate on a fixed random sample
    of training rows. Held-out overlap and held-out genre similarity compare
    each held-out row to the training set in both original and UMAP spaces.
    """
    train = run.embeddings["train"]
    heldout = run.embeddings["heldout"]
    projected_train = coordinates["train"]
    projected_heldout = coordinates["heldout"]
    sample_count = min(sample_size, len(train))
    sample_indices = np.random.default_rng(random_state).choice(
        len(train), size=sample_count, replace=False
    )
    evaluation_k = min(NEIGHBOR_K, sample_count - 1, len(heldout), len(train))
    if evaluation_k < 1:
        raise ValueError(
            "At least two training embeddings and one held-out embedding are required."
        )
    all_ids = np.concatenate([run.movie_ids["train"], run.movie_ids["heldout"]])
    genres = genre_matrix(all_ids)
    train_genres = genres[: len(train)]
    heldout_genres = genres[len(train) :]
    return {
        "metric": METRIC,
        "random_state": random_state,
        "neighbor_k": evaluation_k,
        "trustworthiness": {
            "value": float(
                trustworthiness(
                    train[sample_indices],
                    projected_train[sample_indices],
                    n_neighbors=evaluation_k,
                    metric=METRIC,
                )
            ),
            "scope": "training sample only",
            "sample_size": sample_count,
        },
        "neighbor_overlap": {
            "train": {
                "value": _neighbor_overlap(
                    train[sample_indices], projected_train[sample_indices], evaluation_k
                ),
                "scope": "training sample only",
                "sample_size": sample_count,
            },
            "heldout_to_train": {
                "value": _heldout_neighbor_overlap(
                    train, heldout, projected_train, projected_heldout, evaluation_k
                ),
                "scope": "all held-out rows queried against all training rows",
                "sample_size": int(len(heldout)),
            },
        },
        "genre_neighbor_jaccard": {
            "train_original": average_neighbor_genre_similarity(
                train[sample_indices],
                train_genres[sample_indices],
                k=evaluation_k,
                metric=METRIC,
            ),
            "train_umap": average_neighbor_genre_similarity(
                projected_train[sample_indices],
                train_genres[sample_indices],
                k=evaluation_k,
                metric=METRIC,
            ),
            "train_scope": "training sample only",
            "heldout_original": heldout_neighbor_genre_similarity(
                train,
                heldout,
                train_genres,
                heldout_genres,
                k=evaluation_k,
                metric=METRIC,
            ),
            "heldout_umap": heldout_neighbor_genre_similarity(
                projected_train,
                projected_heldout,
                train_genres,
                heldout_genres,
                k=evaluation_k,
                metric=METRIC,
            ),
            "heldout_scope": "all held-out rows queried against all training rows",
        },
        "interpretation": "Neighborhood preservation measures fidelity to the original embedding geometry; it does not establish improved semantic quality.",
    }


def _metric_or_reason(metric: Any, *args: Any) -> dict[str, Any]:
    """Return a JSON-safe metric value or its mathematical undefined reason."""
    try:
        return {"value": float(metric(*args)), "reason": None}
    except ValueError as exc:
        return {"value": None, "reason": str(exc)}


def _silhouette_with_fixed_sample(
    coordinates: np.ndarray,
    labels: np.ndarray,
    *,
    sample_size: int,
    random_state: int,
) -> dict[str, Any]:
    """Evaluate silhouette on a reproducible subset without changing labels."""
    count = min(sample_size, len(coordinates))
    indices = np.random.default_rng(random_state).choice(
        len(coordinates), size=count, replace=False
    )
    result = _metric_or_reason(silhouette_score, coordinates[indices], labels[indices])
    result.update({"sample_size": int(count), "random_state": random_state})
    return result


def _cluster_metrics(
    coordinates: np.ndarray,
    labels: np.ndarray,
    genres: np.ndarray,
    *,
    random_state: int,
) -> dict[str, Any]:
    """Score a train-fitted clustering on one split of native UMAP coordinates."""
    result = {
        "n_observations": int(len(coordinates)),
        "occupied_clusters": int(len(np.unique(labels))),
        "silhouette": _silhouette_with_fixed_sample(
            coordinates,
            labels,
            sample_size=CLUSTER_SILHOUETTE_SAMPLE_SIZE,
            random_state=random_state,
        ),
        "davies_bouldin": _metric_or_reason(davies_bouldin_index, coordinates, labels),
        "calinski_harabasz": _metric_or_reason(
            calinski_harabasz_index, coordinates, labels
        ),
        "genre_purity": _metric_or_reason(genre_purity, labels, genres),
    }
    return result


def _genre_names(movie_ids: np.ndarray) -> list[str]:
    """Return the vocabulary in exactly the order used by ``genre_matrix``."""
    import pandas as pd

    metadata = pd.read_csv(MOVIES_CLEAN_PATH, usecols=["movieId", "genres"])
    by_id = metadata.set_index("movieId")["genres"]
    values = []
    for movie_id in movie_ids:
        labels = (
            [] if pd.isna(by_id.loc[movie_id]) else str(by_id.loc[movie_id]).split("|")
        )
        values.extend(label for label in labels if label)
    return sorted(set(values))


def _cluster_details(
    labels: np.ndarray,
    genres: np.ndarray,
    genre_names: list[str],
    split: str,
    method: str,
) -> list[dict[str, Any]]:
    """Describe occupied clusters while retaining multi-label genre proportions."""
    rows = []
    for cluster_id in sorted(int(value) for value in np.unique(labels)):
        members = genres[labels == cluster_id]
        proportions = members.mean(axis=0)
        dominant_index = int(np.argmax(proportions))
        distribution = {
            genre_names[index]: float(value)
            for index, value in enumerate(proportions)
            if value > 0
        }
        rows.append(
            {
                "method": method,
                "split": split,
                "cluster_id": cluster_id,
                "sample_count": int(len(members)),
                "dominant_genre": genre_names[dominant_index],
                "dominant_genre_proportion": float(proportions[dominant_index]),
                "genre_distribution": json.dumps(distribution, sort_keys=True),
            }
        )
    return rows


def evaluate_umap_clusters(
    run: EmbeddingRun,
    coordinates: dict[str, np.ndarray],
    *,
    random_state: int = RANDOM_STATE,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    """Evaluate train-fitted K-Means/GMM on 2D UMAP train and held-out rows.

    The validation split is intentionally absent: neither UMAP fitting nor any
    clustering, semantic, or projection metric below uses it.
    """
    train, heldout = coordinates["train"], coordinates["heldout"]
    if (
        train.ndim != 2
        or heldout.ndim != 2
        or train.shape[1] != 2
        or heldout.shape[1] != 2
    ):
        raise ValueError(
            "Cluster evaluation requires native two-dimensional UMAP coordinates."
        )
    if len(train) < CLUSTER_COUNT:
        raise ValueError(
            f"K-Means and GMM require at least {CLUSTER_COUNT} training rows."
        )
    if len(heldout) < 1:
        raise ValueError("At least one held-out row is required for evaluation.")
    movie_ids = np.concatenate([run.movie_ids["train"], run.movie_ids["heldout"]])
    genres = genre_matrix(movie_ids)
    train_genres, heldout_genres = genres[: len(train)], genres[len(train) :]
    genre_names = _genre_names(movie_ids)
    if len(genre_names) != genres.shape[1]:
        raise ValueError(
            "Genre vocabulary is inconsistent with the aligned genre matrix."
        )
    kmeans = KMeans(n_clusters=CLUSTER_COUNT, n_init=10, random_state=random_state).fit(
        train
    )
    gmm = GaussianMixture(n_components=CLUSTER_COUNT, random_state=random_state).fit(
        train
    )
    models = {
        "kmeans": (kmeans, kmeans.labels_, kmeans.predict(heldout)),
        "gmm": (gmm, gmm.predict(train), gmm.predict(heldout)),
    }
    methods: dict[str, Any] = {}
    details: dict[str, list[dict[str, Any]]] = {}
    for name, (_model, train_labels, heldout_labels) in models.items():
        methods[name] = {
            "fit_split": "train",
            "parameters": {
                "n_clusters": CLUSTER_COUNT,
                "random_state": random_state,
                **(
                    {"n_init": 10}
                    if name == "kmeans"
                    else {"n_components": CLUSTER_COUNT}
                ),
            },
            "train": _cluster_metrics(
                train, train_labels, train_genres, random_state=random_state
            ),
            "heldout": _cluster_metrics(
                heldout, heldout_labels, heldout_genres, random_state=random_state
            ),
        }
        details[name] = _cluster_details(
            train_labels, train_genres, genre_names, "train", name
        ) + _cluster_details(
            heldout_labels, heldout_genres, genre_names, "heldout", name
        )
    neighbor_k = min(NEIGHBOR_K, len(train) - 1)
    fidelity = evaluate_umap(run, coordinates, random_state=random_state)
    report = {
        "status": "success",
        "coordinate_space": "native 2D UMAP coordinates",
        "excluded_split": "validation",
        "projection_fidelity": {
            "training_trustworthiness": fidelity["trustworthiness"],
            "training_neighbor_overlap": fidelity["neighbor_overlap"]["train"],
            "heldout_to_training_neighbor_overlap": fidelity["neighbor_overlap"][
                "heldout_to_train"
            ],
            "interpretation": fidelity["interpretation"],
        },
        "semantic_neighbor_metrics": {
            "metric": METRIC,
            "neighbor_k": neighbor_k,
            "train_to_train_excluding_self": average_neighbor_genre_similarity(
                train, train_genres, k=neighbor_k, metric=METRIC
            ),
            "heldout_to_train": heldout_neighbor_genre_similarity(
                train,
                heldout,
                train_genres,
                heldout_genres,
                k=neighbor_k,
                metric=METRIC,
            ),
        },
        "clustering": methods,
    }
    return report, details


def _metric_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten numeric/null train and held-out metrics for a portable CSV."""
    rows = []
    for method, values in report["clustering"].items():
        for split in ("train", "heldout"):
            metrics = values[split]
            for name in (
                "silhouette",
                "davies_bouldin",
                "calinski_harabasz",
                "genre_purity",
            ):
                entry = metrics[name]
                rows.append(
                    {
                        "category": "clustering",
                        "method": method,
                        "split": split,
                        "metric": name,
                        "value": entry["value"],
                        "reason": entry["reason"],
                        "sample_size": entry.get("sample_size"),
                        "random_state": entry.get("random_state"),
                    }
                )
            rows.append(
                {
                    "category": "clustering",
                    "method": method,
                    "split": split,
                    "metric": "occupied_clusters",
                    "value": metrics["occupied_clusters"],
                    "reason": None,
                    "sample_size": None,
                    "random_state": None,
                }
            )
    semantic = report["semantic_neighbor_metrics"]
    for split, key in (
        ("train", "train_to_train_excluding_self"),
        ("heldout", "heldout_to_train"),
    ):
        rows.append(
            {
                "category": "semantic",
                "method": None,
                "split": split,
                "metric": "genre_jaccard_at_10",
                "value": semantic[key],
                "reason": None,
                "sample_size": None,
                "random_state": None,
            }
        )
    return rows


def write_cluster_evaluation(
    run: EmbeddingRun,
    configuration: UmapConfiguration,
    *,
    force: bool = False,
) -> Path:
    """Write opt-in clustering evaluation without changing visualization files."""
    if configuration.n_components != 2:
        raise ValueError(
            "Cluster evaluation is defined only for 2D UMAP configurations."
        )
    destination = (
        run.path / "umap_visualization" / configuration.label / "cluster_evaluation"
    )
    if destination.exists() and not force:
        raise FileExistsError(
            f"{destination} already exists; use --force to replace it."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    _mapper, coordinates = fit_and_transform_umap(run.embeddings, configuration)
    report, details = evaluate_umap_clusters(run, coordinates)
    temporary = Path(
        tempfile.mkdtemp(prefix=".cluster-evaluation-", dir=destination.parent)
    )
    try:
        (temporary / "train_test_metrics.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        with (temporary / "train_test_metrics.csv").open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(_metric_rows(report)[0]))
            writer.writeheader()
            writer.writerows(_metric_rows(report))
        fields = [
            "method",
            "split",
            "cluster_id",
            "sample_count",
            "dominant_genre",
            "dominant_genre_proportion",
            "genre_distribution",
        ]
        for method, rows in details.items():
            with (temporary / f"{method}_cluster_details.csv").open(
                "w", newline=""
            ) as output:
                writer = csv.DictWriter(output, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
        if destination.exists():
            shutil.rmtree(destination)
        temporary.replace(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def _display_data(
    run: EmbeddingRun, coordinates: dict[str, np.ndarray], splits: tuple[str, ...]
) -> Any:
    import pandas as pd

    frames = []
    for split in splits:
        metadata = metadata_for_movie_ids(run.movie_ids[split], MOVIES_CLEAN_PATH)
        values = coordinates[split]
        frame = metadata.assign(split=split, UMAP1=values[:, 0], UMAP2=values[:, 1])
        if values.shape[1] == 3:
            frame = frame.assign(UMAP3=values[:, 2])
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def experiment_label(experiment_path: Path) -> str:
    """Return a concise chart label while preserving the baseline's historic name."""
    if experiment_path.name == "AE_A":
        return "AE-A baseline"
    if experiment_path.name == "AE_A_large":
        return "AE-A Large"
    return experiment_path.name.replace("_", " ")


def _plot_title(configuration: UmapConfiguration, label: str) -> str:
    return (
        f"{label} UMAP ({configuration.n_components}D; "
        f"n_neighbors={configuration.n_neighbors}, min_dist={configuration.min_dist:g})"
    )


def _save_plots(
    data: Any, configuration: UmapConfiguration, destination: Path, label: str
) -> None:
    """Save required interactive plots and the static 2D view when applicable."""
    try:
        import plotly.express as px
    except ImportError as exc:
        raise ImportError(
            "Plotly is required for interactive UMAP visualizations."
        ) from exc
    colors = genre_color_map(MOVIES_CLEAN_PATH)
    hover = {
        "movieId": True,
        "genres": True,
        "split": True,
        "UMAP1": ":.4f",
        "UMAP2": ":.4f",
        "primary_genre": False,
    }
    title = _plot_title(configuration, label)
    if configuration.n_components == 2:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(10, 8))
        for genre in sorted(data["primary_genre"].unique()):
            subset = data[data["primary_genre"] == genre]
            axis.scatter(
                subset["UMAP1"],
                subset["UMAP2"],
                s=12,
                alpha=0.7,
                color=colors[genre],
                label=genre,
                linewidths=0,
            )
        axis.set(xlabel="UMAP1", ylabel="UMAP2", title=title)
        axis.legend(title="Primary genre", bbox_to_anchor=(1.02, 1), loc="upper left")
        figure.savefig(destination / "umap_2d.png", dpi=180, bbox_inches="tight")
        plt.close(figure)
        chart = px.scatter(
            data,
            x="UMAP1",
            y="UMAP2",
            color="primary_genre",
            symbol="split",
            color_discrete_map=colors,
            hover_name="title",
            hover_data=hover,
            title=title,
        )
        chart.write_html(
            destination / "umap_2d.html", include_plotlyjs="cdn", full_html=True
        )
    else:
        hover["UMAP3"] = ":.4f"
        chart = px.scatter_3d(
            data,
            x="UMAP1",
            y="UMAP2",
            z="UMAP3",
            color="primary_genre",
            symbol="split",
            color_discrete_map=colors,
            hover_name="title",
            hover_data=hover,
            title=title,
        )
        chart.update_traces(marker={"size": 3, "opacity": 0.75})
        chart.write_html(
            destination / "umap_3d.html", include_plotlyjs="cdn", full_html=True
        )


def output_complete(destination: Path, n_components: int) -> bool:
    """Return whether metrics and all plots required by this setting exist."""
    files = {"metrics.json", "umap_2d.png", "umap_2d.html"}
    if n_components == 3:
        files = {"metrics.json", "umap_3d.html"}
    return all((destination / name).is_file() for name in files)


def run_configuration(
    run: EmbeddingRun,
    configuration: UmapConfiguration,
    *,
    splits: tuple[str, ...] = ("train",),
    force: bool = False,
    output_root: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Create one UMAP setting atomically without changing training artifacts."""
    output_root = output_root or run.path / "umap_visualization"
    destination = output_root / configuration.label
    if destination.exists() and not force:
        raise FileExistsError(
            f"{destination} already exists; use --force to replace it."
        )
    output_root.mkdir(parents=True, exist_ok=True)
    _, coordinates = fit_and_transform_umap(run.embeddings, configuration)
    report = {
        "configuration": {
            "n_components": configuration.n_components,
            "n_neighbors": configuration.n_neighbors,
            "min_dist": configuration.min_dist,
            "metric": METRIC,
            "random_state": RANDOM_STATE,
            "fit_split": "train",
            "transform_splits": ["val", "heldout"],
        },
        "evaluation": evaluate_umap(run, coordinates),
    }
    data = _display_data(run, coordinates, splits)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{configuration.label}-", dir=output_root)
    )
    try:
        _save_plots(data, configuration, temporary, experiment_label(run.path))
        (temporary / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
        if destination.exists():
            shutil.rmtree(destination)
        temporary.replace(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination, report


def write_comparison(
    reports: list[tuple[UmapConfiguration, dict[str, Any]],], output_root: Path
) -> Path:
    """Write one flat comparison table for settings completed in this invocation."""
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / "comparison.csv"
    rows = []
    for config, report in reports:
        metrics = report["evaluation"]
        jaccard = metrics["genre_neighbor_jaccard"]
        rows.append(
            {
                "n_components": config.n_components,
                "n_neighbors": config.n_neighbors,
                "min_dist": config.min_dist,
                "metric": METRIC,
                "random_state": RANDOM_STATE,
                "trustworthiness_train_sample": metrics["trustworthiness"]["value"],
                "trustworthiness_sample_size": metrics["trustworthiness"][
                    "sample_size"
                ],
                "neighbor_overlap_train_sample": metrics["neighbor_overlap"]["train"][
                    "value"
                ],
                "neighbor_overlap_heldout_to_train": metrics["neighbor_overlap"][
                    "heldout_to_train"
                ]["value"],
                "genre_jaccard_train_original": jaccard["train_original"],
                "genre_jaccard_train_umap": jaccard["train_umap"],
                "genre_jaccard_heldout_original": jaccard["heldout_original"],
                "genre_jaccard_heldout_umap": jaccard["heldout_umap"],
            }
        )
    with destination.open("w", newline="") as output:
        writer = csv.DictWriter(
            output, fieldnames=list(rows[0]) if rows else ["n_components"]
        )
        writer.writeheader()
        writer.writerows(rows)
    return destination


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all", action="store_true", help="Run all six configurations."
    )
    parser.add_argument(
        "--n-components", type=int, nargs="+", choices=(2, 3), default=(2, 3)
    )
    parser.add_argument(
        "--experiment",
        type=Path,
        default=AE_A_PATH,
        help="Embedding experiment directory (default: results/dimensionality_reduction/AE_A).",
    )
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=("train",))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--evaluate-clusters",
        action="store_true",
        help="Opt in to train-fitted clustering/semantic evaluation for one 2D setting.",
    )
    parser.add_argument("--n-neighbors", type=int, choices=(15, 30, 100))
    parser.add_argument("--min-dist", type=float, choices=(0.1, 0.3))
    return parser.parse_args()


def _selected_cluster_configuration(args: argparse.Namespace) -> UmapConfiguration:
    """Resolve the one explicit 2D setting allowed by ``--evaluate-clusters``."""
    if len(args.n_components) != 1 or args.n_components[0] != 2:
        raise ValueError("--evaluate-clusters requires exactly --n-components 2.")
    if (args.n_neighbors is None) != (args.min_dist is None):
        raise ValueError("--n-neighbors and --min-dist must be provided together.")
    neighbors = 100 if args.n_neighbors is None else args.n_neighbors
    min_dist = 0.3 if args.min_dist is None else args.min_dist
    configuration = UmapConfiguration(2, neighbors, min_dist)
    if (configuration.n_neighbors, configuration.min_dist) not in HYPERPARAMETERS:
        raise ValueError(
            "Requested UMAP hyperparameters are not a supported configuration."
        )
    return configuration


def main() -> None:
    args = _arguments()
    experiment_path = args.experiment.resolve()
    output_root = experiment_path / "umap_visualization"
    if args.evaluate_clusters:
        try:
            configuration = _selected_cluster_configuration(args)
            destination = output_root / configuration.label / "cluster_evaluation"
            if args.dry_run:
                print(f"would evaluate: {destination}")
                return
            if args.skip_completed and all(
                (destination / name).is_file()
                for name in (
                    "train_test_metrics.json",
                    "train_test_metrics.csv",
                    "kmeans_cluster_details.csv",
                    "gmm_cluster_details.csv",
                )
            ):
                print(f"skipped: {destination}")
                return
            run = load_experiment(experiment_path)
            write_cluster_evaluation(run, configuration, force=args.force)
            print(f"success: {destination}")
            return
        except Exception as exc:
            raise SystemExit(f"failed: {exc}") from exc
    selected = configurations(tuple(args.n_components))
    if args.dry_run:
        for configuration in selected:
            print(f"would run: {output_root / configuration.label}")
        return
    try:
        run = load_experiment(experiment_path)
        reports = []
        for configuration in selected:
            destination = output_root / configuration.label
            if args.skip_completed and output_complete(
                destination, configuration.n_components
            ):
                print(f"skipped: {destination}")
                continue
            _, report = run_configuration(
                run,
                configuration,
                splits=tuple(args.splits),
                force=args.force,
                output_root=output_root,
            )
            reports.append((configuration, report))
            print(f"success: {destination}")
        if reports:
            write_comparison(reports, output_root)
    except Exception as exc:
        raise SystemExit(f"failed: {exc}") from exc


if __name__ == "__main__":
    main()

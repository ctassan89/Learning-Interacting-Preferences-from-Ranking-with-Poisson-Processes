"""Unified, safe post-processing for AE, VAE, and Beta-VAE embedding runs."""

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
from sklearn.mixture import GaussianMixture

from metrics.metrics import (
    average_neighbor_genre_similarity,
    calinski_harabasz_index,
    davies_bouldin_index,
    fit_pca_projection,
    fit_hdbscan,
    genre_purity,
    heldout_neighbor_genre_similarity,
    heldout_placement,
    project_embeddings,
    silhouette_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"
REQUIRED_ARTIFACTS = {
    "model.pt",
    "config.json",
    "history.csv",
    "embeddings_train.npz",
    "embeddings_val.npz",
    "embeddings_heldout.npz",
    "feature_builder.joblib",
}
K_CLUSTERS = 10
K_NEIGHBORS = 10
RANDOM_STATE = 42


@dataclass
class ExperimentRun:
    """Validated immutable training artifacts for one experiment directory."""

    path: Path
    config: dict[str, Any]
    embeddings: dict[str, np.ndarray]
    movie_ids: dict[str, np.ndarray]


def discover_experiments(
    results_root: Path = RESULTS_ROOT,
) -> tuple[list[Path], list[str]]:
    """Recursively find complete experiment directories, including nested Beta-VAE runs."""
    valid, invalid = [], []
    if not results_root.exists():
        return valid, invalid
    for config_path in results_root.rglob("config.json"):
        run_path = config_path.parent
        present = {path.name for path in run_path.iterdir() if path.is_file()}
        if REQUIRED_ARTIFACTS.issubset(present):
            valid.append(run_path)
        elif any(name.startswith("embeddings_") for name in present):
            invalid.append(f"{run_path}: incomplete training artifacts")
    return sorted(valid), sorted(invalid)


def _load_embedding_archive(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as archive:
        if set(archive.files) != {"embeddings", "movieId"}:
            raise ValueError(f"{path}: expected embeddings and movieId arrays.")
        embeddings, movie_ids = archive["embeddings"], archive["movieId"]
    if embeddings.ndim != 2 or movie_ids.ndim != 1 or len(embeddings) != len(movie_ids):
        raise ValueError(f"{path}: embeddings and movieId arrays are misaligned.")
    if not np.isfinite(embeddings).all() or len(np.unique(movie_ids)) != len(movie_ids):
        raise ValueError(f"{path}: non-finite embeddings or duplicate movie IDs.")
    return embeddings, movie_ids


def load_experiment(path: Path) -> ExperimentRun:
    """Load and validate artifacts without modifying a training run."""
    path = path.resolve()
    missing = [name for name in REQUIRED_ARTIFACTS if not (path / name).is_file()]
    if missing:
        raise ValueError(
            f"{path}: missing required artifacts: {', '.join(sorted(missing))}"
        )
    config = json.loads((path / "config.json").read_text())
    for key in ("experiment", "feature_groups", "input_dim"):
        if key not in config:
            raise ValueError(f"{path}: config.json is missing {key!r}.")
    embeddings, movie_ids = {}, {}
    for split in ("train", "val", "heldout"):
        embeddings[split], movie_ids[split] = _load_embedding_archive(
            path / f"embeddings_{split}.npz"
        )
    dimensions = {values.shape[1] for values in embeddings.values()}
    if len(dimensions) != 1:
        raise ValueError(f"{path}: embedding dimensions disagree across splits.")
    latent_dim = dimensions.pop()
    if "latent_dim" in config and config["latent_dim"] != latent_dim:
        raise ValueError(f"{path}: embedding dimensions disagree with config.json.")
    if "latent_dim" not in config:
        config["latent_dim"] = int(latent_dim)
        config["metadata_warning"] = "latent_dim inferred from embedding archives"
    combined_ids = np.concatenate(list(movie_ids.values()))
    if len(np.unique(combined_ids)) != len(combined_ids):
        raise ValueError(f"{path}: train, validation, and held-out IDs overlap.")
    if "BetaVAE" in str(config["experiment"]):
        beta = config.get("beta")
        if not isinstance(beta, (int, float)) or not np.isfinite(beta) or beta < 0:
            raise ValueError(f"{path}: BetaVAE config has an invalid beta value.")
    return ExperimentRun(path, config, embeddings, movie_ids)


def genre_matrix(
    movie_ids: np.ndarray,
    movies_clean_path: Path | None = None,
    catalogue_path: Path | None = None,
) -> np.ndarray:
    """Align multi-label pipe-separated genres to embedding movie IDs by key."""
    import pandas as pd

    movies_clean_path = (
        movies_clean_path or PROJECT_ROOT / "data" / "processed" / "movies_clean.csv"
    )
    catalogue_path = (
        catalogue_path or PROJECT_ROOT / "data" / "processed" / "catalogue.csv"
    )
    metadata = pd.read_csv(movies_clean_path, usecols=["movieId", "genres"])
    catalogue = pd.read_csv(catalogue_path, usecols=["movieId"])
    if metadata["movieId"].duplicated().any():
        raise ValueError("movies_clean.csv has duplicate movie IDs.")
    if catalogue["movieId"].duplicated().any():
        raise ValueError("catalogue.csv has duplicate movie IDs.")
    by_id = metadata.set_index("movieId")["genres"]
    missing = (
        set(movie_ids)
        .difference(by_id.index)
        .union(set(movie_ids).difference(catalogue["movieId"]))
    )
    if missing:
        raise ValueError("Embedding movie IDs are missing from movies_clean.csv.")
    genre_lists = [
        (
            [genre for genre in str(by_id.loc[movie_id]).split("|") if genre]
            if pd.notna(by_id.loc[movie_id])
            else []
        )
        for movie_id in movie_ids
    ]
    vocabulary = sorted({genre for values in genre_lists for genre in values})
    positions = {genre: index for index, genre in enumerate(vocabulary)}
    matrix = np.zeros((len(movie_ids), len(vocabulary)), dtype=int)
    for row, values in enumerate(genre_lists):
        for genre in values:
            matrix[row, positions[genre]] = 1
    return matrix


def _score_cluster(
    embeddings: np.ndarray, labels: np.ndarray, genres: np.ndarray
) -> dict[str, Any]:
    """Return defined clustering metrics and null plus reason for undefined metrics."""
    result: dict[str, Any] = {
        "n_observations": int(len(labels)),
        "n_noise": int((labels == -1).sum()),
    }
    for name, metric in (
        ("silhouette", silhouette_score),
        ("davies_bouldin", davies_bouldin_index),
        ("calinski_harabasz", calinski_harabasz_index),
        ("genre_purity", genre_purity),
    ):
        try:
            result[name] = float(
                metric(labels, genres)
                if name == "genre_purity"
                else metric(embeddings, labels)
            )
        except ValueError as exc:
            result[name] = None
            result[f"{name}_reason"] = str(exc)
    return result


def evaluate_run(run: ExperimentRun, *, use_hdbscan: bool = True) -> dict[str, Any]:
    """Evaluate original train embeddings and train-fitted held-out placement."""
    train, heldout = run.embeddings["train"], run.embeddings["heldout"]
    all_ids = np.concatenate([run.movie_ids["train"], run.movie_ids["heldout"]])
    all_genres = genre_matrix(all_ids)
    train_genres = all_genres[: len(train)]
    heldout_genres = all_genres[len(train) :]
    if len(train) < 2:
        raise ValueError(
            "At least two training embeddings are required for clustering."
        )
    k = min(K_CLUSTERS, len(train))
    neighbors = min(K_NEIGHBORS, len(train) - 1)
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(train)
    gmm = GaussianMixture(n_components=k, random_state=RANDOM_STATE).fit(train)
    methods = {
        "kmeans": _score_cluster(train, kmeans.labels_, train_genres),
        "gmm": _score_cluster(train, gmm.predict(train), train_genres),
    }
    if use_hdbscan:
        try:
            labels = fit_hdbscan(train, min_cluster_size=5)
            methods["hdbscan"] = _score_cluster(train, labels, train_genres)
            methods["hdbscan"]["n_clusters"] = int(
                len(set(labels)) - (1 if -1 in labels else 0)
            )
        except (ImportError, ValueError) as exc:
            methods["hdbscan"] = {"available": False, "reason": str(exc)}
    semantic = {
        "train_observations": int(len(train)),
        "heldout_observations": int(len(heldout)),
        "neighbors_per_observation": neighbors,
        "train_cosine_genre_jaccard": average_neighbor_genre_similarity(
            train, train_genres, k=neighbors, metric="cosine"
        ),
        "train_euclidean_genre_jaccard": average_neighbor_genre_similarity(
            train, train_genres, k=neighbors, metric="euclidean"
        ),
        "heldout_cosine_genre_jaccard": heldout_neighbor_genre_similarity(
            train, heldout, train_genres, heldout_genres, k=neighbors, metric="cosine"
        ),
        "heldout_euclidean_genre_jaccard": heldout_neighbor_genre_similarity(
            train,
            heldout,
            train_genres,
            heldout_genres,
            k=neighbors,
            metric="euclidean",
        ),
    }
    placement = {
        "kmeans": heldout_placement(
            train,
            heldout,
            k_neighbors=neighbors,
            clustering_model=kmeans,
            train_genres=train_genres,
            heldout_genres=heldout_genres,
        ),
        "gmm": heldout_placement(
            train,
            heldout,
            k_neighbors=neighbors,
            clustering_model=gmm,
            train_genres=train_genres,
            heldout_genres=heldout_genres,
        ),
    }
    return {
        "status": "success",
        "config": run.config,
        "clustering": methods,
        "semantic": semantic,
        "parameters": {
            "k_clusters": k,
            "k_neighbors": neighbors,
            "random_state": RANDOM_STATE,
        },
        "heldout_placement": placement,
        "reconstruction": {
            "available": False,
            "reason": "Not computed: post-processing does not refit or transform features.",
        },
    }


def _save_pca_plot(run: ExperimentRun, destination: Path) -> None:
    """Save a 2D PCA plot fit only on training embeddings."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pca, train = fit_pca_projection(run.embeddings["train"], n_components=2)
    plt.figure(figsize=(7, 6))
    for split, values in (
        ("train", train),
        ("val", pca.transform(run.embeddings["val"])),
        ("heldout", pca.transform(run.embeddings["heldout"])),
    ):
        plt.scatter(values[:, 0], values[:, 1], s=5, alpha=0.5, label=split)
    plt.legend()
    plt.title("PCA fitted on training embeddings")
    plt.savefig(destination / "pca.png", dpi=150, bbox_inches="tight")
    plt.close()


def _save_tsne_plot(run: ExperimentRun, destination: Path) -> None:
    """Save a reproducible training-only t-SNE plot (sklearn t-SNE has no transform)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train = run.embeddings["train"]
    coordinates = project_embeddings(
        train,
        method="tsne",
        n_components=2,
        random_state=RANDOM_STATE,
        perplexity=min(30.0, float(len(train) - 1)),
    )
    plt.figure(figsize=(7, 6))
    plt.scatter(coordinates[:, 0], coordinates[:, 1], s=5, alpha=0.5)
    plt.title("t-SNE on training embeddings only")
    plt.savefig(destination / "tsne.png", dpi=150, bbox_inches="tight")
    plt.close()


def _save_umap_plot(run: ExperimentRun, destination: Path) -> bool:
    """Save UMAP fit on train and transformed for validation/held-out when available."""
    try:
        import umap
    except ImportError:
        return False
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model = umap.UMAP(n_components=2, random_state=RANDOM_STATE)
    train = model.fit_transform(run.embeddings["train"])
    plt.figure(figsize=(7, 6))
    for split, values in (
        ("train", train),
        ("val", model.transform(run.embeddings["val"])),
        ("heldout", model.transform(run.embeddings["heldout"])),
    ):
        plt.scatter(values[:, 0], values[:, 1], s=5, alpha=0.5, label=split)
    plt.legend()
    plt.title("UMAP fitted on training embeddings")
    plt.savefig(destination / "umap.png", dpi=150, bbox_inches="tight")
    plt.close()
    return True


def write_postprocessing(
    run: ExperimentRun,
    report: dict[str, Any],
    *,
    force: bool = False,
    no_plots: bool = False,
    no_tsne: bool = False,
    no_umap: bool = False,
) -> Path:
    """Atomically write only post-processing outputs, never training artifacts."""
    destination = run.path / "postprocessing"
    if destination.exists() and not force:
        raise FileExistsError(
            f"{destination} already exists; use --force to replace it."
        )
    temporary = Path(tempfile.mkdtemp(prefix=".postprocessing-", dir=run.path))
    try:
        (temporary / "metrics.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        with (temporary / "clustering.csv").open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=["method", "metric", "value"])
            writer.writeheader()
            for method, metrics in report["clustering"].items():
                for metric, value in metrics.items():
                    if isinstance(value, (int, float)) or value is None:
                        writer.writerow(
                            {"method": method, "metric": metric, "value": value}
                        )
        with (temporary / "neighbors.csv").open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=["metric", "value"])
            writer.writeheader()
            writer.writerows(
                {"metric": key, "value": value}
                for key, value in report["semantic"].items()
            )
        if not no_plots:
            _save_pca_plot(run, temporary)
            if not no_tsne:
                _save_tsne_plot(run, temporary)
            if not no_umap:
                _save_umap_plot(run, temporary)
        if destination.exists():
            shutil.rmtree(destination)
        temporary.replace(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def _flatten_summary(run: ExperimentRun, report: dict[str, Any]) -> dict[str, Any]:
    config = run.config
    row = {
        "experiment": config["experiment"],
        "input_dim": config["input_dim"],
        "latent_dim": config["latent_dim"],
        "beta": config.get("beta"),
        "status": report["status"],
    }
    for method in ("kmeans", "gmm", "hdbscan"):
        for key, value in report["clustering"].get(method, {}).items():
            if isinstance(value, (int, float)) or value is None:
                row[f"{method}_{key}"] = value
    row.update(report["semantic"])
    return row


def postprocessing_complete(path: Path) -> bool:
    """Return whether core post-processing outputs are present and parseable."""
    required = ("metrics.json", "clustering.csv", "neighbors.csv")
    if not all((path / name).is_file() for name in required):
        return False
    try:
        return (
            json.loads((path / "metrics.json").read_text()).get("status") == "success"
        )
    except (OSError, json.JSONDecodeError):
        return False


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--experiment", type=Path)
    group.add_argument("--all", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-hdbscan", action="store_true")
    parser.add_argument("--no-tsne", action="store_true")
    parser.add_argument("--no-umap", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run requested evaluations; dry-run performs discovery only."""
    args = _arguments()
    paths, invalid = discover_experiments() if args.all else ([args.experiment], [])
    failures, summaries = list(invalid), []
    for path in paths:
        try:
            run = load_experiment(path)
            output = run.path / "postprocessing"
            if args.skip_completed and postprocessing_complete(output):
                print(f"skipped: {run.path}")
                continue
            if args.dry_run:
                print(f"would evaluate: {run.path}")
                continue
            report = evaluate_run(run, use_hdbscan=not args.no_hdbscan)
            write_postprocessing(
                run,
                report,
                force=args.force,
                no_plots=args.no_plots,
                no_tsne=args.no_tsne,
                no_umap=args.no_umap,
            )
            summaries.append(_flatten_summary(run, report))
            print(f"success: {run.path}")
        except Exception as exc:
            failures.append(f"{path}: {exc}")
            print(f"failed: {path}: {exc}")
    if summaries:
        columns = sorted({key for row in summaries for key in row})
        with (RESULTS_ROOT / "postprocessing_summary.csv").open(
            "w", newline=""
        ) as output:
            writer = csv.DictWriter(output, fieldnames=columns)
            writer.writeheader()
            writer.writerows(summaries)
    if failures:
        raise SystemExit(
            f"{len(failures)} invalid or failed run(s): " + "; ".join(failures)
        )


if __name__ == "__main__":
    main()

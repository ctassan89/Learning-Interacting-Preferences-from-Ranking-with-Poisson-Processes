"""Create train-fitted PCA visualizations for saved movie embedding experiments.

The original embedding archives are read only.  PCA is fitted on the training
split and that same model transforms any requested validation or held-out rows.
"""

import argparse
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"
MOVIES_CLEAN_PATH = PROJECT_ROOT / "data" / "processed" / "movies_clean.csv"
SPLITS = ("train", "val", "heldout")
OUTPUT_FILES = ("pca_2d.png", "pca_3d.html", "explained_variance.json")
UNKNOWN_GENRE = "Unknown"

# Tableau's categorical palette is stable and has enough distinct colours for
# the MovieLens genre vocabulary.  Colours are assigned to sorted genre names.
GENRE_COLORS = (
    "#4E79A7",
    "#F28E2B",
    "#E15759",
    "#76B7B2",
    "#59A14F",
    "#EDC948",
    "#B07AA1",
    "#FF9DA7",
    "#9C755F",
    "#BAB0AC",
    "#1F77B4",
    "#FF7F0E",
    "#2CA02C",
    "#D62728",
    "#9467BD",
    "#8C564B",
    "#E377C2",
    "#7F7F7F",
    "#BCBD22",
    "#17BECF",
)


@dataclass(frozen=True)
class EmbeddingRun:
    """Embedding data loaded from one experiment directory."""

    path: Path
    embeddings: dict[str, np.ndarray]
    movie_ids: dict[str, np.ndarray]


def load_embedding_archive(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load and validate one ``embeddings_<split>.npz`` archive."""
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"embeddings", "movieId"}:
            raise ValueError(f"{path}: expected embeddings and movieId arrays.")
        embeddings = archive["embeddings"]
        movie_ids = archive["movieId"]
    if embeddings.ndim != 2 or movie_ids.ndim != 1:
        raise ValueError(f"{path}: embeddings must be 2D and movieId must be 1D.")
    if len(embeddings) != len(movie_ids):
        raise ValueError(f"{path}: embeddings and movieId arrays are misaligned.")
    if not np.isfinite(embeddings).all() or len(np.unique(movie_ids)) != len(movie_ids):
        raise ValueError(f"{path}: non-finite embeddings or duplicate movie IDs.")
    return embeddings, movie_ids


def load_experiment(path: Path) -> EmbeddingRun:
    """Load the three saved splits without reading model or training artifacts."""
    path = path.resolve()
    embeddings, movie_ids = {}, {}
    for split in SPLITS:
        archive_path = path / f"embeddings_{split}.npz"
        if not archive_path.is_file():
            raise ValueError(f"{path}: missing {archive_path.name}.")
        embeddings[split], movie_ids[split] = load_embedding_archive(archive_path)
    dimensions = {values.shape[1] for values in embeddings.values()}
    if len(dimensions) != 1:
        raise ValueError(f"{path}: embedding dimensions disagree across splits.")
    all_ids = np.concatenate([movie_ids[split] for split in SPLITS])
    if len(np.unique(all_ids)) != len(all_ids):
        raise ValueError(f"{path}: movie IDs overlap across splits.")
    if len(embeddings["train"]) < 3:
        raise ValueError(f"{path}: at least three training rows are required for PCA.")
    return EmbeddingRun(path=path, embeddings=embeddings, movie_ids=movie_ids)


def primary_genre(genres: object) -> str:
    """Return the first non-empty pipe-separated genre, or ``Unknown``.

    This deliberately uses source order only for visualization; the complete
    genre string remains unchanged in metadata and hover text.
    """
    if pd.isna(genres):
        return UNKNOWN_GENRE
    labels = [label.strip() for label in str(genres).split("|") if label.strip()]
    return labels[0] if labels else UNKNOWN_GENRE


def genre_color_map(movies_clean_path: Path = MOVIES_CLEAN_PATH) -> dict[str, str]:
    """Build one deterministic genre-to-colour mapping for every experiment."""
    metadata = pd.read_csv(movies_clean_path, usecols=["genres"])
    genres = sorted({primary_genre(value) for value in metadata["genres"]})
    return {
        genre: GENRE_COLORS[index % len(GENRE_COLORS)]
        for index, genre in enumerate(genres)
    }


def metadata_for_movie_ids(
    movie_ids: np.ndarray, movies_clean_path: Path = MOVIES_CLEAN_PATH
) -> pd.DataFrame:
    """Return title and complete genre labels in the exact supplied ID order."""
    metadata = pd.read_csv(movies_clean_path, usecols=["movieId", "title", "genres"])
    if metadata["movieId"].duplicated().any():
        raise ValueError(f"{movies_clean_path}: duplicate movieId values.")
    indexed = metadata.set_index("movieId")
    requested = pd.Index(movie_ids, name="movieId")
    missing = requested.difference(indexed.index)
    if not missing.empty:
        raise ValueError(
            f"{movies_clean_path}: missing metadata for movie IDs {missing.tolist()}."
        )
    aligned = indexed.loc[requested].reset_index()
    aligned["primary_genre"] = aligned["genres"].map(primary_genre)
    return aligned


def fit_and_transform_pca(
    embeddings: dict[str, np.ndarray],
) -> tuple[PCA, dict[str, np.ndarray]]:
    """Fit a three-component PCA only on training embeddings and transform all splits."""
    pca = PCA(n_components=3)
    coordinates = {"train": pca.fit_transform(embeddings["train"])}
    for split in ("val", "heldout"):
        coordinates[split] = pca.transform(embeddings[split])
    return pca, coordinates


def explained_variance_report(pca: PCA) -> dict[str, object]:
    """Return JSON-safe per-component and cumulative train-fitted variance."""
    ratios = [float(value) for value in pca.explained_variance_ratio_]
    return {
        "fit_split": "train",
        "n_components": 3,
        "explained_variance_ratio": {
            f"PC{index + 1}": value for index, value in enumerate(ratios)
        },
        "cumulative_explained_variance": {
            "PC1_PC2": float(sum(ratios[:2])),
            "PC1_PC3": float(sum(ratios[:3])),
        },
    }


def _title(report: dict[str, object], dimensions: int) -> str:
    ratios = report["explained_variance_ratio"]
    cumulative = report["cumulative_explained_variance"]
    assert isinstance(ratios, dict) and isinstance(cumulative, dict)
    suffix = (
        f"PC1 {ratios['PC1']:.1%}, PC2 {ratios['PC2']:.1%}; "
        f"cumulative {cumulative['PC1_PC2']:.1%}"
    )
    if dimensions == 3:
        suffix += (
            f"; PC3 {ratios['PC3']:.1%}; " f"cumulative {cumulative['PC1_PC3']:.1%}"
        )
    return f"PCA fit on training embeddings — {suffix}"


def _display_data(
    run: EmbeddingRun,
    coordinates: dict[str, np.ndarray],
    splits: tuple[str, ...],
    movies_clean_path: Path,
) -> pd.DataFrame:
    rows = []
    for split in splits:
        metadata = metadata_for_movie_ids(run.movie_ids[split], movies_clean_path)
        values = coordinates[split]
        frame = metadata.assign(
            split=split, PC1=values[:, 0], PC2=values[:, 1], PC3=values[:, 2]
        )
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def save_2d_plot(
    data: pd.DataFrame,
    color_map: dict[str, str],
    report: dict[str, object],
    destination: Path,
) -> None:
    """Save a genre-coloured static PC1/PC2 scatter plot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(10, 8))
    for genre in sorted(data["primary_genre"].unique()):
        subset = data[data["primary_genre"] == genre]
        axis.scatter(
            subset["PC1"],
            subset["PC2"],
            s=12,
            alpha=0.7,
            color=color_map[genre],
            label=genre,
            linewidths=0,
        )
    axis.set(xlabel="PC1", ylabel="PC2", title=_title(report, 2))
    axis.legend(title="Primary genre", bbox_to_anchor=(1.02, 1), loc="upper left")
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_3d_plot(
    data: pd.DataFrame,
    color_map: dict[str, str],
    report: dict[str, object],
    destination: Path,
) -> None:
    """Save an interactive Plotly PC1/PC2/PC3 scatter plot."""
    try:
        import plotly.express as px
    except ImportError as exc:
        raise ImportError(
            "Plotly is required for the interactive 3D plot; install requirements.txt."
        ) from exc
    figure = px.scatter_3d(
        data,
        x="PC1",
        y="PC2",
        z="PC3",
        color="primary_genre",
        color_discrete_map=color_map,
        symbol="split",
        hover_name="title",
        hover_data={"movieId": True, "genres": True, "primary_genre": False},
        title=_title(report, 3),
    )
    figure.update_traces(marker={"size": 3, "opacity": 0.75})
    figure.update_layout(legend_title_text="Primary genre")
    figure.write_html(destination, include_plotlyjs="cdn", full_html=True)


def visualization_complete(destination: Path) -> bool:
    """Return whether all visualization outputs are present and parseable."""
    if not all((destination / name).is_file() for name in OUTPUT_FILES):
        return False
    try:
        payload = json.loads((destination / "explained_variance.json").read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("fit_split") == "train" and payload.get("n_components") == 3


def write_visualization(
    run: EmbeddingRun,
    *,
    splits: tuple[str, ...] = ("train",),
    movies_clean_path: Path = MOVIES_CLEAN_PATH,
    force: bool = False,
) -> Path:
    """Atomically create PCA outputs inside the experiment directory."""
    destination = run.path / "pca_visualization"
    if destination.exists() and not force:
        raise FileExistsError(
            f"{destination} already exists; use --force to replace it."
        )
    pca, coordinates = fit_and_transform_pca(run.embeddings)
    report = explained_variance_report(pca)
    data = _display_data(run, coordinates, splits, movies_clean_path)
    colors = genre_color_map(movies_clean_path)
    missing_colours = set(data["primary_genre"]).difference(colors)
    if missing_colours:
        raise ValueError(f"No colours configured for genres: {sorted(missing_colours)}")
    temporary = Path(tempfile.mkdtemp(prefix=".pca-visualization-", dir=run.path))
    try:
        save_2d_plot(data, colors, report, temporary / "pca_2d.png")
        save_3d_plot(data, colors, report, temporary / "pca_3d.html")
        (temporary / "explained_variance.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        if destination.exists():
            shutil.rmtree(destination)
        temporary.replace(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def discover_experiments(results_root: Path = RESULTS_ROOT) -> list[Path]:
    """Find experiment directories containing every required embedding split."""
    if not results_root.exists():
        return []
    return sorted(
        path.parent
        for path in results_root.rglob("embeddings_train.npz")
        if all((path.parent / f"embeddings_{split}.npz").is_file() for split in SPLITS)
    )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--experiment", type=Path, help="Experiment directory.")
    group.add_argument("--all", action="store_true", help="Visualize every experiment.")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=SPLITS,
        default=("train",),
        help="Splits to display (default: train).",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the command-line PCA visualization workflow."""
    args = _arguments()
    paths = discover_experiments() if args.all else [args.experiment]
    failures = []
    for path in paths:
        try:
            run = load_experiment(path)
            output = run.path / "pca_visualization"
            if args.skip_completed and visualization_complete(output):
                print(f"skipped: {run.path}")
                continue
            if args.dry_run:
                print(f"would visualize: {run.path}")
                continue
            write_visualization(run, splits=tuple(args.splits), force=args.force)
            print(f"success: {run.path}")
        except Exception as exc:
            failures.append(f"{path}: {exc}")
            print(f"failed: {path}: {exc}")
    if failures:
        raise SystemExit(f"{len(failures)} failed run(s): " + "; ".join(failures))


if __name__ == "__main__":
    main()

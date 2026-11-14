"""Train, evaluate, and visualize the eight native 2D/3D movie embeddings.

This module is deliberately separate from the 32-dimensional experiment
scripts. It reuses their data preparation and training loops, but makes the
latent dimension an explicit per-run configuration.
"""

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.dimensionality_reduction import postprocess
from experiments.dimensionality_reduction.common import (
    BATCH_SIZE,
    DATASET_VERSION,
    EARLY_STOPPING_PATIENCE,
    FEATURE_CONFIGURATIONS,
    HIDDEN_DIM,
    LEARNING_RATE,
    MAX_EPOCHS,
    SEED,
    PreparedData,
    extract_embeddings,
    extract_vae_embeddings,
    prepare_experiment_data,
    set_seed,
    train_autoencoder,
    train_beta_vae,
    train_vae,
)
from experiments.dimensionality_reduction.visualize_pca import (
    MOVIES_CLEAN_PATH,
    genre_color_map,
    load_experiment,
    metadata_for_movie_ids,
)
from models.AE import autoencoder
from models.BetaVAE import BetaVAE
from models.VAE import VAE

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOW_DIM_ROOT = PROJECT_ROOT / "results" / "dimensionality_reduction" / "LowDim"
SPLITS = ("train", "val", "heldout")
TRAINING_ARTIFACTS = {
    "model.pt",
    "config.json",
    "history.csv",
    "loss_curve.png",
    "embeddings_train.npz",
    "embeddings_val.npz",
    "embeddings_heldout.npz",
    "feature_builder.joblib",
}


@dataclass(frozen=True)
class LowDimConfig:
    """One of the fixed low-dimensional experiment combinations."""

    model: str
    features: str
    latent_dim: int
    beta: float | None = None

    @property
    def experiment_name(self) -> str:
        return f"{self.model}_{self.features}"

    @property
    def feature_groups(self) -> tuple[str, ...]:
        return FEATURE_CONFIGURATIONS[self.features]

    @property
    def path(self) -> Path:
        base = LOW_DIM_ROOT / self.experiment_name
        if self.model == "BetaVAE":
            assert self.beta is not None
            base = base / f"beta_{self.beta:g}"
        return base / f"latent_{self.latent_dim}"


CONFIGURATIONS = tuple(
    [
        LowDimConfig("AE", features, dimension)
        for features in ("A", "B")
        for dimension in (2, 3)
    ]
    + [LowDimConfig("VAE", "A", dimension) for dimension in (2, 3)]
    + [LowDimConfig("BetaVAE", "A", dimension, beta=0.1) for dimension in (2, 3)]
)


def select_configurations(
    *,
    model: str | None = None,
    features: str | None = None,
    latent_dim: int | None = None,
    beta: float | None = None,
) -> list[LowDimConfig]:
    """Filter the fixed experiment table and reject a selection that matches none."""
    selected = [
        config
        for config in CONFIGURATIONS
        if (model is None or config.model == model)
        and (features is None or config.features == features)
        and (latent_dim is None or config.latent_dim == latent_dim)
        and (beta is None or config.beta == beta)
    ]
    if not selected:
        raise ValueError(
            "No configured low-dimensional experiment matches the filters."
        )
    return selected


def build_model(config: LowDimConfig, input_dim: int) -> torch.nn.Module:
    """Construct the configured native-dimension model without altering model classes."""
    if config.model == "AE":
        return autoencoder(input_dim, HIDDEN_DIM, config.latent_dim)
    if config.model == "VAE":
        return VAE(input_dim, HIDDEN_DIM, config.latent_dim)
    if config.model == "BetaVAE":
        assert config.beta is not None
        return BetaVAE(input_dim, HIDDEN_DIM, config.latent_dim, beta=config.beta)
    raise ValueError(f"Unsupported model family: {config.model}")


def training_complete(path: Path) -> bool:
    """Return true only for a complete, internally consistent training directory."""
    if not all((path / name).is_file() for name in TRAINING_ARTIFACTS):
        return False
    try:
        config = json.loads((path / "config.json").read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return config.get("latent_dim") in (2, 3)


def latent_visualization_complete(path: Path, latent_dim: int) -> bool:
    """Return true when every visualization required for the native dimension exists."""
    required = {"latent_2d.png", "latent_2d.html"}
    if latent_dim == 3:
        required = {"latent_2d.png", "latent_3d.html"}
    return all((path / "latent_visualization" / name).is_file() for name in required)


def _save_loss_curve(path: Path, history: list[dict[str, float]], model: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 5))
    if model == "AE":
        axis.plot(
            [row["epoch"] for row in history],
            [row["train_loss"] for row in history],
            label="Training MSE",
        )
        axis.plot(
            [row["epoch"] for row in history],
            [row["val_loss"] for row in history],
            label="Validation MSE",
        )
        axis.set_ylabel("MSE")
    else:
        axis.plot(
            [row["epoch"] for row in history],
            [row["train_total_loss"] for row in history],
            label="Training total",
        )
        axis.plot(
            [row["epoch"] for row in history],
            [row["val_total_loss"] for row in history],
            label="Validation total",
        )
        axis.set_ylabel(f"{model} objective")
    axis.set(xlabel="Epoch", title=f"{model} training and validation")
    axis.legend()
    axis.grid(True)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _reconstruct(
    model: torch.nn.Module, features: np.ndarray, device: torch.device
) -> np.ndarray:
    """Reconstruct deterministically; VAE families decode posterior means."""
    model.eval()
    with torch.no_grad():
        tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
        if isinstance(model, autoencoder):
            reconstructed = model(tensor)
        else:
            mean, _ = model.encoder(tensor)
            reconstructed = model.decoder(mean)
    return reconstructed.cpu().numpy()


def _reconstruction_mse(
    model: torch.nn.Module, prepared: PreparedData, device: torch.device
) -> dict[str, float]:
    return {
        split: float(
            np.mean(
                (
                    prepared.features[split]
                    - _reconstruct(model, prepared.features[split], device)
                )
                ** 2
            )
        )
        for split in ("train", "val")
    }


def _write_training_outputs(
    config: LowDimConfig,
    model: torch.nn.Module,
    prepared: PreparedData,
    history: list[dict[str, float]],
    best_epoch: int,
    best_validation: dict[str, float],
    device: torch.device,
) -> None:
    """Write an artifact-compatible low-dimensional run with its true latent size."""
    import joblib

    path = config.path
    path.mkdir(parents=True, exist_ok=True)
    checkpoint: dict[str, Any] = {
        "state_dict": model.state_dict(),
        "architecture": {
            "input_dim": prepared.input_dim,
            "hidden_dim": HIDDEN_DIM,
            "latent_dim": config.latent_dim,
        },
    }
    if config.beta is not None:
        checkpoint["beta"] = config.beta
    torch.save(checkpoint, path / "model.pt")
    metadata: dict[str, Any] = {
        "experiment": config.experiment_name,
        "model_family": config.model,
        "dataset_version": DATASET_VERSION,
        "feature_configuration": config.features,
        "feature_groups": list(config.feature_groups),
        "input_dim": prepared.input_dim,
        "block_dimensions": prepared.block_dimensions,
        "block_weights": prepared.block_weights,
        "hidden_dim": HIDDEN_DIM,
        "latent_dim": config.latent_dim,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "random_seed": SEED,
        "dataset_sizes": {name: len(ids) for name, ids in prepared.movie_ids.items()},
        "best_epoch": best_epoch,
        "reconstruction_mse": _reconstruction_mse(model, prepared, device),
        **best_validation,
    }
    if config.beta is not None:
        metadata["beta"] = config.beta
    (path / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with (path / "history.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    _save_loss_curve(path / "loss_curve.png", history, config.model)
    extractor = extract_embeddings if config.model == "AE" else extract_vae_embeddings
    for split in SPLITS:
        np.savez_compressed(
            path / f"embeddings_{split}.npz",
            embeddings=extractor(model, prepared.features[split], device),
            movieId=prepared.movie_ids[split],
        )
    joblib.dump(prepared.builder, path / "feature_builder.joblib")


def train_configuration(config: LowDimConfig, *, force: bool = False) -> Path:
    """Train one selected configuration and create its artifact-compatible output."""
    if config.path.exists() and any(config.path.iterdir()):
        if not force:
            raise FileExistsError(
                f"Refusing to overwrite existing results: {config.path}"
            )
        shutil.rmtree(config.path)
    set_seed(SEED)
    prepared = prepare_experiment_data(config.feature_groups)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config, prepared.input_dim).to(device)
    if config.model == "AE":
        history, best_epoch, best_loss = train_autoencoder(
            model, prepared.features["train"], prepared.features["val"], device=device
        )
        best_validation = {"best_validation_loss": best_loss}
    elif config.model == "VAE":
        history, best_epoch, best_loss = train_vae(
            model, prepared.features["train"], prepared.features["val"], device=device
        )
        best_validation = {"best_validation_total_loss": best_loss}
    else:
        history, best_epoch, metrics = train_beta_vae(
            model, prepared.features["train"], prepared.features["val"], device=device
        )
        best_validation = {
            "best_validation_reconstruction_loss": metrics["val_reconstruction_loss"],
            "best_validation_kl_loss": metrics["val_kl_loss"],
            "best_validation_total_loss": metrics["val_total_loss"],
        }
    _write_training_outputs(
        config, model, prepared, history, best_epoch, best_validation, device
    )
    return config.path


def postprocess_configuration(
    config: LowDimConfig, *, force: bool = False
) -> dict[str, Any]:
    """Evaluate native coordinates with the shared post-processing implementation."""
    run = postprocess.load_experiment(config.path)
    destination = config.path / "postprocessing"
    if postprocess.postprocessing_complete(destination) and not force:
        return json.loads((destination / "metrics.json").read_text())
    report = postprocess.evaluate_run(run)
    postprocess.write_postprocessing(run, report, force=force, no_plots=True)
    return report


def _native_display_data(path: Path, splits: tuple[str, ...]) -> tuple[int, Any]:
    run = load_experiment(path)
    latent_dim = run.embeddings["train"].shape[1]
    if latent_dim not in (2, 3):
        raise ValueError(f"{path}: expected native 2D or 3D embeddings.")
    import pandas as pd

    frames = []
    for split in splits:
        metadata = metadata_for_movie_ids(run.movie_ids[split], MOVIES_CLEAN_PATH)
        values = run.embeddings[split]
        frame = metadata.assign(split=split, z1=values[:, 0], z2=values[:, 1])
        if latent_dim == 3:
            frame = frame.assign(z3=values[:, 2])
        frames.append(frame)
    return latent_dim, pd.concat(frames, ignore_index=True)


def write_native_visualization(
    config: LowDimConfig,
    *,
    splits: tuple[str, ...] = ("train",),
    force: bool = False,
) -> Path:
    """Create native-coordinate charts without PCA or any other projection."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        import plotly.express as px
    except ImportError as exc:
        raise ImportError(
            "Plotly is required for native interactive visualizations."
        ) from exc

    latent_dim, data = _native_display_data(config.path, splits)
    destination = config.path / "latent_visualization"
    if destination.exists() and not force:
        raise FileExistsError(
            f"{destination} already exists; use --force to replace it."
        )
    colours = genre_color_map(MOVIES_CLEAN_PATH)
    title = f"{config.experiment_name}, native latent dim {latent_dim}"
    if config.beta is not None:
        title += f", beta={config.beta:g}"
    temporary = destination.with_name(f".latent-visualization-{destination.name}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        figure, axis = plt.subplots(figsize=(10, 8))
        for genre in sorted(data["primary_genre"].unique()):
            subset = data[data["primary_genre"] == genre]
            axis.scatter(
                subset["z1"],
                subset["z2"],
                s=12,
                alpha=0.7,
                color=colours[genre],
                label=genre,
                linewidths=0,
            )
        axis.set(xlabel="z1", ylabel="z2", title=f"{title} — native coordinate view")
        axis.legend(title="Primary genre", bbox_to_anchor=(1.02, 1), loc="upper left")
        figure.savefig(temporary / "latent_2d.png", dpi=180, bbox_inches="tight")
        plt.close(figure)
        hover = {
            "movieId": True,
            "genres": True,
            "split": True,
            "z1": ":.4f",
            "z2": ":.4f",
            "primary_genre": False,
        }
        if latent_dim == 2:
            chart = px.scatter(
                data,
                x="z1",
                y="z2",
                color="primary_genre",
                symbol="split",
                color_discrete_map=colours,
                hover_name="title",
                hover_data=hover,
                title=title,
            )
            chart.write_html(
                temporary / "latent_2d.html", include_plotlyjs="cdn", full_html=True
            )
        else:
            hover["z3"] = ":.4f"
            chart = px.scatter_3d(
                data,
                x="z1",
                y="z2",
                z="z3",
                color="primary_genre",
                symbol="split",
                color_discrete_map=colours,
                hover_name="title",
                hover_data=hover,
                title=title,
            )
            chart.update_traces(marker={"size": 3, "opacity": 0.75})
            chart.write_html(
                temporary / "latent_3d.html", include_plotlyjs="cdn", full_html=True
            )
        if destination.exists():
            shutil.rmtree(destination)
        temporary.replace(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def _summary_row(
    config: LowDimConfig, report: dict[str, Any] | None, source: str
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "source": source,
        "experiment": config.experiment_name,
        "model": config.model,
        "features": config.features,
        "beta": config.beta,
        "latent_dim": config.latent_dim,
    }
    if source == "low_dim" and (config.path / "config.json").is_file():
        run_config = json.loads((config.path / "config.json").read_text())
        reconstruction = run_config.get("reconstruction_mse", {})
        row["train_reconstruction_mse"] = reconstruction.get("train")
        row["val_reconstruction_mse"] = reconstruction.get("val")
    else:
        row["train_reconstruction_mse"] = None
        row["val_reconstruction_mse"] = None
    if report:
        kmeans = report.get("clustering", {}).get("kmeans", {})
        semantic = report.get("semantic", {})
        row.update(
            {
                "kmeans_silhouette": kmeans.get("silhouette"),
                "kmeans_davies_bouldin": kmeans.get("davies_bouldin"),
                "kmeans_calinski_harabasz": kmeans.get("calinski_harabasz"),
                "genre_purity": kmeans.get("genre_purity"),
                "train_cosine_genre_jaccard": semantic.get(
                    "train_cosine_genre_jaccard"
                ),
                "heldout_cosine_genre_jaccard": semantic.get(
                    "heldout_cosine_genre_jaccard"
                ),
                "heldout_euclidean_genre_jaccard": semantic.get(
                    "heldout_euclidean_genre_jaccard"
                ),
            }
        )
    return row


def write_comparative_summary() -> Path:
    """Write a separate low-dimensional comparison; never touch the 32D summary."""
    rows = []
    for config in CONFIGURATIONS:
        low_metrics = config.path / "postprocessing" / "metrics.json"
        if low_metrics.is_file():
            rows.append(
                _summary_row(config, json.loads(low_metrics.read_text()), "low_dim")
            )
        baseline = (
            PROJECT_ROOT
            / "results"
            / "dimensionality_reduction"
            / config.experiment_name
        )
        if config.model == "BetaVAE":
            baseline = baseline / f"beta_{config.beta:g}"
        metrics_path = baseline / "postprocessing" / "metrics.json"
        if metrics_path.is_file():
            rows.append(
                _summary_row(
                    config, json.loads(metrics_path.read_text()), "baseline_32d"
                )
            )
    LOW_DIM_ROOT.mkdir(parents=True, exist_ok=True)
    destination = LOW_DIM_ROOT / "postprocessing_summary.csv"
    columns = (
        sorted({key for row in rows for key in row})
        if rows
        else ["source", "experiment"]
    )
    with destination.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return destination


def run_configuration(
    config: LowDimConfig,
    *,
    splits: tuple[str, ...] = ("train",),
    dry_run: bool = False,
    skip_completed: bool = False,
    force: bool = False,
    only_postprocess: bool = False,
    only_visualize: bool = False,
) -> None:
    """Run the requested resumable stages in train, post-process, visualize order."""
    if dry_run:
        print(f"would run: {config.path}")
        return
    if (
        not only_postprocess
        and not only_visualize
        and not training_complete(config.path)
    ):
        train_configuration(config, force=force)
    elif not training_complete(config.path):
        raise ValueError(f"{config.path}: training artifacts are required first.")
    if not only_visualize:
        if not (
            skip_completed
            and postprocess.postprocessing_complete(config.path / "postprocessing")
        ):
            postprocess_configuration(config, force=force)
    if not only_postprocess:
        if not (
            skip_completed
            and latent_visualization_complete(config.path, config.latent_dim)
        ):
            write_native_visualization(config, splits=splits, force=force)
    write_comparative_summary()


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all", action="store_true", help="Run all eight configurations."
    )
    parser.add_argument("--model", choices=("AE", "VAE", "BetaVAE"))
    parser.add_argument("--features", choices=("A", "B"))
    parser.add_argument("--latent-dim", type=int, choices=(2, 3))
    parser.add_argument("--beta", type=float)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=SPLITS,
        default=("train",),
        help="Splits to display in native-coordinate visualizations (default: train).",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--force", action="store_true")
    stage = parser.add_mutually_exclusive_group()
    stage.add_argument("--only-postprocess", action="store_true")
    stage.add_argument("--only-visualize", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    try:
        configurations = select_configurations(
            model=args.model,
            features=args.features,
            latent_dim=args.latent_dim,
            beta=args.beta,
        )
        for config in configurations:
            run_configuration(
                config,
                splits=tuple(args.splits),
                dry_run=args.dry_run,
                skip_completed=args.skip_completed,
                force=args.force,
                only_postprocess=args.only_postprocess,
                only_visualize=args.only_visualize,
            )
            if not args.dry_run:
                print(f"success: {config.path}")
    except Exception as exc:
        raise SystemExit(f"failed: {exc}") from exc


if __name__ == "__main__":
    main()

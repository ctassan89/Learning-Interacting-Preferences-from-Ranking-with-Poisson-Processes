"""Run the AE-A encoder-width ablation without changing the baseline run.

The baseline is the already-saved ``results/dimensionality_reduction/AE_A`` experiment.  This module
trains only the medium and large variants, using the same leakage-free v2
features and AE training loop as AE-A, then evaluates and visualizes each
saved run.  It deliberately never writes to ``results/dimensionality_reduction/AE_A``.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import torch
from torch import nn

from experiments.dimensionality_reduction.common import (
    BATCH_SIZE,
    DATASET_VERSION,
    EARLY_STOPPING_PATIENCE,
    FEATURE_CONFIGURATIONS,
    LATENT_DIM,
    LEARNING_RATE,
    MAX_EPOCHS,
    PROJECT_ROOT,
    SEED,
    PreparedData,
    _save_history,
    _save_loss_curve,
    extract_embeddings,
    prepare_experiment_data,
    set_seed,
    train_autoencoder,
)
from experiments.dimensionality_reduction.postprocess import (
    evaluate_run,
    load_experiment,
    write_postprocessing,
)
from experiments.dimensionality_reduction.visualize_pca import (
    load_experiment as load_embedding_experiment,
)
from experiments.dimensionality_reduction.visualize_pca import write_visualization


ABLATION_ROOT = (
    PROJECT_ROOT / "results" / "dimensionality_reduction" / "ArchitectureAblation"
)
BASELINE_DIR = PROJECT_ROOT / "results" / "dimensionality_reduction" / "AE_A"
INPUT_DIM = 248


@dataclass(frozen=True)
class Architecture:
    """One encoder/decoder layout in the AE-A ablation."""

    key: str
    experiment: str
    encoder_hidden_dims: tuple[int, ...]
    decoder_hidden_dims: tuple[int, ...]

    @property
    def result_dir(self) -> Path:
        return ABLATION_ROOT / self.experiment

    def description(self, input_dim: int = INPUT_DIM) -> str:
        return " -> ".join(
            str(width)
            for width in (
                input_dim,
                *self.encoder_hidden_dims,
                LATENT_DIM,
                *self.decoder_hidden_dims,
                input_dim,
            )
        )


ARCHITECTURES = {
    "medium": Architecture("medium", "AE_A_medium", (256, 128), (128, 256)),
    "large": Architecture("large", "AE_A_large", (512, 256, 128), (128, 256, 512)),
}


class ConfigurableAutoencoder(nn.Module):
    """Minimal fully-connected AE used only for non-baseline ablations."""

    def __init__(
        self,
        input_dim: int,
        encoder_hidden_dims: Iterable[int],
        latent_dim: int = LATENT_DIM,
        decoder_hidden_dims: Iterable[int] | None = None,
    ) -> None:
        super().__init__()
        encoder_hidden_dims = tuple(encoder_hidden_dims)
        decoder_hidden_dims = tuple(decoder_hidden_dims or ())
        if not encoder_hidden_dims or not decoder_hidden_dims:
            raise ValueError("Encoder and decoder must both have hidden layers.")
        if min(input_dim, latent_dim, *encoder_hidden_dims, *decoder_hidden_dims) <= 0:
            raise ValueError("Layer dimensions must be positive.")
        self.encoder = self._network((input_dim, *encoder_hidden_dims, latent_dim))
        self.decoder = self._network((latent_dim, *decoder_hidden_dims, input_dim))
        self.criterion = nn.MSELoss()

    @staticmethod
    def _network(dimensions: tuple[int, ...]) -> nn.Sequential:
        layers: list[nn.Module] = []
        for index, (source, target) in enumerate(zip(dimensions, dimensions[1:])):
            layers.append(nn.Linear(source, target))
            if index != len(dimensions) - 2:
                layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(values))

    def loss(self, values: torch.Tensor, reconstruction: torch.Tensor) -> torch.Tensor:
        return self.criterion(reconstruction, values)


def parameter_count(model: nn.Module) -> int:
    """Return the number of learned scalar parameters in a model."""
    return sum(parameter.numel() for parameter in model.parameters())


def _architecture_payload(architecture: Architecture, input_dim: int) -> dict[str, Any]:
    return {
        "input_dim": input_dim,
        "latent_dim": LATENT_DIM,
        "encoder_hidden_dims": list(architecture.encoder_hidden_dims),
        "decoder_hidden_dims": list(architecture.decoder_hidden_dims),
        "encoder_layers": [input_dim, *architecture.encoder_hidden_dims, LATENT_DIM],
        "decoder_layers": [LATENT_DIM, *architecture.decoder_hidden_dims, input_dim],
    }


def _ensure_empty_or_replace(path: Path, *, force: bool) -> None:
    """Protect completed ablations unless the caller explicitly requests replacement."""
    if path.exists() and any(path.iterdir()):
        if not force:
            raise FileExistsError(
                f"Refusing to overwrite existing ablation artifacts: {path}. "
                "Use --force to replace this ablation only."
            )
        shutil.rmtree(path)


def save_ablation_outputs(
    model: ConfigurableAutoencoder,
    prepared: PreparedData,
    history: list[dict[str, float]],
    best_epoch: int,
    best_loss: float,
    architecture: Architecture,
    *,
    device: torch.device,
) -> Path:
    """Save a self-contained non-baseline run with aligned embedding archives."""
    destination = architecture.result_dir
    destination.mkdir(parents=True, exist_ok=True)
    specification = _architecture_payload(architecture, prepared.input_dim)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "architecture": specification,
            "parameter_count": parameter_count(model),
        },
        destination / "model.pt",
    )
    config = {
        "experiment": architecture.experiment,
        "dataset_version": DATASET_VERSION,
        "feature_groups": list(FEATURE_CONFIGURATIONS["A"]),
        "input_dim": prepared.input_dim,
        "block_dimensions": prepared.block_dimensions,
        "block_weights": prepared.block_weights,
        "architecture": architecture.description(prepared.input_dim),
        "model_type": "ConfigurableAutoencoder",
        "architecture_details": specification,
        "latent_dim": LATENT_DIM,
        "parameter_count": parameter_count(model),
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "random_seed": SEED,
        "dataset_sizes": {name: len(ids) for name, ids in prepared.movie_ids.items()},
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
    }
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    _save_history(destination / "history.csv", history)
    _save_loss_curve(destination / "loss_curve.png", history)
    for split, features in prepared.features.items():
        np.savez_compressed(
            destination / f"embeddings_{split}.npz",
            embeddings=extract_embeddings(model, features, device),
            movieId=prepared.movie_ids[split],
        )
    joblib.dump(prepared.builder, destination / "feature_builder.joblib")
    return destination


def run_model(architecture: Architecture, *, force: bool = False) -> Path:
    """Train, evaluate, and visualize one non-baseline configuration."""
    _ensure_empty_or_replace(architecture.result_dir, force=force)
    set_seed(SEED)
    prepared = prepare_experiment_data(FEATURE_CONFIGURATIONS["A"])
    if prepared.input_dim != INPUT_DIM:
        raise ValueError(
            f"AE-A ablation requires {INPUT_DIM} inputs, got {prepared.input_dim}."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConfigurableAutoencoder(
        prepared.input_dim,
        architecture.encoder_hidden_dims,
        LATENT_DIM,
        architecture.decoder_hidden_dims,
    ).to(device)
    history, best_epoch, best_loss = train_autoencoder(
        model, prepared.features["train"], prepared.features["val"], device=device
    )
    destination = save_ablation_outputs(
        model,
        prepared,
        history,
        best_epoch,
        best_loss,
        architecture,
        device=device,
    )
    run = load_experiment(destination)
    report = evaluate_run(run)
    write_postprocessing(run, report)
    write_visualization(load_embedding_experiment(destination), splits=("train",))
    return destination


def _nested(data: dict[str, Any], *keys: str) -> Any:
    value: Any = data
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _baseline_parameter_count(config: dict[str, Any]) -> int | None:
    """Compute the baseline count from its documented fixed architecture."""
    input_dim, hidden_dim, latent_dim = (
        config.get("input_dim"),
        config.get("hidden_dim"),
        config.get("latent_dim"),
    )
    if not all(
        isinstance(value, int) and value > 0
        for value in (input_dim, hidden_dim, latent_dim)
    ):
        return None
    dimensions = (
        input_dim,
        hidden_dim,
        hidden_dim // 2,
        latent_dim,
        hidden_dim // 2,
        hidden_dim,
        input_dim,
    )
    return sum((left + 1) * right for left, right in zip(dimensions, dimensions[1:]))


def _comparison_row(path: Path, label: str, architecture: str) -> dict[str, Any]:
    config = json.loads((path / "config.json").read_text())
    metrics_path = path / "postprocessing" / "metrics.json"
    pca_path = path / "pca_visualization" / "explained_variance.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.is_file() else {}
    pca = json.loads(pca_path.read_text()) if pca_path.is_file() else {}
    count = config.get("parameter_count")
    if count is None and label == "AE-A baseline":
        count = _baseline_parameter_count(config)
    return {
        "model": label,
        "architecture": architecture,
        "parameter_count": count,
        "latent_dim": config.get("latent_dim"),
        "best_validation_mse": config.get("best_validation_loss"),
        "training_cosine_genre_jaccard": _nested(
            metrics, "semantic", "train_cosine_genre_jaccard"
        ),
        "heldout_cosine_genre_jaccard": _nested(
            metrics, "semantic", "heldout_cosine_genre_jaccard"
        ),
        "heldout_euclidean_genre_jaccard": _nested(
            metrics, "semantic", "heldout_euclidean_genre_jaccard"
        ),
        "kmeans_silhouette": _nested(metrics, "clustering", "kmeans", "silhouette"),
        "kmeans_genre_purity": _nested(metrics, "clustering", "kmeans", "genre_purity"),
        "gmm_silhouette": _nested(metrics, "clustering", "gmm", "silhouette"),
        "pca_cumulative_explained_variance_pc1_pc3": _nested(
            pca, "cumulative_explained_variance", "PC1_PC3"
        ),
    }


def write_comparison() -> Path:
    """Write the three-run report only when every required run is available."""
    paths = (BASELINE_DIR, *(item.result_dir for item in ARCHITECTURES.values()))
    missing = [str(path) for path in paths if not (path / "config.json").is_file()]
    if missing:
        raise FileNotFoundError("Cannot compare incomplete runs: " + ", ".join(missing))
    rows = [
        _comparison_row(
            BASELINE_DIR, "AE-A baseline", "248 -> 128 -> 64 -> 32 -> 64 -> 128 -> 248"
        ),
        *[
            _comparison_row(
                item.result_dir, f"AE-A {item.key.title()}", item.description()
            )
            for item in ARCHITECTURES.values()
        ],
    ]
    ABLATION_ROOT.mkdir(parents=True, exist_ok=True)
    destination = ABLATION_ROOT / "comparison.csv"
    with destination.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return destination


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Run Medium then Large.")
    group.add_argument("--model", choices=tuple(ARCHITECTURES), help="Run one variant.")
    parser.add_argument(
        "--force", action="store_true", help="Replace only selected ablation outputs."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print planned work without training."
    )
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> None:
    args = parse_args(arguments)
    selected = list(ARCHITECTURES.values()) if args.all else [ARCHITECTURES[args.model]]
    if args.dry_run:
        for architecture in selected:
            print(f"would run: {architecture.experiment} -> {architecture.result_dir}")
        print(f"would compare with baseline: {BASELINE_DIR}")
        return
    for architecture in selected:
        print(f"running: {architecture.experiment}")
        run_model(architecture, force=args.force)
    try:
        print(f"comparison: {write_comparison()}")
    except FileNotFoundError as exc:
        # A single-model invocation is useful before the other variant exists.
        print(f"comparison deferred: {exc}")


if __name__ == "__main__":
    main()

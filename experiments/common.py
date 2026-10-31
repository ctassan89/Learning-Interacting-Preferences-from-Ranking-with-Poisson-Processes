"""
Experiment A — Autoencoder training on movie content features.

Feature configuration A:
    - Genres: multi-hot encoding.
    - Keywords: 128-dimensional deterministic random-code encoding.
    - Overview + tagline: TF-IDF compressed using TruncatedSVD
      (up to 100 components).

    Every selected block uses Feature Matrix v2 variance normalization and
    weights: genres 1.5, keywords 1.0, and text 0.5.

Objective:
    Learn a 32-dimensional latent representation of movies
    using only content-related information.

Model:
    Fully connected Autoencoder.
    Encoder: input_dim -> 128 -> 64 -> 32.
    Decoder: 32 -> 64 -> 128 -> input_dim.
    Hidden activation: ReLU.
    Reconstruction loss: Mean Squared Error (MSE).

Data:
    Use the existing cleaned movie catalogue and Feature Matrix v2 pipeline.
    Preserve the predefined held-out movie split.
    Create a separate validation subset from the remaining movies.
    Never use validation or held-out observations to fit preprocessing or
    neural-network parameters.

Outputs:
    Model checkpoint, training configuration, loss history,
    training/validation curve, and movie embeddings for all splits.
"""

import copy
import csv
import json
import random
from argparse import ArgumentParser
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from math import isfinite
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.AE import autoencoder
from models.BetaVAE import BetaVAE
from models.VAE import VAE

SEED = 42
HIDDEN_DIM = 128
LATENT_DIM = 32
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
MAX_EPOCHS = 200
EARLY_STOPPING_PATIENCE = 20
VALIDATION_FRACTION = 0.15
FEATURE_CONFIGURATIONS = {
    "A": ("genres", "keywords", "text"),
    "B": ("genres", "keywords", "text", "cast", "directors", "writers"),
    "C": (
        "genres",
        "keywords",
        "text",
        "cast",
        "directors",
        "writers",
        "companies",
        "language",
    ),
    "D": (
        "genres",
        "keywords",
        "text",
        "cast",
        "directors",
        "writers",
        "companies",
        "language",
        "collection",
        "numeric",
    ),
}
FEATURE_GROUPS = FEATURE_CONFIGURATIONS["A"]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "AE_A"
DATASET_VERSION = "v2"
LIST_COLUMNS = (
    "genres",
    "keywords",
    "cast",
    "directors",
    "writers",
    "companies",
    "collection",
)
REQUIRED_CLEAN_COLUMNS = {
    "movieId",
    "tmdbId",
    "title",
    *LIST_COLUMNS,
    "overview_tagline",
    "language",
    "year",
    "runtime",
    "budget",
    "split",
}


@dataclass
class PreparedData:
    """Leakage-free features and movie identifiers for each experiment split."""

    features: dict[str, np.ndarray]
    movie_ids: dict[str, np.ndarray]
    builder: Any
    input_dim: int
    block_dimensions: dict[str, int]
    block_weights: dict[str, float]


def set_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible experiment runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_movie_indices(
    movie_ids: np.ndarray,
    predefined_split: np.ndarray,
    *,
    seed: int = SEED,
    validation_fraction: float = VALIDATION_FRACTION,
) -> dict[str, np.ndarray]:
    """Split predefined training movies into reproducible train/validation sets."""
    movie_ids = np.asarray(movie_ids)
    predefined_split = np.asarray(predefined_split)
    if movie_ids.ndim != 1 or predefined_split.shape != movie_ids.shape:
        raise ValueError(
            "movie_ids and predefined_split must be aligned one-dimensional arrays."
        )
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    if not np.all(np.isin(predefined_split, ("train", "heldout"))):
        raise ValueError("predefined_split must contain only 'train' and 'heldout'.")

    train_pool = np.flatnonzero(predefined_split == "train")
    heldout = np.flatnonzero(predefined_split == "heldout")
    if len(train_pool) < 2 or len(heldout) == 0:
        raise ValueError(
            "The predefined split needs at least two train and one held-out movie."
        )
    n_validation = max(1, int(np.ceil(len(train_pool) * validation_fraction)))
    if n_validation >= len(train_pool):
        raise ValueError("Validation split would leave no training movies.")
    permutation = np.random.default_rng(seed).permutation(train_pool)
    validation = np.sort(permutation[:n_validation])
    training = np.sort(permutation[n_validation:])
    return {"train": training, "val": validation, "heldout": heldout}


def select_feature_groups(
    matrix: np.ndarray,
    block_slices: dict[str, list[int] | tuple[int, int]],
    feature_groups: tuple[str, ...],
) -> np.ndarray:
    """Select configured Feature Matrix v2 blocks from builder-provided slices."""
    missing = set(feature_groups).difference(block_slices)
    if missing:
        raise ValueError(
            f"Experiment A feature blocks are unavailable: {sorted(missing)}."
        )
    matrix = np.asarray(matrix)
    if matrix.ndim != 2:
        raise ValueError("Feature matrix must be two-dimensional.")
    selected = []
    for group in feature_groups:
        start, stop = block_slices[group]
        if not 0 <= start < stop <= matrix.shape[1]:
            raise ValueError(f"Invalid slice for feature block '{group}'.")
        selected.append(matrix[:, start:stop])
    return np.hstack(selected).astype(np.float32, copy=False)


def select_experiment_a_features(
    matrix: np.ndarray, block_slices: dict[str, list[int] | tuple[int, int]]
) -> np.ndarray:
    """Backward-compatible selector for Experiment A's content blocks."""
    return select_feature_groups(matrix, block_slices, FEATURE_GROUPS)


def fit_experiment_builder(
    builder: Any, movie_data: Any, train_indices: np.ndarray
) -> Any:
    """Fit preprocessing only on the experiment-training rows."""
    return builder.fit(movie_data.iloc[np.asarray(train_indices)])


def _parse_pipe_list(value: Any) -> list[str]:
    """Restore a FeatureBuilder list column from ``movies_clean.csv``."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return [item for item in str(value).split("|") if item]


def load_cleaned_movies(processed_dir: Path = PROCESSED_DIR) -> Any:
    """Reconstruct the tidy v2 FeatureBuilder input from processed CSV files.

    ``movies_clean.csv`` stores the exact list fields consumed by v2 as
    pipe-separated text. It stores the text input as the same combined
    overview/tagline string used by the original ``tidy`` function.
    """
    import pandas as pd

    clean_path = processed_dir / "movies_clean.csv"
    catalogue_path = processed_dir / "catalogue.csv"
    if not clean_path.is_file() or not catalogue_path.is_file():
        raise FileNotFoundError(
            "Experiment A requires data/processed/movies_clean.csv and catalogue.csv."
        )
    movies = pd.read_csv(clean_path)
    catalogue = pd.read_csv(catalogue_path)
    missing = REQUIRED_CLEAN_COLUMNS.difference(movies.columns)
    if missing:
        raise ValueError(
            f"movies_clean.csv is missing required columns: {sorted(missing)}"
        )
    if not {"movieId", "split"}.issubset(catalogue.columns):
        raise ValueError("catalogue.csv must contain movieId and split columns.")
    if movies["movieId"].duplicated().any() or catalogue["movieId"].duplicated().any():
        raise ValueError("movieId values must be unique in both processed CSV files.")
    if set(movies["movieId"]) != set(catalogue["movieId"]):
        raise ValueError(
            "movies_clean.csv and catalogue.csv contain different movie IDs."
        )

    movies = (
        movies.set_index("movieId", drop=False)
        .loc[catalogue["movieId"]]
        .reset_index(drop=True)
    )
    if not np.array_equal(movies["split"].to_numpy(), catalogue["split"].to_numpy()):
        raise ValueError(
            "The cleaned-data split does not match the canonical catalogue split."
        )
    for column in LIST_COLUMNS:
        movies[column] = movies[column].map(_parse_pipe_list)
    movies["text"] = movies["overview_tagline"].fillna("").astype(str).str.strip()
    movies["language"] = movies["language"].fillna("unknown").replace("", "unknown")
    for column in ("year", "runtime", "budget"):
        movies[column] = pd.to_numeric(movies[column], errors="coerce")
    movies["runtime"] = movies["runtime"].replace(0, np.nan)
    movies["budget"] = movies["budget"].replace(0, np.nan)
    if movies["movieId"].isna().any() or movies["tmdbId"].isna().any():
        raise ValueError(
            "movies_clean.csv must provide movieId and tmdbId for every movie."
        )
    return movies


def _validate_prepared_data(prepared: PreparedData) -> None:
    """Reject split, ID, or finite-value errors before AE training starts."""
    all_ids: list[np.ndarray] = []
    for split in ("train", "val", "heldout"):
        matrix = prepared.features[split]
        movie_ids = prepared.movie_ids[split]
        if matrix.ndim != 2 or matrix.shape[1] != prepared.input_dim:
            raise ValueError(
                f"{split} features have an invalid Experiment A input shape."
            )
        if len(matrix) != len(movie_ids) or len(np.unique(movie_ids)) != len(movie_ids):
            raise ValueError(f"{split} features and movie IDs are not one-to-one.")
        if not np.isfinite(matrix).all():
            raise ValueError(f"{split} features contain non-finite values.")
        all_ids.append(movie_ids)
    if len(np.unique(np.concatenate(all_ids))) != sum(len(ids) for ids in all_ids):
        raise ValueError(
            "Experiment train, validation, and held-out movie IDs overlap."
        )


def prepare_experiment_data(
    feature_groups: tuple[str, ...] = FEATURE_GROUPS,
    processed_dir: Path = PROCESSED_DIR,
) -> PreparedData:
    """Create leakage-free v2 Experiment A features from cleaned processed data.

    A fresh v2 ``FeatureBuilder`` is fitted only on Experiment A training
    movies. The shared builder and matrix are not used as training inputs.
    """
    from scripts.build_features import FeatureBuilder

    movies = load_cleaned_movies(processed_dir)
    movie_ids = movies["movieId"].to_numpy()
    split_indices = split_movie_indices(movie_ids, movies["split"].to_numpy())
    builder = fit_experiment_builder(FeatureBuilder(), movies, split_indices["train"])
    full_matrix, block_slices, _ = builder.transform(movies)
    selected_matrix = select_feature_groups(full_matrix, block_slices, feature_groups)
    block_dimensions = {
        group: block_slices[group][1] - block_slices[group][0]
        for group in feature_groups
    }
    prepared = PreparedData(
        features={
            name: selected_matrix[indices] for name, indices in split_indices.items()
        },
        movie_ids={name: movie_ids[indices] for name, indices in split_indices.items()},
        builder=builder,
        input_dim=selected_matrix.shape[1],
        block_dimensions=block_dimensions,
        block_weights={
            group: float(builder.weights[group]) for group in feature_groups
        },
    )
    _validate_prepared_data(prepared)
    return prepared


def restore_best_state(
    model: torch.nn.Module, state_dict: dict[str, torch.Tensor]
) -> None:
    """Restore an in-memory best checkpoint onto ``model``."""
    model.load_state_dict(state_dict)


def train_autoencoder(
    model: autoencoder,
    train_features: np.ndarray,
    val_features: np.ndarray,
    *,
    device: torch.device,
    max_epochs: int = MAX_EPOCHS,
    patience: int = EARLY_STOPPING_PATIENCE,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
) -> tuple[list[dict[str, float]], int, float]:
    """Train with validation early stopping and restore the best model state."""
    train_tensor = torch.as_tensor(train_features, dtype=torch.float32)
    val_tensor = torch.as_tensor(val_features, dtype=torch.float32)
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(
        TensorDataset(val_tensor), batch_size=batch_size, shuffle=False
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history: list[dict[str, float]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    best_epoch = -1
    stale_epochs = 0

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = model.loss(batch, model(batch))
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch)
        train_loss /= len(train_tensor)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                val_loss += model.loss(batch, model(batch)).item() * len(batch)
        val_loss /= len(val_tensor)
        history.append(
            {"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss}
        )

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    if best_state is None:
        raise RuntimeError(
            "Training completed without producing a validation checkpoint."
        )
    restore_best_state(model, best_state)
    return history, best_epoch, best_loss


def extract_embeddings(
    model: autoencoder, features: np.ndarray, device: torch.device
) -> np.ndarray:
    """Extract encoder outputs in input-row order from a restored model."""
    model.eval()
    with torch.no_grad():
        tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
        return model.encoder(tensor).cpu().numpy()


def _save_history(path: Path, history: list[dict[str, float]]) -> None:
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(history)


def _save_loss_curve(path: Path, history: list[dict[str, float]]) -> None:
    import matplotlib.pyplot as plt

    epochs = [entry["epoch"] for entry in history]
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [entry["train_loss"] for entry in history], label="Training MSE")
    plt.plot(epochs, [entry["val_loss"] for entry in history], label="Validation MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Experiment A: Autoencoder training and validation loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def save_experiment_outputs(
    model: autoencoder,
    prepared: PreparedData,
    history: list[dict[str, float]],
    best_epoch: int,
    best_loss: float,
    *,
    device: torch.device,
    experiment_name: str = "AE_A",
    feature_groups: tuple[str, ...] = FEATURE_GROUPS,
    results_dir: Path = RESULTS_DIR,
) -> None:
    """Write the required checkpoint, preprocessing object, and experiment artifacts."""
    import joblib

    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": model.state_dict(),
        "architecture": {
            "input_dim": prepared.input_dim,
            "hidden_dim": HIDDEN_DIM,
            "latent_dim": LATENT_DIM,
        },
    }
    torch.save(checkpoint, results_dir / "model.pt")
    configuration = {
        "experiment": experiment_name,
        "dataset_version": DATASET_VERSION,
        "feature_groups": list(feature_groups),
        "input_dim": prepared.input_dim,
        "block_dimensions": prepared.block_dimensions,
        "block_weights": prepared.block_weights,
        "hidden_dim": HIDDEN_DIM,
        "latent_dim": LATENT_DIM,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "random_seed": SEED,
        "dataset_sizes": {name: len(ids) for name, ids in prepared.movie_ids.items()},
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
    }
    (results_dir / "config.json").write_text(json.dumps(configuration, indent=2) + "\n")
    _save_history(results_dir / "history.csv", history)
    _save_loss_curve(results_dir / "loss_curve.png", history)
    for split, features in prepared.features.items():
        np.savez_compressed(
            results_dir / f"embeddings_{split}.npz",
            embeddings=extract_embeddings(model, features, device),
            movieId=prepared.movie_ids[split],
        )
    joblib.dump(prepared.builder, results_dir / "feature_builder.joblib")


def run_ae(experiment: str, feature_groups: tuple[str, ...]) -> None:
    """Run one configured AE experiment without sharing result directories."""
    set_seed()
    prepared = prepare_experiment_data(feature_groups)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = autoencoder(prepared.input_dim, HIDDEN_DIM, LATENT_DIM).to(device)
    history, best_epoch, best_loss = train_autoencoder(
        model, prepared.features["train"], prepared.features["val"], device=device
    )
    save_experiment_outputs(
        model,
        prepared,
        history,
        best_epoch,
        best_loss,
        device=device,
        experiment_name=experiment,
        feature_groups=feature_groups,
        results_dir=PROJECT_ROOT / "results" / experiment,
    )


def _vae_batch_losses(
    model: VAE, batch: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-sample reconstruction, KL, and total v2 VAE losses."""
    reconstruction, mean, logvar = model(batch)
    reconstruction_sum = model.criterion(batch, reconstruction)
    kl_sum = -0.5 * torch.sum(1 + logvar - mean**2 - torch.exp(logvar))
    return (
        reconstruction_sum / len(batch),
        kl_sum / len(batch),
        (reconstruction_sum + kl_sum) / len(batch),
    )


def train_vae(
    model: VAE,
    train_features: np.ndarray,
    val_features: np.ndarray,
    *,
    device: torch.device,
    max_epochs: int = MAX_EPOCHS,
    patience: int = EARLY_STOPPING_PATIENCE,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
) -> tuple[list[dict[str, float]], int, float]:
    """Train a VAE with sample-weighted losses and restore best validation state."""
    train_tensor = torch.as_tensor(train_features, dtype=torch.float32)
    val_tensor = torch.as_tensor(val_features, dtype=torch.float32)
    train_loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    val_loader = DataLoader(TensorDataset(val_tensor), batch_size=batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history: list[dict[str, float]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_loss, best_epoch, stale_epochs = float("inf"), -1, 0

    for epoch in range(max_epochs):
        totals = {
            "train_reconstruction_loss": 0.0,
            "train_kl_loss": 0.0,
            "train_total_loss": 0.0,
        }
        model.train()
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            reconstruction, kl, total = _vae_batch_losses(model, batch)
            total.backward()
            optimizer.step()
            for key, value in zip(totals, (reconstruction, kl, total), strict=True):
                totals[key] += value.item() * len(batch)
        for key in totals:
            totals[key] /= len(train_tensor)

        validation = {
            "val_reconstruction_loss": 0.0,
            "val_kl_loss": 0.0,
            "val_total_loss": 0.0,
        }
        model.eval()
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                reconstruction, kl, total = _vae_batch_losses(model, batch)
                for key, value in zip(
                    validation, (reconstruction, kl, total), strict=True
                ):
                    validation[key] += value.item() * len(batch)
        for key in validation:
            validation[key] /= len(val_tensor)
        row = {"epoch": epoch + 1, **totals, **validation}
        history.append(row)
        if validation["val_total_loss"] < best_loss:
            best_loss, best_epoch = validation["val_total_loss"], epoch + 1
            best_state, stale_epochs = copy.deepcopy(model.state_dict()), 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    if best_state is None:
        raise RuntimeError(
            "Training completed without producing a validation checkpoint."
        )
    restore_best_state(model, best_state)
    return history, best_epoch, best_loss


def extract_vae_embeddings(
    model: VAE, features: np.ndarray, device: torch.device
) -> np.ndarray:
    """Return deterministic posterior-mean embeddings, not sampled latents."""
    model.eval()
    with torch.no_grad():
        tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
        mean, _ = model.encoder(tensor)
    return mean.cpu().numpy()


def run_vae(experiment: str, feature_groups: tuple[str, ...]) -> None:
    """Run one configured VAE experiment and save isolated experiment artifacts."""
    set_seed()
    prepared = prepare_experiment_data(feature_groups)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(prepared.input_dim, HIDDEN_DIM, LATENT_DIM).to(device)
    history, best_epoch, best_loss = train_vae(
        model, prepared.features["train"], prepared.features["val"], device=device
    )
    results_dir = PROJECT_ROOT / "results" / experiment
    results_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "architecture": {
                "input_dim": prepared.input_dim,
                "hidden_dim": HIDDEN_DIM,
                "latent_dim": LATENT_DIM,
            },
        },
        results_dir / "model.pt",
    )
    configuration = {
        "experiment": experiment,
        "dataset_version": DATASET_VERSION,
        "feature_groups": list(feature_groups),
        "input_dim": prepared.input_dim,
        "block_dimensions": prepared.block_dimensions,
        "block_weights": prepared.block_weights,
        "architecture": "input_dim -> 128 -> 64 -> latent(32) -> 64 -> 128 -> input_dim",
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "random_seed": SEED,
        "dataset_sizes": {name: len(ids) for name, ids in prepared.movie_ids.items()},
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
    }
    (results_dir / "config.json").write_text(json.dumps(configuration, indent=2) + "\n")
    fields = list(history[0])
    with (results_dir / "history.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    plt.figure(figsize=(8, 5))
    plt.plot(
        epochs, [row["train_total_loss"] for row in history], label="Training total"
    )
    plt.plot(
        epochs, [row["val_total_loss"] for row in history], label="Validation total"
    )
    plt.xlabel("Epoch")
    plt.ylabel("VAE loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(results_dir / "loss_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    for split, features in prepared.features.items():
        np.savez_compressed(
            results_dir / f"embeddings_{split}.npz",
            embeddings=extract_vae_embeddings(model, features, device),
            movieId=prepared.movie_ids[split],
        )
    import joblib

    joblib.dump(prepared.builder, results_dir / "feature_builder.joblib")


def normalize_beta(beta: float) -> str:
    """Return a stable, filesystem-safe label for a finite non-negative beta."""
    if isinstance(beta, bool):
        raise ValueError("beta must be a finite, non-negative real number.")
    try:
        value = float(beta)
        decimal = Decimal(str(beta))
    except (TypeError, ValueError, InvalidOperation) as exc:
        raise ValueError("beta must be a finite, non-negative real number.") from exc
    if not isfinite(value) or value < 0:
        raise ValueError("beta must be a finite, non-negative real number.")
    normalized = format(decimal.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    normalized = normalized or "0"
    if len(normalized) > 32 or any(
        character not in "0123456789." for character in normalized
    ):
        raise ValueError(
            "beta cannot be represented safely as a result-directory name."
        )
    return normalized


def beta_results_dir(experiment: str, beta: float) -> Path:
    """Return the isolated, canonical output directory for one beta value."""
    return PROJECT_ROOT / "results" / experiment / f"beta_{normalize_beta(beta)}"


def ensure_empty_results_dir(results_dir: Path) -> None:
    """Reject a non-empty output directory before a run can overwrite artifacts."""
    if results_dir.exists() and any(results_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing results: {results_dir}")


def parse_beta_argument(arguments: list[str] | None = None) -> float:
    """Parse and validate the optional ``--beta`` command-line argument."""
    parser = ArgumentParser()
    parser.add_argument("--beta", type=float, default=1.0)
    beta = parser.parse_args(arguments).beta
    normalize_beta(beta)
    return beta


def _beta_vae_batch_losses(
    model: BetaVAE, batch: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the normalized reconstruction, KL, and beta-weighted losses."""
    reconstructed, mean, logvar = model(batch)
    return model.loss_components(batch, reconstructed, mean, logvar)


def train_beta_vae(
    model: BetaVAE,
    train_features: np.ndarray,
    val_features: np.ndarray,
    *,
    device: torch.device,
    max_epochs: int = MAX_EPOCHS,
    patience: int = EARLY_STOPPING_PATIENCE,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
) -> tuple[list[dict[str, float]], int, dict[str, float]]:
    """Train Beta-VAE with sample-weighted metrics and total-loss early stopping."""
    train_tensor = torch.as_tensor(train_features, dtype=torch.float32)
    val_tensor = torch.as_tensor(val_features, dtype=torch.float32)
    train_loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    val_loader = DataLoader(TensorDataset(val_tensor), batch_size=batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history: list[dict[str, float]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, float] | None = None
    stale_epochs = 0

    for epoch in range(max_epochs):
        train_metrics = {
            "train_reconstruction_loss": 0.0,
            "train_kl_loss": 0.0,
            "train_total_loss": 0.0,
        }
        model.train()
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            reconstruction, kl, total = _beta_vae_batch_losses(model, batch)
            total.backward()
            optimizer.step()
            for key, value in zip(
                train_metrics, (reconstruction, kl, total), strict=True
            ):
                train_metrics[key] += value.item() * len(batch)
        for key in train_metrics:
            train_metrics[key] /= len(train_tensor)

        validation_metrics = {
            "val_reconstruction_loss": 0.0,
            "val_kl_loss": 0.0,
            "val_total_loss": 0.0,
        }
        model.eval()
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                reconstruction, kl, total = _beta_vae_batch_losses(model, batch)
                for key, value in zip(
                    validation_metrics, (reconstruction, kl, total), strict=True
                ):
                    validation_metrics[key] += value.item() * len(batch)
        for key in validation_metrics:
            validation_metrics[key] /= len(val_tensor)
        row = {"epoch": epoch + 1, **train_metrics, **validation_metrics}
        history.append(row)
        if (
            best_metrics is None
            or row["val_total_loss"] < best_metrics["val_total_loss"]
        ):
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = validation_metrics.copy()
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    if best_state is None or best_metrics is None:
        raise RuntimeError(
            "Training completed without producing a validation checkpoint."
        )
    restore_best_state(model, best_state)
    best_epoch = next(
        row["epoch"]
        for row in history
        if row["val_total_loss"] == best_metrics["val_total_loss"]
    )
    return history, best_epoch, best_metrics


def run_beta_vae(experiment: str, feature_groups: tuple[str, ...], beta: float) -> None:
    """Run one Beta-VAE configuration, refusing to overwrite existing artifacts."""
    beta_label = normalize_beta(beta)
    results_dir = beta_results_dir(experiment, beta)
    ensure_empty_results_dir(results_dir)
    set_seed()
    prepared = prepare_experiment_data(feature_groups)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BetaVAE(prepared.input_dim, HIDDEN_DIM, LATENT_DIM, beta=beta).to(device)
    history, best_epoch, best_metrics = train_beta_vae(
        model, prepared.features["train"], prepared.features["val"], device=device
    )
    import joblib
    import matplotlib.pyplot as plt

    results_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "architecture": {
                "input_dim": prepared.input_dim,
                "hidden_dim": HIDDEN_DIM,
                "latent_dim": LATENT_DIM,
            },
            "beta": model.beta,
        },
        results_dir / "model.pt",
    )
    configuration = {
        "experiment": experiment,
        "dataset_version": DATASET_VERSION,
        "feature_groups": list(feature_groups),
        "input_dim": prepared.input_dim,
        "block_dimensions": prepared.block_dimensions,
        "block_weights": prepared.block_weights,
        "beta": model.beta,
        "beta_label": beta_label,
        "architecture": "input_dim -> 128 -> 64 -> latent(32) -> 64 -> 128 -> input_dim",
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "random_seed": SEED,
        "dataset_sizes": {name: len(ids) for name, ids in prepared.movie_ids.items()},
        "best_epoch": best_epoch,
        "best_validation_reconstruction_loss": best_metrics["val_reconstruction_loss"],
        "best_validation_kl_loss": best_metrics["val_kl_loss"],
        "best_validation_total_loss": best_metrics["val_total_loss"],
    }
    (results_dir / "config.json").write_text(json.dumps(configuration, indent=2) + "\n")
    with (results_dir / "history.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    epochs = [row["epoch"] for row in history]
    plt.figure(figsize=(8, 5))
    plt.plot(
        epochs, [row["train_total_loss"] for row in history], label="Training total"
    )
    plt.plot(
        epochs, [row["val_total_loss"] for row in history], label="Validation total"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Beta-VAE loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(results_dir / "loss_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    for split, features in prepared.features.items():
        np.savez_compressed(
            results_dir / f"embeddings_{split}.npz",
            embeddings=extract_vae_embeddings(model, features, device),
            movieId=prepared.movie_ids[split],
        )
    joblib.dump(prepared.builder, results_dir / "feature_builder.joblib")

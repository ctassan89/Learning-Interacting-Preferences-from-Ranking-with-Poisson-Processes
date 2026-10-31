"""
Experiment A — Autoencoder training on movie content features.

Feature configuration A:
    - Genres: multi-hot encoding.
    - Keywords: multi-hot encoding compressed using TruncatedSVD
      (up to 50 components).
    - Overview + tagline: TF-IDF compressed using TruncatedSVD
      (up to 100 components).

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
    Use the existing movie catalogue and feature-building pipeline.
    Preserve the predefined held-out movie split.
    Create a separate validation subset from the remaining movies.
    Never use validation or held-out observations to select or fit
    neural-network parameters.

Outputs:
    Model checkpoint, training configuration, loss history,
    training/validation curve, and movie embeddings for all splits.
"""

import copy
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.AE import autoencoder

SEED = 42
HIDDEN_DIM = 128
LATENT_DIM = 32
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
MAX_EPOCHS = 200
EARLY_STOPPING_PATIENCE = 20
VALIDATION_FRACTION = 0.15
FEATURE_GROUPS = ("genres", "keywords", "text")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "AE_A"
REQUIRED_RAW_FILES = {
    "movies_metadata.csv",
    "credits.csv",
    "keywords.csv",
    "links_small.csv",
    "ratings_small.csv",
}


@dataclass
class PreparedData:
    """Leakage-free features and movie identifiers for each experiment split."""

    features: dict[str, np.ndarray]
    movie_ids: dict[str, np.ndarray]
    builder: Any
    input_dim: int


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


def select_experiment_a_features(
    matrix: np.ndarray, block_slices: dict[str, list[int] | tuple[int, int]]
) -> np.ndarray:
    """Select the genres, keywords, and text blocks from builder-provided slices."""
    missing = set(FEATURE_GROUPS).difference(block_slices)
    if missing:
        raise ValueError(
            f"Experiment A feature blocks are unavailable: {sorted(missing)}."
        )
    matrix = np.asarray(matrix)
    if matrix.ndim != 2:
        raise ValueError("Feature matrix must be two-dimensional.")
    selected = []
    for group in FEATURE_GROUPS:
        start, stop = block_slices[group]
        if not 0 <= start < stop <= matrix.shape[1]:
            raise ValueError(f"Invalid slice for feature block '{group}'.")
        selected.append(matrix[:, start:stop])
    return np.hstack(selected).astype(np.float32, copy=False)


def fit_experiment_builder(
    builder: Any, movie_data: Any, train_indices: np.ndarray
) -> Any:
    """Fit preprocessing only on the experiment-training rows."""
    return builder.fit(movie_data.iloc[np.asarray(train_indices)])


def prepare_experiment_data(
    raw_dir: Path = RAW_DIR, processed_dir: Path = PROCESSED_DIR
) -> PreparedData:
    """Create leakage-free Experiment A features from raw movie inputs.

    The checked-in feature matrix cannot be used here: its builder was fitted
    before the internal validation split. This function refits a fresh
    ``FeatureBuilder`` exclusively on Experiment A training movies.
    """
    missing_raw = sorted(
        name for name in REQUIRED_RAW_FILES if not (raw_dir / name).is_file()
    )
    if missing_raw:
        raise FileNotFoundError(
            "Leakage-free Experiment A preprocessing requires the original raw files in "
            f"{raw_dir}: {', '.join(missing_raw)}. The existing processed matrix cannot "
            "be reused after introducing the validation split."
        )
    catalogue_path = processed_dir / "catalogue.csv"
    if not catalogue_path.is_file():
        raise FileNotFoundError(f"Missing predefined split catalogue: {catalogue_path}")

    import pandas as pd
    from scripts.build_features import FeatureBuilder, load, tidy

    catalogue = pd.read_csv(catalogue_path)
    if not {"movieId", "split"}.issubset(catalogue.columns):
        raise ValueError("catalogue.csv must contain movieId and split columns.")
    raw_movies, _, _ = load(raw_dir)
    movies = tidy(raw_movies).set_index("movieId", drop=False)
    if movies.index.has_duplicates or catalogue["movieId"].duplicated().any():
        raise ValueError("movieId values must be unique in raw data and catalogue.csv.")
    missing_ids = catalogue.loc[~catalogue["movieId"].isin(movies.index), "movieId"]
    if not missing_ids.empty:
        raise ValueError("Raw data does not match catalogue.csv movie IDs.")
    movies = movies.loc[catalogue["movieId"].to_numpy()].reset_index(drop=True)
    movie_ids = catalogue["movieId"].to_numpy()
    split_indices = split_movie_indices(movie_ids, catalogue["split"].to_numpy())

    builder = fit_experiment_builder(FeatureBuilder(), movies, split_indices["train"])
    full_matrix, block_slices, _ = builder.transform(movies)
    selected_matrix = select_experiment_a_features(full_matrix, block_slices)
    return PreparedData(
        features={
            name: selected_matrix[indices] for name, indices in split_indices.items()
        },
        movie_ids={name: movie_ids[indices] for name, indices in split_indices.items()},
        builder=builder,
        input_dim=selected_matrix.shape[1],
    )


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
        "experiment": "AE_A",
        "feature_groups": list(FEATURE_GROUPS),
        "input_dim": prepared.input_dim,
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


def main() -> None:
    """Run Experiment A after the user has provided the raw movie dataset."""
    set_seed()
    prepared = prepare_experiment_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = autoencoder(prepared.input_dim, HIDDEN_DIM, LATENT_DIM).to(device)
    history, best_epoch, best_loss = train_autoencoder(
        model, prepared.features["train"], prepared.features["val"], device=device
    )
    save_experiment_outputs(
        model, prepared, history, best_epoch, best_loss, device=device
    )


if __name__ == "__main__":
    main()

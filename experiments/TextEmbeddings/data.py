"""Data loading for the text-embedding experiments (independent of other methods' code).

Reads the shared processed files only:
    data/processed/movies_clean.csv   one readable row per movie (lists joined by "|")
    data/processed/catalogue.csv      the shared film list and train/held-out split

Movies are returned in catalogue order, with list fields parsed into Python lists and
a ``text`` column (overview + tagline). Nothing is fitted here.
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 2026
VALIDATION_FRACTION = 0.15
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
LIST_COLUMNS = (
    "genres",
    "keywords",
    "cast",
    "directors",
    "writers",
    "companies",
    "collection",
)
REQUIRED_COLUMNS = {"movieId", "title", "overview_tagline", "language", "year", "split"}


def parse_pipe_list(value) -> list[str]:
    """'a|b|c' -> ['a', 'b', 'c']; missing or empty -> []."""
    if not isinstance(value, str):
        return []
    return [item for item in value.split("|") if item]


def load_movies(processed_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Cleaned movies in catalogue order; fails loudly on any inconsistency."""
    movies = pd.read_csv(Path(processed_dir) / "movies_clean.csv")
    catalogue = pd.read_csv(Path(processed_dir) / "catalogue.csv")
    missing = (REQUIRED_COLUMNS | set(LIST_COLUMNS)) - set(movies.columns)
    if missing:
        raise ValueError(f"movies_clean.csv is missing columns: {sorted(missing)}")
    for name, df in (("movies_clean.csv", movies), ("catalogue.csv", catalogue)):
        if df["movieId"].duplicated().any():
            raise ValueError(f"{name} contains duplicated movieIds.")
    if set(movies["movieId"]) != set(catalogue["movieId"]):
        raise ValueError("movies_clean.csv and catalogue.csv contain different movies.")

    movies = (
        movies.set_index("movieId", drop=False)
        .loc[catalogue["movieId"]]
        .reset_index(drop=True)
    )
    if not np.array_equal(movies["split"].to_numpy(), catalogue["split"].to_numpy()):
        raise ValueError("The split in movies_clean.csv differs from catalogue.csv.")
    if not movies["split"].isin(["train", "heldout"]).all():
        raise ValueError("split must contain only 'train' and 'heldout'.")
    for column in LIST_COLUMNS:
        movies[column] = movies[column].map(parse_pipe_list)
    movies["text"] = movies["overview_tagline"].fillna("").astype(str).str.strip()
    movies["language"] = movies["language"].fillna("").astype(str)
    return movies


def validation_split(
    split: np.ndarray,
    fraction: float = VALIDATION_FRACTION,
    seed: int = SEED,
) -> dict[str, np.ndarray]:
    """Split the TRAINING movies into sub-train / validation row positions (used only
    to choose hyper-parameters). Held-out movies are never included."""
    split = np.asarray(split)
    if not 0 < fraction < 1:
        raise ValueError("fraction must be between 0 and 1.")
    train = np.flatnonzero(split == "train")
    n_val = max(1, int(np.ceil(len(train) * fraction)))
    if n_val >= len(train):
        raise ValueError("The validation subset would leave no training movies.")
    permuted = np.random.default_rng(seed).permutation(train)
    return {"train": np.sort(permuted[n_val:]), "val": np.sort(permuted[:n_val])}

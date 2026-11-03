"""Runner shared by the text-embedding experiments.

One run = one encoder x one input variant, named ``text_<encoder>_<variant>``:

1. build the input text of the 9,025 catalogue movies (``text_inputs``);
2. embed them (``encoders``); the fitted encoder (LSA) is fitted on
   training movies only, with their hyper-parameters chosen on a validation
   subset of the training movies;
3. save ``results/TextEmbeddings/<name>/embedding.npz`` in the shared format
   (plus ``config.json``, ``sample_texts.txt`` and, for fitted encoders, the
   fitted encoder), and reuse it on later runs unless ``--force`` is given;
4. evaluate with the shared ``scripts/evaluate_embedding.py`` (rows in
   ``results/metrics.csv``, maps, neighbours), with UMAP 2D/3D fitted on train;
5. compute the extra metrics (``extra_metrics``) on the native space and on the
   same UMAP 2D/3D projections -> ``results/TextEmbeddings/extra_metrics.csv``.

Run from the repository root, e.g.
    python -m experiments.TextEmbeddings.embed_mpnet --input all
"""

import argparse
import json
import time
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from experiments.TextEmbeddings import data
from experiments.TextEmbeddings.encoders import FITTED_ENCODERS, make_encoder
from experiments.TextEmbeddings.extra_metrics import extra_metrics, genre_matrix
from experiments.TextEmbeddings.text_inputs import INPUT_VARIANTS, build_texts

SEED = 2026  # same seed as the shared evaluator (UMAP, KMeans)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
TEXT_RESULTS_DIR = RESULTS_DIR / "TextEmbeddings"
EXTRA_METRICS_PATH = TEXT_RESULTS_DIR / "extra_metrics.csv"
SAMPLE_TITLES = ("Toy Story", "The Godfather", "Pulp Fiction", "The Matrix")


def load_movies() -> pd.DataFrame:
    """Shared cleaned movies, in catalogue order, with the canonical split."""
    return data.load_movies()


@lru_cache(maxsize=2)
def pretrained_encoder(encoder_key: str):
    """Load a pretrained encoder once per process (shared by the input variants)."""
    return make_encoder(encoder_key)


def run_name(encoder_key: str, variant: str) -> str:
    return f"text_{encoder_key}_{variant}"


def umap_spaces(Z: np.ndarray, train: np.ndarray) -> dict[str, np.ndarray]:
    """UMAP 2D and 3D fitted on train movies, exactly as the shared evaluator
    does it (same parameters and seed), so the extra metrics describe the same
    projections as ``results/metrics.csv``."""
    import umap

    return {
        f"umap{c}d": umap.UMAP(n_components=c, random_state=SEED)
        .fit(Z[train])
        .transform(Z)
        for c in (2, 3)
    }


# ----------------------------------------------------------------------------- fitted encoders
def select_params(
    encoder_key: str, texts: list[str], movies: pd.DataFrame, G: np.ndarray
) -> tuple[dict, list[dict]]:
    """Pick the configuration of a fitted encoder (LSA) on a validation
    subset of the TRAINING movies (held-out movies are never used). Score: genre
    Jaccard between each validation movie and its 10 nearest sub-train movies."""
    from metrics.metrics import heldout_neighbor_genre_similarity

    encoder_class, grid = FITTED_ENCODERS[encoder_key]
    split = data.validation_split(movies["split"].to_numpy())
    sub_train, val = split["train"], split["val"]
    scores = []
    for params in grid:
        encoder = encoder_class(**params).fit([texts[i] for i in sub_train])
        Z_tr = encoder.embed([texts[i] for i in sub_train])
        Z_val = encoder.embed([texts[i] for i in val])
        score = heldout_neighbor_genre_similarity(
            Z_tr, Z_val, G[sub_train], G[val], k=10, metric="cosine"
        )
        scores.append({**params, "val_genre_jaccard": float(score)})
        print(f"  {encoder_key} {params}: validation genre Jaccard {score:.4f}")
    best = max(scores, key=lambda s: s["val_genre_jaccard"])
    return {k: best[k] for k in grid[0]}, scores


# ----------------------------------------------------------------------------- outputs
def _write_samples(path: Path, movies: pd.DataFrame, texts: list[str]) -> None:
    lines = []
    for title in SAMPLE_TITLES:
        hit = np.flatnonzero(movies["title"].to_numpy() == title)
        if len(hit):
            lines.append(f"[{title}]\n{texts[hit[0]]}\n")
    path.write_text("\n".join(lines))


def _save_extra_rows(rows: list[dict], name: str, path: Path = EXTRA_METRICS_PATH):
    """Append the rows of one run, replacing earlier rows of the same run."""
    new = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old[old["method"] != name], new], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(path, index=False, float_format="%.4f")


def embed_catalogue(
    encoder_key: str,
    variant: str,
    movies: pd.DataFrame,
    *,
    encoder=None,
    force: bool = False,
) -> np.ndarray:
    """Embed every catalogue movie (or load the cached embedding)."""
    from scripts.evaluate_embedding import load_embedding, save_embedding

    name = run_name(encoder_key, variant)
    run_dir = TEXT_RESULTS_DIR / name
    path = run_dir / "embedding.npz"
    ids = movies["movieId"].to_numpy()
    if path.exists() and not force:
        Z, saved_ids = load_embedding(path)
        if np.array_equal(saved_ids, ids):
            print(f"{name}: using cached {path.relative_to(PROJECT_ROOT)}")
            return Z
        print(f"{name}: cached movie IDs differ from the catalogue, re-embedding")

    run_dir.mkdir(parents=True, exist_ok=True)
    texts = build_texts(movies, variant)
    train = (movies["split"] == "train").to_numpy()
    start = time.time()
    extra_config = {}
    if encoder_key in FITTED_ENCODERS:
        G = genre_matrix(movies["genres"])
        params, grid = select_params(encoder_key, texts, movies, G)
        # final fit on all training movies with the selected parameters
        encoder = FITTED_ENCODERS[encoder_key][0](**params).fit(
            [t for t, is_train in zip(texts, train) if is_train]
        )
        extra_config = {"selected_params": params, "grid": grid}
    elif encoder is None:
        encoder = pretrained_encoder(encoder_key)
    Z = encoder.embed(texts)
    config = {
        "name": name,
        "input_variant": variant,
        **encoder.config(),
        **extra_config,
        "dim": int(Z.shape[1]),
        "n_movies": int(len(Z)),
        "n_train": int(train.sum()),
        "truncated_share": encoder.truncated_share(texts),
        "mean_words": float(np.mean([len(t.split()) for t in texts])),
        "seconds": round(time.time() - start, 1),
    }
    save_embedding(path, Z, ids)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    _write_samples(run_dir / "sample_texts.txt", movies, texts)
    if encoder_key in FITTED_ENCODERS:
        joblib.dump(encoder, run_dir / "encoder.joblib")
    print(
        f"{name}: {Z.shape[0]} x {Z.shape[1]} in {config['seconds']} s, "
        f"truncated texts {config['truncated_share']:.1%}"
    )
    return Z


def evaluate_run(
    name: str,
    Z: np.ndarray,
    movies: pd.DataFrame,
    *,
    use_umap: bool = True,
    extras: bool = True,
) -> None:
    """Shared evaluation plus the extra metrics, for one embedding."""
    from scripts.evaluate_embedding import evaluate

    ids = movies["movieId"].to_numpy()
    evaluate(Z, ids, name, out_dir=RESULTS_DIR, use_umap=use_umap)
    if not extras:
        return
    train = (movies["split"] == "train").to_numpy()
    G = genre_matrix(movies["genres"])
    spaces = {"native": Z, **(umap_spaces(Z, train) if use_umap else {})}
    rows = []
    for space, S in spaces.items():
        row = {"method": name, "space": space, "dim": S.shape[1]}
        row.update(
            extra_metrics(
                S,
                train,
                G,
                movies["collection"],
                movies["directors"],
                Z_native=None if space == "native" else Z,
            )
        )
        rows.append(row)
        if space != "native":
            np.savez_compressed(
                TEXT_RESULTS_DIR / name / f"{space}.npz", Z=S, movieId=ids
            )
    _save_extra_rows(rows, name)
    print(f"saved: {EXTRA_METRICS_PATH.relative_to(PROJECT_ROOT)}")


def run_text_embedding(
    encoder_key: str,
    variant: str,
    *,
    movies: pd.DataFrame | None = None,
    encoder=None,
    force: bool = False,
    use_umap: bool = True,
    evaluate: bool = True,
) -> np.ndarray:
    movies = load_movies() if movies is None else movies
    Z = embed_catalogue(encoder_key, variant, movies, encoder=encoder, force=force)
    if evaluate:
        evaluate_run(run_name(encoder_key, variant), Z, movies, use_umap=use_umap)
    return Z


def embed_new_movies(records: list[dict], encoder_key: str, variant: str) -> np.ndarray:
    """Place new movies (dicts with the ``movies_clean.csv`` fields) in the space
    of a run. Pretrained encoders need nothing fitted; LSA uses the encoder
    fitted on training movies, saved by ``embed_catalogue``."""
    texts = build_texts(records, variant)
    if encoder_key in FITTED_ENCODERS:
        path = TEXT_RESULTS_DIR / run_name(encoder_key, variant) / "encoder.joblib"
        encoder = joblib.load(path)
    else:
        encoder = pretrained_encoder(encoder_key)
    return encoder.embed(texts)


def main(encoder_key: str, description: str = "") -> None:
    """Command line shared by the thin ``embed_*.py`` scripts."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--input",
        choices=(*INPUT_VARIANTS, "all"),
        default="all",
        help="input text variant (default: all three)",
    )
    parser.add_argument("--force", action="store_true", help="re-embed even if cached")
    parser.add_argument("--no-umap", action="store_true", help="skip UMAP 2D/3D")
    parser.add_argument(
        "--skip-eval", action="store_true", help="only compute and save embeddings"
    )
    args = parser.parse_args()
    variants = INPUT_VARIANTS if args.input == "all" else (args.input,)
    movies = load_movies()
    for variant in variants:
        run_text_embedding(
            encoder_key,
            variant,
            movies=movies,
            force=args.force,
            use_umap=not args.no_umap,
            evaluate=not args.skip_eval,
        )

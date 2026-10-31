"""Parametric UMAP embedding of the movies (cluster-focused track).

Parametric UMAP = UMAP whose map from features to coordinates is a neural network (the "encoder"),
so a new movie gets its position by passing its feature row through the encoder.

Input : data/processed/features.npz (from build_features.py) and data/processed/catalogue.csv
Fit   : on TRAIN movies only; then every movie (train + held-out) is placed with the encoder.
Output, for each configuration <name>:
    results/embeddings/<name>.npz     coordinates in the shared format (Z, movieId)
    results/models/<name>/encoder.keras   the encoder, to place new movies (see place_new_movies below)
    results/models/<name>/config.json     the settings used
    results/metrics.csv etc.          the shared evaluation (evaluate_embedding.py), unless --no-eval

Usage (from the repository root):
    python3 scripts/embed_parametric_umap.py                         # 2D and 3D, default settings
    python3 scripts/embed_parametric_umap.py --dims 2 --n-neighbors 15 50 --min-dist 0.1 0.5
Every combination of the listed values is fitted. One fit takes several minutes on a laptop CPU.

Settings, in short:
    n_neighbors   how many neighbours define the local structure; larger = more global structure
    min_dist      how tightly points may pack; smaller = more compact clusters
Note: umap-learn also offers a "global correlation" loss to keep large distances; in our tests
(umap-learn 0.5.12, Keras 3) it made training diverge, so it is not used here.
"""
import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")      # hide TensorFlow start-up messages

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_embedding import evaluate, save_embedding  # noqa: E402


def load_inputs(features_path, catalogue_path):
    z = np.load(features_path)
    X, ids = z["X"].astype(np.float32), z["movieId"]
    cat = pd.read_csv(catalogue_path)
    if len(cat) != len(ids) or not (cat.movieId.to_numpy() == ids).all():
        sys.exit("features.npz and catalogue.csv do not have the same movies in the same order: "
                 "regenerate both with build_features.py.")
    return X, ids, (cat.split == "train").to_numpy()


def fit_one(X, train, dim, n_neighbors, min_dist, metric, seed):
    import tensorflow as tf
    from umap.parametric_umap import ParametricUMAP
    tf.keras.utils.set_random_seed(seed)
    model = ParametricUMAP(n_components=dim, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric,
                           random_state=seed,
                           keras_fit_kwargs={"verbose": 0}, verbose=False)
    model.fit(X[train])
    return model, model.encoder.predict(X, verbose=0)


def place_new_movies(feature_rows, model_dir):
    """Coordinates of new movies: feature_rows come from builder.transform(...) of build_features.py."""
    import keras
    encoder = keras.models.load_model(Path(model_dir) / "encoder.keras")
    return encoder.predict(np.asarray(feature_rows, dtype=np.float32), verbose=0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dims", type=int, nargs="+", default=[2, 3])
    p.add_argument("--n-neighbors", type=int, nargs="+", default=[15])
    p.add_argument("--min-dist", type=float, nargs="+", default=[0.1])
    p.add_argument("--metric", default="euclidean")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--features", default=REPO / "data" / "processed" / "features.npz")
    p.add_argument("--catalogue", default=REPO / "data" / "processed" / "catalogue.csv")
    p.add_argument("--no-eval", action="store_true", help="only fit and save, skip the shared evaluation")
    a = p.parse_args()

    X, ids, train = load_inputs(a.features, a.catalogue)
    print(f"{len(ids)} movies, {X.shape[1]} features, {train.sum()} used for fitting")
    configs = list(itertools.product(a.dims, a.n_neighbors, a.min_dist, a.seeds))
    for i, (dim, nn, md, seed) in enumerate(configs, 1):
        name = f"pumap_{dim}d_nn{nn}_md{md:g}" + (f"_s{seed}" if seed else "")
        print(f"\n[{i}/{len(configs)}] {name}: fitting ...", flush=True)
        t0 = time.time()
        model, Z = fit_one(X, train, dim, nn, md, a.metric, seed)
        print(f"  done in {time.time() - t0:.0f} s")
        if not np.isfinite(Z).all():
            print("  WARNING: training diverged (NaN coordinates); nothing saved for this configuration.")
            continue

        save_embedding(REPO / "results" / "embeddings" / f"{name}.npz", Z, ids)
        out = REPO / "results" / "models" / name
        out.mkdir(parents=True, exist_ok=True)
        model.encoder.save(out / "encoder.keras")
        json.dump({"dim": dim, "n_neighbors": nn, "min_dist": md, "metric": a.metric, "seed": seed, "fitted_on": "train movies",
                   "fit_seconds": round(time.time() - t0)}, open(out / "config.json", "w"), indent=2)
        if not a.no_eval:
            evaluate(Z, ids, name)


if __name__ == "__main__":
    main()

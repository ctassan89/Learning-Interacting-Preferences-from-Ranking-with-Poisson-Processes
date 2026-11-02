"""Synthetic tests for post-processing discovery and artifact validation."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from experiments.postprocess import (
    REQUIRED_ARTIFACTS,
    _score_cluster,
    discover_experiments,
    genre_matrix,
    load_experiment,
    write_postprocessing,
)


def make_run(path: Path, *, nested: bool = False) -> Path:
    run = path / "BetaVAE_A" / "beta_2" if nested else path / "AE_A"
    run.mkdir(parents=True)
    config = {
        "experiment": "BetaVAE_A" if nested else "AE_A",
        "feature_groups": ["genres"],
        "input_dim": 20,
        "latent_dim": 2,
        **({"beta": 2.0} if nested else {}),
    }
    (run / "config.json").write_text(json.dumps(config))
    for name in REQUIRED_ARTIFACTS - {
        "config.json",
        "embeddings_train.npz",
        "embeddings_val.npz",
        "embeddings_heldout.npz",
    }:
        (run / name).write_text("fixture")
    for split, ids in (("train", [1, 2, 3]), ("val", [4, 5]), ("heldout", [6, 7])):
        np.savez(
            run / f"embeddings_{split}.npz",
            embeddings=np.ones((len(ids), 2)),
            movieId=ids,
        )
    return run


class PostprocessTests(unittest.TestCase):
    def test_discovers_single_and_nested_runs(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = make_run(root), make_run(root / "nested", nested=True)
            discovered, invalid = discover_experiments(root)
            self.assertEqual(discovered, sorted([first, second]))
            self.assertEqual(invalid, [])

    def test_load_validates_npz_schema_and_split_ids(self):
        with TemporaryDirectory() as directory:
            run_path = make_run(Path(directory))
            run = load_experiment(run_path)
            self.assertEqual(run.embeddings["train"].shape, (3, 2))
            np.savez(
                run_path / "embeddings_val.npz",
                embeddings=np.ones((2, 2)),
                movieId=[1, 5],
            )
            with self.assertRaises(ValueError):
                load_experiment(run_path)

    def test_postprocessing_output_protects_existing_results(self):
        with TemporaryDirectory() as directory:
            run = load_experiment(make_run(Path(directory)))
            report = {"status": "success", "clustering": {}, "semantic": {}}
            destination = write_postprocessing(run, report, no_plots=True)
            self.assertTrue((destination / "metrics.json").is_file())
            with self.assertRaises(FileExistsError):
                write_postprocessing(run, report, no_plots=True)
            write_postprocessing(run, report, force=True, no_plots=True)

    def test_genres_align_by_movie_id_and_preserve_multiple_labels(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            movies = root / "movies_clean.csv"
            catalogue = root / "catalogue.csv"
            movies.write_text("movieId,genres\n2,Comedy|Drama\n1,Action\n")
            catalogue.write_text("movieId\n1\n2\n")
            aligned = genre_matrix(
                np.asarray([1, 2]),
                movies_clean_path=movies,
                catalogue_path=catalogue,
            )
            self.assertEqual(aligned.tolist(), [[1, 0, 0], [0, 1, 1]])

    def test_clustering_metrics_handle_defined_and_undefined_scores(self):
        embeddings = np.vstack([np.zeros((6, 2)), np.ones((6, 2)) * 10])
        genres = np.vstack([np.tile([1, 0], (6, 1)), np.tile([0, 1], (6, 1))])
        scores = _score_cluster(embeddings, np.asarray([0] * 6 + [1] * 6), genres)
        self.assertIsInstance(scores["genre_purity"], float)
        undefined = _score_cluster(embeddings, np.zeros(12, dtype=int), genres)
        self.assertIsNone(undefined["silhouette"])
        self.assertIn("silhouette_reason", undefined)

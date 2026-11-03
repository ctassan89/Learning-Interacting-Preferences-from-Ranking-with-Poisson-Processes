"""Synthetic tests for train-fitted PCA movie embedding visualizations."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.decomposition import PCA

from experiments.visualize_pca import (
    EmbeddingRun,
    _arguments,
    discover_experiments,
    explained_variance_report,
    fit_and_transform_pca,
    genre_color_map,
    load_experiment,
    metadata_for_movie_ids,
    primary_genre,
    visualization_complete,
    write_visualization,
)


def make_run(root: Path) -> Path:
    run = root / "BetaVAE_A" / "beta_0.1"
    run.mkdir(parents=True)
    arrays = {
        "train": (
            np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]]),
            [11, 12, 13, 14],
        ),
        "val": (np.array([[4, 0, 0], [5, 0, 0]]), [15, 16]),
        "heldout": (np.array([[6, 0, 0], [7, 0, 0]]), [17, 18]),
    }
    for split, (embeddings, movie_ids) in arrays.items():
        np.savez(
            run / f"embeddings_{split}.npz", embeddings=embeddings, movieId=movie_ids
        )
    return run


def make_metadata(root: Path) -> Path:
    path = root / "movies_clean.csv"
    path.write_text(
        "movieId,title,genres\n"
        "15,Val Movie,Drama|Comedy\n"
        "11,Train One,Action|Comedy\n"
        "18,Held Two,\n"
        "14,Train Four,Comedy\n"
        "12,Train Two,Animation|Family\n"
        "17,Held One,Science Fiction|Drama\n"
        "13,Train Three,Action\n"
        "16,Val Two,Thriller|Drama\n"
    )
    return path


class VisualizePcaTests(unittest.TestCase):
    def test_pca_fits_only_training_and_transforms_other_splits(self):
        embeddings = {
            "train": np.array([[0, 0, 0], [1, 2, 0], [2, 1, 1], [3, 5, 2.0]]),
            "val": np.array([[100, 50, 25.0]]),
            "heldout": np.array([[-20, 3, 7.0]]),
        }
        pca, transformed = fit_and_transform_pca(embeddings)
        expected = PCA(n_components=3).fit(embeddings["train"])
        np.testing.assert_allclose(pca.mean_, expected.mean_)
        np.testing.assert_allclose(
            transformed["train"], expected.transform(embeddings["train"])
        )
        np.testing.assert_allclose(
            transformed["val"], expected.transform(embeddings["val"])
        )
        np.testing.assert_allclose(
            transformed["heldout"], expected.transform(embeddings["heldout"])
        )

    def test_metadata_aligns_by_movie_id_not_csv_order(self):
        with TemporaryDirectory() as directory:
            metadata = make_metadata(Path(directory))
            aligned = metadata_for_movie_ids(np.array([13, 11, 18]), metadata)
            self.assertEqual(
                aligned["title"].tolist(), ["Train Three", "Train One", "Held Two"]
            )
            self.assertEqual(
                aligned["genres"].fillna("").tolist(), ["Action", "Action|Comedy", ""]
            )

    def test_primary_genre_is_first_source_label_or_unknown(self):
        self.assertEqual(primary_genre("Drama|Comedy"), "Drama")
        self.assertEqual(primary_genre(" | Action | Comedy"), "Action")
        self.assertEqual(primary_genre(""), "Unknown")
        self.assertEqual(primary_genre(np.nan), "Unknown")

    def test_explained_variance_report_uses_train_fitted_pca(self):
        train = np.array([[0, 0, 0], [1, 2, 0], [2, 1, 1], [3, 5, 2.0]])
        pca = PCA(n_components=3).fit(train)
        report = explained_variance_report(pca)
        ratios = pca.explained_variance_ratio_
        self.assertEqual(report["fit_split"], "train")
        self.assertAlmostEqual(report["explained_variance_ratio"]["PC2"], ratios[1])
        self.assertAlmostEqual(
            report["cumulative_explained_variance"]["PC1_PC2"], sum(ratios[:2])
        )
        self.assertAlmostEqual(
            report["cumulative_explained_variance"]["PC1_PC3"], sum(ratios)
        )

    def test_discovery_and_visualization_output_paths(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            run_path, metadata = make_run(root), make_metadata(root)
            self.assertEqual(discover_experiments(root), [run_path])
            run = load_experiment(run_path)
            with patch(
                "experiments.visualize_pca.save_2d_plot",
                side_effect=lambda *args: args[-1].write_bytes(b"png"),
            ) as plot_2d, patch(
                "experiments.visualize_pca.save_3d_plot",
                side_effect=lambda *args: args[-1].write_text("html"),
            ) as plot_3d:
                output = write_visualization(
                    run, movies_clean_path=metadata, splits=("train",)
                )
                plot_2d.assert_called_once()
                plot_3d.assert_called_once()
            self.assertEqual(output, run_path / "pca_visualization")
            self.assertTrue((output / "explained_variance.json").is_file())
            self.assertTrue(visualization_complete(output))
            report = json.loads((output / "explained_variance.json").read_text())
            self.assertEqual(report["n_components"], 3)
            self.assertEqual(
                set(genre_color_map(metadata)),
                {
                    "Action",
                    "Animation",
                    "Comedy",
                    "Drama",
                    "Science Fiction",
                    "Thriller",
                    "Unknown",
                },
            )

    def test_cli_selects_experiment_and_requested_splits(self):
        with patch(
            "sys.argv",
            [
                "visualize_pca",
                "--experiment",
                "results/AE_A",
                "--splits",
                "train",
                "val",
                "--dry-run",
            ],
        ):
            args = _arguments()
        self.assertEqual(args.experiment, Path("results/AE_A"))
        self.assertEqual(args.splits, ["train", "val"])
        self.assertTrue(args.dry_run)


if __name__ == "__main__":
    unittest.main()

"""Synthetic tests for train-fitted AE_A UMAP visualizations."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from experiments.dimensionality_reduction.visualize_pca import EmbeddingRun
from experiments.dimensionality_reduction.visualize_umap import (
    AE_A_PATH,
    UmapConfiguration,
    _arguments,
    _display_data,
    _selected_cluster_configuration,
    _cluster_metrics,
    configurations,
    evaluate_umap_clusters,
    experiment_label,
    evaluate_umap,
    fit_and_transform_umap,
    output_complete,
    run_configuration,
    write_cluster_evaluation,
)


def synthetic_run(dimensions: int = 4) -> EmbeddingRun:
    generator = np.random.default_rng(42)
    embeddings = {
        "train": generator.normal(size=(24, dimensions)),
        "val": generator.normal(size=(6, dimensions)),
        "heldout": generator.normal(size=(8, dimensions)),
    }
    movie_ids = {
        "train": np.arange(1, 25),
        "val": np.arange(25, 31),
        "heldout": np.arange(31, 39),
    }
    return EmbeddingRun(Path("synthetic"), embeddings, movie_ids)


class FakeUmap:
    """Minimal mapper used to assert which rows are passed to fitting."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.fit_rows = None

    def fit_transform(self, values):
        self.fit_rows = values.copy()
        return values[:, : self.kwargs["n_components"]]

    def transform(self, values):
        return values[:, : self.kwargs["n_components"]]


class SpyKMeans:
    """Train-only clusterer double with observable fit/predict inputs."""

    fit_values = None
    predict_values = []

    def __init__(self, **_kwargs):
        pass

    def fit(self, values):
        type(self).fit_values = values.copy()
        self.labels_ = np.arange(len(values)) % 10
        return self

    def predict(self, values):
        type(self).predict_values.append(values.copy())
        return np.arange(len(values)) % 10


class SpyGmm(SpyKMeans):
    fit_values = None
    predict_values = []


class VisualizeUmapTests(unittest.TestCase):
    def test_configuration_table_has_all_dimensions_and_hyperparameters(self):
        selected = configurations()
        self.assertEqual(len(selected), 6)
        self.assertEqual({item.n_components for item in selected}, {2, 3})
        self.assertEqual(
            {(item.n_neighbors, item.min_dist) for item in selected},
            {(15, 0.1), (30, 0.1), (100, 0.3)},
        )

    def test_umap_is_fit_only_on_training_rows_and_transforms_other_splits(self):
        run = synthetic_run()
        config = UmapConfiguration(2, 15, 0.1)
        with patch("umap.UMAP", FakeUmap):
            mapper, coordinates = fit_and_transform_umap(run.embeddings, config)
        np.testing.assert_array_equal(mapper.fit_rows, run.embeddings["train"])
        self.assertEqual(coordinates["train"].shape, (24, 2))
        self.assertEqual(coordinates["val"].shape, (6, 2))
        self.assertEqual(coordinates["heldout"].shape, (8, 2))
        self.assertEqual(mapper.kwargs["metric"], "euclidean")
        self.assertEqual(mapper.kwargs["random_state"], 42)

    def test_evaluation_identity_projection_has_perfect_neighbor_preservation(self):
        run = synthetic_run()
        genres = np.tile([[1, 0], [0, 1]], (16, 1))
        with patch(
            "experiments.dimensionality_reduction.visualize_umap.genre_matrix",
            return_value=genres,
        ):
            report = evaluate_umap(run, run.embeddings, sample_size=20)
        self.assertAlmostEqual(report["trustworthiness"]["value"], 1.0)
        self.assertAlmostEqual(report["neighbor_overlap"]["train"]["value"], 1.0)
        self.assertAlmostEqual(
            report["neighbor_overlap"]["heldout_to_train"]["value"], 1.0
        )
        self.assertEqual(report["trustworthiness"]["scope"], "training sample only")
        self.assertIn("held-out", report["genre_neighbor_jaccard"]["heldout_scope"])

    def test_metadata_is_aligned_by_movie_id_for_plot_rows(self):
        run = synthetic_run(dimensions=3)
        with TemporaryDirectory() as directory:
            metadata = Path(directory) / "movies.csv"
            metadata.write_text(
                "movieId,title,genres\n2,Second,Drama|Comedy\n1,First,Action\n"
            )
            small = EmbeddingRun(
                Path(directory),
                {
                    "train": run.embeddings["train"][:2],
                    "val": run.embeddings["val"][:1],
                    "heldout": run.embeddings["heldout"][:1],
                },
                {
                    "train": np.array([1, 2]),
                    "val": np.array([1]),
                    "heldout": np.array([2]),
                },
            )
            with patch(
                "experiments.dimensionality_reduction.visualize_umap.MOVIES_CLEAN_PATH",
                metadata,
            ):
                data = _display_data(small, small.embeddings, ("train",))
        self.assertEqual(data["title"].tolist(), ["First", "Second"])
        self.assertEqual(data["primary_genre"].tolist(), ["Action", "Drama"])

    def test_output_structure_and_no_overwrite_without_force(self):
        run = synthetic_run()
        config = UmapConfiguration(2, 15, 0.1)
        report = {"evaluation": {}}
        with TemporaryDirectory() as directory:
            output_root = Path(directory) / "umap_visualization"

            def save_plots(_data, _config, destination, _label):
                (destination / "umap_2d.png").write_bytes(b"png")
                (destination / "umap_2d.html").write_text("html")

            with patch(
                "experiments.dimensionality_reduction.visualize_umap.fit_and_transform_umap",
                return_value=(object(), run.embeddings),
            ), patch(
                "experiments.dimensionality_reduction.visualize_umap.evaluate_umap",
                return_value=report,
            ), patch(
                "experiments.dimensionality_reduction.visualize_umap._display_data",
                return_value=object(),
            ), patch(
                "experiments.dimensionality_reduction.visualize_umap._save_plots",
                side_effect=save_plots,
            ):
                destination, _ = run_configuration(run, config, output_root=output_root)
                self.assertTrue(output_complete(destination, 2))
                saved = json.loads((destination / "metrics.json").read_text())
                self.assertEqual(saved["configuration"]["fit_split"], "train")
                with self.assertRaises(FileExistsError):
                    run_configuration(run, config, output_root=output_root)

    def test_cli_experiment_defaults_to_baseline_and_accepts_ablation_path(self):
        with patch("sys.argv", ["visualize_umap", "--dry-run"]):
            default_args = _arguments()
        self.assertEqual(default_args.experiment, AE_A_PATH)
        large_path = Path(
            "results/dimensionality_reduction/ArchitectureAblation/AE_A_large"
        )
        with patch(
            "sys.argv",
            ["visualize_umap", "--experiment", str(large_path), "--n-components", "3"],
        ):
            large_args = _arguments()
        self.assertEqual(large_args.experiment, large_path)
        self.assertEqual(large_args.n_components, [3])
        self.assertEqual(experiment_label(large_path), "AE-A Large")

    def test_cluster_evaluation_fits_only_train_and_predicts_heldout(self):
        run = synthetic_run(dimensions=2)
        coordinates = {
            "train": run.embeddings["train"],
            "heldout": run.embeddings["heldout"],
        }
        genres = np.tile([[1, 0], [0, 1]], (16, 1))
        SpyKMeans.predict_values, SpyGmm.predict_values = [], []
        with patch(
            "experiments.dimensionality_reduction.visualize_umap.KMeans", SpyKMeans
        ), patch(
            "experiments.dimensionality_reduction.visualize_umap.GaussianMixture",
            SpyGmm,
        ), patch(
            "experiments.dimensionality_reduction.visualize_umap.genre_matrix",
            return_value=genres,
        ), patch(
            "experiments.dimensionality_reduction.visualize_umap._genre_names",
            return_value=["Action", "Drama"],
        ):
            report, details = evaluate_umap_clusters(run, coordinates)
        np.testing.assert_array_equal(SpyKMeans.fit_values, coordinates["train"])
        np.testing.assert_array_equal(SpyGmm.fit_values, coordinates["train"])
        np.testing.assert_array_equal(
            SpyKMeans.predict_values[-1], coordinates["heldout"]
        )
        np.testing.assert_array_equal(SpyGmm.predict_values[-1], coordinates["heldout"])
        self.assertEqual(report["coordinate_space"], "native 2D UMAP coordinates")
        self.assertEqual(
            {row["split"] for row in details["kmeans"]}, {"train", "heldout"}
        )

    def test_cluster_evaluation_uses_train_as_heldout_neighbor_reference(self):
        run = synthetic_run(dimensions=2)
        coordinates = {
            "train": run.embeddings["train"],
            "heldout": run.embeddings["heldout"],
        }
        genres = np.tile([[1, 0], [0, 1]], (16, 1))
        with patch(
            "experiments.dimensionality_reduction.visualize_umap.genre_matrix",
            return_value=genres,
        ), patch(
            "experiments.dimensionality_reduction.visualize_umap._genre_names",
            return_value=["Action", "Drama"],
        ), patch(
            "experiments.dimensionality_reduction.visualize_umap.heldout_neighbor_genre_similarity",
            return_value=0.5,
        ) as heldout_similarity:
            evaluate_umap_clusters(run, coordinates)
        np.testing.assert_array_equal(
            heldout_similarity.call_args.args[0], coordinates["train"]
        )
        np.testing.assert_array_equal(
            heldout_similarity.call_args.args[1], coordinates["heldout"]
        )

    def test_cluster_metric_uses_multilabel_purity_and_records_undefined_geometry(self):
        coordinates = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        genres = np.array([[1, 0], [1, 1], [0, 1]])
        defined = _cluster_metrics(
            coordinates, np.array([0, 0, 1]), genres, random_state=42
        )
        self.assertAlmostEqual(defined["genre_purity"]["value"], 1.0)
        undefined = _cluster_metrics(
            coordinates, np.zeros(3, dtype=int), genres, random_state=42
        )
        self.assertIsNone(undefined["silhouette"]["value"])
        self.assertIsNotNone(undefined["silhouette"]["reason"])
        self.assertIsNone(undefined["davies_bouldin"]["value"])

    def test_cluster_evaluation_writes_new_directory_without_touching_umap_artifacts(
        self,
    ):
        run = synthetic_run(dimensions=2)
        configuration = UmapConfiguration(2, 100, 0.3)
        genres = np.tile([[1, 0], [0, 1]], (16, 1))
        with TemporaryDirectory() as directory:
            run = EmbeddingRun(Path(directory), run.embeddings, run.movie_ids)
            original = run.path / "umap_visualization" / configuration.label
            original.mkdir(parents=True)
            (original / "metrics.json").write_text("existing")
            with patch(
                "experiments.dimensionality_reduction.visualize_umap.fit_and_transform_umap",
                return_value=(
                    object(),
                    {
                        "train": run.embeddings["train"],
                        "heldout": run.embeddings["heldout"],
                    },
                ),
            ), patch(
                "experiments.dimensionality_reduction.visualize_umap.genre_matrix",
                return_value=genres,
            ), patch(
                "experiments.dimensionality_reduction.visualize_umap._genre_names",
                return_value=["Action", "Drama"],
            ):
                destination = write_cluster_evaluation(run, configuration)
            self.assertEqual((original / "metrics.json").read_text(), "existing")
            self.assertTrue((destination / "train_test_metrics.json").is_file())
            self.assertTrue((destination / "kmeans_cluster_details.csv").is_file())
            self.assertTrue((destination / "gmm_cluster_details.csv").is_file())

    def test_cluster_cli_selects_only_default_or_explicit_2d_setting(self):
        with patch(
            "sys.argv", ["visualize_umap", "--n-components", "2", "--evaluate-clusters"]
        ):
            default = _selected_cluster_configuration(_arguments())
        self.assertEqual(default, UmapConfiguration(2, 100, 0.3))
        with patch(
            "sys.argv",
            [
                "visualize_umap",
                "--n-components",
                "2",
                "--evaluate-clusters",
                "--n-neighbors",
                "30",
                "--min-dist",
                "0.1",
            ],
        ):
            selected = _selected_cluster_configuration(_arguments())
        self.assertEqual(selected, UmapConfiguration(2, 30, 0.1))
        with patch(
            "sys.argv", ["visualize_umap", "--n-components", "3", "--evaluate-clusters"]
        ):
            with self.assertRaises(ValueError):
                _selected_cluster_configuration(_arguments())


if __name__ == "__main__":
    unittest.main()

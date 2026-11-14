"""Synthetic coverage for the isolated native 2D/3D experiment pipeline."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.dimensionality_reduction.common import (
    PreparedData,
    extract_vae_embeddings,
    fit_experiment_builder,
)
from experiments.dimensionality_reduction.low_dim import (
    CONFIGURATIONS,
    LowDimConfig,
    _write_training_outputs,
    build_model,
    latent_visualization_complete,
    select_configurations,
    training_complete,
    write_comparative_summary,
)
from models.VAE import VAE


class LowDimTests(unittest.TestCase):
    def test_fixed_configuration_table_contains_the_eight_requested_runs(self):
        self.assertEqual(len(CONFIGURATIONS), 8)
        self.assertEqual(
            {
                (item.model, item.features, item.latent_dim, item.beta)
                for item in CONFIGURATIONS
            },
            {
                ("AE", "A", 2, None),
                ("AE", "A", 3, None),
                ("AE", "B", 2, None),
                ("AE", "B", 3, None),
                ("VAE", "A", 2, None),
                ("VAE", "A", 3, None),
                ("BetaVAE", "A", 2, 0.1),
                ("BetaVAE", "A", 3, 0.1),
            },
        )
        self.assertEqual(
            LowDimConfig("AE", "A", 2).feature_groups, ("genres", "keywords", "text")
        )
        self.assertEqual(
            LowDimConfig("AE", "B", 2).feature_groups[-3:],
            ("cast", "directors", "writers"),
        )

    def test_model_construction_and_posterior_means_have_native_dimensions(self):
        for model_name in ("AE", "VAE", "BetaVAE"):
            for dimension in (2, 3):
                config = LowDimConfig(
                    model_name, "A", dimension, 0.1 if model_name == "BetaVAE" else None
                )
                model = build_model(config, input_dim=7)
                if model_name == "AE":
                    result = model.encoder(torch.zeros(4, 7))
                else:
                    result, _ = model.encoder(torch.zeros(4, 7))
                self.assertEqual(tuple(result.shape), (4, dimension))
        vae = VAE(7, 8, 3)
        data = np.random.default_rng(4).normal(size=(5, 7)).astype("float32")
        with torch.no_grad():
            expected, _ = vae.encoder(torch.as_tensor(data))
        np.testing.assert_allclose(
            extract_vae_embeddings(vae, data, torch.device("cpu")), expected.numpy()
        )

    def test_completion_checks_require_all_artifacts_and_correct_visualization_kind(
        self,
    ):
        with TemporaryDirectory() as directory:
            path = Path(directory)
            self.assertFalse(training_complete(path))
            for name in (
                "model.pt",
                "history.csv",
                "loss_curve.png",
                "embeddings_train.npz",
                "embeddings_val.npz",
                "embeddings_heldout.npz",
                "feature_builder.joblib",
            ):
                (path / name).write_text("fixture")
            (path / "config.json").write_text(json.dumps({"latent_dim": 2}))
            self.assertTrue(training_complete(path))
            visualization = path / "latent_visualization"
            visualization.mkdir()
            (visualization / "latent_2d.png").write_text("fixture")
            (visualization / "latent_2d.html").write_text("fixture")
            self.assertTrue(latent_visualization_complete(path, 2))
            self.assertFalse(latent_visualization_complete(path, 3))
            (visualization / "latent_3d.html").write_text("fixture")
            self.assertTrue(latent_visualization_complete(path, 3))

    def test_saved_config_and_checkpoint_keep_the_requested_latent_dimension(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = PreparedData(
                features={
                    split: np.ones((3, 5), dtype=np.float32)
                    for split in ("train", "val", "heldout")
                },
                movie_ids={
                    "train": np.array([1, 2, 3]),
                    "val": np.array([4, 5, 6]),
                    "heldout": np.array([7, 8, 9]),
                },
                builder={"synthetic": True},
                input_dim=5,
                block_dimensions={"genres": 5},
                block_weights={"genres": 1.5},
            )
            config = LowDimConfig("AE", "A", 3)
            model = build_model(config, prepared.input_dim)
            history = [{"epoch": 1, "train_loss": 0.2, "val_loss": 0.1}]
            with patch(
                "experiments.dimensionality_reduction.low_dim.LOW_DIM_ROOT", root
            ):
                _write_training_outputs(
                    config,
                    model,
                    prepared,
                    history,
                    1,
                    {"best_validation_loss": 0.1},
                    torch.device("cpu"),
                )
                output = config.path
            saved = json.loads((output / "config.json").read_text())
            checkpoint = torch.load(output / "model.pt", weights_only=True)
            self.assertEqual(saved["latent_dim"], 3)
            self.assertEqual(checkpoint["architecture"]["latent_dim"], 3)
            self.assertTrue(training_complete(output))

    def test_builder_is_fit_only_on_explicit_training_rows(self):
        class RecordingBuilder:
            def fit(self, rows):
                self.movie_ids = rows["movieId"].tolist()
                return self

        import pandas as pd

        builder = RecordingBuilder()
        movies = pd.DataFrame({"movieId": [10, 20, 30, 40]})
        fitted = fit_experiment_builder(builder, movies, np.array([0, 2]))
        self.assertEqual(fitted.movie_ids, [10, 30])

    def test_cli_filters_and_invalid_filters(self):
        selected = select_configurations(
            model="BetaVAE", features="A", latent_dim=3, beta=0.1
        )
        self.assertEqual(selected, [LowDimConfig("BetaVAE", "A", 3, 0.1)])
        self.assertEqual(len(select_configurations(latent_dim=2)), 4)
        with self.assertRaises(ValueError):
            select_configurations(model="VAE", features="B")

    def test_summary_is_isolated_from_existing_32d_summary(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            low_root = root / "results" / "LowDim"
            config = CONFIGURATIONS[0]
            low_metrics = low_root / "AE_A" / "latent_2" / "postprocessing"
            low_metrics.mkdir(parents=True)
            (low_metrics / "metrics.json").write_text(
                json.dumps(
                    {"clustering": {"kmeans": {"silhouette": 0.2}}, "semantic": {}}
                )
            )
            baseline = root / "results" / "AE_A" / "postprocessing"
            baseline.mkdir(parents=True)
            (baseline / "metrics.json").write_text(
                json.dumps(
                    {"clustering": {"kmeans": {"silhouette": 0.4}}, "semantic": {}}
                )
            )
            with patch(
                "experiments.dimensionality_reduction.low_dim.PROJECT_ROOT", root
            ), patch(
                "experiments.dimensionality_reduction.low_dim.LOW_DIM_ROOT", low_root
            ):
                output = write_comparative_summary()
            self.assertEqual(output, low_root / "postprocessing_summary.csv")
            self.assertTrue(output.is_file())
            self.assertFalse((root / "results" / "postprocessing_summary.csv").exists())

    def test_visualization_uses_native_coordinates_without_pca(self):
        import inspect
        import experiments.dimensionality_reduction.low_dim as low_dim

        source = inspect.getsource(low_dim.write_native_visualization)
        self.assertNotIn("fit_and_transform_pca", source)
        self.assertIn('x="z1"', source)
        self.assertIn('y="z2"', source)
        self.assertIn('z="z3"', source)

    def test_shared_postprocessing_accepts_native_2d_and_3d_embeddings(self):
        from experiments.dimensionality_reduction import postprocess

        generator = np.random.default_rng(42)
        ids = {
            "train": np.arange(100, 112),
            "val": np.arange(112, 116),
            "heldout": np.arange(116, 120),
        }
        for dimension in (2, 3):
            run = postprocess.ExperimentRun(
                path=Path("synthetic"),
                config={
                    "experiment": "AE_A",
                    "feature_groups": ["genres"],
                    "input_dim": 1,
                    "latent_dim": dimension,
                },
                embeddings={
                    split: generator.normal(size=(len(split_ids), dimension))
                    for split, split_ids in ids.items()
                },
                movie_ids=ids,
            )
            with patch(
                "experiments.dimensionality_reduction.postprocess.genre_matrix",
                return_value=np.tile([[1, 0], [0, 1]], (8, 1)),
            ):
                report = postprocess.evaluate_run(run, use_hdbscan=False)
            self.assertEqual(report["status"], "success")
            self.assertIn("kmeans", report["clustering"])


if __name__ == "__main__":
    unittest.main()

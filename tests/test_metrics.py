"""Tests for reusable movie-embedding evaluation metrics."""

import unittest

import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from experiments.common import (
    DATASET_VERSION,
    FEATURE_CONFIGURATIONS,
    FEATURE_GROUPS,
    _vae_batch_losses,
    extract_embeddings,
    extract_vae_embeddings,
    fit_experiment_builder,
    load_cleaned_movies,
    prepare_experiment_data,
    restore_best_state,
    select_experiment_a_features,
    split_movie_indices,
)
from metrics.metrics import (
    average_neighbor_genre_similarity,
    assign_heldout_clusters,
    calinski_harabasz_index,
    davies_bouldin_index,
    evaluate_embeddings,
    evaluate_generalization,
    fit_pca_projection,
    fit_hdbscan,
    genre_purity,
    heldout_placement,
    nearest_neighbors,
    nearest_training_neighbors,
    project_embeddings,
    reconstruction_metrics,
    silhouette_score,
)


EMBEDDINGS = np.array(
    [[0.0, 0.0], [0.0, 0.1], [0.1, 0.0], [5.0, 5.0], [5.0, 5.1], [5.1, 5.0]]
)
LABELS = np.array([0, 0, 0, 1, 1, 1])
GENRES = np.array([[1, 0], [1, 0], [1, 1], [0, 1], [0, 1], [0, 1]])
HELDOUT_EMBEDDINGS = np.array([[0.0, 0.05], [0.1, 0.05], [5.0, 5.05], [5.1, 5.05]])
HELDOUT_LABELS = np.array([0, 0, 1, 1])
HELDOUT_GENRES = np.array([[1, 0], [1, 1], [0, 1], [0, 1]])


class MetricsTests(unittest.TestCase):
    def test_clustering_metrics(self):
        self.assertGreater(silhouette_score(EMBEDDINGS, LABELS), 0.9)
        self.assertLess(davies_bouldin_index(EMBEDDINGS, LABELS), 0.1)
        self.assertGreater(calinski_harabasz_index(EMBEDDINGS, LABELS), 100.0)

    def test_invalid_clustering_configuration_raises(self):
        with self.assertRaises(ValueError):
            silhouette_score(EMBEDDINGS, np.zeros(len(EMBEDDINGS), dtype=int))
        with self.assertRaises(ValueError):
            silhouette_score(EMBEDDINGS[:, 0], LABELS)

    def test_multilabel_genre_purity(self):
        self.assertAlmostEqual(genre_purity(LABELS, GENRES), 1.0)
        mixed_genres = np.array([[1, 0], [0, 1], [1, 1], [0, 1], [1, 0], [0, 1]])
        self.assertAlmostEqual(genre_purity(LABELS, mixed_genres), 2 / 3)

    def test_nearest_neighbors_exclude_self(self):
        indices, distances = nearest_neighbors(EMBEDDINGS, k=1)
        self.assertEqual(indices.shape, (6, 1))
        self.assertEqual(distances.shape, (6, 1))
        self.assertFalse(np.any(indices[:, 0] == np.arange(len(EMBEDDINGS))))
        self.assertGreater(
            average_neighbor_genre_similarity(EMBEDDINGS, GENRES, k=1), 0.5
        )

    def test_reconstruction_metrics_with_feature_blocks(self):
        target = np.array([[0.0, 2.0], [2.0, 0.0]])
        reconstruction = np.array([[1.0, 2.0], [2.0, 2.0]])
        metrics = reconstruction_metrics(target, reconstruction, {"first": slice(0, 1)})
        self.assertAlmostEqual(metrics["mse"], 1.25)
        self.assertAlmostEqual(metrics["mae"], 0.75)
        self.assertAlmostEqual(metrics["feature_blocks"]["first"]["mse"], 0.5)

    def test_noise_is_excluded_from_cluster_metrics(self):
        noisy_labels = np.array([0, 0, 0, 1, 1, -1])
        self.assertGreater(silhouette_score(EMBEDDINGS, noisy_labels), 0.9)
        result = evaluate_embeddings(EMBEDDINGS, noisy_labels, GENRES, k_neighbors=1)
        self.assertEqual(result["clustering"]["n_noise"], 1)
        self.assertEqual(result["clustering"]["n_scored"], 5)

    def test_hdbscan_when_available(self):
        try:
            labels = fit_hdbscan(EMBEDDINGS, min_cluster_size=2)
        except ImportError:
            self.skipTest("HDBSCAN is unavailable in this scikit-learn version")
        self.assertEqual(labels.shape, (len(EMBEDDINGS),))

    def test_pca_projections_and_heldout_transform(self):
        pca, projected = fit_pca_projection(EMBEDDINGS, n_components=2)
        self.assertEqual(projected.shape, (6, 2))
        self.assertEqual(pca.transform(HELDOUT_EMBEDDINGS).shape, (4, 2))
        embeddings_3d = np.column_stack([EMBEDDINGS, np.arange(len(EMBEDDINGS))])
        self.assertEqual(
            project_embeddings(embeddings_3d, "pca", n_components=3).shape, (6, 3)
        )

    def test_tsne_projection_and_invalid_projection_dimension(self):
        projected = project_embeddings(
            EMBEDDINGS, "tsne", n_components=2, random_state=7, perplexity=3
        )
        self.assertEqual(projected.shape, (6, 2))
        with self.assertRaises(ValueError):
            project_embeddings(EMBEDDINGS, "pca", n_components=4)

    def test_kmeans_and_gmm_heldout_assignment(self):
        kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(EMBEDDINGS)
        gmm = GaussianMixture(n_components=2, random_state=0).fit(EMBEDDINGS)
        self.assertEqual(
            assign_heldout_clusters(kmeans, HELDOUT_EMBEDDINGS).shape, (4,)
        )
        self.assertEqual(assign_heldout_clusters(gmm, HELDOUT_EMBEDDINGS).shape, (4,))

    def test_generalization_and_heldout_placement(self):
        indices, distances = nearest_training_neighbors(
            EMBEDDINGS, HELDOUT_EMBEDDINGS, k=2
        )
        self.assertEqual(indices.shape, (4, 2))
        self.assertEqual(distances.shape, (4, 2))
        result = evaluate_generalization(
            EMBEDDINGS,
            HELDOUT_EMBEDDINGS,
            LABELS,
            HELDOUT_LABELS,
            GENRES,
            HELDOUT_GENRES,
            k_neighbors=1,
        )
        self.assertIsNotNone(result["train"]["silhouette_score"])
        self.assertIsNotNone(result["heldout"]["silhouette_score"])
        self.assertIsNotNone(result["silhouette_difference"])
        self.assertIn("average_neighbor_genre_similarity", result["heldout_placement"])

    def test_heldout_placement_assignments_and_hdbscan_limit(self):
        kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(EMBEDDINGS)
        placement = heldout_placement(
            EMBEDDINGS, HELDOUT_EMBEDDINGS, k_neighbors=1, clustering_model=kmeans
        )
        self.assertEqual(len(placement["assigned_clusters"]), len(HELDOUT_EMBEDDINGS))
        self.assertAlmostEqual(
            sum(placement["cluster_assignment_fractions"].values()), 1.0
        )
        try:
            from sklearn.cluster import HDBSCAN
        except ImportError:
            self.skipTest("HDBSCAN is unavailable in this scikit-learn version")
        with self.assertRaises(NotImplementedError):
            assign_heldout_clusters(
                HDBSCAN(min_cluster_size=2).fit(EMBEDDINGS), HELDOUT_EMBEDDINGS
            )

    def test_experiment_a_feature_selection_and_input_dimension(self):
        matrix = np.arange(32, dtype=np.float32).reshape(4, 8)
        slices = {
            "genres": (0, 2),
            "keywords": (2, 5),
            "text": (5, 7),
            "language": (7, 8),
        }
        selected = select_experiment_a_features(matrix, slices)
        self.assertEqual(FEATURE_GROUPS, ("genres", "keywords", "text"))
        self.assertEqual(selected.shape, (4, 7))
        np.testing.assert_array_equal(selected, matrix[:, :7])

    def test_experiment_splits_do_not_overlap_or_touch_heldout(self):
        movie_ids = np.arange(12)
        original = np.array(["train"] * 10 + ["heldout"] * 2)
        splits = split_movie_indices(movie_ids, original, seed=42)
        self.assertEqual(len(splits["val"]), 2)
        self.assertEqual(len(splits["train"]), 8)
        self.assertEqual(set(splits["heldout"]), {10, 11})
        self.assertFalse(set(splits["train"]).intersection(splits["val"]))
        self.assertFalse(set(splits["train"]).intersection(splits["heldout"]))
        self.assertFalse(set(splits["val"]).intersection(splits["heldout"]))

    def test_preprocessing_is_fitted_on_experiment_training_rows_only(self):
        class Rows:
            @property
            def iloc(self):
                return self

            def __getitem__(self, indices):
                return np.asarray(indices)

        class RecordingBuilder:
            def fit(self, rows):
                self.fitted_rows = rows
                return self

        builder = fit_experiment_builder(
            RecordingBuilder(), Rows(), np.array([1, 4, 6])
        )
        np.testing.assert_array_equal(builder.fitted_rows, np.array([1, 4, 6]))

    def test_embedding_dimensions_movie_id_alignment_and_checkpoint_restore(self):
        import torch

        from models.AE import autoencoder

        model = autoencoder(input_dim=7, hidden_dim=8, latent_dim=3)
        original_state = {
            name: value.clone() for name, value in model.state_dict().items()
        }
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(1.0)
        restore_best_state(model, original_state)
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, original_state[name]))

        features = np.arange(35, dtype=np.float32).reshape(5, 7)
        movie_ids = np.array([11, 12, 13, 14, 15])
        embeddings = extract_embeddings(model, features, torch.device("cpu"))
        self.assertEqual(embeddings.shape, (len(movie_ids), 3))
        self.assertEqual(len(movie_ids), len(embeddings))

    def test_v2_cleaned_loader_restores_feature_builder_inputs(self):
        movies = load_cleaned_movies()
        self.assertEqual(DATASET_VERSION, "v2")
        self.assertEqual(len(movies), 9025)
        self.assertEqual(movies["movieId"].tolist()[:3], [1, 2, 3])
        self.assertTrue(all(isinstance(value, list) for value in movies["keywords"]))
        self.assertTrue(all(isinstance(value, list) for value in movies["companies"]))
        self.assertIn("text", movies.columns)
        self.assertEqual(
            movies["split"].value_counts().to_dict(), {"train": 8171, "heldout": 854}
        )

    def test_v2_random_code_keywords_are_deterministic(self):
        from scripts.build_features import CODE_DIM, code_block, name_code

        first = name_code("keywords", "space adventure", CODE_DIM)
        second = name_code("keywords", "space adventure", CODE_DIM)
        np.testing.assert_array_equal(first, second)
        codes = code_block(
            [["space adventure"], ["space adventure"], []],
            [1, 2, 3],
            "keywords",
            CODE_DIM,
        )
        self.assertEqual(codes.shape, (3, 128))
        np.testing.assert_array_equal(codes[0], codes[1])
        self.assertAlmostEqual(np.linalg.norm(codes[2]), 1.0)

    def test_v2_preprocessing_smoke_test_uses_only_experiment_training_movies(self):
        prepared = prepare_experiment_data()
        self.assertEqual(prepared.input_dim, 248)
        self.assertEqual(
            prepared.block_dimensions,
            {"genres": 20, "keywords": 128, "text": 100},
        )
        self.assertEqual(
            prepared.block_weights,
            {"genres": 1.5, "keywords": 1.0, "text": 0.5},
        )
        self.assertEqual(prepared.features["train"].shape, (6945, 248))
        self.assertEqual(prepared.features["val"].shape, (1226, 248))
        self.assertEqual(prepared.features["heldout"].shape, (854, 248))
        self.assertTrue(np.isfinite(prepared.features["train"]).all())

    def test_all_experiment_configurations_select_expected_v2_dimensions(self):
        widths = {
            "text": 100,
            "genres": 20,
            "language": 11,
            "keywords": 128,
            "cast": 128,
            "directors": 128,
            "writers": 128,
            "companies": 128,
            "collection": 128,
            "numeric": 3,
        }
        expected = {"A": 248, "B": 632, "C": 771, "D": 902}
        start = 0
        slices = {}
        for name, width in widths.items():
            slices[name] = (start, start + width)
            start += width
        matrix = np.zeros((3, start), dtype=np.float32)
        from experiments.common import select_feature_groups

        for name, groups in FEATURE_CONFIGURATIONS.items():
            selected = select_feature_groups(matrix, slices, groups)
            self.assertEqual(selected.shape, (3, expected[name]))

    def test_vae_forward_mean_embeddings_and_loss_accounting(self):
        import torch

        from models.VAE import VAE

        model = VAE(input_dim=7, hidden_dim=8, latent_dim=3)
        batch = torch.arange(35, dtype=torch.float32).reshape(5, 7)
        reconstruction, mean, logvar = model(batch)
        self.assertEqual(reconstruction.shape, batch.shape)
        self.assertEqual(mean.shape, (5, 3))
        self.assertEqual(logvar.shape, (5, 3))
        reconstruction_loss, kl_loss, total_loss = _vae_batch_losses(model, batch)
        self.assertTrue(torch.allclose(total_loss, reconstruction_loss + kl_loss))
        embeddings = extract_vae_embeddings(model, batch.numpy(), torch.device("cpu"))
        self.assertEqual(embeddings.shape, (5, 3))
        model.eval()
        with torch.no_grad():
            expected_mean, _ = model.encoder(batch)
        np.testing.assert_allclose(embeddings, expected_mean.numpy())


if __name__ == "__main__":
    unittest.main()

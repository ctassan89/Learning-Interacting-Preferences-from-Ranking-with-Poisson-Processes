"""Tests for the text-embedding experiments (no network, no model downloads)."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from experiments.TextEmbeddings.encoders import l2_normalize
from experiments.TextEmbeddings.extra_metrics import (
    anisotropy,
    extra_metrics,
    genre_matrix,
    knn_genre_f1,
    placement_gap,
    same_entity_nn,
)
from experiments.TextEmbeddings.text_inputs import (
    TOP_CAST,
    TOP_KEYWORDS,
    build_text,
    build_texts,
)

MOVIE = {
    "title": "Toy Story",
    "year": 1995,
    "genres": ["Animation", "Comedy", "Family"],
    "directors": ["John Lasseter"],
    "cast": ["Tom Hanks", "Tim Allen", "Don Rickles", "Jim Varney", "Wallace Shawn"],
    "keywords": ["jealousy", "toy", "boy", "friendship", "friends", "rivalry", "x"],
    "writers": ["A", "B", "C", "D"],
    "companies": ["Pixar Animation Studios"],
    "collection": ["Toy Story Collection"],
    "language": "en",
    "text": "Led by Woody, Andy's toys live happily in his room.",
}


class TextInputTests(unittest.TestCase):
    def test_overview_variant_and_title_fallback(self):
        self.assertEqual(build_text(MOVIE, "overview"), MOVIE["text"])
        empty = {**MOVIE, "text": ""}
        self.assertEqual(build_text(empty, "overview"), "Toy Story")

    def test_meta_core_fields_order_and_truncation(self):
        text = build_text(MOVIE, "meta_core")
        self.assertTrue(text.startswith("Title: Toy Story (1995)."))
        self.assertTrue(text.endswith("Plot: " + MOVIE["text"]))
        order = [text.index(f) for f in ("Genres:", "Director:", "Cast:", "Keywords:")]
        self.assertEqual(order, sorted(order))
        self.assertIn(", ".join(MOVIE["cast"][:TOP_CAST]) + ".", text)
        self.assertNotIn(MOVIE["cast"][TOP_CAST], text)
        self.assertIn(", ".join(MOVIE["keywords"][:TOP_KEYWORDS]) + ".", text)
        self.assertNotIn("rivalry", text)
        self.assertNotIn("Language", text)  # rich-only fields

    def test_meta_rich_adds_fields_before_plot(self):
        text = build_text(MOVIE, "meta_rich")
        for field in ("Language: English.", "Writers: A, B, C.", "Collection:"):
            self.assertIn(field, text)
        self.assertLess(text.index("Collection:"), text.index("Plot:"))

    def test_empty_fields_are_omitted(self):
        movie = {**MOVIE, "keywords": [], "collection": float("nan"), "cast": None}
        text = build_text(movie, "meta_rich")
        for label in ("Keywords:", "Collection:", "Cast:", "unknown"):
            self.assertNotIn(label, text)

    def test_pipe_strings_are_accepted(self):
        movie = {**MOVIE, "genres": "Animation|Comedy", "cast": "A|B|C|D"}
        text = build_text(movie, "meta_core")
        self.assertIn("Genres: Animation, Comedy.", text)
        self.assertIn("Cast: A, B, C.", text)

    def test_unknown_variant(self):
        with self.assertRaises(ValueError):
            build_text(MOVIE, "abstract")

    def test_build_texts_on_list(self):
        self.assertEqual(len(build_texts([MOVIE, MOVIE], "overview")), 2)


def clustered_data(seed=0, n=300, d=8):
    """Three well separated clusters, one genre each; every 5th movie held out."""
    rng = np.random.default_rng(seed)
    labels = np.arange(n) % 3
    centres = rng.normal(0, 10, (3, d))
    Z = centres[labels] + rng.normal(0, 0.5, (n, d))
    genres = [["Drama", "Comedy", "Horror"][g] for g in labels]
    train = np.arange(n) % 5 != 0
    return Z, train, genre_matrix(genres), labels


class ExtraMetricTests(unittest.TestCase):
    def test_genre_matrix_handles_strings_lists_and_missing(self):
        G = genre_matrix(["A|B", ["B"], float("nan"), ""])
        self.assertEqual(G.shape, (4, 2))
        self.assertEqual(G.sum(axis=1).tolist(), [2, 1, 0, 0])

    def test_knn_genre_f1_is_perfect_on_separated_clusters(self):
        Z, train, G, _ = clustered_data()
        f1 = knn_genre_f1(Z, train, G)
        self.assertAlmostEqual(f1["genre_f1_micro_heldout"], 1.0)
        self.assertAlmostEqual(f1["genre_f1_micro_train"], 1.0)

    def test_placement_gap(self):
        Z, train, _, _ = clustered_data()
        self.assertLess(abs(placement_gap(Z, train) - 1), 0.5)
        far = Z.copy()
        far[~train] += 100  # held-out movies placed far from everything
        self.assertGreater(placement_gap(far, train), 10)

    def test_same_entity_nn(self):
        Z, _, _, labels = clustered_data()
        entities = [[f"saga{g}"] for g in labels]
        self.assertAlmostEqual(same_entity_nn(Z, entities), 1.0)
        self.assertTrue(np.isnan(same_entity_nn(Z, [[] for _ in labels])))

    def test_anisotropy(self):
        rng = np.random.default_rng(0)
        self.assertLess(abs(anisotropy(rng.normal(size=(500, 50)))), 0.05)
        self.assertGreater(anisotropy(rng.normal(size=(500, 50)) + 10), 0.9)

    def test_extra_metrics_native_and_projected(self):
        Z, train, G, labels = clustered_data()
        entities = [[f"saga{g}"] for g in labels]
        native = extra_metrics(Z, train, G, entities, entities)
        self.assertIn("anisotropy", native)
        self.assertNotIn("hdbscan_n_clusters", native)
        self.assertGreater(native["gmm_purity_ml"], 0.95)
        projected = extra_metrics(Z[:, :2], train, G, entities, entities, Z_native=Z)
        self.assertIn("trustworthiness", projected)
        self.assertIn("hdbscan_n_clusters", projected)
        self.assertTrue(0 <= projected["trustworthiness"] <= 1)

    def test_l2_normalize(self):
        Z = l2_normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
        self.assertAlmostEqual(float(np.linalg.norm(Z[0])), 1.0, places=6)
        self.assertTrue(np.isfinite(Z).all())


class SavedEmbeddingTests(unittest.TestCase):
    def test_saved_embedding_round_trip_and_alignment(self):
        from scripts.evaluate_embedding import align, load_embedding, save_embedding
        import pandas as pd

        catalogue = pd.DataFrame({"movieId": [10, 20, 30]})
        Z = l2_normalize(np.eye(3)[[2, 0, 1]])
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "run" / "embedding.npz"
            save_embedding(path, Z, [30, 10, 20])
            Z_loaded, ids = load_embedding(path)
        aligned = align(Z_loaded, ids, catalogue)
        np.testing.assert_allclose(aligned, np.eye(3)[[0, 1, 2]], atol=1e-6)


class DataTests(unittest.TestCase):
    def write_files(self, tmp, catalogue_split=("train", "heldout", "train")):
        import pandas as pd

        movies = pd.DataFrame(
            {
                "movieId": [3, 1, 2],
                "title": ["C", "A", "B"],
                "overview_tagline": [" plot c ", None, "plot b"],
                "language": ["en", "fr", None],
                "year": [2000, 1990, 1995],
                "split": ["train", "train", "heldout"],
                **{
                    c: ["x|y", None, "z"]
                    for c in (
                        "genres",
                        "keywords",
                        "cast",
                        "directors",
                        "writers",
                        "companies",
                        "collection",
                    )
                },
            }
        )
        movies.to_csv(Path(tmp) / "movies_clean.csv", index=False)
        pd.DataFrame({"movieId": [1, 2, 3], "split": list(catalogue_split)}).to_csv(
            Path(tmp) / "catalogue.csv", index=False
        )

    def test_load_movies_order_lists_and_text(self):
        from experiments.TextEmbeddings.data import load_movies

        with TemporaryDirectory() as tmp:
            self.write_files(tmp)
            movies = load_movies(Path(tmp))
        self.assertEqual(movies["movieId"].tolist(), [1, 2, 3])  # catalogue order
        self.assertEqual(movies["genres"].tolist(), [[], ["z"], ["x", "y"]])
        self.assertEqual(movies["text"].tolist(), ["", "plot b", "plot c"])

    def test_load_movies_rejects_a_different_split(self):
        from experiments.TextEmbeddings.data import load_movies

        with TemporaryDirectory() as tmp:
            self.write_files(tmp, catalogue_split=("heldout", "heldout", "train"))
            with self.assertRaises(ValueError):
                load_movies(Path(tmp))

    def test_validation_split_uses_only_train_and_is_reproducible(self):
        from experiments.TextEmbeddings.data import validation_split

        split = np.array(["train"] * 90 + ["heldout"] * 10)
        a, b = validation_split(split), validation_split(split)
        np.testing.assert_array_equal(a["val"], b["val"])
        self.assertEqual(len(a["val"]), 14)  # ceil(0.15 * 90)
        self.assertFalse(set(a["train"]) & set(a["val"]))
        self.assertTrue((np.concatenate([a["train"], a["val"]]) < 90).all())

    def test_real_processed_data_if_present(self):
        from experiments.TextEmbeddings.data import PROCESSED_DIR, load_movies

        if not (PROCESSED_DIR / "movies_clean.csv").exists():
            self.skipTest("data/processed not available")
        movies = load_movies()
        self.assertEqual(len(movies), 9025)
        self.assertEqual(int((movies["split"] == "train").sum()), 8171)


class LSATests(unittest.TestCase):
    TRAIN = [
        "a detective hunts a killer in the dark city",
        "the detective follows the killer through the city at night",
        "two friends fall in love during a summer in paris",
        "a romantic comedy about love and friends in paris",
    ] * 3

    def test_fits_on_train_only_and_places_new_texts(self):
        from experiments.TextEmbeddings.encoders import LSAEncoder

        encoder = LSAEncoder(n_components=2).fit(self.TRAIN)
        vocabulary = encoder.pipeline.named_steps["tfidf"].vocabulary_
        self.assertNotIn("spaceship", vocabulary)  # only training words
        Z = encoder.embed(["a killer in the city", "love in paris", "spaceship"])
        self.assertEqual(Z.shape, (3, 2))
        self.assertTrue(np.isfinite(Z).all())
        np.testing.assert_allclose(np.linalg.norm(Z[:2], axis=1), 1, atol=1e-5)
        self.assertLess(float(Z[0] @ Z[1]), 0.5)  # different topics are apart
        self.assertEqual(encoder.config()["n_train_texts"], len(self.TRAIN))

    def test_embed_before_fit_fails(self):
        from experiments.TextEmbeddings.encoders import LSAEncoder

        with self.assertRaises(RuntimeError):
            LSAEncoder().embed(["text"])


if __name__ == "__main__":
    unittest.main()

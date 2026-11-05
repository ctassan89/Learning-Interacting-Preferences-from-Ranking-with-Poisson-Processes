"""Text encoders: pretrained sentence-embedding models and an LSA baseline.

Every encoder exposes ``embed(texts) -> np.ndarray`` with one L2-normalized row
per text. Normalized rows make the Euclidean distance used by the shared evaluator
a monotone function of cosine similarity (``|a - b|^2 = 2 - 2 cos(a, b)``).

``embed`` is also how a brand-new movie is placed in the space: build its text
with ``text_inputs.build_text`` and embed it, nothing else is needed.

Pretrained models are used as they are (nothing is fitted on our data, so there
is no leakage). LSA (TF-IDF + truncated SVD) is fitted on training movies only;
afterwards every movie, training ones included, is placed with the same
``transform``. (Doc2Vec was planned as the classical baseline but dropped: its
library, gensim, does not install on Python 3.14.)
"""

from dataclasses import dataclass

import numpy as np

SEED = 2026


@dataclass(frozen=True)
class PretrainedSpec:
    """A Hugging Face sentence-embedding model and how to call it."""

    model_name: str
    prefix: str = ""  # e5 models expect a task prefix before every text


PRETRAINED_MODELS = {
    "minilm": PretrainedSpec("sentence-transformers/all-MiniLM-L6-v2"),
    "mpnet": PretrainedSpec("sentence-transformers/all-mpnet-base-v2"),
    # "query: " is the prefix e5 recommends for symmetric similarity tasks
    "e5": PretrainedSpec("intfloat/e5-base-v2", prefix="query: "),
    "bge": PretrainedSpec("BAAI/bge-base-en-v1.5"),
}


def l2_normalize(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, 1e-12)


class SentenceEncoder:
    """Wrapper around a pretrained ``sentence_transformers`` model."""

    def __init__(self, key: str, batch_size: int = 64, device: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.key = key
        self.spec = PRETRAINED_MODELS[key]
        self.batch_size = batch_size
        self.model = SentenceTransformer(self.spec.model_name, device=device)

    @property
    def max_tokens(self) -> int:
        return int(self.model.max_seq_length)

    def config(self) -> dict:
        return {
            "encoder": "pretrained",
            "model_name": self.spec.model_name,
            "prefix": self.spec.prefix,
            "max_seq_length": self.max_tokens,
            "normalized": True,
        }

    def embed(self, texts: list[str]) -> np.ndarray:
        inputs = [self.spec.prefix + text for text in texts]
        Z = self.model.encode(
            inputs,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(inputs) > 1000,
        )
        return l2_normalize(Z)

    def truncated_share(self, texts: list[str]) -> float:
        """Share of texts longer than the model's maximum input length."""
        tokenizer = self.model.tokenizer
        lengths = [
            len(tokenizer(self.spec.prefix + text)["input_ids"]) for text in texts
        ]
        return float(np.mean(np.asarray(lengths) > self.max_tokens))


# Small grid for LSA, chosen on the validation subset of the training movies.
LSA_GRID = (
    {"n_components": 100},
    {"n_components": 200},
    {"n_components": 300},
)


class LSAEncoder:
    """Classical baseline: TF-IDF (unigrams + bigrams) compressed with truncated SVD
    (latent semantic analysis). Vocabulary, IDF weights and SVD are all fitted on
    training texts only; a new movie is placed with ``transform`` (words never seen
    in training are simply ignored)."""

    def __init__(
        self,
        n_components: int = 200,
        min_df: int = 2,
        max_df: float = 0.5,
        ngram_range: tuple[int, int] = (1, 2),
        seed: int = SEED,
    ):
        self.params = {
            "n_components": n_components,
            "min_df": min_df,
            "max_df": max_df,
            "ngram_range": list(ngram_range),
        }
        self.seed = seed
        self.pipeline = None
        self.n_train_texts = 0

    @property
    def max_tokens(self) -> int | None:
        return None  # no truncation

    def config(self) -> dict:
        vocabulary = (
            len(self.pipeline.named_steps["tfidf"].vocabulary_) if self.pipeline else 0
        )
        svd = self.pipeline.named_steps["svd"] if self.pipeline else None
        return {
            "encoder": "lsa",
            **self.params,
            "stop_words": "english",
            "sublinear_tf": True,
            "vocabulary_size": vocabulary,
            "explained_variance": (
                float(svd.explained_variance_ratio_.sum()) if svd is not None else None
            ),
            "seed": self.seed,
            "n_train_texts": self.n_train_texts,
            "normalized": True,
        }

    def fit(self, train_texts: list[str]) -> "LSAEncoder":
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import Pipeline

        tfidf = TfidfVectorizer(
            min_df=self.params["min_df"],
            max_df=self.params["max_df"],
            ngram_range=tuple(self.params["ngram_range"]),
            stop_words="english",
            sublinear_tf=True,
            strip_accents="unicode",
        )
        svd = TruncatedSVD(
            n_components=self.params["n_components"], random_state=self.seed
        )
        self.pipeline = Pipeline([("tfidf", tfidf), ("svd", svd)]).fit(train_texts)
        self.n_train_texts = len(train_texts)
        return self

    def embed(self, texts: list[str]) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("LSAEncoder must be fitted before embedding.")
        return l2_normalize(self.pipeline.transform(texts))

    def truncated_share(self, texts: list[str]) -> float:
        return 0.0


# Encoders that must be fitted on training movies (key -> class, parameter grid).
FITTED_ENCODERS = {
    "lsa": (LSAEncoder, LSA_GRID),
}


def make_encoder(key: str, **kwargs):
    """Encoder by key: a key of PRETRAINED_MODELS or of FITTED_ENCODERS."""
    if key in FITTED_ENCODERS:
        return FITTED_ENCODERS[key][0](**kwargs)
    if key not in PRETRAINED_MODELS:
        raise ValueError(
            f"Unknown encoder {key!r}; use one of "
            f"{sorted(PRETRAINED_MODELS) + sorted(FITTED_ENCODERS)}."
        )
    return SentenceEncoder(key, **kwargs)

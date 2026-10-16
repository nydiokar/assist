"""Embedding adapters (v1 §4.5, v2 §D1).

EmbedderProtocol is the stable seam. HashEmbedder is a deterministic, dependency-free
embedder for cheap/offline validation and CI. E5Embedder is the production baseline
(intfloat/multilingual-e5-small) loaded lazily only when selected.

Both honor asymmetric query/passage prefixes via embed_queries / embed_passages so the
pipeline exercises the exact E5 contract even under the fake (v1 §4.5).
"""

from __future__ import annotations

import hashlib
from typing import Protocol, runtime_checkable

import numpy as np

from .config import EmbeddingConfig
from .schema import EmbeddingProfile


@runtime_checkable
class EmbedderProtocol(Protocol):
    @property
    def dim(self) -> int: ...

    @property
    def profile(self) -> EmbeddingProfile: ...

    def embed_passages(self, texts: list[str]) -> np.ndarray: ...

    def embed_queries(self, texts: list[str]) -> np.ndarray: ...


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class HashEmbedder:
    """Deterministic bag-of-hashed-tokens embedding. Not semantic, but stable and cheap.

    Token hashing into `dim` buckets with sub-token character n-grams gives enough signal
    for exact-term and near-duplicate retrieval validation offline. Prefixes are respected
    so the asymmetric contract is exercised.
    """

    def __init__(self, cfg: EmbeddingConfig) -> None:
        self._cfg = cfg
        self._dim = cfg.hash_dim
        self._profile = cfg.profile()

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def profile(self) -> EmbeddingProfile:
        return self._profile

    def _tokenize(self, text: str) -> list[str]:
        lowered = text.lower()
        words = [w for w in "".join(c if c.isalnum() else " " for c in lowered).split() if w]
        grams: list[str] = []
        for w in words:
            grams.append(w)
            if len(w) >= 4:
                grams.extend(w[i : i + 3] for i in range(len(w) - 2))
        return grams

    def _embed(self, texts: list[str], prefix: str) -> np.ndarray:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in self._tokenize(prefix + text):
                h = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
                idx = int.from_bytes(h[:4], "little") % self._dim
                sign = 1.0 if h[4] & 1 else -1.0
                out[i, idx] += sign
        return _l2_normalize(out)

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts, "passage: ")

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts, "query: ")


class E5Embedder:
    """Production baseline: intfloat/multilingual-e5-small via sentence-transformers.

    Lazily imports sentence-transformers so the package is importable without the heavy
    dependency. Applies required asymmetric prefixes and guards the 512-token limit.
    """

    def __init__(self, cfg: EmbeddingConfig) -> None:
        self._cfg = cfg
        self._profile = cfg.profile()
        self._model = None  # lazy

    def _ensure(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        self._model = SentenceTransformer(
            self._cfg.model, revision=self._cfg.revision
        )
        self._model.max_seq_length = self._cfg.max_tokens

    @property
    def dim(self) -> int:
        return self._cfg.dim

    @property
    def profile(self) -> EmbeddingProfile:
        return self._profile

    def _encode(self, texts: list[str], prefix: str) -> np.ndarray:
        self._ensure()
        assert self._model is not None
        prefixed = [prefix + t for t in texts]
        vecs = self._model.encode(
            prefixed,
            batch_size=self._cfg.batch_size,
            normalize_embeddings=self._cfg.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, self._cfg.passage_prefix)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, self._cfg.query_prefix)


def build_embedder(cfg: EmbeddingConfig) -> EmbedderProtocol:
    if cfg.provider == "hash":
        return HashEmbedder(cfg)
    if cfg.provider == "e5":
        return E5Embedder(cfg)
    raise ValueError(f"unknown embedding.provider: {cfg.provider}")

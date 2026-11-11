"""Deterministic bag-of-hashed-tokens embedder. Test/dev/CI only (A01: never a release profile;
`Config.assert_release_safe()` rejects it)."""

from __future__ import annotations

import hashlib

import numpy as np

from ..core.vectors import l2_normalize, validate_matrix
from ..domain.config import EmbeddingConfig
from ..domain.models import EmbeddingProfile


class HashEmbedder:
    """Deterministic bag-of-hashed-tokens embedding. Test/dev only (A01: not a release profile)."""

    def __init__(self, cfg: EmbeddingConfig) -> None:
        self._dim = cfg.hash_dim
        self._profile = cfg.profile()
        self._max_tokens = 2048

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def profile(self) -> EmbeddingProfile:
        return self._profile

    @property
    def max_input_tokens(self) -> int:
        return self._max_tokens

    def _tokenize(self, text: str) -> list[str]:
        lowered = text.lower()
        words = [w for w in "".join(c if c.isalnum() else " " for c in lowered).split() if w]
        grams: list[str] = []
        for w in words:
            grams.append(w)
            if len(w) >= 4:
                grams.extend(w[i : i + 3] for i in range(len(w) - 2))
        return grams

    def token_count(self, text: str) -> int:
        lowered = text.lower()
        return len([w for w in "".join(c if c.isalnum() else " " for c in lowered).split() if w])

    def _embed(self, texts: list[str], prefix: str) -> np.ndarray:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in self._tokenize(prefix + text):
                h = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
                idx = int.from_bytes(h[:4], "little") % self._dim
                sign = 1.0 if h[4] & 1 else -1.0
                out[i, idx] += sign
        out = l2_normalize(out)
        validate_matrix(out, self._dim, normalize=True)
        return out

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts, "passage: ")

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts, "query: ")

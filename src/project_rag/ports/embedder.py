"""Embedder port (spec A03 §15.3). S1 keeps the prototype shape; S3 extends it for HTTP
providers (profile fingerprint fields, batched token counts)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from ..domain.models import EmbeddingProfile

Matrix = npt.NDArray[np.float32]


@runtime_checkable
class Embedder(Protocol):
    @property
    def dim(self) -> int: ...

    @property
    def profile(self) -> EmbeddingProfile: ...

    @property
    def max_input_tokens(self) -> int: ...

    def token_count(self, text: str) -> int: ...

    def embed_passages(self, texts: list[str]) -> Matrix:
        """A (len(texts), dim) float32 matrix; unit-norm rows when the profile normalizes."""
        ...

    def embed_queries(self, texts: list[str]) -> Matrix: ...

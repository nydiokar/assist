"""Pure vector helpers shared by embedder adapters (A01 §D output checks)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

Matrix = npt.NDArray[np.float32]


def l2_normalize(mat: Matrix) -> Matrix:
    """Return a row-wise unit-norm copy; all-zero rows stay zero."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    safe = np.where(norms == 0, 1.0, norms)
    return (mat / safe).astype(mat.dtype, copy=False)


def validate_matrix(mat: Matrix, dim: int, normalize: bool) -> None:
    """A01 §D output checks: finite, correct dim, consistent normalization."""
    if mat.ndim != 2 or mat.shape[1] != dim:
        raise ValueError(f"embedding shape {mat.shape} != (*, {dim})")
    if not np.isfinite(mat).all():
        raise ValueError("embedding contains non-finite values")
    if normalize and mat.shape[0]:
        norms = np.linalg.norm(mat, axis=1)
        nonzero = norms[norms > 1e-8]
        if nonzero.size and not np.allclose(nonzero, 1.0, atol=1e-3):
            raise ValueError("normalized embeddings are not unit-norm")

"""Embedder port contract."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from project_rag.adapters.embed_hash import HashEmbedder
from project_rag.domain.config import EmbeddingConfig
from project_rag.ports.embedder import Embedder
from project_rag.wiring import load_config


def _hash() -> Embedder:
    return HashEmbedder(EmbeddingConfig(provider="hash", hash_dim=256))


def _qwen3() -> Embedder:
    torch = pytest.importorskip("torch")
    pytest.importorskip("sentence_transformers")
    from project_rag.adapters.embed_torch import Qwen3Embedder
    from project_rag.adapters.gpu_env import prepare_gpu_env

    prepare_gpu_env()
    if not torch.cuda.is_available():
        pytest.skip("Qwen3 reference needs the WSL ROCm/CUDA venv")
    # The pinned A01/A02 reference profile (bf16 inference, FP32 storage, pinned HF revision).
    cfg = load_config(Path(__file__).resolve().parents[2] / "configs" / "yp_qwen4b_gpu.yaml")
    return Qwen3Embedder(cfg.embedding)


ADAPTERS: list[object] = [
    pytest.param(_hash, id="hash"),
    pytest.param(_qwen3, id="torch_qwen3", marks=pytest.mark.live),
]

TEXTS = ["Проектът има бюджет 49 697,00 EUR.", "The final report is due 31 January 2026.", "x"]


@pytest.fixture(params=ADAPTERS)
def embedder(request: pytest.FixtureRequest) -> Embedder:
    factory: Callable[[], Embedder] = request.param
    return factory()


def test_satisfies_protocol_and_profile(embedder: Embedder) -> None:
    assert isinstance(embedder, Embedder)
    assert embedder.dim == embedder.profile.dim
    assert embedder.max_input_tokens > 0


@pytest.mark.parametrize("which", ["passages", "queries"])
def test_shape_dtype_finite_and_normalised(embedder: Embedder, which: str) -> None:
    fn = embedder.embed_passages if which == "passages" else embedder.embed_queries
    mat = fn(TEXTS)
    assert mat.shape == (len(TEXTS), embedder.dim)
    assert mat.dtype == np.float32
    assert np.isfinite(mat).all()
    if embedder.profile.normalize:
        assert np.allclose(np.linalg.norm(mat, axis=1), 1.0, atol=1e-5)


def test_deterministic(embedder: Embedder) -> None:
    a = embedder.embed_passages(TEXTS)
    b = embedder.embed_passages(TEXTS)
    assert np.allclose(a, b, atol=1e-6)


def test_token_count_is_positive_for_text(embedder: Embedder) -> None:
    assert all(embedder.token_count(t) >= 1 for t in TEXTS)

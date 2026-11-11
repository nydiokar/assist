"""bf16 inference yields norms ~0.998; the adapter must store exact unit-norm FP32 vectors."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from project_rag.adapters.embed_torch import Qwen3Embedder
from project_rag.domain.config import EmbeddingConfig, Qwen3Config


class _FakeBf16Model:
    prompts: ClassVar[dict] = {"query": "q"}

    def encode(self, texts, **kw):
        rng = np.random.default_rng(0)
        v = rng.normal(size=(len(texts), 1024)).astype(np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True) * 0.9985  # bf16-style drift


def test_low_precision_vectors_are_renormalized_in_fp32() -> None:
    cfg = EmbeddingConfig(provider="qwen3", qwen3=Qwen3Config(dim=1024))
    emb = Qwen3Embedder(cfg)
    emb._model = _FakeBf16Model()
    out = emb.embed_passages(["a", "b", "c"])
    assert out.dtype == np.float32
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-6)

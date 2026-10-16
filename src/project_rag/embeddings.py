"""Embedding adapters (v1 §4.5, v2 §D1, Amendment A01 §§A/D).

EmbedderProtocol is the stable seam. Implementations:
  - HashEmbedder : deterministic, dependency-free; test/dev + CI only (not a release profile).
  - Qwen3Embedder: A01 production candidate (Qwen/Qwen3-Embedding-4B, 2560-dim, CUDA, last-token
                   pooling, official query prompt, NO document prefix, FP32-stored vectors).
  - E5Embedder   : legacy comparator (query:/passage: prefixes). Not the default under A01.

All implementations expose a token_count() so chunking can measure+guard truncation (A01 §F).
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

    @property
    def max_input_tokens(self) -> int: ...

    def token_count(self, text: str) -> int: ...

    def embed_passages(self, texts: list[str]) -> np.ndarray: ...

    def embed_queries(self, texts: list[str]) -> np.ndarray: ...


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def _validate_matrix(mat: np.ndarray, dim: int, normalize: bool) -> None:
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
        out = _l2_normalize(out)
        _validate_matrix(out, self._dim, normalize=True)
        return out

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts, "passage: ")

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts, "query: ")


class Qwen3Embedder:
    """A01 production candidate. Requires the `qwen3` extra + CUDA. No silent CPU fallback.

    Uses the officially supported Sentence Transformers integration:
      model.encode(docs)                         # plain document encoding, no prefix
      model.encode(queries, prompt_name="query") # official query instruction
    Pooling (last-token), attention masks and L2 normalization are handled by the ST integration.
    Vectors are stored FP32 regardless of FP16/BF16 inference precision.
    """

    def __init__(self, cfg: EmbeddingConfig) -> None:
        self._cfg = cfg
        self._q = cfg.qwen3
        self._profile = cfg.profile()
        self._model = None  # lazy
        self._has_query_prompt = True

    def _ensure(self) -> None:
        if self._model is not None:
            return
        import torch  # noqa: PLC0415
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        torch.set_num_threads(self._cfg.threads)
        if self._q.device == "cuda" and not torch.cuda.is_available():
            # A01 §A2/§E/§J: fail visibly, do NOT silently fall back to CPU or another model.
            raise RuntimeError(
                "Qwen3 profile requires CUDA but torch.cuda.is_available() is False. "
                "Run scripts/feasibility_smoke.py; do not silently substitute device/model.")
        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                 "float32": torch.float32}.get(self._q.inference_dtype, torch.float16)
        self._model = SentenceTransformer(
            self._q.model, revision=self._q.revision, device=self._q.device,
            model_kwargs={"torch_dtype": dtype},
            tokenizer_kwargs={"padding_side": self._q.tokenizer_padding_side},
        )
        self._model.max_seq_length = self._q.max_input_tokens
        prompts = getattr(self._model, "prompts", {}) or {}
        self._has_query_prompt = self._q.query_prompt_name in prompts

    @property
    def dim(self) -> int:
        return self._q.dim

    @property
    def profile(self) -> EmbeddingProfile:
        return self._profile

    @property
    def max_input_tokens(self) -> int:
        return self._q.max_input_tokens

    def token_count(self, text: str) -> int:
        self._ensure()
        assert self._model is not None
        return len(self._model.tokenizer.encode(text, add_special_tokens=True))

    def _encode(self, texts: list[str], *, is_query: bool) -> np.ndarray:
        self._ensure()
        assert self._model is not None
        kwargs = dict(batch_size=self._q.batch_size,
                      normalize_embeddings=self._q.normalize,
                      convert_to_numpy=True, show_progress_bar=False)
        if is_query and self._has_query_prompt:
            kwargs["prompt_name"] = self._q.query_prompt_name
        vecs = np.asarray(self._model.encode(texts, **kwargs), dtype=np.float32)
        if self._q.normalize:
            # bf16/fp16 inference leaves norms ~0.998 (measured, cosine to fp32 = 0.9998);
            # re-normalize in FP32 so stored vectors are exactly unit-norm (A01 §D, A02).
            vecs = _l2_normalize(vecs)
        _validate_matrix(vecs, self._q.dim, normalize=self._q.normalize)
        return vecs  # stored as FP32 (A01 §D)

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, is_query=False)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, is_query=True)


class E5Embedder:
    """Legacy comparator: intfloat/multilingual-e5-small. Not the A01 default."""

    def __init__(self, cfg: EmbeddingConfig) -> None:
        self._cfg = cfg
        self._e = cfg.e5
        self._profile = cfg.profile()
        self._model = None

    def _ensure(self) -> None:
        if self._model is not None:
            return
        import torch  # noqa: PLC0415
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        torch.set_num_threads(self._cfg.threads)
        self._model = SentenceTransformer(self._e.model, revision=self._e.revision)
        self._model.max_seq_length = self._e.max_tokens

    @property
    def dim(self) -> int:
        return self._e.dim

    @property
    def profile(self) -> EmbeddingProfile:
        return self._profile

    @property
    def max_input_tokens(self) -> int:
        return self._e.max_tokens

    def token_count(self, text: str) -> int:
        self._ensure()
        assert self._model is not None
        return len(self._model.tokenizer.encode(text, add_special_tokens=True))

    def _encode(self, texts: list[str], prefix: str) -> np.ndarray:
        self._ensure()
        assert self._model is not None
        vecs = self._model.encode([prefix + t for t in texts], batch_size=self._e.batch_size,
                                  normalize_embeddings=self._e.normalize,
                                  convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, self._e.passage_prefix)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, self._e.query_prefix)


def build_embedder(cfg: EmbeddingConfig) -> EmbedderProtocol:
    if cfg.provider == "hash":
        return HashEmbedder(cfg)
    if cfg.provider == "qwen3":
        return Qwen3Embedder(cfg)
    if cfg.provider == "e5":
        return E5Embedder(cfg)
    raise ValueError(f"unknown embedding.provider: {cfg.provider}")

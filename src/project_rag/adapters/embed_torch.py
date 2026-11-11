"""Torch / sentence-transformers embedder adapters (v1 §4.5, A01 §§A/D). `dev-gpu` extra only.

  - Qwen3Embedder: A01 reference (Qwen/Qwen3-Embedding-4B, 2560-dim, CUDA/ROCm, last-token
                   pooling, official query prompt, NO document prefix, FP32-stored vectors).
  - E5Embedder   : legacy comparator (query:/passage: prefixes).

Both expose token_count() so chunking can measure and guard truncation (A01 §F).
"""

from __future__ import annotations

import numpy as np

from ..core.vectors import l2_normalize as _l2_normalize
from ..core.vectors import validate_matrix as _validate_matrix
from ..domain.config import EmbeddingConfig
from ..domain.models import EmbeddingProfile
from .gpu_env import describe_gpu_env, prepare_gpu_env


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
        prepare_gpu_env()  # must precede `import torch` (ROCm-on-WSL needs HSA_ENABLE_DXG_DETECTION)
        import torch
        from sentence_transformers import SentenceTransformer

        torch.set_num_threads(self._cfg.threads)
        if self._q.device == "cuda" and not torch.cuda.is_available():
            # A01 §A2/§E/§J: fail visibly, do NOT silently fall back to CPU or another model.
            raise RuntimeError(
                "Qwen3 profile requires CUDA but torch.cuda.is_available() is False. "
                + describe_gpu_env(torch.version.hip, torch.version.cuda)
                + " Run scripts/feasibility_smoke.py; do not silently substitute device/model."
            )
        dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }.get(self._q.inference_dtype, torch.float16)
        self._model = SentenceTransformer(
            self._q.model,
            revision=self._q.revision,
            device=self._q.device,
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
        kwargs = {
            "batch_size": self._q.batch_size,
            "normalize_embeddings": self._q.normalize,
            "convert_to_numpy": True,
            "show_progress_bar": False,
        }
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
        import torch
        from sentence_transformers import SentenceTransformer

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
        vecs = self._model.encode(
            [prefix + t for t in texts],
            batch_size=self._e.batch_size,
            normalize_embeddings=self._e.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, self._e.passage_prefix)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, self._e.query_prefix)

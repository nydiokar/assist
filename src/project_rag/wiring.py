"""Composition root (spec A03 §15.3): the only module that knows which adapter implements which
port. Grows into build(settings) -> Services as the service layer lands."""

from __future__ import annotations

from .domain.config import EmbeddingConfig, RerankConfig, VectorStoreConfig
from .ports.embedder import Embedder
from .ports.reranker import Reranker
from .ports.vector import VectorIndex


def build_embedder(cfg: EmbeddingConfig) -> Embedder:
    if cfg.provider == "hash":
        from .adapters.embed_hash import HashEmbedder

        return HashEmbedder(cfg)
    if cfg.provider == "qwen3":
        from .adapters.embed_torch import Qwen3Embedder

        return Qwen3Embedder(cfg)
    if cfg.provider == "e5":
        from .adapters.embed_torch import E5Embedder

        return E5Embedder(cfg)
    raise ValueError(f"unknown embedding.provider: {cfg.provider}")


def build_vector_store(cfg: VectorStoreConfig) -> VectorIndex:
    if cfg.provider == "memory":
        from .adapters.vector_memory import InMemoryVectorStore

        return InMemoryVectorStore(cfg)
    if cfg.provider == "chroma":
        from .adapters.vector_chroma import ChromaVectorStore

        return ChromaVectorStore(cfg)
    raise ValueError(f"unknown vector_store.provider: {cfg.provider}")


def build_reranker(cfg: RerankConfig) -> Reranker | None:
    if not cfg.enabled:
        return None
    from .adapters.rerank_torch import CrossEncoderReranker

    return CrossEncoderReranker(cfg)

"""Composition root (spec A03 §15.3): the only module that knows which adapter implements which
port. `build(config)` returns the services plus the resources they share."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .domain.config import Config, EmbeddingConfig, RerankConfig, VectorStoreConfig
from .ports.cache import EmbeddingCache
from .ports.catalog import Catalog
from .ports.embedder import Embedder
from .ports.lexical import LexicalIndex
from .ports.reranker import Reranker
from .ports.vector import VectorIndex
from .services.ingest import IngestService
from .services.search import SearchService


def load_config(path: str | Path | None = None) -> Config:
    """The one place configuration is read from disk (R7)."""
    from .adapters.config_yaml import load_config as _load

    return _load(path)


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


@dataclass(frozen=True)
class Services:
    """Built services and the shared resources behind them (closed together)."""

    config: Config
    ingest: IngestService
    search: SearchService
    embedder: Embedder
    vector: VectorIndex
    lexical: LexicalIndex
    catalog: Catalog

    def close(self) -> None:
        self.lexical.close()
        self.catalog.close()


def build(config: Config) -> Services:
    from .adapters.corpus_local_fs import LocalFsCorpus
    from .adapters.parse_registry import ParserRegistry
    from .adapters.sqlite.catalog import SqliteCatalog
    from .adapters.sqlite.embedding_cache import SqliteEmbeddingCache
    from .adapters.sqlite.fts5 import Fts5LexicalIndex
    from .adapters.stage_cache_fs import FsArtifactStore

    embedder = build_embedder(config.embedding)
    vector = build_vector_store(config.vector_store)
    lexical = Fts5LexicalIndex(config.lexical_store.path)
    catalog = SqliteCatalog(config.lexical_store.path)
    parser = ParserRegistry(config.parsing)
    corpus = LocalFsCorpus()
    cache_root = Path(config.lexical_store.path).resolve().parent / "cache"

    def open_embedding_cache() -> EmbeddingCache:
        return SqliteEmbeddingCache(cache_root / "embeddings.sqlite3")

    ingest = IngestService(
        config,
        corpus=corpus,
        parser=parser,
        embedder=embedder,
        vector=vector,
        lexical=lexical,
        catalog=catalog,
        artifacts=FsArtifactStore(cache_root),
        open_embedding_cache=open_embedding_cache,
    )
    search = SearchService(
        config,
        embedder=embedder,
        reranker=build_reranker(config.retrieval.rerank),
        vector=vector,
        lexical=lexical,
        catalog=catalog,
        corpus=corpus,
        parser=parser,
    )
    return Services(
        config=config,
        ingest=ingest,
        search=search,
        embedder=embedder,
        vector=vector,
        lexical=lexical,
        catalog=catalog,
    )

"""Config loading and typed access (v2 §D5)."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from .schema import EmbeddingProfile

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "retrieval.yaml"


class EmbeddingConfig(BaseModel):
    provider: str = "hash"
    hash_dim: int = 384
    model: str = "intfloat/multilingual-e5-small"
    revision: str = "main"
    dim: int = 384
    max_tokens: int = 512
    query_prefix: str = "query: "
    passage_prefix: str = "passage: "
    normalize: bool = True
    pooling: str = "mean"
    batch_size: int = 32
    threads: int = 8

    def effective_dim(self) -> int:
        return self.hash_dim if self.provider == "hash" else self.dim

    def profile(self) -> EmbeddingProfile:
        if self.provider == "hash":
            return EmbeddingProfile(
                provider="hash",
                model="hash-deterministic",
                revision="v1",
                dim=self.hash_dim,
                normalize=True,
                pooling="hash",
                query_prefix="",
                passage_prefix="",
            )
        return EmbeddingProfile(
            provider="e5",
            model=self.model,
            revision=self.revision,
            dim=self.dim,
            normalize=self.normalize,
            pooling=self.pooling,
            query_prefix=self.query_prefix,
            passage_prefix=self.passage_prefix,
        )


class ParsingConfig(BaseModel):
    backend: str = "native"
    detect_scanned_pdf: bool = True


class VectorStoreConfig(BaseModel):
    provider: str = "memory"
    collection: str = "rag_chunks_e5small_v1"
    distance: str = "cosine"
    persist_path: str = "local_data/chroma"


class LexicalStoreConfig(BaseModel):
    path: str = "local_data/manifest.sqlite3"


class ChunkingConfig(BaseModel):
    target_tokens: int = 300
    max_tokens: int = 480
    overlap_tokens: int = 40
    table_rows_per_chunk: int = 20


class FusionConfig(BaseModel):
    method: str = "rrf"
    k: int = 60


class RetrievalConfig(BaseModel):
    top_k: int = 8
    candidates_per_channel: int = 16
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    max_source_chars: int = 6000


class SecurityConfig(BaseModel):
    max_query_chars: int = 2000
    max_result_excerpt_chars: int = 1200


class Config(BaseModel):
    schema_version: int = 1
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    parsing: ParsingConfig = Field(default_factory=ParsingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    lexical_store: LexicalStoreConfig = Field(default_factory=LexicalStoreConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)

    def assert_release_safe(self) -> list[str]:
        """Release guard (v2 §D1/D3): fakes are not valid production backends."""
        problems: list[str] = []
        if self.embedding.provider == "hash":
            problems.append("embedding.provider=hash is test-only; use e5 for release")
        if self.vector_store.provider == "memory":
            problems.append("vector_store.provider=memory is test-only; use chroma for release")
        return problems


def load_config(path: str | Path | None = None) -> Config:
    p = Path(path) if path else DEFAULT_CONFIG_PATH
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return Config.model_validate(raw)

"""Config loading and typed access (v2 §D5)."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from .schema import EmbeddingProfile

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "retrieval.yaml"


class Qwen3Config(BaseModel):
    """A01 §D Qwen3 embedding-profile contract. Production dense candidate."""

    model: str = "Qwen/Qwen3-Embedding-4B"
    revision: str = "PIN_EXACT_RESOLVED_HF_COMMIT_AT_BOOTSTRAP"
    dim: int = 2560
    normalize: bool = True
    vector_storage_dtype: str = "float32"
    inference_dtype: str = "float16"   # bf16 permitted after tested hardware compatibility
    device: str = "cuda"               # fail visibly if unavailable; no silent fallback
    max_input_tokens: int = 2048       # operational cap, NOT the 32768 model max
    document_prompt: str | None = None  # no prefix for indexed passages
    query_prompt_name: str = "query"
    query_prompt_version: str = "qwen3_query_v1"
    tokenizer_padding_side: str = "left"
    batch_size: int = 4


class E5Config(BaseModel):
    """Legacy E5 profile, retained as an optional comparator (A01 supersedes as default)."""

    model: str = "intfloat/multilingual-e5-small"
    revision: str = "main"
    dim: int = 384
    max_tokens: int = 512
    query_prefix: str = "query: "
    passage_prefix: str = "passage: "
    normalize: bool = True
    pooling: str = "mean"
    batch_size: int = 32


class EmbeddingConfig(BaseModel):
    # provider: hash (test/dev) | qwen3 (A01 production candidate) | e5 (legacy comparator)
    provider: str = "hash"
    hash_dim: int = 2560               # match Qwen3 dim so store schema is provider-agnostic
    threads: int = 8
    qwen3: Qwen3Config = Field(default_factory=Qwen3Config)
    e5: E5Config = Field(default_factory=E5Config)

    def effective_dim(self) -> int:
        if self.provider == "hash":
            return self.hash_dim
        if self.provider == "qwen3":
            return self.qwen3.dim
        return self.e5.dim

    def profile(self) -> EmbeddingProfile:
        if self.provider == "hash":
            return EmbeddingProfile(
                provider="hash", model="hash-deterministic", revision="v1",
                dim=self.hash_dim, normalize=True, pooling="hash",
                query_prompt_version="hash_query_v1",
            )
        if self.provider == "qwen3":
            q = self.qwen3
            return EmbeddingProfile(
                provider="qwen3", model=q.model, revision=q.revision, dim=q.dim,
                normalize=q.normalize, pooling="last_token",
                query_prompt_name=q.query_prompt_name,
                query_prompt_version=q.query_prompt_version,
                inference_dtype=q.inference_dtype, device=q.device,
                max_input_tokens=q.max_input_tokens,
            )
        e = self.e5
        return EmbeddingProfile(
            provider="e5", model=e.model, revision=e.revision, dim=e.dim,
            normalize=e.normalize, pooling=e.pooling,
            query_prefix=e.query_prefix, passage_prefix=e.passage_prefix,
        )


class ParsingConfig(BaseModel):
    backend: str = "native"
    detect_scanned_pdf: bool = True


class VectorStoreConfig(BaseModel):
    provider: str = "memory"
    collection: str = "rag_chunks_qwen3_4b_2560_v1"  # A01 §G rename
    distance: str = "cosine"
    persist_path: str = "local_data/chroma"


class LexicalStoreConfig(BaseModel):
    path: str = "local_data/manifest.sqlite3"


class ChunkingConfig(BaseModel):
    # A01 §F: evidence-preserving target 350-700, operational cap 2048 (hypotheses).
    target_tokens: int = 500
    max_tokens: int = 2048
    overlap_tokens: int = 60
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
        """Release guard (v2 §D1/D3, A01): fakes are not valid production backends.

        A01 production dense candidate is qwen3; e5 remains an allowed comparator.
        """
        problems: list[str] = []
        if self.embedding.provider == "hash":
            problems.append("embedding.provider=hash is test-only; use qwen3 for release (A01)")
        if (self.embedding.provider == "qwen3"
                and "PIN_EXACT" in self.embedding.qwen3.revision):
            problems.append("embedding.qwen3.revision not pinned; resolve HF commit at bootstrap")
        if self.vector_store.provider == "memory":
            problems.append("vector_store.provider=memory is test-only; use chroma for release")
        return problems


def load_config(path: str | Path | None = None) -> Config:
    p = Path(path) if path else DEFAULT_CONFIG_PATH
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return Config.model_validate(raw)

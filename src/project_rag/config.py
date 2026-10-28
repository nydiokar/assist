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


class RerankConfig(BaseModel):
    """Cross-encoder reranker over the fused candidate pool (spec v1 §7, deferred until measured).

    Enabled only after the WP3 k-sweep (2026-10-27) showed answers sit in the candidate pool
    (doc 0.94 / evidence 0.82) but are mis-ranked at @5 (0.71 / 0.59) — the exact precondition
    the spec requires before adopting a reranker. Rank-only RRF cannot fix this; a content-aware
    scorer can. Off by default; query-time only (no reindex). Offline model from local HF cache.
    """

    enabled: bool = False
    model: str = "BAAI/bge-reranker-v2-m3"   # multilingual (BG+EN); pin revision when enabling
    revision: str = "main"
    device: str = "cpu"
    top_n: int = 12                          # rerank this many fused candidates (F19 sweep peak)
    batch_size: int = 16
    max_length: int = 512


class AbstentionConfig(BaseModel):
    """Retrieval-side evidence-sufficiency gate (spec v1 §6.2 "Abstention", WP4 precondition; F16).

    Emits an explicit "insufficient evidence" verdict when the best retrieved passage is not
    actually relevant to the query, so a downstream LLM returns "Not established from accessible
    project documents" (v1 §5.2) instead of confabulating over plausible-but-irrelevant chunks.

    Signal: the top-1 relevance score of the active ranker. `rerank_score` (cross-encoder, a
    calibrated 0-1 relevance probability) is the trustworthy signal — measured 2026-10-28 it
    cleanly separates the 11 no_answer questions (top1 ≤ 0.38, 10 of them ≤ 0.26) from answerable
    ones. `fusion_score` (raw RRF) is NOT calibrated (v1 §4.6: Chroma distances / rank-fusion
    scores are not confidence) and is only a weak fallback when the reranker is off; the gate
    warns when it has to use it.

    `min_top_score` is the measured operating point, not a guess: a threshold sweep (DECISIONS /
    EVAL_LOG 2026-10-28) put the knee on a flat 0.26-0.32 plateau — 0.26 abstains on 10/11
    no_answer while wrongly abstaining on only 2 real answerable hits (yp14, yp22).

    `suppress_on_insufficient` makes the response return the verdict INSTEAD of passages (the
    spec's "explicit no-evidence response"); off = passages are still returned and the verdict is
    advisory metadata (retrieval-first discipline, v1 §2.2 — the retriever flags, the answerer
    decides). Either way `SearchResponse.evidence` always carries the computed verdict.
    """

    enabled: bool = False
    signal: str = "rerank_score"          # rerank_score (calibrated) | fusion_score (weak fallback)
    min_top_score: float = 0.26           # measured knee (F19 reranker, 2026-10-28)
    suppress_on_insufficient: bool = True  # return verdict instead of passages when insufficient


class QueryExpansionConfig(BaseModel):
    """Cross-language query expansion (F11/F14, spec v1 §4.7). Query-time, no reindex, additive.

    Appends a small fixed local domain glossary of counterpart-language anchor terms when the query
    is in the minority script, so a short BG table row with no English token can still be fetched
    from an EN query (measured: yp09 miss->rank 1, scripts/probe_crosslang.py). NOT translation (no
    egress) and NOT a model swap (F11 disposition). Off by default. Honest limit: only glossary
    terms are expanded — a curated project lexicon, not general MT.
    """

    enabled: bool = False
    corpus_script: str = "cyrillic"   # dominant script of the indexed corpus (yspdb = Bulgarian)


class ContextExpansionConfig(BaseModel):
    """Parent-section context expansion (spec v1 §7). Query-time, no reindex, additive.

    When on, each served result gets `expanded_context`: the parent section the chunk came from
    (blocks sharing its heading; sheet for tables; page for PDFs), re-read via the path-safe
    `open_source`. This widens the evidence window handed to a WP4 answerer so it can recover the
    exact answer from the surrounding section when the top chunk alone did not carry it — the
    standard production move for the evidence@5 chunk-granularity gap (measured: span@5 0.718 vs
    evidence@5 0.667, +2q yp28/yp33; probe_span_coverage.py). Off by default; the chunk's own
    verbatim `excerpt` is never replaced.
    """

    enabled: bool = False
    max_chars: int = 4000


class RetrievalConfig(BaseModel):
    top_k: int = 8
    candidates_per_channel: int = 16
    # Max chunks from one source file allowed in the result window. WP3 2026-10-27 measured the
    # sweet spot at 3: looser (4=old top_k//2) lets sibling chunks crowd out the evidence chunk
    # (yp28/yp35), tighter (1-2) kills genuine multi-chunk-evidence questions. evidence@5 peaks
    # at cap=3 (0.706 vs 0.676 uncapped). None = fall back to legacy top_k//2.
    per_file_cap: int | None = 3
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    rerank: RerankConfig = Field(default_factory=RerankConfig)
    abstention: AbstentionConfig = Field(default_factory=AbstentionConfig)
    context_expansion: ContextExpansionConfig = Field(default_factory=ContextExpansionConfig)
    query_expansion: QueryExpansionConfig = Field(default_factory=QueryExpansionConfig)
    max_source_chars: int = 6000


class ClassificationConfig(BaseModel):
    """Document class by path prefix. `reference` = third-party material (laws, guides) that
    is not the project's own record; retrieval can filter on it (WP0 finding, owner-approved)."""

    reference_prefixes: list[str] = Field(default_factory=list)
    # Synthesize `doc_class=filesystem` presence chunks for folders holding unparseable files
    # (owner-requested 2026-10-28; DECISIONS). Off by default — a change forces a new generation
    # (it is folded into the pipeline fingerprint, ADR-009).
    filesystem_evidence: bool = False


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
    classification: ClassificationConfig = Field(default_factory=ClassificationConfig)

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

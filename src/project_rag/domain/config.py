"""Typed, immutable configuration models (v2 §D5; spec A03 §15.3 R4/R7).

Loading YAML is an effect and lives outside the domain; `with_overrides` derives a new config
from dotted keys without mutating the original.
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field

from .models import EmbeddingProfile


class _Cfg(BaseModel):
    model_config = ConfigDict(frozen=True)


class Qwen3Config(_Cfg):
    """A01 §D Qwen3 embedding-profile contract. Production dense candidate."""

    model: str = "Qwen/Qwen3-Embedding-4B"
    revision: str = "PIN_EXACT_RESOLVED_HF_COMMIT_AT_BOOTSTRAP"
    dim: int = 2560
    normalize: bool = True
    vector_storage_dtype: str = "float32"
    inference_dtype: str = "float16"  # bf16 permitted after tested hardware compatibility
    device: str = "cuda"  # fail visibly if unavailable; no silent fallback
    max_input_tokens: int = 2048  # operational cap, NOT the 32768 model max
    document_prompt: str | None = None  # no prefix for indexed passages
    query_prompt_name: str = "query"
    query_prompt_version: str = "qwen3_query_v1"
    tokenizer_padding_side: str = "left"
    batch_size: int = 4


class E5Config(_Cfg):
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


class RetryPolicy(_Cfg):
    """Bounded exponential backoff for transient provider failures (timeouts, 5xx, refused)."""

    attempts: int = 3  # total tries, first one included
    backoff_s: float = 0.5  # wait before retry n is backoff_s * 2**(n-1), capped
    max_backoff_s: float = 4.0


class OpenAICompatEmbedConfig(_Cfg):
    """llama-server (or any OpenAI-compatible) embeddings endpoint (spec §6.1, A1).

    The Qwen3 query instruct prompt is applied CLIENT-side to queries only (llama-server does not
    apply sentence-transformers' prompt_name); passages get no prefix. Vectors are L2-normalised
    client-side. `gguf_sha256` + `quant` are part of the profile fingerprint (A4).
    """

    base_url: str = "http://127.0.0.1:8081"
    model: str = "qwen3-embedding-4b"  # name sent in the request; the served GGUF decides
    model_id: str = "Qwen/Qwen3-Embedding-4B-GGUF"  # fingerprint identity
    revision: str = "f4602530db1d980e16da9d7d3a70294cf5c190be"
    gguf_file: str = "Qwen3-Embedding-4B-Q8_0.gguf"
    gguf_sha256: str = ""  # pinned from configs/models.lock.yaml; empty = not release-safe
    quant: str = "Q8_0"
    dim: int = 2560
    normalize: bool = True
    pooling: str = "last"
    # Exact template of the qwen3_query_v1 profile (Qwen3-Embedding-4B
    # config_sentence_transformers.json, prompts.query), prepended with no separator.
    query_prompt: str = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
    query_prompt_version: str = "qwen3_query_v1"
    # Appended to every input when the server does not add it itself (the HF tokenizer appends
    # <|endoftext|> and last-token pooling reads it). Set from the S3 parity measurement.
    append_text: str = ""
    max_input_tokens: int = 2048
    max_batch: int = 16
    timeout_s: float = 60.0
    connect_timeout_s: float = 3.0
    api_key: str = ""  # hosted fallback only (L5); set from the environment, never in YAML
    retry: RetryPolicy = Field(default_factory=RetryPolicy)


class EmbeddingConfig(_Cfg):
    # provider: hash (test/dev) | openai_compat (v3: llama-server) | qwen3 (torch reference)
    #           | e5 (legacy comparator)
    provider: str = "hash"
    hash_dim: int = 2560  # match Qwen3 dim so store schema is provider-agnostic
    threads: int = 8
    qwen3: Qwen3Config = Field(default_factory=Qwen3Config)
    e5: E5Config = Field(default_factory=E5Config)
    openai_compat: OpenAICompatEmbedConfig = Field(default_factory=OpenAICompatEmbedConfig)

    def effective_dim(self) -> int:
        if self.provider == "hash":
            return self.hash_dim
        if self.provider == "qwen3":
            return self.qwen3.dim
        if self.provider == "openai_compat":
            return self.openai_compat.dim
        return self.e5.dim

    def profile(self) -> EmbeddingProfile:
        if self.provider == "hash":
            return EmbeddingProfile(
                provider="hash",
                model="hash-deterministic",
                revision="v1",
                dim=self.hash_dim,
                normalize=True,
                pooling="hash",
                query_prompt_version="hash_query_v1",
            )
        if self.provider == "qwen3":
            q = self.qwen3
            return EmbeddingProfile(
                provider="qwen3",
                model=q.model,
                revision=q.revision,
                dim=q.dim,
                normalize=q.normalize,
                pooling="last_token",
                query_prompt_name=q.query_prompt_name,
                query_prompt_version=q.query_prompt_version,
                inference_dtype=q.inference_dtype,
                device=q.device,
                max_input_tokens=q.max_input_tokens,
            )
        if self.provider == "openai_compat":
            o = self.openai_compat
            return EmbeddingProfile(
                provider="openai_compat",
                model=o.model_id,
                revision=o.revision,
                dim=o.dim,
                normalize=o.normalize,
                pooling=o.pooling,
                query_prompt_name="query",
                query_prompt_version=o.query_prompt_version,
                max_input_tokens=o.max_input_tokens,
                gguf_sha256=o.gguf_sha256 or None,
                quant=o.quant,
            )
        e = self.e5
        return EmbeddingProfile(
            provider="e5",
            model=e.model,
            revision=e.revision,
            dim=e.dim,
            normalize=e.normalize,
            pooling=e.pooling,
            query_prefix=e.query_prefix,
            passage_prefix=e.passage_prefix,
        )


class OcrConfig(_Cfg):
    """OCR for scanned PDFs (ocrmypdf) and images (tesseract), spec §6.5 / A11.

    Image text is kept only when it has >= `min_chars` characters at >= `min_mean_conf` mean word
    confidence (tesseract 0-100); otherwise the image is NO_TEXT (event photos). The thresholds are
    tuned on the YP corpus in S4 and recorded in DECISIONS.
    """

    enabled: bool = False
    languages: str = "bul+eng"
    min_chars: int = 40
    min_mean_conf: float = 60.0
    image_extensions: list[str] = Field(
        default_factory=lambda: [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]
    )
    timeout_s: float = 300.0


class ParsingConfig(_Cfg):
    backend: str = "native"
    detect_scanned_pdf: bool = True
    ocr: OcrConfig = Field(default_factory=OcrConfig)
    antiword: bool = False  # legacy .doc via antiword (D-10); needs the binary
    antiword_timeout_s: float = 60.0


class IngestConfig(_Cfg):
    """Exclusions and isolation policy (S4). Exclusions mirror the rclone filters (S8, D-9):
    excluded files never enter the catalog, the generation id or coverage."""

    exclude_extensions: list[str] = Field(
        default_factory=lambda: [
            ".mp4",
            ".mov",
            ".m4a",
            ".mp3",
            ".wav",
            ".avi",
            ".mkv",
            ".webm",
            ".m4v",
            ".wmv",
        ]
    )
    exclude_name_globs: list[str] = Field(
        default_factory=lambda: [
            "~$*",
            ".~*",
            "*.tmp",
            "*.lock",
            "*~",
            "Thumbs.db",
            "desktop.ini",
            ".DS_Store",
        ]
    )
    max_file_bytes: int = 200_000_000
    # A run aborts (nothing published, previous generation stays) when embedding fails for at
    # least `abort_min_failures` documents AND more than this fraction of those needing vectors.
    max_embed_failure_fraction: float = 0.25
    abort_min_failures: int = 3


class VectorStoreConfig(_Cfg):
    # memory (tests) | chroma (D-3 choice, S2) | sqlite_vec (D-3 spike, same SQLite file)
    provider: str = "memory"
    collection: str = "rag_chunks_qwen3_4b_2560_v1"  # A01 §G rename
    distance: str = "cosine"
    persist_path: str = "local_data/chroma"


class LexicalStoreConfig(_Cfg):
    path: str = "local_data/manifest.sqlite3"


class ChunkingConfig(_Cfg):
    # A01 §F: evidence-preserving target 350-700, operational cap 2048 (hypotheses).
    target_tokens: int = 500
    max_tokens: int = 2048
    overlap_tokens: int = 60
    table_rows_per_chunk: int = 20


class FusionConfig(_Cfg):
    method: str = "rrf"
    k: int = 60


class OpenAICompatRerankConfig(_Cfg):
    """llama-server `--reranking` endpoint, Jina/Cohere-style `POST /v1/rerank` (spec §6.1, A2)."""

    base_url: str = "http://127.0.0.1:8082"
    model: str = "bge-reranker-v2-m3"
    gguf_sha256: str = ""
    # llama-server returns raw logits; the torch CrossEncoder returned sigmoid probabilities and
    # the abstention threshold (0.26) is on that scale. S3 parity: sigmoid(served) spans the same
    # 0-0.9991 range as the reference on the fixture pairs.
    score_transform: str = "sigmoid"  # sigmoid | none
    timeout_s: float = 30.0
    connect_timeout_s: float = 3.0
    retry: RetryPolicy = Field(default_factory=RetryPolicy)


class RerankConfig(_Cfg):
    """Cross-encoder reranker over the fused candidate pool (spec v1 §7, deferred until measured).

    Enabled only after the WP3 k-sweep (2026-10-27) showed answers sit in the candidate pool
    (doc 0.94 / evidence 0.82) but are mis-ranked at @5 (0.71 / 0.59) — the exact precondition
    the spec requires before adopting a reranker. Rank-only RRF cannot fix this; a content-aware
    scorer can. Off by default; query-time only (no reindex). Offline model from local HF cache.
    """

    enabled: bool = False
    provider: str = "torch"  # torch (dev reference) | openai_compat (v3: llama-server /v1/rerank)
    model: str = "BAAI/bge-reranker-v2-m3"  # multilingual (BG+EN); pin revision when enabling
    revision: str = "main"
    device: str = "cpu"
    top_n: int = 12  # rerank this many fused candidates (F19 sweep peak)
    batch_size: int = 16
    max_length: int = 512
    openai_compat: OpenAICompatRerankConfig = Field(default_factory=OpenAICompatRerankConfig)


class AbstentionConfig(_Cfg):
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
    signal: str = "rerank_score"  # rerank_score (calibrated) | fusion_score (weak fallback)
    min_top_score: float = 0.26  # measured knee (F19 reranker, 2026-10-28)
    suppress_on_insufficient: bool = True  # return verdict instead of passages when insufficient


class QueryExpansionConfig(_Cfg):
    """Cross-language query expansion (F11/F14, spec v1 §4.7). Query-time, no reindex, additive.

    Appends a small fixed local domain glossary of counterpart-language anchor terms when the query
    is in the minority script, so a short BG table row with no English token can still be fetched
    from an EN query (measured: yp09 miss->rank 1, scripts/probe_crosslang.py). NOT translation (no
    egress) and NOT a model swap (F11 disposition). Off by default. Honest limit: only glossary
    terms are expanded — a curated project lexicon, not general MT.
    """

    enabled: bool = False
    corpus_script: str = "cyrillic"  # dominant script of the indexed corpus (yspdb = Bulgarian)


class ContextExpansionConfig(_Cfg):
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


class RetrievalConfig(_Cfg):
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


class ClassificationConfig(_Cfg):
    """Document class by path prefix. `reference` = third-party material (laws, guides) that
    is not the project's own record; retrieval can filter on it (WP0 finding, owner-approved)."""

    reference_prefixes: list[str] = Field(default_factory=list)
    # Synthesize `doc_class=filesystem` presence chunks for folders holding unparseable files
    # (owner-requested 2026-10-28; DECISIONS). Off by default — a change forces a new generation
    # (it is folded into the pipeline fingerprint, ADR-009).
    filesystem_evidence: bool = False


class SecurityConfig(_Cfg):
    max_query_chars: int = 2000
    max_result_excerpt_chars: int = 1200


class Config(_Cfg):
    schema_version: int = 1
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    parsing: ParsingConfig = Field(default_factory=ParsingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    lexical_store: LexicalStoreConfig = Field(default_factory=LexicalStoreConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    classification: ClassificationConfig = Field(default_factory=ClassificationConfig)
    ingest: IngestConfig = Field(default_factory=IngestConfig)

    def assert_release_safe(self) -> list[str]:
        """Release guard (v2 §D1/D3, A01): fakes are not valid production backends.

        A01 production dense candidate is qwen3; e5 remains an allowed comparator.
        """
        problems: list[str] = []
        if self.embedding.provider == "hash":
            problems.append("embedding.provider=hash is test-only; use qwen3 for release (A01)")
        if self.embedding.provider == "qwen3" and "PIN_EXACT" in self.embedding.qwen3.revision:
            problems.append("embedding.qwen3.revision not pinned; resolve HF commit at bootstrap")
        if (
            self.embedding.provider == "openai_compat"
            and not self.embedding.openai_compat.gguf_sha256
        ):
            problems.append(
                "embedding.openai_compat.gguf_sha256 not pinned (configs/models.lock.yaml)"
            )
        if self.vector_store.provider == "memory":
            problems.append("vector_store.provider=memory is test-only; use chroma for release")
        return problems


def with_overrides(cfg: Config, overrides: Mapping[str, object]) -> Config:
    """Return a new Config with dotted keys replaced (e.g. ``{"retrieval.top_k": 5}``)."""
    data: dict[str, object] = cfg.model_dump()
    for dotted, value in overrides.items():
        *parents, leaf = dotted.split(".")
        node: dict[str, object] = data
        for name in parents:
            child = node[name]
            if not isinstance(child, dict):
                raise KeyError(f"{dotted}: {name} is not a section")
            node = child  # pyright: ignore[reportUnknownVariableType]
        if leaf not in node:
            raise KeyError(f"unknown config key: {dotted}")
        node[leaf] = value
    return Config.model_validate(data)

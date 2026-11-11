"""Domain models: document, block, chunk, citation, tool responses, traces and artifacts.

Identity levels follow v1 §4.3. All models are Pydantic and frozen (spec A03 §15.3 R4): code that
needs a changed value builds a new one with `model_copy(update=...)`.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Frozen(BaseModel):
    """Base for every domain model: immutable after construction (R4)."""

    model_config = ConfigDict(frozen=True)


class DocState(StrEnum):
    """Document state machine (v1 §4.1)."""

    DISCOVERED = "DISCOVERED"
    PARSING = "PARSING"
    PARSED = "PARSED"
    INDEXING = "INDEXING"
    READY = "READY"
    FAILED = "FAILED"
    UNSUPPORTED = "UNSUPPORTED"
    QUARANTINED = "QUARANTINED"
    OCR_REQUIRED = "OCR_REQUIRED"


class BlockType(StrEnum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE = "table"
    TABLE_ROW = "table_row"
    CELL = "cell"


class SourceLocation(Frozen):
    """Provenance address back to the original source (v1 §4.3)."""

    page: int | None = None  # PDF page (1-based)
    sheet: str | None = None  # XLSX worksheet name
    row: int | None = None  # CSV/XLSX row (1-based)
    cell: str | None = None  # XLSX cell ref, e.g. "B7"
    heading: str | None = None  # nearest heading text
    block_path: str | None = None  # structural path, e.g. "sec2/para3"
    char_start: int | None = None
    char_end: int | None = None


class Block(Frozen):
    """A logical block extracted from a parsed document."""

    block_type: BlockType
    text: str
    location: SourceLocation = Field(default_factory=SourceLocation)


class ParsedDocument(Frozen):
    """Output of a parser adapter (v1 §4.2)."""

    source_path: str  # relative path within project root
    state: DocState
    blocks: list[Block] = Field(default_factory=list)
    diagnostics: list[str] = Field(default_factory=list)


class Chunk(Frozen):
    """An embeddable, citable unit (v1 §4.4 key design).

    `embed_text` is the clean text used for embeddings (may carry heading context);
    `citation_text` is the original verbatim text used for citations/display.
    """

    chunk_id: str
    project_id: str
    source_id: str
    version_id: str
    document_id: str  # == source_id; kept for response contract clarity
    embed_text: str  # alias: A01 retrieval_text (enriched, auditable)
    citation_text: str  # verbatim original; citations refer to THIS, never embed_text
    block_type: BlockType
    heading: str | None = None
    location: SourceLocation = Field(default_factory=SourceLocation)
    chunk_ordinal: int = 0
    source_path: str = ""
    source_type: str = ""  # file extension without dot
    doc_class: str = (
        "project_record"  # project_record | reference | filesystem (config classification)
    )
    token_count: int = 0  # measured tokens of embed_text (A01 §F)
    truncated: bool = False  # True if embed_text was token-truncated (A01 §F fail check)

    @property
    def retrieval_text(self) -> str:
        """A01 §F name for the enriched text used for embeddings."""
        return self.embed_text


class EmbeddingProfile(Frozen):
    """Pinned embedding identity (v1 §4.5 safeguard 1, A01 §D). Persisted per collection.

    Supports two query-conditioning styles:
      - prefix style (legacy E5): query_prefix/passage_prefix non-empty.
      - prompt style (Qwen3): query_prompt_name/query_prompt_version set, no prefixes.
    """

    provider: str
    model: str
    revision: str
    dim: int
    normalize: bool
    pooling: str
    query_prefix: str = ""
    passage_prefix: str = ""
    query_prompt_name: str | None = None
    query_prompt_version: str | None = None
    inference_dtype: str | None = None  # recorded for drift diagnosis (A01 §C)
    device: str | None = None
    max_input_tokens: int | None = None

    def key(self) -> str:
        qp = self.query_prompt_version or self.query_prefix
        return (
            f"{self.provider}:{self.model}:{self.revision}:{self.dim}:"
            f"{self.pooling}:{int(self.normalize)}:{qp}"
        )


# ---- Tool response contracts (v1 §3.2) ----


class RetrievedResult(Frozen):
    chunk_id: str
    document_id: str
    source_path: str
    source_location: SourceLocation
    excerpt: str
    rank: int
    retrieval_channels: list[str]
    score_details: dict[str, float] = Field(default_factory=dict)
    expanded_context: str = ""  # parent-section context (v1 §7, query-time; empty unless on)


class EvidenceVerdict(Frozen):
    """Retrieval-side evidence-sufficiency decision (v1 §6.2 abstention gate, F16).

    `sufficient=False` means the best retrieved passage is not relevant enough to ground an
    answer; a downstream answerer should say "Not established from accessible project documents"
    (v1 §5.2) rather than answer from the returned chunks. `enabled=False` means no gate was
    applied (verdict is informational only). Scores are ranking/relevance evidence, never a
    probability of factual correctness (v1 §3.2).
    """

    enabled: bool = False
    sufficient: bool = True
    signal: str = ""  # score_details key used (rerank_score | fusion_score)
    top_score: float | None = None  # top-1 relevance score observed
    threshold: float | None = None  # min_top_score the signal was compared against


class SearchResponse(Frozen):
    query_id: str
    project_id: str
    index_generation: str
    retriever_version: str = "hybrid_v1"
    results: list[RetrievedResult] = Field(default_factory=list)
    evidence: EvidenceVerdict = Field(default_factory=EvidenceVerdict)
    warnings: list[str] = Field(default_factory=list)


class SourceResponse(Frozen):
    project_id: str
    document_id: str
    source_path: str
    location: SourceLocation | None
    excerpt: str
    truncated: bool = False
    warnings: list[str] = Field(default_factory=list)


class FileReport(Frozen):
    source_path: str
    state: DocState
    source_id: str
    version_id: str | None = None
    n_blocks: int = 0
    n_chunks: int = 0
    diagnostics: list[str] = Field(default_factory=list)


class IngestReport(Frozen):
    project_id: str
    index_generation: str
    dry_run: bool = False
    files: list[FileReport] = Field(default_factory=list)
    added: int = 0
    modified: int = 0
    deleted: int = 0
    unchanged: int = 0
    embedding_profile_key: str = ""
    warnings: list[str] = Field(default_factory=list)


class StatusReport(Frozen):
    project_id: str
    index_generation: str | None
    n_documents: int
    n_chunks: int
    states: dict[str, int] = Field(default_factory=dict)
    embedding_profile_key: str = ""
    collection: str = ""
    healthy: bool = True
    warnings: list[str] = Field(default_factory=list)


# ---- Discovery, channel hits, stage artifacts, traces ----


class DiscoveredFile(Frozen):
    """One file found under a project root (v1 §4.1)."""

    source_path: str  # relative to root, forward-slash normalized
    abs_path: str
    ext: str
    size: int
    mtime: float  # diagnostic only (v1 §4.1)
    content_hash: str  # SHA-256
    supported: bool


class HitPayload(Frozen):
    """What a lexical/vector channel returns per chunk so results build without a second lookup."""

    document_id: str
    source_path: str
    heading: str = ""
    location: SourceLocation = Field(default_factory=SourceLocation)
    citation_text: str = ""


class ChannelHit(Frozen):
    """One ranked candidate from a retrieval channel. `score` is channel-native (cosine distance
    or bm25, lower is better) and is never used as confidence (v1 §4.6)."""

    chunk_id: str
    score: float
    payload: HitPayload


class ParseArtifact(Frozen):
    artifact_id: str  # == parse_key (identity for downstream chunk key)
    source_path: str
    source_content_hash: str
    doc: ParsedDocument


class ChunkArtifact(Frozen):
    artifact_id: str  # == chunk_key
    parsed_artifact_id: str
    chunks: list[Chunk]


class QueryTrace(Frozen):
    """Structured query diagnostics (v1 §5.2). No raw PII: the query is hashed."""

    query_id: str
    tool: str
    project_id: str
    query_hash: str
    index_generation: str
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    channels: dict[str, int] = Field(default_factory=dict)  # channel -> candidate count
    result_count: int = 0
    timings_ms: dict[str, float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class DocumentRecord(Frozen):
    """One catalog row: a source document's state within a generation."""

    project_id: str
    source_id: str
    source_path: str
    content_hash: str
    version_id: str
    state: str
    generation: str
    n_chunks: int = 0
    diagnostics: str = ""


class OcrText(Frozen):
    """OCR output for one image: the text and the engine's mean word confidence (0-100)."""

    text: str
    mean_conf: float


class FeedbackSignal(Frozen):
    """A quality signal recorded by the service or the agent (spec §6.9). No raw query text."""

    kind: str  # abstain | empty | degraded | user_dissatisfied | rescue
    project_id: str
    at: datetime
    query_hash: str = ""
    detail: str = ""

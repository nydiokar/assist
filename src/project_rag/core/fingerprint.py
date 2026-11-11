"""Deterministic identities: source/version/generation ids, stage cache keys, query ids.

Pure functions over strings and frozen config (v1 §4.3, ADR-009, A01 §C). Every id is a truncated
sha256 so re-running on identical input yields identical identity.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

from ..domain.config import ChunkingConfig, ClassificationConfig, ParsingConfig
from .chunking import CHUNKER_VERSION

PARSER_VERSION = "native_v1"


def _sha(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def text_hash(text: str) -> str:
    """Full sha256 of a text (embedding-cache key component, ledger F13)."""
    return _sha(text)


def source_id(source_path: str) -> str:
    return "src_" + _sha(source_path)[:20]


def version_id(content_hash: str, parser_backend: str) -> str:
    return "v_" + _sha(f"{content_hash}|{parser_backend}|{CHUNKER_VERSION}")[:20]


def pipeline_fingerprint(
    parsing: ParsingConfig,
    chunking: ChunkingConfig,
    classification: ClassificationConfig,
    embedding_profile_key: str,
) -> str:
    """Everything that changes the derived index: chunks AND their vectors. Folded into the
    generation id, so a different embedding profile (provider, model, GGUF sha256, quant, pooling,
    prompt version, dim) is a new generation (ADR-009, A4)."""
    return "|".join(
        [
            CHUNKER_VERSION,
            parsing.model_dump_json(),  # backend + OCR/antiword switches and thresholds (S4)
            chunking.model_dump_json(),
            classification.model_dump_json(),
            embedding_profile_key,
        ]
    )


def generation_id(project_id: str, hashes: Iterable[str], pipeline: str) -> str:
    """Identity of the whole derived index: sources AND the pipeline that produced it.
    A chunker/parser/config change must yield a new generation (ADR-009), not 'no changes'."""
    raw = project_id + "|" + pipeline + "|" + "|".join(sorted(hashes))
    return "g_" + _sha(raw)[:16]


def stage_fp(*parts: str) -> str:
    return _sha("|".join(parts))[:32]


def parse_key(source_content_hash: str, parser_cfg: ParsingConfig) -> str:
    return stage_fp("parse", source_content_hash, PARSER_VERSION, parser_cfg.model_dump_json())


def chunk_key(parsed_artifact_id: str, chunk_cfg: ChunkingConfig) -> str:
    return stage_fp("chunk", parsed_artifact_id, CHUNKER_VERSION, chunk_cfg.model_dump_json())


def chunk_lineage_key(key: str, project_id: str, source_id: str, version_id: str) -> str:
    """Chunk identity also depends on project/source/version lineage (A01 §C)."""
    return stage_fp(key, project_id, source_id, version_id)


def query_hash(query: str) -> str:
    return _sha(query)[:16]


def new_query_id(project_id: str, query: str, nonce: str) -> str:
    return "q_" + _sha(f"{project_id}|{query}|{nonce}")[:16]


def classify_doc(source_path: str, reference_prefixes: tuple[str, ...]) -> str:
    """`reference` for third-party material under a configured prefix, else `project_record`."""
    if reference_prefixes and source_path.startswith(reference_prefixes):
        return "reference"
    return "project_record"

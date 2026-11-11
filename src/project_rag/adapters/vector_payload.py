"""Per-chunk metadata stored by vector adapters and its mapping back to a HitPayload.

Metadata stored per chunk: project_id, document_id, version_id, generation, source_path, heading,
source_type, doc_class, location_json, citation_text (v1 §4.6 collection model).
"""

from __future__ import annotations

from collections.abc import Mapping

from ..domain.models import Chunk, HitPayload, SourceLocation


def chunk_metadata(c: Chunk, generation: str) -> dict[str, str]:
    return {
        "chunk_id": c.chunk_id,
        "project_id": c.project_id,
        "document_id": c.document_id,
        "version_id": c.version_id,
        "generation": generation,
        "source_path": c.source_path,
        "heading": c.heading or "",
        "source_type": c.source_type,
        "doc_class": c.doc_class,
        "location_json": c.location.model_dump_json(),
        "citation_text": c.citation_text,
    }


def hit_payload(meta: Mapping[str, object]) -> HitPayload:
    return HitPayload(
        document_id=str(meta["document_id"]),
        source_path=str(meta["source_path"]),
        heading=str(meta.get("heading", "")),
        location=SourceLocation.model_validate_json(str(meta["location_json"])),
        citation_text=str(meta["citation_text"]),
    )

"""Shared helpers for the port contract suites (spec A03 §15.3 R3).

Every adapter of a port runs the same suite. Adapters that need a GPU, model weights or a live
server are marked `live` and skip unless their dependency is importable and usable.
"""

from __future__ import annotations

from project_rag.domain.models import BlockType, Chunk, SourceLocation


def make_chunk(
    cid: str,
    text: str,
    *,
    project_id: str = "p1",
    source_path: str = "a.md",
    doc_class: str = "project_record",
) -> Chunk:
    return Chunk(
        chunk_id=cid,
        project_id=project_id,
        source_id="src_" + source_path,
        version_id="v1",
        document_id="src_" + source_path,
        embed_text=text,
        citation_text=text,
        block_type=BlockType.PARAGRAPH,
        heading="H",
        location=SourceLocation(heading="H", block_path=cid),
        source_path=source_path,
        source_type="md",
        doc_class=doc_class,
    )

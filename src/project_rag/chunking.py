"""Structure-aware + table-aware chunking (v1 §4.4, Amendment A01 §F).

Free-text blocks under a heading are grouped toward target_tokens; oversized sections split
at paragraph boundaries. Table rows are grouped into row-group chunks with the heading/sheet
context repeated (semantic overlap, not blind token duplication). Numeric values stay attached
to their column header because the parser already emits "header=value" rows.

A01 §F: actual model tokens are counted before encoding via an injected token_counter; chunks
exceeding max_tokens are marked truncated=True (a failed ingestion check surfaced upstream) and
their enriched embed_text (A01 retrieval_text) is capped — citation_text keeps the verbatim
original. chunk_id is a deterministic hash (v1 §4.3) — never a sequential int.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable

from .config import ChunkingConfig
from .schema import Block, BlockType, Chunk, ParsedDocument, SourceLocation

CHUNKER_VERSION = "structure_v2"  # A01 §F token-aware chunking


def _approx_tokens(text: str) -> int:
    # ~1.3 tokens/word proxy; conservative for mixed Bulgarian/English.
    return max(1, int(len(text.split()) * 1.3))


def _chunk_id(project_id: str, source_id: str, version_id: str, block_path: str,
              ordinal: int) -> str:
    raw = f"{project_id}|{source_id}|{version_id}|{block_path}|{ordinal}|{CHUNKER_VERSION}"
    return "ck_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def chunk_document(doc: ParsedDocument, *, project_id: str, source_id: str,
                   version_id: str, source_type: str, cfg: ChunkingConfig,
                   token_counter: Callable[[str], int] | None = None) -> list[Chunk]:
    count_tokens = token_counter or _approx_tokens
    text_blocks = [b for b in doc.blocks if b.block_type in
                   {BlockType.PARAGRAPH, BlockType.LIST_ITEM, BlockType.HEADING}]
    table_blocks = [b for b in doc.blocks if b.block_type in
                    {BlockType.TABLE_ROW, BlockType.TABLE, BlockType.CELL}]

    chunks: list[Chunk] = []
    ordinal = 0

    def emit(embed_text: str, citation_text: str, btype: BlockType, heading: str | None,
             loc: SourceLocation) -> None:
        nonlocal ordinal
        bp = loc.block_path or f"ord{ordinal}"
        cid = _chunk_id(project_id, source_id, version_id, bp, ordinal)
        n_tok = count_tokens(embed_text)
        truncated = n_tok > cfg.max_tokens
        final_embed = embed_text
        if truncated:
            # Cap enriched embed_text by words proportional to the token overflow; citation stays whole.
            words = embed_text.split()
            keep = max(1, int(len(words) * cfg.max_tokens / max(1, n_tok)))
            final_embed = " ".join(words[:keep])
            n_tok = count_tokens(final_embed)
        chunks.append(Chunk(
            chunk_id=cid, project_id=project_id, source_id=source_id, version_id=version_id,
            document_id=source_id, embed_text=final_embed, citation_text=citation_text,
            block_type=btype, heading=heading, location=loc, chunk_ordinal=ordinal,
            source_path=doc.source_path, source_type=source_type,
            token_count=n_tok, truncated=truncated,
        ))
        ordinal += 1

    # Free text: accumulate under nearest heading toward target_tokens.
    buf: list[Block] = []
    buf_tokens = 0
    cur_heading: str | None = None

    def flush_text() -> None:
        nonlocal buf, buf_tokens
        if not buf:
            return
        body = "\n".join(b.text for b in buf)
        citation = body
        embed = f"{cur_heading}\n{body}" if cur_heading else body
        loc = SourceLocation(
            heading=cur_heading,
            page=buf[0].location.page,
            block_path=buf[0].location.block_path,
            sheet=buf[0].location.sheet,
        )
        emit(embed, citation, BlockType.PARAGRAPH, cur_heading, loc)
        buf = []
        buf_tokens = 0

    for b in text_blocks:
        if b.block_type == BlockType.HEADING:
            flush_text()
            cur_heading = b.text
            continue
        t = count_tokens(b.text)
        if buf and buf_tokens + t > cfg.target_tokens:
            flush_text()
        buf.append(b)
        buf_tokens += t
        if buf_tokens >= cfg.max_tokens:
            flush_text()
    flush_text()

    # Tables: group rows, repeat heading/sheet context per chunk.
    group: list[Block] = []

    def flush_table() -> None:
        nonlocal group
        if not group:
            return
        ctx = group[0].location.heading or group[0].location.sheet or ""
        body = "\n".join(b.text for b in group)
        embed = f"{ctx}\n{body}" if ctx else body
        loc = SourceLocation(
            heading=group[0].location.heading, sheet=group[0].location.sheet,
            row=group[0].location.row, block_path=group[0].location.block_path,
            page=group[0].location.page,
        )
        emit(embed, body, BlockType.TABLE, group[0].location.heading, loc)
        group = []

    for b in table_blocks:
        group.append(b)
        if len(group) >= cfg.table_rows_per_chunk:
            flush_table()
    flush_table()

    return chunks

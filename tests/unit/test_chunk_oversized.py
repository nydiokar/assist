"""Oversized blocks (survey rows, PDF pages) must be split, never silently truncated (ledger F5)."""

from __future__ import annotations

from project_rag.core.chunking import chunk_document
from project_rag.domain.config import ChunkingConfig
from project_rag.domain.models import Block, BlockType, DocState, ParsedDocument, SourceLocation


def test_giant_xlsx_row_is_split_not_truncated() -> None:
    words = " ".join(f"дума{i}" for i in range(6000))
    row = Block(
        block_type=BlockType.TABLE_ROW,
        text=f"[Анкета] Отговор={words}",
        location=SourceLocation(sheet="Анкета", row=7, block_path="Анкета.row7"),
    )
    doc = ParsedDocument(source_path="a.xlsx", state=DocState.PARSED, blocks=[row])
    cfg = ChunkingConfig()
    chunks = chunk_document(
        doc, project_id="p", source_id="s", version_id="v", source_type="xlsx", cfg=cfg
    )
    assert len(chunks) > 1
    assert not any(c.truncated for c in chunks)
    assert all(c.token_count <= cfg.max_tokens for c in chunks)
    joined = " ".join(c.citation_text for c in chunks)
    assert "дума0 " in joined and "дума5999" in joined  # first and last word both kept
    assert len({c.chunk_id for c in chunks}) == len(chunks)  # distinct deterministic ids


def test_repeated_page_boilerplate_is_dropped_but_content_kept() -> None:
    blocks: list[Block] = []
    for page in range(1, 9):
        blocks.append(
            Block(
                block_type=BlockType.PARAGRAPH,
                text="Form ID KA154 Deadline 08 Oct 2024",
                location=SourceLocation(block_path=f"b{page}h"),
            )
        )
        blocks.append(
            Block(
                block_type=BlockType.PARAGRAPH,
                text=f"{page} / 8",
                location=SourceLocation(block_path=f"b{page}n"),
            )
        )
        blocks.append(
            Block(
                block_type=BlockType.PARAGRAPH,
                text=f"Unique content of page {page} budget 49 697,00",
                location=SourceLocation(block_path=f"b{page}c"),
            )
        )
    doc = ParsedDocument(source_path="p.docx", state=DocState.PARSED, blocks=blocks)
    chunks = chunk_document(
        doc, project_id="p", source_id="s", version_id="v", source_type="docx", cfg=ChunkingConfig()
    )
    text = " ".join(c.citation_text for c in chunks)
    assert "Form ID" not in text and "/ 8" not in text
    assert "Unique content of page 3" in text and "49 697,00" in text

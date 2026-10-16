"""Oversized blocks (survey rows, PDF pages) must be split, never silently truncated (ledger F5)."""

from __future__ import annotations

from project_rag.chunking import chunk_document
from project_rag.config import ChunkingConfig
from project_rag.schema import Block, BlockType, DocState, ParsedDocument, SourceLocation


def test_giant_xlsx_row_is_split_not_truncated() -> None:
    words = " ".join(f"дума{i}" for i in range(6000))
    row = Block(block_type=BlockType.TABLE_ROW, text=f"[Анкета] Отговор={words}",
                location=SourceLocation(sheet="Анкета", row=7, block_path="Анкета.row7"))
    doc = ParsedDocument(source_path="a.xlsx", state=DocState.PARSED, blocks=[row])
    cfg = ChunkingConfig()
    chunks = chunk_document(doc, project_id="p", source_id="s", version_id="v",
                            source_type="xlsx", cfg=cfg)
    assert len(chunks) > 1
    assert not any(c.truncated for c in chunks)
    assert all(c.token_count <= cfg.max_tokens for c in chunks)
    joined = " ".join(c.citation_text for c in chunks)
    assert "дума0 " in joined and "дума5999" in joined       # first and last word both kept
    assert len({c.chunk_id for c in chunks}) == len(chunks)  # distinct deterministic ids

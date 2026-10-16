"""Optional Docling parser adapter (v1 §4.2, ADR-002). Loaded only when the `docling` extra
is installed and parsing.backend == 'docling'. Import of this module is guarded by parsers.py,
so a missing dependency degrades to native parsers rather than erroring.

Docling preserves richer structure (tables, reading order) than the native adapters. Validate
per-format fidelity on the real corpus before relying on it (v1 §4.2 parsing gate).
"""

from __future__ import annotations

from .schema import Block, BlockType, DocState, ParsedDocument, SourceLocation

_DOCLING_EXTS = {".pdf", ".docx", ".xlsx"}


class DoclingParser:
    def __init__(self) -> None:
        from docling.document_converter import DocumentConverter  # noqa: PLC0415

        self._converter = DocumentConverter()

    def can_parse(self, ext: str) -> bool:
        return ext in _DOCLING_EXTS

    def parse(self, abs_path: str, source_path: str) -> ParsedDocument:
        try:
            result = self._converter.convert(abs_path)
        except Exception as e:  # noqa: BLE001
            return ParsedDocument(source_path=source_path, state=DocState.FAILED,
                                  diagnostics=[f"docling convert failed: {e}"])
        doc = result.document
        blocks: list[Block] = []
        # Docling exposes a markdown/structured export; use it as the normalized block stream.
        # Kept intentionally simple — fidelity tuning happens against the real corpus (WP1 gate).
        markdown = doc.export_to_markdown()
        heading: str | None = None
        for i, line in enumerate(markdown.splitlines()):
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                heading = s.lstrip("#").strip()
                blocks.append(Block(block_type=BlockType.HEADING, text=heading,
                                    location=SourceLocation(block_path=f"h{i}")))
            elif s.startswith("|"):
                blocks.append(Block(block_type=BlockType.TABLE_ROW, text=s.strip("|").strip(),
                                    location=SourceLocation(heading=heading, block_path=f"t{i}")))
            else:
                blocks.append(Block(block_type=BlockType.PARAGRAPH, text=s,
                                    location=SourceLocation(heading=heading, block_path=f"p{i}")))
        if not blocks:
            return ParsedDocument(source_path=source_path, state=DocState.OCR_REQUIRED,
                                  diagnostics=["docling produced no text; possible scan"])
        return ParsedDocument(source_path=source_path, state=DocState.PARSED, blocks=blocks)

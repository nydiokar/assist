"""OCR-aware document parsing (S4, spec §6.5): native/antiword parsing first, OCR when needed.

- Images (when OCR is on): tesseract text, kept only if `core.ingest_policy.ocr_keep` accepts it,
  else NO_TEXT (event photos).
- PDFs the native parser marks OCR_REQUIRED: ocrmypdf page texts become page-addressed blocks;
  nothing usable -> NO_TEXT.
- An OCR failure is FAILED with the reason, for that document only.
The result is cached by the parse stage (the OCR config is part of the parse cache key).
"""

from __future__ import annotations

from ..core.ingest_policy import ocr_keep
from ..domain.config import ParsingConfig
from ..domain.errors import OcrFailed
from ..domain.models import Block, BlockType, DocState, ParsedDocument, SourceLocation
from ..ports.ocr import OcrEngine
from ..ports.parser import DocumentParser


class OcrAwareParser:
    def __init__(self, parser: DocumentParser, ocr: OcrEngine | None, cfg: ParsingConfig) -> None:
        self._parser = parser
        self._ocr = ocr if cfg.ocr.enabled else None
        self._cfg = cfg
        self._image_exts = frozenset(e.lower() for e in cfg.ocr.image_extensions)

    def supported_extensions(self) -> frozenset[str]:
        exts = set(self._parser.supported_extensions())
        if self._ocr is not None:
            exts |= self._image_exts
        return frozenset(exts)

    def parse(self, abs_path: str, source_path: str, ext: str) -> ParsedDocument:
        if self._ocr is not None and ext in self._image_exts:
            return self._ocr_image(self._ocr, abs_path, source_path)
        doc = self._parser.parse(abs_path, source_path, ext)
        if self._ocr is not None and doc.state == DocState.OCR_REQUIRED:
            return self._ocr_pdf(self._ocr, abs_path, source_path, doc)
        return doc

    def _ocr_image(self, ocr: OcrEngine, abs_path: str, source_path: str) -> ParsedDocument:
        try:
            result = ocr.ocr_image(abs_path)
        except OcrFailed as e:
            return ParsedDocument(
                source_path=source_path, state=DocState.FAILED, diagnostics=[f"ocr failed: {e}"]
            )
        diag = f"ocr_image chars={len(result.text.strip())} mean_conf={result.mean_conf:.1f}"
        if not ocr_keep(result.text, result.mean_conf, self._cfg.ocr):
            return ParsedDocument(
                source_path=source_path,
                state=DocState.NO_TEXT,
                diagnostics=[diag, "below OCR keep threshold"],
            )
        paragraphs = [p.strip() for p in result.text.split("\n\n") if p.strip()]
        blocks = [
            Block(
                block_type=BlockType.PARAGRAPH,
                text=p,
                location=SourceLocation(block_path=f"ocr.p{i}"),
            )
            for i, p in enumerate(paragraphs)
        ]
        return ParsedDocument(
            source_path=source_path, state=DocState.PARSED, blocks=blocks, diagnostics=[diag]
        )

    def _ocr_pdf(
        self, ocr: OcrEngine, abs_path: str, source_path: str, native: ParsedDocument
    ) -> ParsedDocument:
        try:
            pages = ocr.ocr_pdf(abs_path)
        except OcrFailed as e:
            return ParsedDocument(
                source_path=source_path,
                state=DocState.FAILED,
                diagnostics=[*native.diagnostics, f"ocr failed: {e}"],
            )
        blocks: list[Block] = []
        for pageno, page in enumerate(pages, start=1):
            for i, para in enumerate(p.strip() for p in page.split("\n\n")):
                if para:
                    blocks.append(
                        Block(
                            block_type=BlockType.PARAGRAPH,
                            text=" ".join(para.split()),
                            location=SourceLocation(page=pageno, block_path=f"ocr.p{pageno}.{i}"),
                        )
                    )
        chars = sum(len(b.text) for b in blocks)
        diag = f"ocr_pdf pages={len(pages)} chars={chars}"
        if chars < self._cfg.ocr.min_chars:
            return ParsedDocument(
                source_path=source_path,
                state=DocState.NO_TEXT,
                diagnostics=[*native.diagnostics, diag, "below OCR keep threshold"],
            )
        return ParsedDocument(
            source_path=source_path,
            state=DocState.PARSED,
            blocks=blocks,
            diagnostics=[*native.diagnostics, diag],
        )

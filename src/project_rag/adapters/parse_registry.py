"""Parser dispatch by extension (implements the DocumentParser port). Native backend by default;
Docling optional (v2 §D2) and degrades to native when its extra is missing."""

from __future__ import annotations

from ..domain.config import ParsingConfig
from ..domain.models import DocState, ParsedDocument
from ..ports.parser import FormatParser
from .parse_antiword import AntiwordParser
from .parse_native import CsvParser, DocxParser, PdfParser, TextParser, XlsxParser


class ParserRegistry:
    """Dispatches by extension. Default native backend; Docling optional (v2 §D2)."""

    def __init__(self, cfg: ParsingConfig) -> None:
        self._cfg = cfg
        self._parsers: list[FormatParser] = [
            TextParser(),
            PdfParser(detect_scanned=cfg.detect_scanned_pdf),
            DocxParser(),
            XlsxParser(),
            CsvParser(),
        ]
        if cfg.antiword:
            self._parsers.append(AntiwordParser(timeout_s=cfg.antiword_timeout_s))

    def supported_extensions(self) -> frozenset[str]:
        exts = {".pdf", ".docx", ".xlsx", ".csv", ".txt", ".md"}
        if self._cfg.antiword:
            exts.add(".doc")
        return frozenset(exts)

    def parse(self, abs_path: str, source_path: str, ext: str) -> ParsedDocument:
        if self._cfg.backend == "docling":
            doc = self._try_docling(abs_path, source_path, ext)
            if doc is not None:
                return doc
        for parser in self._parsers:
            if parser.can_parse(ext):
                return parser.parse(abs_path, source_path)
        return ParsedDocument(
            source_path=source_path,
            state=DocState.UNSUPPORTED,
            diagnostics=[f"no parser for extension {ext}"],
        )

    def _try_docling(self, abs_path: str, source_path: str, ext: str) -> ParsedDocument | None:
        try:
            from .parse_docling import DoclingParser
        except ImportError:
            return None
        p = DoclingParser()
        if not p.can_parse(ext):
            return None
        return p.parse(abs_path, source_path)

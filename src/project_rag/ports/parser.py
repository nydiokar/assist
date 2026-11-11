"""Parser ports. Parsers report content problems as document states, never by raising."""

from __future__ import annotations

from typing import Protocol

from ..domain.models import ParsedDocument


class FormatParser(Protocol):
    """One format (or family): `can_parse` by extension, `parse` one file."""

    def can_parse(self, ext: str) -> bool: ...

    def parse(self, abs_path: str, source_path: str) -> ParsedDocument: ...


class DocumentParser(Protocol):
    """Dispatches any file to the right format parser (UNSUPPORTED when none applies)."""

    def parse(self, abs_path: str, source_path: str, ext: str) -> ParsedDocument: ...

    def supported_extensions(self) -> frozenset[str]:
        """Lower-case extensions (with dot) this parser turns into text."""
        ...

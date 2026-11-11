"""Parser port contracts: content problems become document states, never exceptions."""

from __future__ import annotations

from pathlib import Path

import pytest

from project_rag.adapters.parse_native import (
    CsvParser,
    DocxParser,
    PdfParser,
    TextParser,
    XlsxParser,
)
from project_rag.adapters.parse_registry import ParserRegistry
from project_rag.domain.config import ParsingConfig
from project_rag.domain.models import DocState
from project_rag.ports.parser import DocumentParser, FormatParser

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "synthetic" / "ngo_project_demo"

FORMAT_PARSERS: list[tuple[FormatParser, str]] = [
    (TextParser(), ".md"),
    (PdfParser(), ".pdf"),
    (DocxParser(), ".docx"),
    (XlsxParser(), ".xlsx"),
    (CsvParser(), ".csv"),
]


@pytest.fixture(params=[pytest.param(ParserRegistry(ParsingConfig()), id="native_registry")])
def document_parser(request: pytest.FixtureRequest) -> DocumentParser:
    return request.param


@pytest.mark.parametrize("path", sorted(FIXTURES.iterdir()), ids=lambda p: p.name)
def test_fixture_files_parse_to_a_terminal_state(
    document_parser: DocumentParser, path: Path
) -> None:
    doc = document_parser.parse(str(path), path.name, path.suffix.lower())
    assert doc.source_path == path.name
    assert doc.state in {DocState.PARSED, DocState.OCR_REQUIRED}
    if doc.state == DocState.PARSED:
        assert doc.blocks, "a parsed fixture yields blocks"
        assert all(b.text is not None for b in doc.blocks)


@pytest.mark.parametrize("ext", [".pdf", ".docx", ".xlsx"])
def test_corrupt_binary_is_failed_not_raised(
    document_parser: DocumentParser, tmp_path: Path, ext: str
) -> None:
    bad = tmp_path / f"corrupt{ext}"
    bad.write_bytes(b"\x00\x01 not a real document \xff" * 10)
    doc = document_parser.parse(str(bad), bad.name, ext)
    assert doc.state == DocState.FAILED
    assert doc.diagnostics


def test_unknown_extension_is_unsupported(document_parser: DocumentParser, tmp_path: Path) -> None:
    f = tmp_path / "clip.mp4"
    f.write_bytes(b"\x00")
    assert document_parser.parse(str(f), f.name, ".mp4").state == DocState.UNSUPPORTED


@pytest.mark.parametrize(("parser", "ext"), FORMAT_PARSERS, ids=lambda x: str(x))
def test_format_parser_claims_its_extension_only(parser: FormatParser, ext: str) -> None:
    assert parser.can_parse(ext)
    assert not parser.can_parse(".mp4")

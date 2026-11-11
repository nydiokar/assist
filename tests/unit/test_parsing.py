"""Parsing coverage/fidelity (v1 §6.3)."""

from __future__ import annotations

from project_rag.config import ParsingConfig
from project_rag.parsers import ParserRegistry
from project_rag.schema import BlockType, DocState


def _reg() -> ParserRegistry:
    return ParserRegistry(ParsingConfig())


def test_markdown_headings_and_body(corpus_root):
    doc = _reg().parse(f"{corpus_root}/proposal.md", "proposal.md", ".md")
    assert doc.state == DocState.PARSED
    headings = [b.text for b in doc.blocks if b.block_type == BlockType.HEADING]
    assert "Project indicators" in headings
    bodies = [b for b in doc.blocks if b.block_type == BlockType.PARAGRAPH]
    assert any("Target: 120" in b.text for b in bodies)
    # body carries its nearest heading as provenance
    assert any(b.location.heading == "Project indicators" for b in bodies)


def test_pdf_scanned_detected_as_ocr_required(corpus_root):
    doc = _reg().parse(f"{corpus_root}/scanned_placeholder.pdf", "scanned_placeholder.pdf", ".pdf")
    assert doc.state == DocState.OCR_REQUIRED


def test_xlsx_multisheet_numeric_and_none_cache(corpus_root):
    doc = _reg().parse(f"{corpus_root}/indicators.xlsx", "indicators.xlsx", ".xlsx")
    assert doc.state == DocState.PARSED
    rows = [b.text for b in doc.blocks if b.block_type == BlockType.TABLE_ROW]
    assert any("target=120" in r for r in rows)
    # None cached formula result must NOT become 0
    garden_row = next(r for r in rows if "gardens" in r)
    assert "reported_actual=" in garden_row
    assert "reported_actual=0" not in garden_row
    assert any(b.location.sheet == "Indicators" for b in doc.blocks)


def test_csv_bulgarian_and_rows(corpus_root):
    doc = _reg().parse(f"{corpus_root}/registrations.csv", "registrations.csv", ".csv")
    assert doc.state == DocState.PARSED
    rows = [b for b in doc.blocks if b.block_type == BlockType.TABLE_ROW]
    assert len(rows) == 6  # 5 unique + 1 intentional duplicate
    assert all(b.location.row is not None for b in rows)


def test_unsupported_extension_reported(tmp_path):
    f = tmp_path / "thing.xyz"
    f.write_text("data")
    doc = _reg().parse(str(f), "thing.xyz", ".xyz")
    assert doc.state == DocState.UNSUPPORTED

"""DOCX table fidelity found on the real corpus (WP0 parsing gate)."""

from __future__ import annotations

from pathlib import Path

import docx

from project_rag.config import ParsingConfig
from project_rag.parsers import ParserRegistry
from project_rag.schema import BlockType


def _parse(path: Path):
    return ParserRegistry(ParsingConfig()).parse(str(path), path.name, ".docx")


def test_merged_cells_emitted_once_and_order_preserved(tmp_path: Path) -> None:
    d = docx.Document()
    d.add_paragraph("Before table")
    t = d.add_table(rows=3, cols=3)
    t.cell(0, 0).merge(t.cell(0, 2)).text = "Activity programme"
    t.cell(1, 0).text, t.cell(1, 1).text = "Dates", "04-07.07.2025"
    t.cell(2, 0).text, t.cell(2, 1).text = "Venue", "Ямбол"
    d.add_paragraph("After table")
    p = tmp_path / "a.docx"
    d.save(p)

    blocks = _parse(p).blocks
    texts = [b.text for b in blocks]
    assert texts[0] == "Before table" and texts[-1] == "After table"
    rows = [b for b in blocks if b.block_type == BlockType.TABLE_ROW]
    assert rows[0].text == "Activity programme"          # merged title once, not x3
    assert any("Dates | 04-07.07.2025" in r.text for r in rows)  # first data row not eaten as header


def test_real_header_row_is_used_as_header(tmp_path: Path) -> None:
    d = docx.Document()
    t = d.add_table(rows=3, cols=2)
    for c, v in zip(t.rows[0].cells, ["Indicator", "Target"], strict=True):
        c.text = v
    t.rows[1].cells[0].text, t.rows[1].cells[1].text = "People trained", "120"
    t.rows[2].cells[0].text, t.rows[2].cells[1].text = "Sessions", "8"
    p = tmp_path / "b.docx"
    d.save(p)
    rows = [b.text for b in _parse(p).blocks if b.block_type == BlockType.TABLE_ROW]
    assert rows == ["Indicator: People trained, Target: 120", "Indicator: Sessions, Target: 8"]

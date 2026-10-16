"""XLSX header detection found on the real corpus (sheets with title rows / loose notes)."""

from __future__ import annotations

from pathlib import Path

from openpyxl import Workbook

from project_rag.config import ParsingConfig
from project_rag.parsers import ParserRegistry


def _rows(path: Path) -> list[str]:
    doc = ParserRegistry(ParsingConfig()).parse(str(path), path.name, ".xlsx")
    return [b.text for b in doc.blocks]


def test_title_rows_above_header_are_not_the_header(tmp_path: Path) -> None:
    wb = Workbook()
    ws = wb.active
    ws.title = "Резултати"
    ws.append(["Резултати от проекта"])
    ws.append([])
    ws.append(["Какво", "брой заложени", "Брой постигнати"])
    ws.append(["Обучени млади", 120, 74])
    p = tmp_path / "t.xlsx"
    wb.save(p)
    rows = _rows(p)
    assert rows[0] == "[Резултати] Резултати от проекта"
    assert rows[1] == "[Резултати] Какво=Обучени млади, брой заложени=120, Брой постигнати=74"


def test_sheet_without_table_keeps_every_cell(tmp_path: Path) -> None:
    wb = Workbook()
    ws = wb.active
    ws.append(["Идеи"])
    ws.append(["Пост за дебата"])
    p = tmp_path / "n.xlsx"
    wb.save(p)
    assert _rows(p) == ["[Sheet] Идеи", "[Sheet] Пост за дебата"]

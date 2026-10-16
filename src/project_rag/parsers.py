"""Format-specific parser adapters (v1 §4.2, v2 §D2).

ParserProtocol is the stable seam. Native adapters (pypdf/python-docx/openpyxl/csv/text)
are the default backend; Docling is an optional extra behind the same protocol.

Each parser returns a ParsedDocument with logical blocks carrying provenance
(page/sheet/row/cell/heading). Failures are captured as diagnostics and a FAILED/
UNSUPPORTED/OCR_REQUIRED state rather than raising past the orchestrator.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Protocol

from .config import ParsingConfig
from .schema import Block, BlockType, DocState, ParsedDocument, SourceLocation


class ParserProtocol(Protocol):
    def can_parse(self, ext: str) -> bool: ...

    def parse(self, abs_path: str, source_path: str) -> ParsedDocument: ...


# ---- Native adapters ----


class TextParser:
    def can_parse(self, ext: str) -> bool:
        return ext in {".txt", ".md"}

    def parse(self, abs_path: str, source_path: str) -> ParsedDocument:
        text = Path(abs_path).read_text(encoding="utf-8", errors="replace")
        is_md = source_path.endswith(".md")
        blocks: list[Block] = []
        heading: str | None = None
        if is_md:
            # Line-aware: markdown headings are standalone lines starting with '#'.
            para_buf: list[str] = []
            idx = 0

            def flush() -> None:
                nonlocal para_buf, idx
                if para_buf:
                    blocks.append(Block(block_type=BlockType.PARAGRAPH, text=" ".join(para_buf),
                                        location=SourceLocation(heading=heading,
                                                                block_path=f"p{idx}")))
                    idx += 1
                    para_buf = []

            for line in text.splitlines():
                stripped = line.strip()
                if not stripped:
                    flush()
                    continue
                if stripped.startswith("#"):
                    flush()
                    heading = stripped.lstrip("#").strip()
                    blocks.append(Block(block_type=BlockType.HEADING, text=heading,
                                        location=SourceLocation(block_path=f"h{idx}")))
                    idx += 1
                else:
                    para_buf.append(stripped)
            flush()
        else:
            for i, para in enumerate(_split_paragraphs(text)):
                blocks.append(Block(block_type=BlockType.PARAGRAPH, text=para.strip(),
                                    location=SourceLocation(heading=heading, block_path=f"p{i}")))
        return ParsedDocument(source_path=source_path, state=DocState.PARSED, blocks=blocks)


class PdfParser:
    def __init__(self, detect_scanned: bool = True) -> None:
        self._detect_scanned = detect_scanned

    def can_parse(self, ext: str) -> bool:
        return ext == ".pdf"

    def parse(self, abs_path: str, source_path: str) -> ParsedDocument:
        try:
            from pypdf import PdfReader  # noqa: PLC0415
        except ImportError as e:  # pragma: no cover
            return ParsedDocument(source_path=source_path, state=DocState.FAILED,
                                  diagnostics=[f"pypdf missing: {e}"])
        try:
            reader = PdfReader(abs_path)
        except Exception as e:  # noqa: BLE001
            return ParsedDocument(source_path=source_path, state=DocState.FAILED,
                                  diagnostics=[f"pdf open failed: {e}"])
        if reader.is_encrypted:
            return ParsedDocument(source_path=source_path, state=DocState.QUARANTINED,
                                  diagnostics=["encrypted PDF"])
        blocks: list[Block] = []
        total_chars = 0
        for pageno, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception as e:  # noqa: BLE001
                blocks.append(Block(block_type=BlockType.PARAGRAPH, text="",
                                    location=SourceLocation(page=pageno)))
                _ = e
                continue
            total_chars += len(text.strip())
            for i, para in enumerate(_split_paragraphs(text, strict_blank=True)):
                if para.strip():
                    blocks.append(Block(block_type=BlockType.PARAGRAPH, text=para.strip(),
                                        location=SourceLocation(page=pageno, block_path=f"p{pageno}.{i}")))
        if self._detect_scanned and total_chars < 20 and len(reader.pages) > 0:
            return ParsedDocument(source_path=source_path, state=DocState.OCR_REQUIRED,
                                  diagnostics=["likely scanned PDF: negligible extractable text"])
        return ParsedDocument(source_path=source_path, state=DocState.PARSED, blocks=blocks)


class DocxParser:
    def can_parse(self, ext: str) -> bool:
        return ext == ".docx"

    def parse(self, abs_path: str, source_path: str) -> ParsedDocument:
        try:
            import docx  # noqa: PLC0415
        except ImportError as e:  # pragma: no cover
            return ParsedDocument(source_path=source_path, state=DocState.FAILED,
                                  diagnostics=[f"python-docx missing: {e}"])
        try:
            doc = docx.Document(abs_path)
        except Exception as e:  # noqa: BLE001
            return ParsedDocument(source_path=source_path, state=DocState.FAILED,
                                  diagnostics=[f"docx open failed: {e}"])
        from docx.table import Table  # noqa: PLC0415
        from docx.text.paragraph import Paragraph  # noqa: PLC0415

        blocks: list[Block] = []
        heading: str | None = None
        n_para: int = 0
        n_table: int = 0
        # Walk the body in document order so tables stay next to their surrounding text.
        for child in doc.element.body.iterchildren():
            tag: str = child.tag.rsplit("}", 1)[-1]
            if tag == "p":
                para = Paragraph(child, doc)
                idx: int = n_para
                n_para += 1
                txt = para.text.strip()
                if not txt:
                    continue
                style = (para.style.name or "").lower() if para.style else ""
                if style.startswith("heading") or style == "title":
                    heading = txt
                    blocks.append(Block(block_type=BlockType.HEADING, text=txt,
                                        location=SourceLocation(block_path=f"para{idx}")))
                else:
                    blocks.append(Block(block_type=BlockType.PARAGRAPH, text=txt,
                                        location=SourceLocation(heading=heading,
                                                                block_path=f"para{idx}")))
            elif tag == "tbl":
                blocks.extend(_docx_table_blocks(Table(child, doc), n_table, heading))
                n_table += 1
        return ParsedDocument(source_path=source_path, state=DocState.PARSED, blocks=blocks)


def _docx_table_blocks(table: object, ti: int, heading: str | None) -> list[Block]:
    """Render a DOCX table as row blocks.

    Merged cells are repeated by python-docx for every grid column they span; dedupe by the
    underlying XML cell so a merged title is emitted once. The first row is used as a header
    only when it looks like one (>=3 rows, as wide as the widest row, all cells distinct,
    non-empty and short);
    otherwise rows are rendered as plain ' | '-joined cells so form-style/layout tables
    keep their first data row.
    """
    rows: list[list[str]] = []
    for row in table.rows:  # type: ignore[attr-defined]
        seen: set[int] = set()
        cells: list[str] = []
        for c in row.cells:
            if id(c._tc) in seen:
                continue
            seen.add(id(c._tc))
            cells.append(" ".join(c.text.split()))
        rows.append(cells)
    if not rows:
        return []
    head: list[str] = rows[0]
    has_header: bool = (
        len(rows) >= 3
        and len(head) >= 2
        and len(head) == max(len(r) for r in rows)
        and all(head)
        and len(set(head)) == len(head)
        and max(len(h) for h in head) <= 60
    )
    out: list[Block] = []
    start: int = 1 if has_header else 0
    for ri in range(start, len(rows)):
        cells = rows[ri]
        if not any(cells):
            continue
        if has_header:
            text = ", ".join(f"{h}: {v}" for h, v in zip(head, cells, strict=False) if v)
        else:
            text = " | ".join(v for v in cells if v)
        out.append(Block(block_type=BlockType.TABLE_ROW, text=text,
                         location=SourceLocation(heading=heading,
                                                 block_path=f"table{ti}.row{ri + 1}", row=ri + 1)))
    return out


class XlsxParser:
    def can_parse(self, ext: str) -> bool:
        return ext == ".xlsx"

    def parse(self, abs_path: str, source_path: str) -> ParsedDocument:
        try:
            from openpyxl import load_workbook  # noqa: PLC0415
            from openpyxl.utils import get_column_letter  # noqa: PLC0415
        except ImportError as e:  # pragma: no cover
            return ParsedDocument(source_path=source_path, state=DocState.FAILED,
                                  diagnostics=[f"openpyxl missing: {e}"])
        try:
            # data_only=False keeps formulas visible; cached values read separately.
            wb = load_workbook(abs_path, read_only=True, data_only=True)
        except Exception as e:  # noqa: BLE001
            return ParsedDocument(source_path=source_path, state=DocState.FAILED,
                                  diagnostics=[f"xlsx open failed: {e}"])
        blocks: list[Block] = []
        diagnostics: list[str] = []
        for ws in wb.worksheets:
            rows = [r for r in ws.iter_rows(values_only=False)]
            hdr_idx: int | None = _xlsx_header_index([[c.value for c in r] for r in rows])
            if hdr_idx is None:
                hdr_idx = -1  # no table-like header: every non-empty row is a plain row
            header: list[str] = (
                [str(c.value).strip() if c.value is not None else "" for c in rows[hdr_idx]]
                if hdr_idx >= 0 else [])
            for r_idx, row in enumerate(rows, start=1):
                values = [c.value for c in row]
                if all(v is None or str(v).strip() == "" for v in values):
                    continue
                if r_idx - 1 == hdr_idx:
                    continue
                # None cached result must NOT be treated as 0 (v1 §4.2 XLSX rule).
                pairs: list[str] = []
                tabular: bool = hdr_idx >= 0 and r_idx - 1 > hdr_idx
                for col_idx, v in enumerate(values, start=1):
                    shown = "" if v is None else str(v).strip()
                    h: str = header[col_idx - 1] if tabular and col_idx <= len(header) else ""
                    if tabular and h:
                        pairs.append(f"{h}={shown}")
                    elif shown:  # title/loose cell, or a column without a header
                        pairs.append(f"{get_column_letter(col_idx)}{r_idx}={shown}" if tabular
                                     else shown)
                text = f"[{ws.title}] " + ", ".join(pairs)
                blocks.append(Block(block_type=BlockType.TABLE_ROW, text=text,
                                    location=SourceLocation(sheet=ws.title, row=r_idx,
                                                            block_path=f"{ws.title}.row{r_idx}")))
        wb.close()
        return ParsedDocument(source_path=source_path, state=DocState.PARSED,
                              blocks=blocks, diagnostics=diagnostics)


def _xlsx_header_index(grid: list[list[object]]) -> int | None:
    """First row within the top 30 that looks like a column header: >=2 non-empty string cells,
    covering at least half of the widest row seen, with distinct values. Sheets that are
    titles/loose notes (e.g. one-cell rows) get None."""
    top = grid[:30]
    widths: list[int] = [sum(1 for v in r if v is not None and str(v).strip()) for r in top]
    if not widths or max(widths) < 2:
        return None
    for i, r in enumerate(top):
        cells = [str(v).strip() for v in r if v is not None and str(v).strip()]
        strings = [c for c in cells if not _looks_numeric(c)]
        if (len(cells) >= 2 and len(strings) == len(cells) and len(set(cells)) == len(cells)
                and len(cells) * 2 >= max(widths)):
            return i
    return None


def _looks_numeric(s: str) -> bool:
    try:
        float(s.replace(",", "."))
    except ValueError:
        return False
    return True


class CsvParser:
    def can_parse(self, ext: str) -> bool:
        return ext == ".csv"

    def parse(self, abs_path: str, source_path: str) -> ParsedDocument:
        raw = Path(abs_path).read_bytes()
        encoding = "utf-8"
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            encoding = "cp1251"  # common for Bulgarian legacy files
            text = raw.decode(encoding, errors="replace")
        try:
            dialect = csv.Sniffer().sniff(text[:4096])
        except csv.Error:
            dialect = csv.excel
        reader = csv.reader(io.StringIO(text), dialect)
        rows = list(reader)
        if not rows:
            return ParsedDocument(source_path=source_path, state=DocState.PARSED,
                                  diagnostics=["empty csv"])
        header = [h.strip() for h in rows[0]]
        blocks: list[Block] = []
        for r_idx, row in enumerate(rows[1:], start=2):
            if not any(cell.strip() for cell in row):
                continue
            pairs = ", ".join(f"{h}={v.strip()}" for h, v in zip(header, row, strict=False))
            blocks.append(Block(block_type=BlockType.TABLE_ROW, text=pairs,
                                location=SourceLocation(row=r_idx, block_path=f"row{r_idx}")))
        return ParsedDocument(source_path=source_path, state=DocState.PARSED, blocks=blocks,
                              diagnostics=[f"encoding={encoding}"])


def _split_paragraphs(text: str, strict_blank: bool = False) -> list[str]:
    """Split on blank lines. `strict_blank` (PDF): only truly empty lines break paragraphs;
    pypdf emits whitespace-only lines between justified words ("word \n \n word"), which
    would otherwise shatter text into one block per word (seen on the real corpus)."""
    parts: list[str] = []
    buf: list[str] = []
    for line in text.splitlines():
        is_blank: bool = (line == "") if strict_blank else not line.strip()
        if not is_blank:
            if not line.strip():
                continue
            buf.append(line.strip())
        elif buf:
            parts.append(" ".join(buf))
            buf = []
    if buf:
        parts.append(" ".join(buf))
    return parts


class ParserRegistry:
    """Dispatches by extension. Default native backend; Docling optional (v2 §D2)."""

    def __init__(self, cfg: ParsingConfig) -> None:
        self._cfg = cfg
        self._parsers: list[ParserProtocol] = [
            TextParser(),
            PdfParser(detect_scanned=cfg.detect_scanned_pdf),
            DocxParser(),
            XlsxParser(),
            CsvParser(),
        ]

    def parse(self, abs_path: str, source_path: str, ext: str) -> ParsedDocument:
        if self._cfg.backend == "docling":
            doc = self._try_docling(abs_path, source_path, ext)
            if doc is not None:
                return doc
        for parser in self._parsers:
            if parser.can_parse(ext):
                return parser.parse(abs_path, source_path)
        return ParsedDocument(source_path=source_path, state=DocState.UNSUPPORTED,
                              diagnostics=[f"no parser for extension {ext}"])

    def _try_docling(self, abs_path: str, source_path: str, ext: str) -> ParsedDocument | None:
        try:
            from .parsers_docling import DoclingParser  # noqa: PLC0415
        except ImportError:
            return None
        p = DoclingParser()
        if not p.can_parse(ext):
            return None
        return p.parse(abs_path, source_path)

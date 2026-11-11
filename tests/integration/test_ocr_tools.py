"""OCR and antiword adapters against the real binaries (S4). Skips where a binary is absent; CI
installs tesseract (bul+eng), ocrmypdf and antiword so these run there. Synthetic fixtures only."""

from __future__ import annotations

import os
import shutil
import stat
import sys
from pathlib import Path

import pytest

from project_rag.adapters.ocr_tesseract import TesseractOcr, parse_tesseract_tsv
from project_rag.adapters.parse_antiword import AntiwordParser
from project_rag.adapters.parse_registry import ParserRegistry
from project_rag.domain.config import OcrConfig, ParsingConfig
from project_rag.domain.errors import OcrFailed
from project_rag.domain.models import DocState
from project_rag.services.parsing import OcrAwareParser

OCR = Path(__file__).resolve().parents[1] / "fixtures" / "synthetic" / "ocr"
needs_tesseract = pytest.mark.skipif(shutil.which("tesseract") is None, reason="no tesseract")
needs_ocrmypdf = pytest.mark.skipif(shutil.which("ocrmypdf") is None, reason="no ocrmypdf")


def _parser() -> OcrAwareParser:
    cfg = ParsingConfig(ocr=OcrConfig(enabled=True, min_chars=40, min_mean_conf=60.0))
    return OcrAwareParser(ParserRegistry(cfg), TesseractOcr(languages="bul+eng"), cfg)


def test_tsv_parsing_groups_lines_and_paragraphs() -> None:
    head = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext"
    rows = [
        "5\t1\t1\t1\t1\t1\t0\t0\t1\t1\t90\tНационална",
        "5\t1\t1\t1\t1\t2\t0\t0\t1\t1\t80\tсреща",
        "5\t1\t1\t1\t2\t1\t0\t0\t1\t1\t70\t2025",
        "5\t1\t2\t1\t1\t1\t0\t0\t1\t1\t60\tRegistration",
        "4\t1\t2\t1\t1\t0\t0\t0\t1\t1\t-1\t",
        "5\t1\t2\t1\t1\t2\t0\t0\t1\t1\t-1\t",
    ]
    out = parse_tesseract_tsv("\n".join([head, *rows]))
    assert out.text == "Национална среща 2025\n\nRegistration"
    assert out.mean_conf == 75.0


@needs_tesseract
def test_image_with_text_is_indexed_and_photo_is_no_text() -> None:
    p = _parser()
    poster = p.parse(str(OCR / "poster_bg_en.png"), "poster_bg_en.png", ".png")
    assert poster.state == DocState.PARSED
    text = " ".join(b.text for b in poster.blocks)
    assert "Национална среща" in text and "attendance sheet" in text
    assert poster.diagnostics[0].startswith("ocr_image chars=")
    photo = p.parse(str(OCR / "photo_no_text.png"), "photo_no_text.png", ".png")
    assert photo.state == DocState.NO_TEXT


@needs_tesseract
def test_unreadable_image_fails_with_reason(tmp_path: Path) -> None:
    bad = tmp_path / "broken.png"
    bad.write_bytes(b"\x89PNG\r\n\x1a\n not really an image")
    doc = _parser().parse(str(bad), "broken.png", ".png")
    assert doc.state == DocState.FAILED and doc.diagnostics[0].startswith("ocr failed")


@needs_tesseract
@needs_ocrmypdf
def test_scanned_pdf_gets_page_addressed_ocr_text() -> None:
    doc = _parser().parse(str(OCR / "scanned_bg_en.pdf"), "scanned_bg_en.pdf", ".pdf")
    assert doc.state == DocState.PARSED
    assert any("ocr_pdf pages=1" in d for d in doc.diagnostics)
    assert all(b.location.page == 1 for b in doc.blocks)
    assert "Регистрация" in " ".join(b.text for b in doc.blocks)


def test_missing_binary_is_ocr_failed() -> None:
    with pytest.raises(OcrFailed, match="not installed"):
        TesseractOcr()._run(["definitely-not-a-binary-xyz", "x"])


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell stub")
def test_antiword_adapter_states(tmp_path: Path) -> None:
    ok = tmp_path / "antiword_ok"
    ok.write_text("#!/bin/sh\nprintf 'First paragraph\\nwraps here\\n\\nSecond one\\n'\n")
    fail = tmp_path / "antiword_fail"
    fail.write_text("#!/bin/sh\necho 'not a Word document' >&2\nexit 1\n")
    empty = tmp_path / "antiword_empty"
    empty.write_text("#!/bin/sh\nexit 0\n")
    for f in (ok, fail, empty):
        f.chmod(f.stat().st_mode | stat.S_IEXEC)
    doc = AntiwordParser(binary=str(ok)).parse("x.doc", "x.doc")
    assert doc.state == DocState.PARSED
    assert [b.text for b in doc.blocks] == ["First paragraph wraps here", "Second one"]
    assert "parser=antiword" in doc.diagnostics
    bad = AntiwordParser(binary=str(fail)).parse("x.doc", "x.doc")
    assert bad.state == DocState.FAILED and "exit 1" in bad.diagnostics[0]
    assert AntiwordParser(binary=str(empty)).parse("x.doc", "x.doc").state == DocState.NO_TEXT
    assert AntiwordParser(binary="no-such-antiword").parse("x.doc", "x.doc").diagnostics == [
        "antiword not installed (no-such-antiword)"
    ]
    assert os.environ.get("PATH") is not None

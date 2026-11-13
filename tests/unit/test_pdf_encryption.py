"""Encrypted PDFs (S5): permissions-only encryption is parsed, a real user password quarantines."""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader, PdfWriter

from project_rag.adapters.parse_native import PdfParser
from project_rag.domain.models import DocState

# The digital-text page of the synthetic OCR fixture would be OCR input; any page with a text
# layer works. Build one from the fixture corpus's placeholder by re-encrypting it.
_SRC = Path(__file__).resolve().parents[1] / "fixtures" / "synthetic" / "ngo_project_demo"


def _encrypted(tmp_path: Path, user_password: str) -> Path:
    writer = PdfWriter(clone_from=PdfReader(_SRC / "scanned_placeholder.pdf"))
    writer.encrypt(user_password=user_password, owner_password="owner-secret", algorithm="AES-128")
    out = tmp_path / f"enc_{bool(user_password)}.pdf"
    with out.open("wb") as fh:
        writer.write(fh)
    return out


def test_permissions_only_encryption_is_opened(tmp_path: Path) -> None:
    p = _encrypted(tmp_path, user_password="")
    assert PdfReader(p).is_encrypted
    doc = PdfParser(detect_scanned=False).parse(str(p), "enc.pdf")
    assert doc.state == DocState.PARSED
    assert "encrypted PDF opened with the empty user password" in doc.diagnostics


def test_user_password_quarantines(tmp_path: Path) -> None:
    p = _encrypted(tmp_path, user_password="needs-this")
    doc = PdfParser(detect_scanned=False).parse(str(p), "enc.pdf")
    assert doc.state == DocState.QUARANTINED
    assert doc.diagnostics == ["encrypted PDF (user password required)"]

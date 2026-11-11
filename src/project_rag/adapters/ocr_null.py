"""Null OCR engine: OCR not configured. Every call raises OcrFailed (callers record `no_text`)."""

from __future__ import annotations

from ..domain.errors import OcrFailed
from ..domain.models import OcrText


class NullOcr:
    def ocr_pdf(self, path: str) -> list[str]:
        raise OcrFailed(f"OCR not configured: {path}")

    def ocr_image(self, path: str) -> OcrText:
        raise OcrFailed(f"OCR not configured: {path}")

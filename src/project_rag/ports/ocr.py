"""OCR port (tesseract/ocrmypdf adapter arrives in S4; `null` until then)."""

from __future__ import annotations

from typing import Protocol

from ..domain.models import OcrText


class OcrEngine(Protocol):
    def ocr_pdf(self, path: str) -> str:
        """OCR a scanned PDF; return the path of a text-layered copy. Raises OcrFailed."""
        ...

    def ocr_image(self, path: str) -> OcrText:
        """OCR one image. Raises OcrFailed."""
        ...

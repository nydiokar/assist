"""OCR port: `tesseract` adapter (ocrmypdf + tesseract) and `null`."""

from __future__ import annotations

from typing import Protocol

from ..domain.models import OcrText


class OcrEngine(Protocol):
    def ocr_pdf(self, path: str) -> list[str]:
        """OCR a scanned PDF; return the recognised text of each page, in page order.
        Raises OcrFailed. (Spec §15.3 sketches a path to a text-layered copy; page texts keep
        temp files inside the adapter and give the parser page provenance directly.)"""
        ...

    def ocr_image(self, path: str) -> OcrText:
        """OCR one image. Raises OcrFailed."""
        ...

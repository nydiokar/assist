"""OCR engine over the tesseract and ocrmypdf binaries (spec §6.5, A11). Languages `bul+eng`.

- `ocr_image`: `tesseract <img> stdout -l <langs> tsv`; words with conf >= 0 make the text
  (lines in reading order, blank line between paragraphs) and their mean confidence.
- `ocr_pdf`: `ocrmypdf --force-ocr --sidecar` in a temporary directory; the sidecar's form feeds
  split the pages. Temp files never leave the adapter.
Any failure (missing binary, non-zero exit, timeout, unreadable output) raises `OcrFailed`.
"""

from __future__ import annotations

import csv
import io
import shutil
import subprocess
import tempfile
from pathlib import Path

from ..domain.errors import OcrFailed
from ..domain.models import OcrText


class TesseractOcr:
    def __init__(self, *, languages: str = "bul+eng", timeout_s: float = 300.0, jobs: int = 2):
        self._languages = languages
        self._timeout_s = timeout_s
        self._jobs = jobs

    def _run(self, args: list[str]) -> subprocess.CompletedProcess[bytes]:
        exe = shutil.which(args[0])
        if exe is None:
            raise OcrFailed(f"{args[0]} not installed")
        try:
            proc = subprocess.run(
                [exe, *args[1:]], capture_output=True, timeout=self._timeout_s, check=False
            )
        except subprocess.TimeoutExpired as e:
            raise OcrFailed(f"{args[0]} timed out after {self._timeout_s:.0f}s") from e
        if proc.returncode != 0:
            err = proc.stderr.decode("utf-8", errors="replace").strip()[-300:]
            raise OcrFailed(f"{args[0]} exit {proc.returncode}: {err}")
        return proc

    def ocr_image(self, path: str) -> OcrText:
        proc = self._run(["tesseract", path, "stdout", "-l", self._languages, "tsv"])
        return parse_tesseract_tsv(proc.stdout.decode("utf-8", errors="replace"))

    def ocr_pdf(self, path: str) -> list[str]:
        with tempfile.TemporaryDirectory(prefix="assist_ocr_") as tmp:
            out_pdf = Path(tmp) / "out.pdf"
            sidecar = Path(tmp) / "out.txt"
            self._run(
                [
                    "ocrmypdf",
                    "--force-ocr",
                    "--language",
                    self._languages,
                    "--sidecar",
                    str(sidecar),
                    "--jobs",
                    str(self._jobs),
                    "--quiet",
                    path,
                    str(out_pdf),
                ]
            )
            if not sidecar.is_file():
                raise OcrFailed("ocrmypdf produced no sidecar text")
            text = sidecar.read_text(encoding="utf-8", errors="replace")
        return [page.strip() for page in text.split("\f")]


def parse_tesseract_tsv(tsv: str) -> OcrText:
    """Text + mean word confidence from tesseract TSV output (level 5 rows are words)."""
    lines: dict[tuple[int, int, int, int], list[str]] = {}
    confs: list[float] = []
    reader = csv.DictReader(io.StringIO(tsv), delimiter="\t", quoting=csv.QUOTE_NONE)
    for row in reader:
        try:
            level = int(row.get("level") or 0)
            conf = float(row.get("conf") or -1)
        except ValueError:
            continue
        word = (row.get("text") or "").strip()
        if level != 5 or conf < 0 or not word:
            continue
        key = (
            int(row["page_num"]),
            int(row["block_num"]),
            int(row["par_num"]),
            int(row["line_num"]),
        )
        lines.setdefault(key, []).append(word)
        confs.append(conf)
    paragraphs: dict[tuple[int, int, int], list[str]] = {}
    for (page, block, par, _line), words in sorted(lines.items()):
        paragraphs.setdefault((page, block, par), []).append(" ".join(words))
    text = "\n\n".join(" ".join(ls) for _, ls in sorted(paragraphs.items()))
    mean = sum(confs) / len(confs) if confs else 0.0
    return OcrText(text=text, mean_conf=round(mean, 2))

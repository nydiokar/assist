"""OCR survey for threshold tuning (S4, spec §6.5: tune N chars / M mean confidence on YP).

Runs tesseract on every image and ocrmypdf on every PDF the native parser marks OCR_REQUIRED,
under a project root, and writes one JSON row per file: chars, mean_conf, words, and a short
excerpt. The output quotes corpus text, so it goes to the gitignored state dir only. Run inside
the assist image (tools + bul/eng traineddata):

  docker run --rm -v "<yspdb>:/corpus/yp:ro" -v "<local_data/v3_yp>:/state" --entrypoint python \
      assist:dev scripts/ocr_survey.py --root /corpus/yp --out /state/ocr_survey.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from project_rag.adapters.corpus_local_fs import discover
from project_rag.adapters.ocr_tesseract import TesseractOcr
from project_rag.adapters.parse_registry import ParserRegistry
from project_rag.domain.config import ParsingConfig
from project_rag.domain.errors import OcrFailed
from project_rag.domain.models import DocState

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--languages", default="bul+eng")
    args = ap.parse_args()

    ocr = TesseractOcr(languages=args.languages, timeout_s=600)
    registry = ParserRegistry(ParsingConfig())
    rows: list[dict[str, object]] = []
    files = discover(args.root)
    t0 = time.monotonic()
    for f in files:
        row: dict[str, object] = {"path": f.source_path, "ext": f.ext, "size": f.size}
        start = time.monotonic()
        try:
            if f.ext in IMAGE_EXTS:
                res = ocr.ocr_image(f.abs_path)
                row.update(
                    kind="image",
                    chars=len(res.text.strip()),
                    words=len(res.text.split()),
                    mean_conf=res.mean_conf,
                    excerpt=" ".join(res.text.split())[:240],
                )
            elif f.ext == ".pdf":
                doc = registry.parse(f.abs_path, f.source_path, f.ext)
                if doc.state != DocState.OCR_REQUIRED:
                    continue
                pages = ocr.ocr_pdf(f.abs_path)
                text = "\n".join(pages)
                row.update(
                    kind="scanned_pdf",
                    pages=len(pages),
                    chars=len(text.strip()),
                    words=len(text.split()),
                    excerpt=" ".join(text.split())[:240],
                )
            else:
                continue
        except OcrFailed as e:
            row.update(kind="error", error=str(e))
        row["seconds"] = round(time.monotonic() - start, 2)
        rows.append(row)
        print(
            f"{len(rows):4d} {row.get('kind')} chars={row.get('chars')} conf={row.get('mean_conf')} {row['seconds']}s"
        )
    Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"done: {len(rows)} files in {time.monotonic() - t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()

"""WP0: read-only corpus inventory + parsing-gate report (spec v1 §9 WP0, §4.2).

Writes inventory.json and parse_report.json under --out (default local_data/wp0/).
Output contains real file names, so it lives in gitignored local_data/ only.

    .\\.venv\\Scripts\\python.exe -X utf8 scripts\\wp0_inventory.py --root yspdb
"""

from __future__ import annotations

import argparse
import collections
import json
import time
from pathlib import Path

from pydantic import BaseModel, Field

from project_rag.adapters.corpus_local_fs import discover
from project_rag.adapters.parse_registry import ParserRegistry
from project_rag.domain.config import ParsingConfig
from project_rag.domain.models import DiscoveredFile

MEDIA: frozenset[str] = frozenset({".mp4", ".mov", ".m4a", ".mp3", ".wav", ".avi"})
IMAGES: frozenset[str] = frozenset({".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tif"})
LEGACY: frozenset[str] = frozenset({".doc", ".xls", ".ppt", ".pptx"})


class FileRow(BaseModel):
    source_path: str
    ext: str
    size: int
    content_hash: str
    scope: str  # text | legacy_office | image | media | other
    duplicate_of: str | None = None
    state: str | None = None
    n_blocks: int = 0
    n_chars: int = 0
    n_empty_blocks: int = 0
    parse_seconds: float = 0.0
    diagnostics: list[str] = Field(default_factory=list)


def classify(d: DiscoveredFile) -> str:
    if d.supported:
        return "text"
    if d.ext in LEGACY:
        return "legacy_office"
    if d.ext in IMAGES:
        return "image"
    if d.ext in MEDIA:
        return "media"
    return "other"


def build_rows(root: Path, parse: bool) -> list[FileRow]:
    reg: ParserRegistry = ParserRegistry(ParsingConfig())
    first_seen: dict[str, str] = {}
    rows: list[FileRow] = []
    for d in discover(root):
        row = FileRow(
            source_path=d.source_path,
            ext=d.ext,
            size=d.size,
            content_hash=d.content_hash,
            scope=classify(d),
        )
        if d.content_hash in first_seen:
            row.duplicate_of = first_seen[d.content_hash]
        else:
            first_seen[d.content_hash] = d.source_path
        if parse and row.scope == "text":
            t0: float = time.perf_counter()
            try:
                doc = reg.parse(d.abs_path, d.source_path, d.ext)
                row.state = doc.state.value
                row.n_blocks = len(doc.blocks)
                row.n_chars = sum(len(b.text) for b in doc.blocks)
                row.n_empty_blocks = sum(1 for b in doc.blocks if not b.text.strip())
                row.diagnostics = doc.diagnostics[:5]
            except Exception as e:  # noqa: BLE001 - a parser crash is a gate finding, not a stop
                row.state = "CRASH"
                row.diagnostics = [f"{type(e).__name__}: {e}"[:300]]
            row.parse_seconds = round(time.perf_counter() - t0, 3)
        rows.append(row)
    return rows


def summarize(rows: list[FileRow]) -> dict[str, object]:
    by_scope: dict[str, dict[str, int]] = collections.defaultdict(lambda: {"files": 0, "bytes": 0})
    for r in rows:
        by_scope[r.scope]["files"] += 1
        by_scope[r.scope]["bytes"] += r.size
    states: collections.Counter[str] = collections.Counter(
        f"{r.ext}:{r.state}" for r in rows if r.state is not None
    )
    return {
        "files": len(rows),
        "by_scope": by_scope,
        "exact_duplicates": sum(1 for r in rows if r.duplicate_of),
        "parse_states_by_ext": dict(sorted(states.items())),
        "text_files_zero_blocks": [
            r.source_path
            for r in rows
            if r.scope == "text" and r.state == "PARSED" and r.n_blocks == 0
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="local_data/wp0")
    ap.add_argument("--no-parse", action="store_true")
    args = ap.parse_args()
    out: Path = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[FileRow] = build_rows(Path(args.root), parse=not args.no_parse)
    summary: dict[str, object] = summarize(rows)
    (out / "inventory.json").write_text(
        json.dumps(
            {"summary": summary, "files": [r.model_dump() for r in rows]},
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

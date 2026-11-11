"""Dump the PARSED text of a corpus file (what the retriever indexes), for authoring eval labels
from ground truth. Reads yspdb/ read-only. No LLM, no index.

Usage: dump_parsed.py "<relative path under yspdb>" [--grep TERM] [--max CHARS]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from project_rag.config import load_config
from project_rag.discovery import _long_path
from project_rag.parsers import ParserRegistry


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--root", default="yspdb")
    ap.add_argument(
        "--grep", default=None, help="print only blocks containing this term (+context)"
    )
    ap.add_argument("--max", type=int, default=12000)
    args = ap.parse_args()

    reg = ParserRegistry(load_config(None).parsing)
    sp = args.path
    full = _long_path(str((Path(args.root) / sp).resolve()))
    doc = reg.parse(full, sp, Path(sp).suffix.lower())
    txt = "\n".join(b.text for b in doc.blocks)
    print(f"# {sp}  state={doc.state} blocks={len(doc.blocks)} chars={len(txt)}")
    if args.grep:
        term = args.grep
        for b in doc.blocks:
            if term.lower() in b.text.lower():
                print("----", b.location.model_dump(exclude_none=True))
                print(b.text[:1500])
    else:
        print(txt[: args.max])


if __name__ == "__main__":
    main()

"""Author eval labels from ground truth: read a parsed file, locate candidate answer strings,
print each with its block location so a question can be targeted at the file that proves it.

Terms are read from a UTF-8 file (--terms) or stdin, one per line — never passed on the shell
argv (avoids Cyrillic mangling through Git Bash -> WSL).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from project_rag.config import load_config
from project_rag.discovery import _long_path
from project_rag.parsers import ParserRegistry


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--root", default="yspdb")
    ap.add_argument("--terms", default=None, help="UTF-8 file of search terms, one per line")
    args = ap.parse_args()

    terms = (Path(args.terms).read_text(encoding="utf-8").splitlines()
             if args.terms else sys.stdin.read().splitlines())
    terms = [t.strip() for t in terms if t.strip()]

    reg = ParserRegistry(load_config(None).parsing)
    full = _long_path(str((Path(args.root) / args.path).resolve()))
    doc = reg.parse(full, args.path, Path(args.path).suffix.lower())
    full_txt = "\n".join(b.text for b in doc.blocks)
    print(f"# {args.path}  state={doc.state} blocks={len(doc.blocks)} chars={len(full_txt)}\n")

    for term in terms:
        present = term in full_txt
        print(f"=== TERM: {term!r}  present_in_parsed={present}")
        if not present:
            continue
        for b in doc.blocks:
            if term in b.text:
                loc = b.location.model_dump(exclude_none=True)
                snippet = b.text.strip().replace("\n", " ")
                print(f"    loc={loc}")
                print(f"    «{snippet[:400]}»")
                break
        print()


if __name__ == "__main__":
    main()

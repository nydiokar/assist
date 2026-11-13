"""S5 task 2: carry a reviewed gold set onto a changed corpus, corpus-verified (no LLM).

For every answerable question, each expected source is re-found in the new corpus root:
  1. same relative path, all expected substrings present          -> kept
  2. else any parseable file whose text contains ALL substrings    -> moved (best name match first)
  3. else                                                          -> unresolved (owner decides)
Unresolved questions and filesystem questions whose folder is gone are left out of the scored
file and listed in the change list only.
Each kept/moved source gets `expect_source_hashes[path] = sha256` (spec §6.8.4). Filesystem and
no-answer questions are carried over; filesystem folders are re-checked for existence.

Writes the new question file and a label-change list (JSON). Both quote corpus facts, so both
belong in gitignored locations (eval/yp_*.jsonl, local_data/).

  uv run python -X utf8 scripts/relabel_gold.py --root yspdb_all \
      --questions eval/yp_questions_v2.jsonl --out eval/yp_questions_v3.jsonl \
      --changes local_data/yp_all/label_changes_v3.json
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path

from pydantic import BaseModel

from project_rag.adapters.corpus_local_fs import _long_path
from project_rag.adapters.parse_registry import ParserRegistry
from project_rag.domain.config import ParsingConfig

TEXT_EXTS: frozenset[str] = frozenset({".docx", ".xlsx", ".pdf", ".txt", ".md", ".csv"})


class LabelChange(BaseModel):
    id: str
    status: str  # kept | moved | unresolved | carried | folder_missing
    old_paths: list[str]
    new_paths: list[str]
    note: str = ""


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _walk(root: Path) -> list[str]:
    base = _long_path(str(root.resolve()))
    out: list[str] = []
    for dp, _dn, fn in os.walk(base):
        for f in fn:
            out.append(os.path.relpath(os.path.join(dp, f), base).replace("\\", "/"))
    return sorted(out)


def _texts(root: Path, rels: list[str]) -> dict[str, str]:
    reg = ParserRegistry(ParsingConfig(detect_scanned_pdf=False))
    texts: dict[str, str] = {}
    for rel in rels:
        ext = Path(rel).suffix.lower()
        if ext not in TEXT_EXTS:
            continue
        doc = reg.parse(_long_path(str((root / rel).resolve())), rel, ext)
        texts[rel] = "\n".join(b.text for b in doc.blocks)
    return texts


def _name_score(old: str, new: str) -> float:
    return difflib.SequenceMatcher(None, Path(old).name.lower(), Path(new).name.lower()).ratio()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--changes", required=True)
    args = ap.parse_args()

    root = Path(args.root)
    rels = _walk(root)
    dirs = {str(Path(r).parent).replace("\\", "/") for r in rels}
    dirs |= {"/".join(r.split("/")[:i]) for r in rels for i in range(1, r.count("/") + 1)}
    texts = _texts(root, rels)

    rows = [
        json.loads(line)
        for line in Path(args.questions).read_text("utf-8").splitlines()
        if line.strip()
    ]
    out_rows: list[dict[str, object]] = []
    changes: list[LabelChange] = []
    for q in rows:
        olds: list[str] = list(q.get("expect_source_paths", []))
        subs: list[str] = list(q.get("expect_substrings", []))
        if q["kind"] == "no_answer" or not olds:
            out_rows.append(q)
            changes.append(
                LabelChange(id=q["id"], status="carried", old_paths=olds, new_paths=olds)
            )
            continue
        if q["kind"] == "filesystem":
            present = [p for p in olds if p in dirs]
            if len(present) == len(olds):  # a missing folder is not scored; listed for the owner
                out_rows.append(q)
            changes.append(
                LabelChange(
                    id=q["id"],
                    status="carried" if len(present) == len(olds) else "folder_missing",
                    old_paths=olds,
                    new_paths=present,
                )
            )
            continue
        kept = [p for p in olds if p in texts]
        uncovered = [s for s in subs if not any(s in texts[p] for p in kept)]
        new_paths: list[str] = list(kept)
        status = "kept" if len(kept) == len(olds) else "moved"
        if uncovered:
            hits = [r for r, t in texts.items() if all(s in t for s in uncovered)]
            if not hits:  # substrings split over several files: one file per substring
                per = [[r for r, t in texts.items() if s in t] for s in uncovered]
                hits = [] if any(not h for h in per) else [h[0] for h in per]
            if not hits:
                changes.append(
                    LabelChange(
                        id=q["id"],
                        status="unresolved",
                        old_paths=olds,
                        new_paths=kept,
                        note="uncovered: " + "; ".join(uncovered),
                    )
                )
                continue  # not scored; listed for the owner
            missing = [p for p in olds if p not in texts] or olds
            best = max(
                hits, key=lambda r: (max(_name_score(o, r) for o in missing), -r.count("/"), r)
            )
            new_paths.append(best)
            status = "moved"
        new_paths = list(dict.fromkeys(new_paths))
        hashes = {p: _sha256(_long_path(str((root / p).resolve()))) for p in new_paths}
        changes.append(LabelChange(id=q["id"], status=status, old_paths=olds, new_paths=new_paths))
        out_rows.append({**q, "expect_source_paths": new_paths, "expect_source_hashes": hashes})

    Path(args.out).write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out_rows), encoding="utf-8"
    )
    Path(args.changes).parent.mkdir(parents=True, exist_ok=True)
    Path(args.changes).write_text(
        json.dumps([c.model_dump() for c in changes], ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    counts: dict[str, int] = {}
    for c in changes:
        counts[c.status] = counts.get(c.status, 0) + 1
    print(json.dumps(counts))


if __name__ == "__main__":
    main()

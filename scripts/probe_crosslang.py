"""Task B diagnostic: does appending BG anchor terms to an EN query fetch the BG table row?

Pure localization before building anything. Runs yp09 (EN→BG short numeric row, the true
cross-language miss) and yp07 (already recovered) three ways and prints whether the expected
source reaches top-5 and at what rank:
  1. raw query (baseline)
  2. query + a BG gloss of its key domain terms (the proposed cheap fix)
  3. the pure BG terms alone (upper bound: what the row would need)
No LLM, no translation API — the gloss is a fixed local term list. WSL ROCm GPU.
"""

from __future__ import annotations

import argparse

from project_rag.api import ProjectRAG

PROBES = {
    # id: (raw EN query, BG gloss to append, expected source, expected substring)
    "yp09": (
        "How many decision makers were planned and attended the national meeting?",
        "Вземащи решения брой заложени постигнати национална среща",
        "Overview_Debate 2025.xlsx",
        "Вземащи решения, брой заложени=10.0",
    ),
    "yp07": (
        "How many advocacy campaigns by participants were planned and how many achieved?",
        "застъпнически кампании брой заложени постигнати",
        "Overview_Debate 2025.xlsx",
        "брой заложени=5.0, Брой постигнати=1.0",
    ),
}


def _rank(
    rag: ProjectRAG, project: str, query: str, want_path: str
) -> tuple[int | None, list[str]]:
    res = rag.search_project(project, query, top_k=5)
    paths = [r.source_path for r in res.results]
    rank = next((i for i, p in enumerate(paths, 1) if p == want_path), None)
    return rank, paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--project", default="ypq")
    ap.add_argument("--root", default="yspdb")
    args = ap.parse_args()
    rag = ProjectRAG(config_path=args.config)
    rag.register_project_root(args.project, args.root)
    try:
        for qid, (raw, gloss, path, _sub) in PROBES.items():
            r0, _ = _rank(rag, args.project, raw, path)
            r1, _ = _rank(rag, args.project, f"{raw} {gloss}", path)
            r2, _ = _rank(rag, args.project, gloss, path)
            print(f"{qid}: raw_rank={r0}  raw+gloss_rank={r1}  gloss_only_rank={r2}")
    finally:
        rag.close()


if __name__ == "__main__":
    main()

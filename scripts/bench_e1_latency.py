"""E1 step 5 (S5 task 5): query latency per pipeline stage on the gold questions. No LLM.

Runs every question once as warm-up, then `--repeats` timed passes through the production
pipeline of the given config (hybrid + rerank + abstention). Stage times come from the
SearchService telemetry hook (embed, dense, lexical, fuse, rerank, ...); `total` is wall time per
search. Prints p50/p95/max per stage and writes JSON. Run it with the model containers limited to
the target host size (e.g. `docker update --cpus 4 --memory 12g ...`) to get host-like numbers.

  uv run python -X utf8 scripts/bench_e1_latency.py --config configs/yp_all_q8_0.yaml \
      --project youth-participation --root "<corpus>/Youth Participation" \
      --questions eval/yp_questions_v3.jsonl --out local_data/yp_all/e1_latency_q8.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Mapping
from pathlib import Path

from project_rag.api import ProjectRAG
from project_rag.services.eval import load_questions


class _Recorder:
    """Telemetry port implementation that keeps stage observations in memory."""

    def __init__(self) -> None:
        self.stages: dict[str, list[float]] = {}
        self.enabled = False

    def observe(self, stage: str, seconds: float) -> None:
        if self.enabled:
            self.stages.setdefault(stage, []).append(seconds)

    def count(self, name: str, labels: Mapping[str, str] | None = None) -> None:
        return None

    def gauge(self, name: str, value: float, labels: Mapping[str, str] | None = None) -> None:
        return None


def _summary(xs: list[float]) -> dict[str, float]:
    s = sorted(xs)
    p95 = s[min(len(s) - 1, round(0.95 * (len(s) - 1)))]
    return {
        "n": len(s),
        "p50_s": round(statistics.median(s), 4),
        "p95_s": round(p95, 4),
        "max_s": round(s[-1], 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    queries = [q.query for q in load_questions(args.questions)]
    rag = ProjectRAG(config_path=args.config)
    rec = _Recorder()
    rag.services.search._telemetry = rec  # pyright: ignore[reportPrivateUsage] - bench hook
    rag.register_project_root(args.project, args.root)
    totals: list[float] = []
    degraded = 0
    try:
        for q in queries:  # warm-up: model servers, caches, SQLite pages
            rag.search_project(args.project, q)
        rec.enabled = True
        for _ in range(args.repeats):
            for q in queries:
                t0 = time.perf_counter()
                res = rag.search_project(args.project, q)
                totals.append(time.perf_counter() - t0)
                degraded += bool(res.degraded)
    finally:
        rag.close()
    report = {
        "config": args.config,
        "questions": len(queries),
        "repeats": args.repeats,
        "degraded_searches": degraded,
        "total": _summary(totals),
        "stages": {k: _summary(v) for k, v in sorted(rec.stages.items())},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

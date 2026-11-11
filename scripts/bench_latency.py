"""Latency + memory benchmark for the full hybrid+reranker query pipeline (spec v1 §6.2
Efficiency gate). No LLM. Measures per-query wall time over the v2 question set and reports
p50/p95/mean/max, with a warmup pass so model-load and first-call CUDA graph costs are excluded
from the steady-state numbers (and reported separately as cold-start). Also reports peak process
RSS and, when CUDA/ROCm is live, peak GPU VRAM.

No SLA is invented; the numbers are reported as-measured on this machine (v1 §6.2: "Measured and
acceptable to stakeholder; no premature numeric SLA invented").

Run (WSL ROCm):
  wsl.exe -d Ubuntu-24.04 -- bash -c 'export HSA_ENABLE_DXG_DETECTION=1; cd /mnt/c/Users/Cicada38/Projects/assist && PYTHONPATH=src /home/gorila/venvs/tokens-ingest-rocm/bin/python -X utf8 scripts/bench_latency.py --config configs/yp_qwen4b_gpu_rerank.yaml --project ypq --root yspdb --questions eval/yp_questions_v2.jsonl --out eval/results/yp_v2_latency.json'
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from project_rag.api import ProjectRAG
from project_rag.evaluation import load_questions


def _peak_vram_gb() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 1e9, 3)
    except (ImportError, RuntimeError):
        return None
    return None


def _rss_gb() -> float | None:
    try:
        import resource

        # ru_maxrss is KB on Linux.
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 3)
    except (ImportError, AttributeError):  # no `resource` module on Windows
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--project", default="ypq")
    ap.add_argument("--root", default="yspdb")
    ap.add_argument("--questions", default="eval/yp_questions_v2.jsonl")
    ap.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="timed passes over the question set after warmup (default 3)",
    )
    ap.add_argument("--out", default="eval/results/yp_v2_latency.json")
    args = ap.parse_args()

    questions = load_questions(args.questions)
    rag = ProjectRAG(config_path=args.config)
    rag.register_project_root(args.project, args.root)

    try:
        # Cold start: first query pays model load + first CUDA kernels. Timed separately.
        t0 = time.perf_counter()
        rag.search_project(args.project, questions[0].query, top_k=5)
        cold_ms = round((time.perf_counter() - t0) * 1000, 1)

        # Warmup the rest once (not timed) so steady-state excludes any remaining first-touch cost.
        for q in questions[1:]:
            rag.search_project(args.project, q.query, top_k=5)

        # Timed steady-state passes.
        per_query_ms: list[float] = []
        for _ in range(args.repeats):
            for q in questions:
                t = time.perf_counter()
                rag.search_project(args.project, q.query, top_k=5)
                per_query_ms.append((time.perf_counter() - t) * 1000)
    finally:
        rag.close()

    per_query_ms.sort()
    n = len(per_query_ms)

    def pct(p: float) -> float:
        return round(per_query_ms[min(n - 1, int(p * n))], 1)

    report = {
        "config": args.config,
        "n_queries_timed": n,
        "repeats": args.repeats,
        "cold_start_ms": cold_ms,
        "warm_p50_ms": round(statistics.median(per_query_ms), 1),
        "warm_p95_ms": pct(0.95),
        "warm_mean_ms": round(statistics.mean(per_query_ms), 1),
        "warm_min_ms": round(per_query_ms[0], 1),
        "warm_max_ms": round(per_query_ms[-1], 1),
        "peak_rss_gb": _rss_gb(),
        "peak_vram_gb": _peak_vram_gb(),
    }
    print(json.dumps(report, indent=2))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

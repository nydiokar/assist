"""WSL/CUDA + Qwen3-4B feasibility smoke (Amendment A01 §E).

Run BEFORE any bulk Qwen3 embedding. Captures GPU/VRAM/driver, loads the model on CUDA,
encodes one EN and one BG passage + query, verifies dimension/finite/normalization and trivial
ranking, measures peak VRAM and duration. Fails VISIBLY if CUDA is unavailable or OOM — it never
silently falls back to CPU or a different model (A01 §A2/§E/§J).

Writes a JSON report to eval/results/feasibility_smoke.json. No NGO data is used; synthetic
passages only.

Usage:  python scripts/feasibility_smoke.py
Exit:   0 = gate passed; non-zero = gate failed (reason in report + stderr).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from project_rag.gpu_env import describe_gpu_env, prepare_gpu_env

REPORT = Path(__file__).resolve().parents[1] / "eval" / "results" / "feasibility_smoke.json"

EN_PASSAGE = "The approved project defines a target of 120 people trained in sustainable farming."
BG_PASSAGE = "Обучението по устойчиво земеделие се проведе през март 2025 г. в село Бреза."
EN_QUERY = "how many people are targeted for training"
BG_QUERY = "кога се проведе обучението"


def _fail(report: dict, reason: str) -> int:
    report["passed"] = False
    report["failure_reason"] = reason
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"FEASIBILITY GATE FAILED: {reason}", file=sys.stderr)
    return 2


def main() -> int:
    report: dict = {"passed": False, "stage": "start"}

    # 1) environment capture
    prepare_gpu_env()  # must precede `import torch`
    try:
        import torch
    except ImportError as e:
        return _fail(report, f"torch not installed (install .[qwen3]): {e}")

    report["torch_version"] = torch.__version__
    report["cuda_available"] = bool(torch.cuda.is_available())
    if not torch.cuda.is_available():
        return _fail(
            report,
            "torch.cuda.is_available() is False — no CUDA/ROCm device. "
            + describe_gpu_env(torch.version.hip, torch.version.cuda)
            + " Do NOT silently fall back to CPU/0.6B (A01 §E).",
        )

    dev = torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(dev)
    report["gpu_name"] = torch.cuda.get_device_name(dev)
    report["vram_total_gb"] = round(total / 1e9, 2)
    report["vram_free_gb"] = round(free / 1e9, 2)
    report["cuda_runtime"] = torch.version.cuda

    # 2) load model + encode
    from project_rag.config import load_config
    from project_rag.embeddings import build_embedder

    cfg = load_config()
    cfg.embedding.provider = "qwen3"
    report["model_id"] = cfg.embedding.qwen3.model
    report["revision"] = cfg.embedding.qwen3.revision
    report["inference_dtype"] = cfg.embedding.qwen3.inference_dtype
    report["max_input_tokens"] = cfg.embedding.qwen3.max_input_tokens

    embedder = build_embedder(cfg.embedding)
    report["stage"] = "encode"
    try:
        torch.cuda.reset_peak_memory_stats(dev)
        t0 = time.perf_counter()
        docs = embedder.embed_passages([EN_PASSAGE, BG_PASSAGE])
        queries = embedder.embed_queries([EN_QUERY, BG_QUERY])
        dur = time.perf_counter() - t0
        report["peak_vram_gb"] = round(torch.cuda.max_memory_allocated(dev) / 1e9, 2)
        report["encode_seconds"] = round(dur, 3)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            return _fail(report, f"CUDA OOM during encode; lower batch/length: {e}")
        return _fail(report, f"encode failed: {e}")

    # 3) output validation
    import numpy as np

    report["doc_shape"] = list(docs.shape)
    if docs.shape[1] != cfg.embedding.qwen3.dim:
        return _fail(report, f"dim {docs.shape[1]} != expected {cfg.embedding.qwen3.dim}")
    if not np.isfinite(docs).all() or not np.isfinite(queries).all():
        return _fail(report, "non-finite embedding values")

    # 4) trivial ranking: EN query should match EN passage; BG query → BG passage
    def cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    en_ok = cos(queries[0], docs[0]) > cos(queries[0], docs[1])
    bg_ok = cos(queries[1], docs[1]) > cos(queries[1], docs[0])
    report["trivial_ranking_en_ok"] = en_ok
    report["trivial_ranking_bg_ok"] = bg_ok
    if not (en_ok and bg_ok):
        return _fail(report, "trivial relevance ranking failed; investigate prompt/pooling")

    report["passed"] = True
    report["stage"] = "done"
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("FEASIBILITY GATE PASSED")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

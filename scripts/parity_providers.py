"""S3 parity: llama-server (served GGUF) vs the torch reference on the synthetic fixture corpus.

Two halves, run where their dependencies live, then compared:

  # reference (WSL ROCm venv, torch bf16 Qwen3-4B pinned revision + bge-reranker-v2-m3):
  HF_HUB_OFFLINE=1 PYTHONPATH=src python scripts/parity_providers.py reference --out ref.json
  # served (llama-server pair on 127.0.0.1:8081 / :8082):
  uv run python scripts/parity_providers.py served --out served.json [--append-text '<|endoftext|>']
  uv run python scripts/parity_providers.py compare ref.json served.json

Measured per the build plan S3 exit checks: embedding cosine per passage/query (gate: mean >=
0.99), rerank Spearman over 20 (query, passage) pairs (gate: >= 0.95), token counts per chunk text
(gate: identical) and chunk boundaries under each token counter (gate: identical). Fixture data
only; outputs carry vectors/scores, no corpus text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

from project_rag.adapters.parse_registry import ParserRegistry
from project_rag.core.chunking import chunk_document
from project_rag.domain.config import ChunkingConfig, EmbeddingConfig, ParsingConfig, RerankConfig
from project_rag.domain.models import DocState
from project_rag.ports.embedder import Embedder
from project_rag.ports.reranker import Reranker

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "fixtures" / "synthetic" / "ngo_project_demo"
QUESTIONS = REPO / "eval" / "archive" / "questions_synthetic_gv.jsonl"
# Small target so the fixture yields several chunks and boundaries actually depend on counts.
CHUNKING = ChunkingConfig(target_tokens=60, max_tokens=2048, overlap_tokens=0)


def _docs() -> list[tuple[str, object]]:
    reg = ParserRegistry(ParsingConfig())
    out: list[tuple[str, object]] = []
    for p in sorted(FIXTURES.iterdir()):
        doc = reg.parse(str(p), p.name, p.suffix.lower())
        if doc.state == DocState.PARSED:
            out.append((p.name, doc))
    return out


def _passages() -> list[str]:
    texts: list[str] = []
    for _, doc in _docs():
        texts.extend(b.text for b in doc.blocks if b.text.strip())  # type: ignore[attr-defined]
    return texts


def _queries() -> list[str]:
    lines = QUESTIONS.read_text(encoding="utf-8").splitlines()
    return [json.loads(ln)["query"] for ln in lines if ln.strip()]


def _boundaries(counter: Callable[[str], int]) -> list[str]:
    ids: list[str] = []
    for name, doc in _docs():
        chunks = chunk_document(
            doc,  # type: ignore[arg-type]
            project_id="p",
            source_id=name,
            version_id="v",
            source_type="x",
            cfg=CHUNKING,
            token_counter=counter,
        )
        ids.extend(
            f"{name}#{c.chunk_ordinal}:{hashlib.sha256(c.citation_text.encode()).hexdigest()[:12]}"
            for c in chunks
        )
    return ids


def _pairs() -> list[tuple[str, str]]:
    queries = _queries()[:4]
    passages = [p for p in _passages() if len(p) > 40][:5]
    return [(q, p) for q in queries for p in passages]


def _measure(
    embedder: Embedder, reranker: Reranker, counter: Callable[[str], int]
) -> dict[str, object]:
    passages, queries = _passages(), _queries()
    pairs = _pairs()
    by_query: dict[str, list[str]] = {}
    for q, p in pairs:
        by_query.setdefault(q, []).append(p)
    scores: list[float] = []
    for q, ps in by_query.items():
        scores.extend(reranker.score(q, ps))
    return {
        "passage_vectors": embedder.embed_passages(passages).tolist(),
        "query_vectors": embedder.embed_queries(queries).tolist(),
        "rerank_scores": scores,
        "token_counts": [counter(t) for t in passages],
        "boundaries": _boundaries(counter),
        "n_passages": len(passages),
        "n_queries": len(queries),
        "n_pairs": len(pairs),
    }


def cmd_reference(out: str) -> None:
    from transformers import AutoTokenizer

    from project_rag.adapters.embed_torch import Qwen3Embedder
    from project_rag.adapters.gpu_env import prepare_gpu_env
    from project_rag.adapters.rerank_torch import CrossEncoderReranker
    from project_rag.wiring import load_config

    prepare_gpu_env()
    cfg = load_config(REPO / "configs" / "yp_qwen4b_gpu.yaml")
    q3 = cfg.embedding.qwen3
    tok = AutoTokenizer.from_pretrained(q3.model, revision=q3.revision)

    def hf_count(text: str) -> int:
        return len(tok.encode(text, add_special_tokens=True))

    rr = CrossEncoderReranker(RerankConfig(enabled=True))
    result = _measure(Qwen3Embedder(cfg.embedding), rr, hf_count)
    result["side"] = f"torch {q3.model}@{q3.revision[:12]} {q3.inference_dtype}; bge-reranker-v2-m3"
    Path(out).write_text(json.dumps(result), encoding="utf-8")
    print("wrote", out)


def cmd_served(out: str, append_text: str, transform: str) -> None:
    from project_rag.adapters.embed_openai_compat import OpenAICompatEmbedder
    from project_rag.adapters.rerank_openai_compat import OpenAICompatReranker
    from project_rag.domain.config import OpenAICompatEmbedConfig, OpenAICompatRerankConfig

    ecfg = EmbeddingConfig(
        provider="openai_compat", openai_compat=OpenAICompatEmbedConfig(append_text=append_text)
    )
    embedder = OpenAICompatEmbedder(ecfg)
    rcfg = RerankConfig(
        enabled=True,
        provider="openai_compat",
        openai_compat=OpenAICompatRerankConfig(score_transform=transform),
    )
    result = _measure(embedder, OpenAICompatReranker(rcfg), embedder.token_count)
    result["side"] = f"llama-server append_text={append_text!r} transform={transform}"
    Path(out).write_text(json.dumps(result), encoding="utf-8")
    print("wrote", out)


def _spearman(a: list[float], b: list[float]) -> float:
    def ranks(x: list[float]) -> np.ndarray:
        arr = np.asarray(x)
        order = arr.argsort()
        r = np.empty(len(arr))
        r[order] = np.arange(len(arr), dtype=float)
        for v in np.unique(arr):  # average ranks for ties
            idx = arr == v
            r[idx] = r[idx].mean()
        return r

    ra, rb = ranks(a), ranks(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def cmd_compare(ref_path: str, served_path: str) -> None:
    ref = json.loads(Path(ref_path).read_text(encoding="utf-8"))
    srv = json.loads(Path(served_path).read_text(encoding="utf-8"))
    report: dict[str, object] = {"reference": ref["side"], "served": srv["side"]}
    for kind in ("passage_vectors", "query_vectors"):
        a = np.asarray(ref[kind], dtype=np.float64)
        b = np.asarray(srv[kind], dtype=np.float64)
        cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
        report[kind] = {
            "n": len(cos),
            "mean_cos": round(float(cos.mean()), 5),
            "min_cos": round(float(cos.min()), 5),
            "gate_mean_ge_0.99": bool(cos.mean() >= 0.99),
        }
    rho = _spearman(ref["rerank_scores"], srv["rerank_scores"])
    report["rerank"] = {
        "n_pairs": len(ref["rerank_scores"]),
        "spearman": round(rho, 4),
        "gate_ge_0.95": rho >= 0.95,
        "ref_range": [round(min(ref["rerank_scores"]), 4), round(max(ref["rerank_scores"]), 4)],
        "served_range": [round(min(srv["rerank_scores"]), 4), round(max(srv["rerank_scores"]), 4)],
    }
    ta, tb = ref["token_counts"], srv["token_counts"]
    report["tokens"] = {
        "n": len(ta),
        "identical": ta == tb,
        "mismatches": sum(1 for x, y in zip(ta, tb, strict=True) if x != y),
        "example_diffs": [(x, y) for x, y in zip(ta, tb, strict=True) if x != y][:5],
    }
    report["chunk_boundaries"] = {
        "n_ref": len(ref["boundaries"]),
        "n_served": len(srv["boundaries"]),
        "identical": ref["boundaries"] == srv["boundaries"],
    }
    print(json.dumps(report, indent=1, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("reference")
    r.add_argument("--out", required=True)
    s = sub.add_parser("served")
    s.add_argument("--out", required=True)
    s.add_argument("--append-text", default="")
    s.add_argument("--rerank-transform", default="none", choices=["none", "sigmoid"])
    c = sub.add_parser("compare")
    c.add_argument("ref")
    c.add_argument("served")
    args = ap.parse_args()
    if args.cmd == "reference":
        cmd_reference(args.out)
    elif args.cmd == "served":
        cmd_served(args.out, args.append_text, args.rerank_transform)
    else:
        cmd_compare(args.ref, args.served)


if __name__ == "__main__":
    main()

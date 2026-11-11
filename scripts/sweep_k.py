"""Diagnostic k-sweep (WP3 → WP4 readiness). No LLM, no config/index change.

Question it answers: when hybrid misses, is the answer *retrieved but mis-ranked* (then the
fix is fusion/rerank) or *not retrieved at all* (then the fix is chunking/embedding)?

It bypasses the production top_k clamp (cfg.top_k=8) and the file-diversity dedup, going straight
to each store's candidate pool and a raw RRF fuse, so recall@{5,10,20,pool} is honest. Document-
level recall (expected source path in top-k) and evidence-level (expected substring present in an
expected-source chunk's citation_text within top-k) are both reported.

Run in WSL ROCm:
  wsl.exe -d Ubuntu-24.04 -- bash -c 'export HSA_ENABLE_DXG_DETECTION=1; cd /mnt/c/Users/Cicada38/Projects/assist && PYTHONPATH=src /home/gorila/venvs/tokens-ingest-rocm/bin/python -X utf8 scripts/sweep_k.py'
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from project_rag.api import ProjectRAG
from project_rag.core.fusion import rrf_fuse
from project_rag.domain.models import ChannelHit
from project_rag.services.eval import load_questions

CONFIG = "configs/yp_qwen4b_gpu.yaml"
PROJECT = "ypq"
ROOT = "yspdb"
QUESTIONS = "eval/yp_questions_v2.jsonl"
POOL = 50  # candidates per channel for the sweep (production uses 16)
KS = (5, 10, 20)


def fuse_raw(dense: list[ChannelHit], lexical: list[ChannelHit], k: int) -> list[tuple[str, dict]]:
    """Uncapped RRF order as (chunk_id, payload dict) — the sweep inspects the raw pool."""
    return [
        (c.chunk_id, c.payload.model_dump())
        for c in rrf_fuse({"dense": dense, "lexical": lexical}, k)
    ]


def main() -> None:
    rag = ProjectRAG(config_path=CONFIG)
    rag.register_project_root(PROJECT, ROOT)
    gen = rag.services.catalog.active_generation(PROJECT)
    questions = [
        q for q in load_questions(QUESTIONS) if q.kind != "no_answer" and q.expect_source_paths
    ]

    doc = {k: 0 for k in (*KS, "pool")}
    ev = {k: 0 for k in (*KS, "pool")}
    miss_detail: list[dict] = []

    for q in questions:
        qvec = rag.services.embedder.embed_queries([q.query])[0]
        dense = rag.services.vector.query(PROJECT, gen, np.asarray(qvec), POOL, None)
        lexical = rag.services.lexical.search(PROJECT, gen, q.query, POOL, None)
        fused = fuse_raw(dense, lexical, 60)

        def doc_hit(n: int, q=q, fused=fused) -> bool:
            return any(p.get("source_path", "") in q.expect_source_paths for _c, p in fused[:n])

        def ev_hit(n: int, q=q, fused=fused) -> bool:
            texts = [
                p.get("citation_text", "")
                for _c, p in fused[:n]
                if p.get("source_path", "") in q.expect_source_paths
            ]
            return bool(texts) and all(any(s in t for t in texts) for s in q.expect_substrings)

        for k in KS:
            doc[k] += int(doc_hit(k))
            ev[k] += int(ev_hit(k))
        doc["pool"] += int(doc_hit(len(fused)))
        ev["pool"] += int(ev_hit(len(fused)))

        # rank at which the expected doc / evidence first appears in the full fused list
        drank = next(
            (
                i
                for i, (_c, p) in enumerate(fused, 1)
                if p.get("source_path", "") in q.expect_source_paths
            ),
            None,
        )
        erank = next(
            (
                i
                for i, (_c, p) in enumerate(fused, 1)
                if p.get("source_path", "") in q.expect_source_paths
                and all(s in p.get("citation_text", "") for s in q.expect_substrings)
            ),
            None,
        )
        if not (doc_hit(5) and ev_hit(5)):
            miss_detail.append(
                {
                    "id": q.id,
                    "kind": q.kind,
                    "doc_rank": drank,
                    "evidence_rank": erank,
                    "in_pool_doc": drank is not None,
                    "in_pool_ev": erank is not None,
                }
            )

    n = len(questions)
    print(f"n(scored)={n}  pool={POOL}/channel")
    print("\n           recall@5  @10   @20   pool")
    print(f"  doc      {doc[5]/n:.3f}   {doc[10]/n:.3f} {doc[20]/n:.3f} {doc['pool']/n:.3f}")
    print(f"  evidence {ev[5]/n:.3f}   {ev[10]/n:.3f} {ev[20]/n:.3f} {ev['pool']/n:.3f}")
    print("\nmiss detail (fused rank of expected doc / evidence chunk; None = not in pool):")
    for m in sorted(miss_detail, key=lambda x: int(x["id"][2:])):
        print(
            f"  {m['id']:5} {m['kind']:14} doc_rank={m['doc_rank']}  ev_rank={m['evidence_rank']}"
        )
    rag.close()

    Path("eval/results/yp_v2_sweep.json").write_text(
        json.dumps(
            {"n": n, "pool_per_channel": POOL, "doc": doc, "evidence": ev, "misses": miss_detail},
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

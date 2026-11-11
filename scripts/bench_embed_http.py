"""Embedding throughput pilot against a llama-server (X6 procedure before a long ingest).

Samples chunk texts from a stage-cache chunk directory, embeds them through the
openai_compat adapter in production-sized batches, and prints tokens/s plus the extrapolated
duration for a given total token count. Prints numbers only (no corpus text).

  uv run python scripts/bench_embed_http.py --chunks local_data/yp_qwen4b_gpu/cache/ypq/chunk \
      --base-url http://127.0.0.1:8081 --sample 160 --total-tokens 2211471
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from project_rag.adapters.embed_openai_compat import OpenAICompatEmbedder
from project_rag.domain.config import EmbeddingConfig, OpenAICompatEmbedConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True, help="stage-cache chunk dir (*.json)")
    ap.add_argument("--base-url", default="http://127.0.0.1:8081")
    ap.add_argument("--sample", type=int, default=160)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--total-tokens", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    texts: list[tuple[str, int]] = []
    for f in sorted(Path(args.chunks).glob("*.json")):
        for ch in json.loads(f.read_text(encoding="utf-8"))["chunks"]:
            texts.append((ch["embed_text"], int(ch.get("token_count", 0))))
    random.Random(args.seed).shuffle(texts)
    sample = texts[: args.sample]
    cfg = EmbeddingConfig(
        provider="openai_compat",
        openai_compat=OpenAICompatEmbedConfig(
            base_url=args.base_url, max_batch=args.batch, timeout_s=600
        ),
    )
    emb = OpenAICompatEmbedder(cfg)
    emb.embed_passages([sample[0][0]])  # warm-up
    t0 = time.perf_counter()
    emb.embed_passages([t for t, _ in sample])
    secs = time.perf_counter() - t0
    tokens = sum(n for _, n in sample)
    rate = tokens / secs
    out = {
        "base_url": args.base_url,
        "sample_chunks": len(sample),
        "sample_tokens": tokens,
        "seconds": round(secs, 1),
        "tokens_per_s": round(rate, 1),
        "chunks_per_s": round(len(sample) / secs, 2),
    }
    if args.total_tokens:
        out["extrapolated_minutes"] = round(args.total_tokens / rate / 60, 1)
    print(json.dumps(out))


if __name__ == "__main__":
    main()

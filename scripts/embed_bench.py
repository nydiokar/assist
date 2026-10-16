"""CPU embedding feasibility pilot (A01 §E replacement evidence; see docs/DECISIONS.md).

Measures, on REAL chunks from an ingested index: tokenizer-true token counts, load time and
encode throughput for one model, then extrapolates to the whole index. Does not write vectors.

    .\\.venv\\Scripts\\python.exe -X utf8 scripts\\embed_bench.py --model Qwen/Qwen3-Embedding-0.6B --dtype float32
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def load_chunk_texts(db: str, n_all: int | None) -> tuple[list[str], int]:
    con = sqlite3.connect(db)
    rows = [r[0] for r in con.execute("SELECT body FROM chunk_fts")]
    total = len(rows)
    random.Random(0).shuffle(rows)
    return (rows if n_all is None else rows[:n_all]), total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--db", default="local_data/yp_baseline/manifest.sqlite3")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    texts, total = load_chunk_texts(args.db, args.n)
    dtype = getattr(torch, args.dtype)

    t0 = time.perf_counter()
    model = SentenceTransformer(args.model, device="cpu", model_kwargs={"torch_dtype": dtype},
                                tokenizer_kwargs={"padding_side": "left"})
    model.max_seq_length = args.max_len
    load_s = time.perf_counter() - t0

    toks = [len(model.tokenizer.encode(t, add_special_tokens=True)) for t in texts]
    model.encode(texts[:2], batch_size=2, normalize_embeddings=True)  # warm-up
    t1 = time.perf_counter()
    vecs = np.asarray(model.encode(texts, batch_size=args.batch, normalize_embeddings=True))
    enc_s = time.perf_counter() - t1
    tok_s = sum(min(t, args.max_len) for t in toks) / enc_s
    mean_tok = float(np.mean(toks))
    out = {
        "model": args.model, "dtype": args.dtype, "threads": args.threads, "dim": int(vecs.shape[1]),
        "sample": len(texts), "load_s": round(load_s, 1),
        "tokens_mean": round(mean_tok, 1), "tokens_max": int(max(toks)),
        "n_over_max_len": int(sum(t > args.max_len for t in toks)),
        "encode_s": round(enc_s, 1), "tokens_per_s": round(tok_s, 1),
        "chunks_total": total,
        "est_full_index_hours": round(total * mean_tok / tok_s / 3600, 2),
        "finite": bool(np.isfinite(vecs).all()),
    }
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""E1 step 4 (S5 task 4): GPU vs CPU llama.cpp build of the SAME GGUF on real chunks.

Samples `--n` random chunks of the active generation from the store's FTS table, embeds each text
on both servers (`/v1/embeddings`, L2-normalised client-side like the adapter), and reports the
per-chunk cosine. Gate (spec §6.2 E1 step 4): mean >= 0.999 and min >= 0.995. Output carries
numbers only (no chunk text).

  python scripts/parity_builds.py --db /state/store.sqlite3 --project youth-participation \
      --a http://host.docker.internal:8091 --b http://host.docker.internal:8081 --out parity.json
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
from pathlib import Path

import httpx
import numpy as np
import numpy.typing as npt


def _embed(client: httpx.Client, url: str, texts: list[str]) -> npt.NDArray[np.float32]:
    r = client.post(f"{url}/v1/embeddings", json={"input": texts})
    r.raise_for_status()
    items = sorted(r.json()["data"], key=lambda d: d["index"])
    m = np.asarray([d["embedding"] for d in items], dtype=np.float32)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--a", required=True, help="base URL of build A (e.g. GPU)")
    ap.add_argument("--b", required=True, help="base URL of build B (e.g. CPU)")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    (gen,) = con.execute(
        "SELECT active_generation FROM projects WHERE project_id = ?", (args.project,)
    ).fetchone()
    rows = con.execute(
        "SELECT chunk_id, body FROM chunk_fts WHERE project_id = ? AND generation = ? "
        "ORDER BY chunk_id",
        (args.project, gen),
    ).fetchall()
    sample = random.Random(args.seed).sample(rows, min(args.n, len(rows)))
    texts = [body for _cid, body in sample]
    cos: list[float] = []
    with httpx.Client(timeout=600) as client:
        for i in range(0, len(texts), args.batch):
            part = texts[i : i + args.batch]
            a = _embed(client, args.a, part)
            b = _embed(client, args.b, part)
            cos.extend(float(x) for x in np.sum(a * b, axis=1))
    arr = np.asarray(cos)
    report = {
        "generation": gen,
        "n": int(arr.size),
        "seed": args.seed,
        "mean_cosine": round(float(arr.mean()), 6),
        "min_cosine": round(float(arr.min()), 6),
        "p01_cosine": round(float(np.quantile(arr, 0.01)), 6),
        "gate": {"mean>=0.999": bool(arr.mean() >= 0.999), "min>=0.995": bool(arr.min() >= 0.995)},
    }
    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

"""D-3 spike benchmark (v3 build plan S2.2): sqlite-vec brute-force KNN at N x DIM float32.

Builds a throwaway SQLite file with the production vec0 layout (partition key project_id,
metadata generation/doc_class, cosine distance), inserts N random unit vectors, then times
filtered KNN queries. Reports insert time, query p50/p95/max, peak RSS and file size. Synthetic
vectors only: no corpus content.

  uv run python scripts/bench_vec.py --n 50000 --dim 2560 --queries 200 --k 16
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import sqlite_vec


def _peak_rss_mb() -> float | None:
    try:
        import psutil

        info = psutil.Process().memory_info()
        peak = getattr(info, "peak_wset", None) or info.rss
        return round(peak / 1e6, 1)
    except ImportError:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50_000)
    ap.add_argument("--dim", type=int, default=2560)
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--projects", type=int, default=3, help="partitions sharing the table")
    ap.add_argument(
        "--metric",
        default="cosine",
        choices=["cosine", "l2"],
        help="vec0 distance metric (l2 on unit vectors orders identically to cosine)",
    )
    args = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="bench_vec_"))
    db = tmp / "bench.sqlite3"
    conn = sqlite3.connect(str(db))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        f"CREATE VIRTUAL TABLE v USING vec0(project_id text partition key, "
        f"embedding float[{args.dim}] distance_metric={args.metric}, generation text, doc_class text, "
        f"+chunk_id text)"
    )
    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    batch = 2000
    main_project_rows = 0
    for start in range(0, args.n, batch):
        m = rng.standard_normal((min(batch, args.n - start), args.dim)).astype(np.float32)
        m /= np.linalg.norm(m, axis=1, keepdims=True)
        rows = []
        for i, vec in enumerate(m):
            pid = f"p{(start + i) % args.projects}"
            main_project_rows += pid == "p0"
            rows.append((pid, vec.tobytes(), "g1", "project_record", f"ck_{start + i}"))
        conn.executemany(
            "INSERT INTO v(project_id, embedding, generation, doc_class, chunk_id) "
            "VALUES (?,?,?,?,?)",
            rows,
        )
    conn.commit()
    insert_s = time.perf_counter() - t0

    qs = rng.standard_normal((args.queries, args.dim)).astype(np.float32)
    qs /= np.linalg.norm(qs, axis=1, keepdims=True)
    lat: list[float] = []
    for q in qs:
        t = time.perf_counter()
        hits = conn.execute(
            "SELECT chunk_id, distance FROM v WHERE embedding MATCH ? AND k = ? "
            "AND project_id = ? AND generation = ? ORDER BY distance",
            (q.tobytes(), args.k, "p0", "g1"),
        ).fetchall()
        lat.append((time.perf_counter() - t) * 1000)
        assert len(hits) == args.k
    lat.sort()
    report = {
        "sqlite_version": sqlite3.sqlite_version,
        "sqlite_vec": conn.execute("select vec_version()").fetchone()[0],
        "platform": sys.platform,
        "cpu_count": os.cpu_count(),
        "n": args.n,
        "dim": args.dim,
        "projects": args.projects,
        "rows_in_queried_partition": main_project_rows,
        "k": args.k,
        "metric": args.metric,
        "insert_s": round(insert_s, 2),
        "query_ms_p50": round(statistics.median(lat), 2),
        "query_ms_p95": round(lat[int(0.95 * (len(lat) - 1))], 2),
        "query_ms_max": round(lat[-1], 2),
        "peak_rss_mb": _peak_rss_mb(),
        "db_mb": round(db.stat().st_size / 1e6, 1),
    }
    conn.close()
    for f in tmp.iterdir():
        f.unlink()
    tmp.rmdir()
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

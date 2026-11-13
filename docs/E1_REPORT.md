# E1 — embedder quantisation on the extended YP corpus (spec A03 §6.2, build plan S5)

**Status:** in progress (2026-11-13). Metrics only; no corpus text. Raw outputs are in
`local_data/yp_all/` and `eval/results/` (gitignored).

## Setup

| Item | Value |
|---|---|
| Corpus | `yspdb_all/` (owner-supplied 2026-11-13), 1500 files discovered, 9 excluded (media), 1 curated out (`KA154-YOU-E2373104.pdf`) |
| Pipeline | native parsers (`native_v2`), OCR bul+eng (15 chars / 65 conf), antiword; exact-duplicate collapse; chunk target 500 / max 2048 tokens; hybrid lexical + dense, RRF k=60, per-file cap 3, rerank top 12 (bge-reranker-v2-m3 Q8_0, CPU llama-server, sigmoid), abstention on rerank score 0.26 |
| Embedders | Qwen3-Embedding-4B GGUF Q4_K_M / Q6_K / Q8_0 (rev `f460253`, sha256 pinned in `configs/models.lock.yaml`) on the GPU llama.cpp build b11804 (Vulkan, RX 7900 XT); **reference** torch bf16 (rev `5cf2132`, WSL ROCm) |
| Gold set | `eval/yp_questions_v3.jsonl` (local): v2 carried onto `yspdb_all/` by `scripts/relabel_gold.py` |
| Band | ± 2 questions per primary metric (spec §6.8.1) |

## Ingest

_(filled per run)_

## Gold-set relabel (v2 → v3)

53 questions carried by exact-substring evidence (`scripts/relabel_gold.py`), then checked with
the harness's own `validate_labels` against `yspdb_all/`: **50 scored-type questions, 0 label
problems.** Every kept/moved source carries `expect_source_hashes` (spec §6.8.4).

| Status | n | Question ids |
|---|---|---|
| kept (same path, evidence present) | 26 | — |
| moved (evidence found under a new path) | 13 | yp01–04, yp14, yp17, yp30, yp34 (proposal → `KA154-YOU-E2373104.docx`); yp11, yp12 (`_` → space in the file name); yp27, yp28 (renamed folder); yp45 (one of two sources) |
| carried unchanged (no-answer ×11, folder ×1) | 12 | — |
| folder missing in `yspdb_all/` | 2 | yp52 (handbook photo-responses folder), yp53 (declaration videos folder) — excluded from scoring until the owner decides (owner queue) |

**For the owner to confirm (O2):** the 13 moves above, the 2 exclusions, and whether any of the
11 no-answer questions has become answerable on the extended corpus (checked below against the
Q8_0 index).

## Quality (production pipeline)

_(filled by `scripts/e1_compare.py`)_

## Parity (GPU vs CPU build, same GGUF)

_(filled by `scripts/parity_builds.py`)_

## Host-size latency and throughput

_(filled by `scripts/bench_e1_latency.py` and the CPU ingest pilot)_

## Recommendation

_(filled)_

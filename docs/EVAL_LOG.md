# Evaluation log (append-only)

Each entry: date · corpus snapshot · config · questions · result · what it taught.

## 2026-10-17 — Baseline B: lexical-only (FTS5/BM25), real corpus

- Corpus: `yspdb/`, 334 files; 172 text files ingested → 169 READY, 3 OCR_REQUIRED, 162 unsupported (media/images/legacy); 2,275 chunks (`structure_v3`, hash-embedder token counter).
- Config: `configs/yp_baseline.yaml` (dense channel not used). Questions: `eval/yp_questions.jsonl` (16; labels machine-validated against parsed text; **drafted by the agent from reading the corpus, not yet owner-reviewed**).
- Reproduce: `scripts/eval_yp.py --config configs/yp_baseline.yaml`.
- Result (14 scored, 2 no_answer excluded): **Recall@5 = 0.714, evidence-Recall@5 = 0.429, MRR = 0.643**.
- Failures → ledger: F9 (page boilerplate dilutes chunks), F10 (no Bulgarian stemming), F11 (no cross-language), F12 (right doc, wrong chunk).
- Not measured: abstention (a lexical channel always returns something; needs a score/threshold policy), dense, hybrid.
- Caveat: n=14; differences of one question are 7 points. Do not claim a ranking between methods from <±2 questions.

## 2026-10-17 — Baseline B after F9 fix (boilerplate dropped, `structure_v4`)

- Same corpus/questions/config; 2,270 chunks (-5). Recall@5 0.714 (unchanged), evidence@5 0.429 → 0.500 (+yp03), MRR 0.643 → 0.595 (yp02 fell from rank 1 to 3).
- Taught: one question moved each way — within noise at n=14. The boilerplate fix is justified by mechanism (shorter, cleaner chunks), not by this metric. Also found and fixed a real defect: the generation id ignored chunker/parser config, so a chunker change reported "no changes" and served a stale index (`test_chunking_config_change_forces_new_generation`).

## 2026-10-17 — E5-small (legacy comparator), CPU, real corpus, chunk target 300 E5-tokens (6,118 chunks)

Config `configs/yp_e5.yaml`; questions as above (n=14 scored). Chunking differs from the Baseline-B chunk set (confound, A02).

| Channel | Recall@5 | evidence@5 | MRR |
|---|---|---|---|
| lexical (FTS5) | 0.714 | 0.500 | 0.714 |
| dense (E5-small) | 0.643 | 0.286 | 0.643 |
| hybrid (RRF k=60) | 0.786 | 0.500 | 0.679 |

- Differences are ≤ 2 questions of 14: **no ranking between channels can be claimed.** What is solid: dense alone is not better than lexical on this question set; hybrid recovered the "conflict" (yp12) and reference (yp13) questions that one channel alone missed.
- Still failing in all channels: yp06 (inflection), yp07/yp09 (English question → Bulgarian table), no_answer pair (no abstention policy yet).
- Ingest note: E5 ingest first crashed at Chroma write (`Batch size 6118 > max 5461`); embeddings were cached so the retry took 12 s. Fixed in `ChromaVectorStore` (test `test_chroma_batching.py`).

## 2026-10-17 — Qwen3-Embedding-0.6B (bf16, CPU), chunk target 500 Qwen-tokens (5,496 chunks, 50 min)

Config `configs/yp_qwen06.yaml`; same 16 questions.

| Channel | Recall@5 | evidence@5 | MRR |
|---|---|---|---|
| lexical (this chunk set) | 0.714 | 0.500 | 0.679 |
| dense (Qwen3-0.6B) | 0.714 | 0.500 | 0.554 |
| hybrid (RRF k=60) | 0.643 | 0.643 | 0.643 |

- Dense fixed the inflection question (yp06 `уебинара`≈`Уебинари`, F10) that lexical cannot; it lost yp11 (date).
- Cross-language (yp07, yp09) still fails in every channel with Qwen too. Diagnosis (yp07/yp09 top hits): the target rows live in the `Резултати` sheet of `Overview_Debate 2025.xlsx`; dense retrieval returns long free-text survey chunks and proposal paragraphs instead. Hypothesis F14: small numeric table rows lose to long prose in embedding space.
- Hybrid at n=14 is again within noise of both channels; it gained evidence@5 (more correct passages in the top 5) but lost Recall@5 by one question.

## 2026-10-17 — Experiment: table granularity 20 → 3 rows/chunk (E5, 6,583 chunks)

Result identical on every metric to the 20-row run (lexical .714/.500, dense .643/.286, hybrid .786/.500). **F14's "row-group blending" hypothesis is not supported** for E5; do not change `table_rows_per_chunk` on this evidence.

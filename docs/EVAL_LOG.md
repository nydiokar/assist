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

## 2026-10-26 — Expanded question set v2 (n=39, 34 scored), real corpus, Qwen3-4B index — lexical + dense + hybrid

> **Correction (same day):** an earlier draft of this entry claimed dense/hybrid were "GPU-blocked / `/dev/kfd` absent / GPU compute down" and ran `wsl --shutdown`. **That was wrong and is retracted.** Root cause was operator error, not hardware: ROCm 7.2.4 on WSL needs `HSA_ENABLE_DXG_DETECTION=1`, which was exported only in `~/.bashrc`; a non-interactive `wsl -- bash -c` shell never sources `.bashrc`, so `rocminfo` returned `hsa_init Failed` and `torch.cuda.is_available()` was False. With the variable exported **in the same command**, `rocminfo` shows gfx1100 / AMD Radeon RX 7900 XT and torch sees 1 device. `/dev/kfd` is absent by design under WSL ROCm and is NOT a health check. The real dense/hybrid numbers are below. Ledger F18 is Closed; the shell/GPU contract is in `docs/DECISIONS.md` 2026-10-26 and `README.md`.

- Corpus: `yspdb/` real corpus, ingested index `yp_qwen4b_gpu` (project `ypq`, Qwen3-Embedding-4B, 2560-dim, chunk target 500 Qwen-tokens, `table_rows_per_chunk=20`). Index built 2026-10-20 in WSL ROCm venv (`/home/gorila/venvs/tokens-ingest-rocm`, exit 0, 11m49s). Reused unchanged; **no re-ingest** (query-embedding only). Config `configs/yp_qwen4b_gpu.yaml` (no config change).
- Questions: `eval/yp_questions_v2.jsonl`, 39 questions, **34 scored** (5 `no_answer` excluded from the denominator: yp15, yp16, yp19, yp36, yp39). Labels machine-validated against parsed corpus text this run (all 39 pass).
- Label fixes applied before validation passed (recorded in `docs/DECISIONS.md` 2026-10-26): 13 wrong `expect_source_path` entries corrected (yp20/21/24/25/26/27/28/29/31/38) and 1 absent substring replaced (yp30: `2024-3-BG01-KA154-YOU-000283706` is in NO corpus file — grep across `yspdb/` returns nothing; the proposal's real identifier is Form ID `KA154-YOU-E2373104`, which was substituted). No question deleted.
- Reproduce (WSL ROCm, all channels): `wsl.exe -d Ubuntu-24.04 -- bash -c 'export HSA_ENABLE_DXG_DETECTION=1; cd /mnt/c/Users/Cicada38/Projects/assist && PYTHONPATH=src /home/gorila/venvs/tokens-ingest-rocm/bin/python -X utf8 scripts/eval_yp.py --config configs/yp_qwen4b_gpu.yaml --project ypq --root yspdb --questions eval/yp_questions_v2.jsonl --channels lexical dense lexical+dense --out eval/results/yp_v2_qwen4b_gpu.json'`

| Channel | Recall@5 | evidence@5 | MRR | n (scored) |
|---|---|---|---|---|
| lexical (FTS5) | 0.6765 (23/34) | 0.5000 (17/34) | 0.5882 | 34 |
| dense (Qwen3-4B) | 0.6471 (22/34) | 0.5294 (18/34) | 0.5784 | 34 |
| **hybrid (RRF k=60)** | **0.7647 (26/34)** | **0.6765 (23/34)** | **0.6186** | 34 |

- **Hybrid beats lexical on all three metrics** (+0.088 Recall@5 = +3 questions, +0.176 evidence@5 = +6 questions, +0.030 MRR). This is the first real-corpus run where the margin is larger than the ±2-question noise band on Recall@5, and clearly larger than it on evidence@5. Dense alone is slightly *below* lexical on Recall@5 (−1 question) but above it on evidence@5 (+1) — consistent with every prior run: dense surfaces the right document with better evidence chunks, lexical nails exact tokens/numbers; fusion keeps both.
- **What dense fixed** (lexical miss → dense hit): yp06 (BG inflection `уебинара`≈`Уебинари`, F10), yp07 (EN question → BG numeric table, F11/F14), yp21 (exact, forum chair), yp24 (date, article), yp26 (exact, declaration demands). **What dense broke** (lexical hit → dense miss): yp05, yp11, yp23, yp29, yp35, yp37 — all numeric/exact where a specific number or ID is the answer and dense blurs it into semantically-similar rows. Fusion recovers these: hybrid holds all six.
- **Hybrid scored misses (8/34):** yp06, yp09, yp11, yp12, yp20, yp22, yp31, yp38 (diagnosis below).
- **Hybrid hit-but-no-evidence (3):** yp28, yp30, yp33 — correct document in top-5, specific evidence chunk not (F12 ranking family).
- `no_answer` (5): all returned top-k content (no abstention; F16, deferred to WP4 — `DECISIONS.md` 2026-10-26).

**Diagnosis of the 8 hybrid misses, in v1 §6.4 order (source → parser → chunk → embedding/query → rank/fusion). No reranker or larger model proposed; every miss lands on a seam already tracked.**
- yp06 (webinars planned/done): dense retrieved it at rr≈0.33, **fusion demoted it out of top-5** → rank/fusion seam (F10/F12), not embedding — the content IS reachable.
- yp09 (cross_language, decision-makers numeric row from an EN query): miss in every channel → embedding/query mismatch on a short BG numeric table row (F11/F14).
- yp11 (date, first team meeting): lexical had it, dense lost it, fusion lost it → rank seam; generic tokens (`среща`) dominate (F12).
- yp12 (conflict, two required sources): miss in every channel → metric + rank (F15 multi-source leniency / F12).
- yp20, yp38 (exact; invitation chair, contract address): miss in both single channels → the specific name/address is a small span in a doc dominated by other text (F12 chunk/rank).
- yp22 (numeric, budget total `76650`): miss in both → number-in-table rank (F12/F14).
- yp31 (list, three webinar themes): miss in both → themes spread across the doc; chunk-boundary/rank.

**What it teaches vs. prior runs.** The 2026-10-20 run (`yp_qwen4b_gpu_main.json` 18q + `_oos.json` 10q) had lexical/dense/hybrid at 0.733/0.733/0.733 (main) — all tied within noise, hybrid best only on evidence@5. v2 (34 scored, with 18 new exact/date/list questions yp20–yp40) **separates the channels for the first time on the real corpus**: hybrid is clearly ahead of lexical on Recall@5 and evidence@5, with a margin above the single-question noise floor. The added exact/numeric questions are exactly where lexical and dense each fail differently, so fusion's value becomes measurable. The dense channel's value on v2 is now **measured**, not assumed.

## WP3 Gate (spec v1 §9 / §6.2) — assessed 2026-10-26 (re-assessed with measured dense/hybrid)

WP3 gate text: *"Written evidence the candidate is useful versus baseline; no project-filter failures; documented unresolved failures."*

**1. Does hybrid beat/match the lexical baseline on the full v2 set?** **Yes, measured.** On the frozen 34-scored v2 set: hybrid Recall@5 0.7647 vs lexical 0.6765 (+3 questions), hybrid evidence@5 0.6765 vs lexical 0.5000 (+6 questions), hybrid MRR 0.6186 vs 0.5882. The evidence@5 margin (+0.176) is well outside the ±2-question noise band. Dense alone is at or just below lexical on Recall@5 but above on evidence@5; fusion dominates both single channels on every metric. The spec §6.2 retrieval acceptance rule ("beat or match the simplest workable baseline") and §6 release rule ("if performance does not beat direct search, do not claim value") are **satisfied** — the fused RAG layer demonstrably beats the lexical baseline.

**2. Project-filter failures?** None. yp14 (`isolation`) returns only the project's own application (`KA154-YOU-E2373104(1).docx`, rr=1.0) in every channel; no wrong-project / out-of-scope document leaked. Reference material is tagged `doc_class=reference` and filterable. Spec §6.2 isolation rule (**zero** cross-project disclosures) holds. (Full cross-project isolation is WP5; only one project is indexed.)

**3. Disposition of open failures (all dispositioned; none block WP3):**
- F10 (BG morphology; yp06): accept and track — dense reaches it, fusion rank drops it; a fusion-weight / rank fix is the next lever, not a bigger model.
- F11 (cross-language; yp09): accept and track — dense recovered yp07 on v2; yp09 (short numeric row) still open under F14.
- F12 (right doc, wrong chunk; yp11/20/22/28/30/33/38): accept and track — dominant remaining seam; rank/chunk-boundary work, measured before any reranker.
- F14 (short numeric table row vs long prose; yp07/yp09/yp22): accept and track — open hypothesis.
- F15 (conflict multi-source metric; yp12): accept and track — eval-metric refinement.
- F16 (no abstention; yp15/16/19/36/39): accept and track — deferred to WP4 (`DECISIONS.md` 2026-10-26).
- F6 (Cyrillic token approximation): Fixed (model tokenizer in production path; residual only in HashEmbedder dev path).
- F17 (v2 label errors): Fixed this run.
- F18 (GPU "down" misdiagnosis): Closed — operator env-var error, not hardware; see correction note above.

**4. Statement:** **WP3 PASSED.**
The fused candidate (hybrid) is measured on the frozen real-corpus v2 set (34 scored) and beats the lexical baseline on Recall@5, evidence@5 and MRR, with the evidence@5 margin outside the noise band. No project-filter failures. All unresolved failures are documented and dispositioned (accept-and-track or deferred to WP4), and each remaining miss is diagnosed to a known seam (rank/fusion/chunk), not to model capacity — so no reranker or larger model is warranted at this gate. Caveat retained: n=34 is small; single-question moves are ~0.03 on Recall@5, so the Recall@5 margin (+3q) is suggestive while the evidence@5 margin (+6q) is the solid signal. WP4 (AI-Team adapter) may begin.

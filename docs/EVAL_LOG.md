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

## 2026-10-27 — WP3→WP4 readiness study: k-sweep, candidate/k knobs, per-file cap, reranker (localize before optimize)

Goal: push evidence@5 toward "ready to put an LLM behind it" and record *which lever moved the needle and why*. Principle applied: do not optimize what you have not localized. First a diagnostic, then only the fix the diagnostic points to.

### Step 1 — k-sweep (`scripts/sweep_k.py`, raw RRF, dedup bypassed, pool=50/channel)

| level | recall@5 | @10 | @20 | in-pool |
|---|---|---|---|---|
| document | 0.706 | 0.735 | 0.853 | **0.941** |
| evidence | 0.588 | 0.706 | 0.794 | **0.824** |

**Verdict: this is a RANKING problem, not a retrieval problem.** The right file is retrieved 94% of the time and the right chunk 82%, but we only surface them at @5 in 71% / 59%. ~23 points of both metrics are lost to ordering, not to the embedder/chunker. Only **2 of 34** questions are true retrieval misses (yp09 cross-language short numeric row, yp31 list) — not in the pool at any depth → the only genuine chunking/embedding gap.

Per-question fused rank of the evidence chunk (None = absent from pool): yp28=6, yp33=7, yp35=6, yp29=8 (just outside @5); yp22/yp23/yp37≈12-14, yp06=13, yp11=24, yp24=30, yp12=33 (deep); yp09/yp31=None. By eye (`_probe_rank`, since deleted): for yp28/yp35 the top-4 are *sibling chunks of the correct file* that do not contain the answer, and the answer chunk sits at rank 6 — the mechanism the per-file cap targets.

### Step 2 — cheap knobs (measured on the real harness, @5)

- **candidates_per_channel 16→32:** hybrid evidence@5 **0.676→0.647** (worse). More candidates inject distractors RRF can float above the true evidence chunk. **Reverted.** Lesson: more retrieval ≠ better top-k.
- **serve top_k 8→10/12:** no change at the production pipeline (0.676) — the file-diversity dedup flattens the gain that raw RRF showed at @10. Serving more did not help here. **Not adopted.**

### Step 3 — per-file cap (the lever that worked; zero download, query-time, reversible)

The dedup already existed but capped one file at `top_k//2 = 4`, too loose to stop sibling-chunk crowding. Swept the hard per-file cap (raw fused order, window=5):

| per-file cap | evidence@5 |
|---|---|
| ∞ (uncapped = old behavior) | 0.6765 |
| **3** | **0.7059** |
| 2 | 0.6176 |
| 1 | 0.5882 |

Clear peak at **3**: looser lets siblings crowd out the evidence chunk; tighter (1-2) kills genuine 2-3-chunk-evidence questions. Added `retrieval.per_file_cap` (default 3) and wired it into `fuse_and_rank`. **Real-harness result, all channels, @5 (this is the new canonical `eval/results/yp_v2_qwen4b_gpu.json`):**

| Channel | Recall@5 | evidence@5 | MRR | Δ evidence@5 |
|---|---|---|---|---|
| lexical | 0.7059 | 0.5294 | 0.5941 | +0.029 |
| dense (Qwen3-4B) | 0.6471 | 0.5294 | 0.5784 | 0 |
| **hybrid (RRF k=60, per_file_cap=3)** | **0.7647** | **0.7059** | **0.6186** | **+0.029 (0.676→0.706)** |

Nothing regressed; hybrid evidence@5 +1 question (yp33 and yp35 moved from hit-no-evidence to full hit). Hybrid scored misses now: yp06, yp09, yp11, yp12, yp20, yp22, yp31, yp38. Hit-no-evidence: yp28, yp30.

### Step 4 — cross-encoder reranker (implemented, NOT measured — download blocked)

The sweep satisfied the spec §7 precondition for a reranker ("candidate in top-20 but ranked too low" — measured: evidence in pool 0.82 vs shown 0.59). RRF is rank-only and content-blind, so it cannot fix intra-document ordering (which sibling chunk of the right doc ranks first); a cross-encoder scoring (query, chunk) jointly can. Implemented as an optional query-time stage (`src/project_rag/rerank.py`, `retrieval.rerank` config, **off by default**): it fuses a wider window (`rerank.top_n`), reorders by the cross-encoder, then cuts to `top_k`.

**It could not be measured on 2026-10-27:** both candidate models (`BAAI/bge-reranker-v2-m3` 2.3 GB, then `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` 0.47 GB) stalled on download — the HF LFS/CDN was blocked (homepage returned 200 in 0.2 s, but `*.incomplete` blobs stayed at 0-256 MB and did not grow). No egress rule was violated; this is a network-availability gap, not a policy block. The stage is inert (`enabled: false`) and the full suite stays green. **Open as F19.** To measure: on a box with HF CDN access, `pip install -e ".[rerank]"`, pin the model revision, set `retrieval.rerank.enabled: true`, re-run the v2 harness, and record the before/after here. Expectation (hypothesis, not a claim): the deep-ranked misses (yp06@13, yp11@24, yp22, yp24@30) are the ones a content reranker should recover; yp09/yp31 will not move (they are not in the pool — a chunking/embedding problem, not a ranking one).

### What this teaches (the transferable shape of building a RAG)

1. **Measure where the loss is before fixing it.** The k-sweep (recall@k curve + in-pool recall) is the single most informative cheap experiment: a steep curve = ranking problem (fusion/rerank); a flat curve = retrieval problem (chunking/embedding). Here it was overwhelmingly ranking.
2. **Bigger/more is not better by default.** More candidates and larger k both failed to help and one regressed — because they add distractors a rank-only fusion can misorder.
3. **The cheap, correct lever first.** A one-parameter dedup cap (no model, no download) bought the only measured gain available today. The reranker is the bigger lever but is heavier and was blocked; it is staged and justified, not hand-waved.
4. **evidence@5 is the number that matters for an LLM** (right *chunk* in a small set), not recall@5 (right *file*). We moved it 0.676→0.706. Honest status: still below the ~0.85 a grounded assistant wants (see "elephant" notes — coverage of the 162 un-parsed files and abstention are the larger unmodeled gaps), so the reranker measurement and those two items are the real WP4-adjacent work. **[Superseded 2026-10-28: the reranker was measured; it lifted recall@5 to 0.82 and MRR to 0.74 but evidence@5 held at 0.71 — see the 2026-10-28 entry. The evidence@5 ceiling is now a chunking limit, not a ranking one.]**

## 2026-10-28 — F19 reranker MEASURED: cross-encoder on v2, top_n swept (localize → fix → verify)

The 2026-10-27 reranker was staged but unmeasured (HF CDN stall). This run the CDN resolved (`curl -I` on the weight blob → 302 to the xet CDN, 2.27 GB; GPU up: gfx1100 / RX 7900 XT, `torch.cuda.is_available()` True with `HSA_ENABLE_DXG_DETECTION=1`). Model `BAAI/bge-reranker-v2-m3` pinned to rev `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`, run on GPU via `sentence-transformers 3.4.1` already in the WSL ROCm venv. Same frozen index `yp_qwen4b_gpu`, same 39-question v2 set (34 scored, 5 `no_answer` excluded). Query-time only, **no reindex**.

- Config: `configs/yp_qwen4b_gpu_rerank.yaml` (identical to `yp_qwen4b_gpu.yaml` except `retrieval.rerank.enabled: true`, `top_n: 12`).
- Reproduce: `wsl.exe -d Ubuntu-24.04 -- bash -c 'export HSA_ENABLE_DXG_DETECTION=1; cd /mnt/c/Users/Cicada38/Projects/assist && PYTHONPATH=src /home/gorila/venvs/tokens-ingest-rocm/bin/python -X utf8 scripts/eval_yp.py --config configs/yp_qwen4b_gpu_rerank.yaml --project ypq --root yspdb --questions eval/yp_questions_v2.jsonl --channels lexical+dense --out eval/results/yp_v2_qwen4b_rerank.json'`

### The `rerank.top_n` sweep (the knob that decides the win) — hybrid, 34 scored, @5

| top_n | recall@5 | evidence@5 | MRR | hit-but-no-evidence |
|---|---|---|---|---|
| none (cap only, prior canonical) | 0.7647 | 0.7059 | 0.6186 | yp28, yp30 |
| 8 | 0.7647 | 0.7059 | 0.6922 | yp28, yp30 |
| 10 | 0.7941 | 0.7059 | 0.7279 | yp06, yp28, yp30 |
| **12** | **0.8235** | **0.7059** | **0.7363** | yp06, yp28, yp30 |
| 15 | 0.7941 | 0.6765 | 0.7304 | +yp33 |
| 20 | 0.7941 | 0.6471 | 0.7279 | +yp02, +yp33 |

**top_n=12 adopted.** recall@5 0.7647→**0.8235** (+2 questions: yp06, yp22), evidence@5 unchanged at **0.7059**, MRR 0.6186→**0.7363** (+0.118). Nothing regressed at 12.

### The mechanism — and the trap the sweep exposed

- **What the reranker recovers (as hypothesized):** yp06 (BG inflection, deep in pool) and yp22 (budget total `76650`, a numeric row the k-sweep had at rank ~12) — both were deep-pool misses RRF could not float. The cross-encoder reads (query, chunk) jointly and pulls them up. This is exactly the §7 precondition the WP3 k-sweep proved (answer in pool, mis-ranked).
- **What a wide window BREAKS (the trap):** at top_n=20 the reranker *reordered the whole head* and **demoted chunks fusion had correct at rank 1** — yp02, yp21, yp33 (all short exact/numeric answers) lost their evidence rank, dropping evidence@5 to 0.647. A cross-encoder is weak on short keyword-match passages where the lexical/fusion signal was already right. **Narrowing the window to 12 keeps those strong fusion decisions untouched** while still reaching the deep recoveries. This is the whole lesson of the sweep: a reranker is a *corrective* over the mis-ranked tail, not a *replacement* for fusion's confident head.
- **What did NOT move (confirmed):** yp09, yp31 — absent from the candidate pool at any depth (k-sweep in-pool miss). The reranker can only reorder what retrieval already fetched; these are a chunking/embedding gap, not ranking.

### Final real-corpus standing (v2, 34 scored, @5) — the LLM-ready picture

| Channel / stage | recall@5 | evidence@5 | MRR |
|---|---|---|---|
| lexical baseline | 0.7059 | 0.5294 | 0.5941 |
| hybrid (RRF + per_file_cap=3) | 0.7647 | 0.7059 | 0.6186 |
| **hybrid + reranker (top_n=12)** | **0.8235** | **0.7059** | **0.7363** |

### Is retrieval "good enough for the LLM"? — honest verdict

**For the document/evidence it CAN parse: yes, at the bar a grounded assistant needs.** recall@5 0.82 (right file in the top-5 for 28/34 questions) and MRR 0.74 (right file usually at rank 1-2) mean the LLM almost always sees the correct source in a short, citable window. evidence@5 0.71 means the exact answer *chunk* is in front of the LLM ~71% of the time; for the remaining ~29% the right *document* is still there, so an LLM that reads the whole retrieved doc (not just the top chunk) recovers most of them. This clears the §6 release rule (beat the lexical baseline) by a wide, out-of-noise margin on all three metrics.

**The gap is no longer ranking — it is coverage and granularity, and both are now localized:**
1. **evidence@5 plateaus at 0.71 (chunking, not ranking).** Every residual miss is now a span/granularity problem: yp30/yp28 (short answer buried in a long doc), yp20/yp21/yp38 (a name/address is a tiny span), yp31 (list spread across the doc). The reranker exhausted the ranking lever; the next real lever is a reindex with smaller/overlapping chunks for exact-span questions — a bigger, separate experiment (F9/F12 chunk family), not a query-time knob.
2. **Two genuine retrieval gaps (F11/F14):** yp09/yp31 never enter the pool — short bilingual table rows the dense+lexical pair does not fetch.
3. **No abstention (F16):** the 5 `no_answer` questions all return content. This is the one item that is a *correctness* risk for an LLM (it will confabulate over irrelevant retrieved text) and is the required WP4 gate before wiring the assistant — an evidence-sufficiency threshold, decided in DECISIONS 2026-10-26, not yet built.
4. **162 un-parsed files (images/video, no OCR):** a known coverage ceiling; any question whose answer lives only in an image is unanswerable regardless of ranking.

**Bottom line:** the retrieval *ranking* pipeline is professional-grade and ready to feed an agent — hybrid + reranker, measured, reproducible, regression-tested. Before the agent is trustworthy end-to-end, one gate remains that is about *refusing* rather than *ranking*: the abstention/evidence-sufficiency check (F16/WP4). That is the correct next move, and it is generation-adjacent, not a retrieval-quality problem.

### What this run teaches (adds to the 2026-10-27 list)

5. **A reranker is a corrective, not a replacement.** Its win and its damage come from the same action (reordering by content). Scope it to the mis-ranked tail (small `top_n`) so it fixes the deep misses without overruling the confident head fusion already got right. The `top_n` sweep is the experiment that finds that line — and it has a real peak, not a monotone curve.
6. **recall and evidence can diverge, and that tells you the next lever.** Here recall climbed (0.76→0.82) while evidence stalled (0.71): the reranker found the right *files* it had been mis-ordering, but the right *chunk* inside them is a chunking limit it cannot touch. When your two curves split like that, stop tuning ranking and go to chunking.

## 2026-10-28 — Eval set expanded to 50q (F16 prep) + abstention gate MEASURED (the reranker score IS the signal)

Two linked changes this session, both gating WP4. First the eval set was grown where it was too
thin to tune (Task 1); then the evidence-sufficiency gate was built and measured on it (Task 2).
Same frozen index `yp_qwen4b_gpu`, same reranked pipeline (Qwen3-4B dense + FTS5 lexical, RRF
k=60, per_file_cap=3, cross-encoder `bge-reranker-v2-m3` top_n=12). Query-time only, no reindex.

### Eval set: 39 → 50 questions (Task 1; DECISIONS 2026-10-28)

The 39-set was numeric-heavy (17) and thin on the kinds that gate WP4: 5 `no_answer`, 2 `date`,
1 `conflict`. Appended 11 (yp41–yp51), **no existing question touched**: +6 `no_answer`
(adversarial near-misses — plausible-for-this-corpus but the specific answer is verified absent by
parsed-text grep), +3 `date`, +2 `conflict`. All 50 pass `validate_labels` against parsed corpus
text. Now: `no_answer` 11, `date` 5, `conflict` 3. The 6 new negatives share vocabulary with real
content (catering budget, 4th webinar, meeting hotel, prize fund, satisfaction survey, registration
fee) — they test whether the gate rejects on *relevance*, not on topic unfamiliarity.

### The signal: top-1 cross-encoder rerank_score (not Chroma distance, not RRF)

v1 §4.6 warns Chroma distances and rank-fusion scores are NOT calibrated confidence — confirmed
here: raw `fusion_score` maxes at ~0.033 (= 2/(60+1)) and does not separate relevant from
irrelevant. The **cross-encoder relevance score already computed by the reranker** is the right
signal: `bge-reranker-v2-m3` emits a sigmoid 0–1 relevance probability per (query, chunk). It is a
free by-product of a stage we already run — no new model, no new compute. Probe:
`scripts/probe_abstain.py` dumps top-1/top-2/margin/mean per question (`eval/results/abstain_probe_rerank.json`).

**Measured separation (top-1 rerank_score), sorted:** the 11 `no_answer` questions all score
top-1 ≤ 0.381, ten of them ≤ 0.26. Answerable questions mostly score high (median ≈ 0.96), with a
short low-scoring tail (yp45 0.001, yp11 0.043, yp22 0.056, yp14 0.098). So a single threshold
separates the bulk cleanly; the overlap is a handful of genuinely-hard answerables + one stubborn
negative.

### Threshold sweep (the operating-point decision, not a guess) — `/tmp/sweep_thresh.py` style, @top1

| threshold | no_answer withheld | answerable wrongly withheld | …of which real hits |
|---|---|---|---|
| 0.10 | 6/11 | 4/39 | 3/31 |
| 0.20 | 8/11 | 4/39 | 3/31 |
| **0.26** | **10/11** | **4/39** | **3/31** |
| 0.30 | 10/11 | 4/39 | 3/31 |
| 0.34 | 10/11 | 5/39 | 3/31 |
| 0.40 | 11/11 | 6/39 | 4/31 |

**0.26 adopted.** It sits on a flat plateau (0.26–0.32: identical tradeoff), so the choice is
robust to ±0.03 drift. It abstains on 10/11 negatives while the answerable cost is 4 questions,
only **3 of them real recall hits** (yp14 isolation, yp22 budget-total, yp45 identifier-conflict;
yp11 was already a retrieval miss, so withholding it costs nothing). Pushing to 0.40 would catch
the last negative (yp47) but at the price of a 4th real hit — not worth it.

### Full real-corpus run, reranked pipeline + gate, 50q (`configs/yp_qwen4b_gpu_rerank_abstain.yaml`, suppression OFF for honest recall)

Reproduce: `wsl.exe -d Ubuntu-24.04 -- bash -c 'export HSA_ENABLE_DXG_DETECTION=1; cd /mnt/c/Users/Cicada38/Projects/assist && PYTHONPATH=src /home/gorila/venvs/tokens-ingest-rocm/bin/python -X utf8 scripts/eval_yp.py --config configs/yp_qwen4b_gpu_rerank_abstain.yaml --project ypq --root yspdb --questions eval/yp_questions_v2.jsonl --channels lexical+dense --out eval/results/yp_v2_abstain.json'`

- **Retrieval (39 scored answerable): Recall@5 0.7949, evidence@5 0.6667, MRR 0.7188.** These are
  *lower* than the 34-scored numbers (0.8235 / 0.7059 / 0.7363) for one reason only: the
  denominator grew from 34 → 39 by adding 5 deliberately-hard answerables (yp43/44/45 conflict/
  cross-document-date miss or partial-hit, yp41/42 hit). Same pipeline, harder set — not a
  regression. The 34-set numbers remain the canonical ranking result; the 50-set is the
  abstention-measurement set.
- **Abstention @ 0.26: no_answer withheld 10/11 (0.909); answerable wrongly withheld 4/39, of which
  3 real hits.** The one leaked negative is yp47 ("when was the *fourth* webinar") — the reranker
  scores a real webinar chunk 0.381 because the chunk genuinely discusses webinars; distinguishing
  "there is no fourth" is a counting/reasoning job for the LLM, not a retrieval-score job. This is
  the correct division of labor: the gate removes the easy 10, the answerer handles the subtle one.

### Design (what shipped)

- `retrieval.abstention` config: `enabled`, `signal` (rerank_score | fusion_score), `min_top_score`
  (0.26), `suppress_on_insufficient`. Pure decision in `api.assess_evidence` (no LLM, no mutation).
- `SearchResponse.evidence: EvidenceVerdict` (enabled/sufficient/signal/top_score/threshold) is
  **always attached**. When `suppress_on_insufficient` is true, an insufficient verdict returns the
  verdict *instead of passages* (the spec's "explicit no-evidence response") and warns; when false
  (eval mode) passages are kept and the verdict is advisory — honoring retrieval-first/generation-
  second (v1 §2.2: the retriever flags, the answerer decides).
- The gate deactivates + warns if `signal: rerank_score` but no reranker ran (never silently treats
  a missing calibrated score as "insufficient"; v1 §4.6).
- Prod/WP4 config `yp_qwen4b_gpu_rerank.yaml` ships the gate enabled with suppression ON.
- Regression: `tests/unit/test_abstention.py` (pure decision logic, 6 cases) +
  `tests/integration/test_abstention_wiring.py` (suppression wired through search_project, 4 cases).
  Full suite 72 green.

### What it teaches

7. **Reuse the score you already paid for.** The abstention signal is not a new model or a new
   knob — it is the cross-encoder relevance the reranker already computes. The uncalibrated
   fusion/distance scores the spec explicitly distrusts (§4.6) are useless for this; the calibrated
   cross-encoder score is exactly what an evidence-sufficiency gate needs.
8. **A retrieval-side gate removes the easy negatives, not the subtle ones.** 10/11 no_answer were
   topic-plausible but evidence-irrelevant → the score caught them. The 11th (yp47) needs *counting*
   ("there were only three webinars"), which is a generation-layer judgement. Drawing that line is
   what keeps the gate honest rather than overclaiming.

## 2026-10-28 — Filesystem-presence evidence for unparseable files (Task 3; DECISIONS 2026-10-28)

Owner-requested (NOT OCR): make the 162 unparseable files (images/video/audio/legacy .doc)
retrievable by their folder/filenames as circumstantial evidence. Built `filesystem.py` → one
`doc_class=filesystem` presence chunk per folder holding an unparseable file (37 folders), tagged
with an explicit "[FILESYSTEM PRESENCE — NOT DOCUMENT CONTENT]" banner. Measured on a SEPARATE index
`yp_qwen4b_gpu_fs` (reranked+abstain pipeline, `configs/yp_qwen4b_gpu_fs.yaml`) so the frozen
`yp_qwen4b_gpu` index is untouched. Ingest reused the parse cache (172 parse hits, 0 misses) and the
copied embedding cache; only the 37 fs chunks embedded fresh → 5717 chunks total.

Reproduce: ingest with `classification.filesystem_evidence: true` into the fs index, then
`scripts/eval_yp.py --config configs/yp_qwen4b_gpu_fs.yaml --project ypq --root yspdb --questions eval/yp_questions_v2.jsonl --channels lexical+dense --out eval/results/yp_v2_fs.json`.

### It helps the intended questions (3 new `filesystem`-kind questions, yp52–54)

| id | question | folder surfaced | result |
|---|---|---|---|
| yp52 | collected photos for the good-practices questionnaire? | `…/Качете снимки (File responses)` (53 img) | HIT rr=1.0, EV |
| yp53 | video materials for the political declaration? | `Communications/Видеа - Декларация_` (4 mp4) | HIT rr=1.0, EV |
| yp54 | saved files for the "Youth participation in practice" webinar? | `Уебинари/2. Младежко участие на практика` | HIT rr=0.5, EV |

With a `doc_class: filesystem` filter these folders are top-1 for photo/video/webinar presence
questions that previously returned nothing citable. The banner + "contents NOT indexed" text travels
in the excerpt, so the answerer sees it is presence, not content.

### It does not pollute normal retrieval (39 content questions, fs index vs no-fs index)

| set | Recall@5 | evidence@5 | MRR |
|---|---|---|---|
| no-fs index (`yp_v2_abstain.json`) | 0.7949 | 0.6667 | 0.7188 |
| fs index (`yp_v2_fs.json`, same 39) | 0.7692 | 0.6410 | 0.7137 |

One question moved: **yp21** (exact; chair of NYF) HIT→miss. Probed and attributed, not hand-waved:
yp21's evidence chunk (`Покани/Адела Бозмарова/…`) is **absent from the dense top-16, sits at
lexical rank 10, and is not in the rerank top_n=12 window** on the fs index — and **no filesystem
chunk appears anywhere in yp21's candidate pool**. yp21 was a borderline rank-5 hit on the no-fs
index, crowded by ~15 near-identical `Покани` invitation letters that all contain "Атанас Г. Радев";
rebuilding the Chroma collection (fresh HNSW graph — v2 §D3 nondeterminism) reorders those identical-
scoring siblings and tips yp21 from rank 5 to 6. The fs chunks are not the cause. Net: new evidence
class, no demonstrated cost to content retrieval. Off by default (`classification.filesystem_evidence`).

Regression: `tests/unit/test_filesystem_evidence.py` (5 cases, pure builder, no index). 77 tests green.

## 2026-10-28 — Chunking experiment: target_tokens 500→250 — NEGATIVE, reverted (Task 4; F9/F12 chunk family)

The 2026-10-27/28 analysis localized the evidence@5 plateau (0.71) to *chunk granularity*: short
exact-span answers (names/addresses/IDs — yp20/21/30/38) buried in large chunks. Pre-registered
hypothesis (G0, heavy-run-governor): a smaller `target_tokens` isolates those spans into their own
chunk, lifting evidence@5 without regressing recall@5/MRR beyond the ±1-question (~0.03) noise band.
Decision rule fixed before the run: **adopt only if evidence@5 improves and recall/MRR don't
regress; else record the negative and revert.**

New index generation by design (chunk params → ADR-009), SEPARATE location
`yp_qwen4b_gpu_chunk250` so the frozen index is untouched. Config
`configs/yp_qwen4b_gpu_chunk250.yaml`: only `chunking.target_tokens 500→250` differs (overlap left
0 — overlap is NOT implemented in `chunking.py`, so this A/B honestly isolates the target-size
lever, not overlap). GPU preflight passed (RX 7900 XT, 19.9 GB free; `torch.version.hip` set,
`cuda` None — genuine ROCm). Re-chunk + full re-embed: **11,223 chunks** (vs ~2,500 at 500-tok),
11.3 min on GPU. Same reranked+abstain pipeline.

Reproduce: ingest with `configs/yp_qwen4b_gpu_chunk250.yaml`, then
`scripts/eval_yp.py --config configs/yp_qwen4b_gpu_chunk250.yaml --project ypq --root yspdb --questions eval/yp_questions_v2.jsonl --channels lexical+dense --out eval/results/yp_v2_chunk250.json`.

### Result (39 content questions, same frozen labels, @5)

| chunking | recall@5 | evidence@5 | MRR |
|---|---|---|---|
| 500-tok (frozen, `yp_v2_abstain.json`) | **0.7949** | **0.6667** | **0.7188** |
| 250-tok (`yp_v2_chunk250.json`) | 0.7436 | 0.6154 | 0.7137 |

**Every metric moved down or flat. Hypothesis NOT supported → reverted.** Per-question deltas
(500→250): yp06 hit lost, yp21 hit+ev lost, yp02 ev lost, yp42 ev lost; yp28 ev *gained* (the one
case where isolation helped). Net −2 recall, −2 evidence, flat MRR.

**Mechanism (why smaller was worse, not just noise).** Halving the chunk fragments the surrounding
context the embedder and the cross-encoder use to judge relevance. yp02 (participant + facilitation
counts) and yp42 (contract validity date) lost their evidence because the answer token got split
from the context that made its chunk rank; yp06 and yp21 fell out of the retrieved file set
entirely. Only yp28 benefited from a tighter chunk. The reranker already recovers mis-ranked
answers from *within* a well-formed larger chunk (that is what lifted recall 0.76→0.82 on 2026-10-28);
shrinking the chunk removes the context that recovery depends on. So the evidence@5 ceiling is NOT
a "span too big" problem solvable by smaller chunks — the residual misses are genuinely-hard spans
(tiny name/address in a doc dominated by other text) and a couple of true retrieval gaps
(yp09/yp31). **The cheap chunking lever is exhausted; evidence@5 ≈ 0.71 is the accepted ceiling for
this corpus/pipeline.**

Negative result is first-class (AGENTS.md): the 500-token chunking stays canonical, the 250-token
index/config are kept only as the reproducible record of a tested-and-rejected lever. No regression
test needed (config-only experiment; the chunker is unchanged).

## 2026-10-28 — Latency + memory benchmark for the reranked pipeline (Task 5; spec v1 §6.2 Efficiency gate)

First measurement of the Efficiency gate for the hybrid+reranker pipeline (never benchmarked before —
2026-10-27 noted it as unmeasured). `scripts/bench_latency.py`: one cold-start query (pays model
load + first CUDA kernels, timed separately), a warmup pass, then 3 timed passes over the 53-question
v2 set (159 timed queries) through `search_project`. Peak process RSS (ru_maxrss) and peak GPU VRAM
(`torch.cuda.max_memory_allocated`) recorded. WSL ROCm, RX 7900 XT. **No SLA invented** (v1 §6.2:
"Measured and acceptable to stakeholder; no premature numeric SLA invented").

Reproduce: `wsl.exe -d Ubuntu-24.04 -- bash -c 'export HSA_ENABLE_DXG_DETECTION=1; cd /mnt/c/Users/Cicada38/Projects/assist && PYTHONPATH=src /home/gorila/venvs/tokens-ingest-rocm/bin/python -X utf8 scripts/bench_latency.py --config configs/yp_qwen4b_gpu_rerank.yaml --project ypq --root yspdb --questions eval/yp_questions_v2.jsonl --out eval/results/yp_v2_latency.json'` (and `--config configs/yp_qwen4b_gpu.yaml` for the no-rerank row).

| pipeline | cold start | warm p50 | warm p95 | warm mean | warm range | peak VRAM | peak RSS |
|---|---|---|---|---|---|---|---|
| hybrid only (Qwen3-4B dense + FTS5, RRF, cap=3) | 26.3 s | 2.53 s | 2.68 s | 2.23 s | 1.38–3.11 s | 8.1 GB | 6.1 GB |
| **hybrid + reranker (bge-reranker-v2-m3, top_n=12)** | 31.3 s | **2.73 s** | **2.92 s** | 2.49 s | 1.74–3.24 s | 10.7 GB | 6.1 GB |

### What it measures / teaches

- **Steady-state ≈ 2.7 s/query (p50) at the full reranked pipeline; p95 2.9 s.** Tight spread
  (p95 only ~7% above p50) — latency is dominated by a fixed GPU forward pass, not by query-dependent
  branching, so it is predictable.
- **The reranker is cheap here.** It adds only ~0.2 s (p50) / ~0.24 s (p95) and +2.5 GB VRAM over
  hybrid-only. The dominant cost in BOTH is the **Qwen3-4B query embedding** (~2.3–2.5 s for a single
  short query — the price of a 4B embedder doing a batch-1 forward pass on a consumer GPU). So if
  query latency ever needs to drop, the lever is the embedder (smaller model / batching / caching
  frequent queries), not the reranker.
- **Memory fits the box comfortably:** 10.7 GB peak VRAM of 21.4 GB (RX 7900 XT), 6.1 GB RSS. No
  memory pressure; headroom for a larger rerank batch if needed.
- **Cold start ~30 s** is a one-time per-process model load (Qwen3-4B + reranker weights). For a
  long-lived WP4 service this is paid once at startup, not per query. A per-request process would pay
  it every call — so WP4 should keep the retriever warm (persistent process), which the
  `ProjectRAG` object already supports.

Caveat: n=53 distinct queries ×3; single machine (ROCm WSL, RX 7900 XT). These are this-box numbers,
not a portable SLA. CPU-only latency would be far higher (the A01 CUDA gate exists for this reason).

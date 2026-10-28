# Decisions log

Short, dated, unspecified-step design choices (AGENTS.md "Pivot rule"). Spec-contradicting
changes go in `docs/AMENDMENT_A0N.md`, not here. Failures go in `docs/FAILURE_LEDGER.md`.

## 2026-10-26 — v2 question-label fixes (WP3, Task 1)

The eval validator (`scripts/eval_yp.py:validate_labels`, which checks every `expect_substring`
against the parsed text of its `expect_source_path`) rejected 14 labels in the first author draft
of `eval/yp_questions_v2.jsonl`. Diagnosed each in source→substring order; no question deleted.

- **Path errors (substring present, path wrong)** — corrected to the real `yspdb/` location:
  - yp20, yp21: `Покани/ПОКАНА-АДЕЛА-БОЗМАРОВА.docx` → `Покани/Адела Бозмарова/ПОКАНА-АДЕЛА-БОЗМАРОВА.docx`
  - yp24, yp25: bare filename → `Communications/Статии/Поглед към Национална среща „Младежко участие“ в Ямбол.docx`
  - yp26: bare filename → `Communications/Прессъобщения/ПРЕССЪОБЩЕНИЕ-Декларация.docx`
  - yp27, yp28: `АНАЛИЗ_ММП.docx` → `Проучване някакво___/АНАЛИЗ_ММП.docx`
  - yp29, yp38: `05-YP-2025_Tsveti.docx` → `Administration YP/05-YP-2025_Tsveti.docx`
  - yp31: `Уебинари/Уебинар 2.docx` (does not exist) → `Communications/Постове/Уебинар 2.docx` (contains `Младежкото участие на практика`)
  - All corrected substrings verified present in the parsed text of the corrected source.
- **Absent substring (fabricated value)** — yp30 expected `2024-3-BG01-KA154-YOU-000283706`.
  `grep -rl 000283706 yspdb` returns nothing; the string is in NO parsed corpus file. The
  proposal's actual identifier is the application Form ID `KA154-YOU-E2373104` (present on every
  page). Query reworded to "application form ID" and the substring replaced with
  `KA154-YOU-E2373104`. The grant-agreement number is simply not in this corpus.

After fixes the validator passes all 39 questions. Tracked as ledger **F17** (labels authored
from assumption, not from parsed text — the exact anti-pattern the validator exists to catch).

## 2026-10-26 — Abstention policy for `no_answer` questions (F16) — DEFERRED to WP4

**Current behavior (measured, v2 run 2026-10-26):** retrieval always returns top-k regardless of
relevance. All 5 `no_answer` questions (yp15, yp16, yp19, yp36, yp39 — weather, drones, vegetarians,
buses, vegetarian meals: none in the corpus) returned a full top-5 of plausible-but-irrelevant
chunks. There is no "no evidence" signal. These are excluded from the recall denominator so they
do not distort retrieval metrics, but that is a measurement convenience, not a policy.

**Two candidate policies considered:**
- **(a) Score-threshold cutoff** — abstain when the top fused score (or per-channel best) is below
  a calibrated threshold. Pure-retrieval, no LLM, testable now. Risk: BM25/cosine scores are not
  calibrated across query types or languages; a single threshold that abstains correctly on the 5
  no-answer questions without suppressing the hard-but-answerable ones (yp06/yp07/yp09) is unlikely
  on n=5 negatives — would overfit. Choosing the threshold needs many more negative examples.
- **(b) Evidence-sufficiency check by the LLM at WP4** — the agent inspects the retrieved passages
  and states "no supporting evidence found" rather than answering. Matches the spec's own
  abstention gate (§6.2: "Explicit no-evidence response, not plausible fabricated answer") and the
  retrieval-first/generation-second rule (the retriever is not supposed to decide truth; v1 §2.2).

**Decision:** Defer to **WP4**, policy **(b)**. Reasons, kept brief for a WP3 log:
1. WP3 is retrieval-only; abstention is fundamentally a generation-layer behavior (the retriever
   returning candidates and the answerer declining to answer are different concerns).
2. n=5 negatives cannot calibrate a score threshold without overfitting; a threshold added now
   would be unjustified tuning (AGENTS.md: every deviation defensible with evidence).
3. The spec's abstention acceptance criterion is written against the *answer*, not the ranked list.
Until WP4, `no_answer` questions stay out of the recall denominator and are tracked under F16.
A score-threshold experiment (a) remains a fallback if WP4 evidence-sufficiency proves insufficient.

## 2026-10-26 — Retire the synthetic question file (WP3, Task 7)

`eval/questions.jsonl` held 15 questions against a synthetic "Green Villages" fixture corpus that
is not the real corpus and is not ingested in any real-corpus index. Keeping it beside the real
`yp_questions*.jsonl` sets invites running it against the wrong index. Moved to
`eval/archive/questions_synthetic_gv.jsonl` (kept, not deleted — it is still the fixture for the
synthetic unit/integration tests). Active real-corpus question sets are `eval/yp_questions.jsonl`
(original 16), `eval/yp_questions_oos.jsonl` (out-of-sample 10), and `eval/yp_questions_v2.jsonl`
(expanded 39).

## 2026-10-26 — GPU env is set by code, not by the shell (ROCm-on-WSL)

Evidence: dense/hybrid eval was reported "GPU down" (F18). Real cause: ROCm 7.2.4 under WSL needs
`HSA_ENABLE_DXG_DETECTION=1`; it was exported only in `~/.bashrc`, which non-interactive
`wsl -- bash -c` shells never read, so `rocminfo` → `hsa_init Failed` and
`torch.cuda.is_available()` → False. Same shell with the var: gfx1100 (RX 7900 XT) visible, torch
True. Decision: `project_rag.gpu_env.prepare_gpu_env()` (called before `import torch` in the Qwen3
embedder and `scripts/feasibility_smoke.py`) sets the var when running in WSL with `/dev/dxg`;
`describe_gpu_env()` is appended to the CUDA-gate error so the failure names its own cause. The
gate stays fail-closed (no CPU fallback). Reversible: delete the two calls. Touches A01 §E only as
diagnostics; no behavior change when the GPU is truly absent.

## 2026-10-27 — Retrieval ranking tuning (WP3→WP4 readiness; all evidence in EVAL_LOG 2026-10-27)

Context: the k-sweep localized the loss to *ranking* (answer in pool 0.82–0.94, shown at @5 only
0.59–0.71), not retrieval. Three tuning choices followed, each measured on the frozen v2 set:

- **candidates_per_channel 16→32: REJECTED.** Hybrid evidence@5 regressed 0.676→0.647 — extra
  candidates are distractors a rank-only fusion can float above the true chunk. Kept at 16.
- **serve top_k 8→10/12: NOT ADOPTED.** No change in the production pipeline (the file-diversity
  dedup flattens the gain raw RRF showed at @10).
- **per_file_cap = 3: ADOPTED** (`retrieval.per_file_cap`, default 3, query-time, reversible).
  Swept {∞,3,2,1}; evidence@5 peaks at 3 (0.706 vs 0.676 uncapped; 2→0.618, 1→0.588). Mechanism:
  caps how many chunks of one source file may occupy the result window so sibling chunks cannot
  crowd out the file's evidence chunk (yp28/yp35). Moved yp33/yp35 to full hit, nothing regressed.
  Reversible: set `per_file_cap: null` to restore the legacy `top_k//2` behavior. Design choice
  under spec §4.7 (file-diversity dedup existed; this parameterizes its cap), not a spec change.

- **Cross-encoder reranker: STAGED, not enabled, not measured.** Implemented (`rerank.py` +
  `retrieval.rerank`, off by default) because the k-sweep met the spec §7 precondition
  ("candidate in top-20 but ranked too low"). Could not be measured — HF CDN blocked both model
  downloads (F19). Default-off so it changes nothing until a box with model access enables and
  measures it. This is the next real lever and the one most likely to move evidence@5 further;
  it is deliberately left justified-but-unverified rather than claimed to work.

- **Cross-encoder reranker: ENABLED and MEASURED (supersedes the "STAGED" note above), 2026-10-28.**
  HF CDN reachable this session; `BAAI/bge-reranker-v2-m3` pinned (rev `953dc6f…`) and run on GPU.
  Swept `rerank.top_n` {8,10,12,15,20}: peak at **12** — recall@5 0.7647→0.8235, evidence@5 holds
  0.7059, MRR 0.6186→0.7363. Adopted top_n=12 in `configs/yp_qwen4b_gpu_rerank.yaml` and as the
  `RerankConfig` default. **Why 12, not more:** top_n≥15 lets the cross-encoder demote fusion's
  strong short-chunk head (yp02/yp21/yp33 lose evidence), eroding evidence@5. A reranker is a
  corrective over the mis-ranked tail, not a replacement for the confident head — scope it narrow.
  The reranker is a query-time stage (no reindex), offline (local HF cache, `trust_remote_code=False`),
  pinned revision, regression-tested (`tests/unit/test_rerank.py`, fake scorer, no download).
  Reversible: `retrieval.rerank.enabled: false` restores the per_file_cap-only pipeline.
  Design choice under spec §7 (reranker preconditioned on the measured k-sweep), not a spec change.

# WP4-readiness close-out — the retrieval layer, measured against every applicable gate

> **UPDATE 2026-10-30 — WP4 is now BUILT and proven.** This document is a frozen point-in-time
> *retrieval-readiness* assessment written before wiring; its "stop line / no LLM is called"
> statements describe that moment only. WP4 is now live: RAG is exposed as a stdio MCP server
> (`project_rag.mcp_server`) and a real Claude agent grounds on it end-to-end (see
> `docs/DECISIONS.md` 2026-10-30, ledger F21/F22, `scripts/eval_agent.py`). Read this file for the
> retrieval-gate verdict, not for current WP4 status.

**Date:** 2026-10-28; **round-2 refresh 2026-10-29** · **Verdict:** the retrieval layer is
**ready for an LLM to be wired (WP4)**, with items accepted-and-tracked, not blocking. · **Scope:**
this document assesses the *retrieval* subsystem only. It does **not** cross the stop line — no LLM
is called, no AI-Team tool is registered, no generation/grounding/citation-answer path is built.
Those are WP4, the next person's job.

## Round-2 refresh (2026-10-29) — what A–E changed, and the honest go/no-go

A targeted round of retrieval polish + a trust-the-eval push, then stop (the push-vs-diminishing-
returns decision was already made: one round, no chasing the metric). Verify with `git log` +
`pytest` (90 green at this refresh), not this prose.

| Task | Result | Adopted? |
|---|---|---|
| **A — parent-section context expansion** (v1 §7, query-time, no reindex) | span@5 **0.667→0.718** (+2q: yp28/yp33) on the 39-scored set; attaches surrounding-section text per served chunk for the LLM | **Yes** — on in `yp_qwen4b_gpu_rerank.yaml`; `tests/integration/test_context_expansion.py` |
| **B — cross-language query-expansion glossary** (F11/F14) | **NEGATIVE**: evidence@5 0.619→0.595, yp09 unmoved, yp02 regressed (word-sense miss + common-phrase distractors) | **No** — kept off-by-default as the reproducible record; `tests/unit/test_query_expansion.py` |
| **C — owner-drivable retrieval session** (`scripts/ask.py`) | Warm-retriever CLI: owner asks his own questions, sees citations + evidence verdict + section context, logs a 1–5 usefulness rating to JSONL. The honest eval when there is no gold set, and the demo backbone. | **Yes** — `tests/unit/test_ask_session_log.py` |
| **D — holdout report-section eval** (12 questions from `holdout-report/`, validated vs parsed `yspdb/`) | The better-trusted eval: recall@5 **0.417**, evidence@5 **0.083** — far below the self-authored 0.795/0.667. Exposes the real report-drafting gap (reference-material bleed → use `doc_class` filter; multi-document synthesis → LLM's job). Held for owner review, **not merged**. | artifact `eval/yp_report_questions_candidate.jsonl` |
| **E — this refresh** | A–D folded into the go/no-go below; WP4 wiring investigated in `docs/WP4_WIRING_OPTIONS.md` (no wiring done) | — |

**Updated go/no-go.** **GO for WP4 wiring, unchanged** — the retrieval layer still clears every
applicable §6.2 gate (below), and round 2 added (A) richer LLM-facing context and (C) the honest
owner-eval surface. The **sobering, honest** addition is Task D: on independently-derived
report-section questions the pinpoint-fact strength (recall@5 0.82 on the self-authored set) does
**not** carry to diffuse multi-document synthesis (recall@5 0.42, evidence@5 0.08). That is **not a
retrieval regression and not a wiring blocker** — it localizes the remaining work to the *answerer*:
(i) scope with `doc_class: project_record` to stop reference-material bleed (verified to fix the
rp10/rp11 bleed), (ii) read the whole retrieved doc + parent-section `expanded_context` (Task A), and
(iii) synthesize across the multiple docs retrieval does surface. All three are WP4 answerer
concerns. The retrieval *ranking/chunking* levers are exhausted and documented; the correct next
move is wiring the LLM (see `docs/WP4_WIRING_OPTIONS.md`), with the single owner decision being
egress (local Ollama vs hosted Claude). **Do not spend another round chasing the report-set number
by tuning retrieval — it is a synthesis gap, not a ranking gap.**

Read alongside: `docs/EVAL_LOG.md` (the measured runs, newest entries 2026-10-26→28),
`docs/FAILURE_LEDGER.md` (F1–F19), `docs/DECISIONS.md`. Verify with `git log` and `pytest` (90 green
at the 2026-10-29 refresh), not this prose.

## The pipeline being handed off

```
query → Qwen3-Embedding-4B (dense, 2560-dim, ROCm GPU)  ┐
        SQLite FTS5 BM25 (lexical)                       ├→ RRF fusion (k=60)
                                                         ┘   → per_file_cap=3 dedup
     → cross-encoder rerank (bge-reranker-v2-m3, top_n=12)
     → evidence-sufficiency gate (top-1 rerank_score ≥ 0.26)
     → SearchResponse{results, evidence verdict, trace}
```
Config: `configs/yp_qwen4b_gpu_rerank.yaml` (reranker + abstention gate on, suppression on).
Frozen real-corpus index `yp_qwen4b_gpu` (project `ypq`, `yspdb/`, 334 files → 169 READY).

## Final measured numbers (real corpus, no LLM)

**Ranking** — frozen 34-scored v2 set (the canonical ranking result, EVAL_LOG 2026-10-28):

| stage | recall@5 | evidence@5 | MRR |
|---|---|---|---|
| lexical baseline | 0.706 | 0.529 | 0.594 |
| hybrid (RRF + per_file_cap=3) | 0.765 | 0.706 | 0.619 |
| **hybrid + reranker (top_n=12)** | **0.824** | **0.706** | **0.736** |

**Abstention** — expanded 50-question set (11 no_answer incl. 6 adversarial near-misses), gate on
top-1 rerank_score ≥ 0.26 (EVAL_LOG 2026-10-28): **no_answer correctly withheld 10/11 (0.91)**;
answerable wrongly withheld 4/39, of which **3 real recall hits** (yp14, yp22, yp45).

**Efficiency** — RX 7900 XT, 159 timed queries (EVAL_LOG 2026-10-28): warm **p50 2.73 s / p95 2.92 s**
per query, cold start ~31 s (one-time model load), peak VRAM 10.7 GB, peak RSS 6.1 GB.

## Standing against spec v1 §6.2 gates

Six gates are the retrieval layer's to satisfy before WP4; two (Grounding, Integration) are WP4's
and are explicitly out of scope here.

| §6.2 gate | Verdict | Evidence |
|---|---|---|
| **Parsing** | **PASS** | 169/172 text files READY; 3 OCR_REQUIRED, unparseable reported not silently skipped. Per-format fidelity fixed and tested (F2 DOCX tables, F3/F7 PDF/XLSX, F5 oversized). Tests: `test_docx_tables`, `test_xlsx_headers`, `test_parsing`, `test_chunk_oversized`. **Accepted gap:** 162 image/video/legacy files have no OCR/transcription (out of scope) — now *partially* mitigated by filesystem-presence evidence (F8, Task 3). |
| **Indexing** | **PASS** | Idempotent re-ingest, stale removal on modify/delete/shrink, model/config mismatch refused, chunker-config change forces new generation (ADR-009). Tests: `test_index_lifecycle` (7 cases), `test_a01_stage_reuse`, `test_embedding_cache`, `test_chroma_batching`. |
| **Retrieval** | **PASS** | Hybrid+reranker beats the lexical baseline on all three metrics with the margins well outside the ±1-question (~0.03) noise band (recall +0.118, evidence +0.177, MRR +0.142 vs lexical). Satisfies the §6 release rule ("beat direct/simple search or stop"). Residual misses diagnosed to known seams, not model capacity. Tests: `test_retrieval_isolation_security` (exact/paraphrase/BG/table/empty), `test_rerank`, `test_doc_class_and_channels`. |
| **Isolation** | **PASS** | Zero cross-project leakage: `test_project_isolation_zero_leakage` (two projects, wrong-project query returns nothing). Server-side project scope (never trusts client `project_id`), reference material filterable by `doc_class`. Path-traversal rejected (`test_open_source_path_traversal_rejected`). Caveat: full cross-project proof is WP5 (only one real project indexed). |
| **Abstention** | **PASS** (F16 closed) | Retrieval-side evidence-sufficiency gate on the calibrated cross-encoder score; withholds passages + emits an explicit "insufficient evidence" verdict. Measured 10/11 no_answer abstained at a cost of 3 real-hit false abstains. Tests: `test_abstention` (6), `test_abstention_wiring` (4). |
| **Efficiency** | **PASS (measured, no SLA)** | p50 2.73 s / p95 2.92 s per query, 10.7 GB VRAM, 6.1 GB RSS on this box. Reported as-measured per §6.2 ("no premature numeric SLA invented"). Dominant cost is the 4B embedder, not the reranker (+0.2 s). |
| Grounding | **N/A — WP4** | Requires an LLM answering; stop line. |
| Integration | **N/A — WP4** | Requires AI-Team tool registration; stop line. |

## Accepted-and-tracked (do not block WP4; the LLM handles or inherits them)

1. **evidence@5 ceiling ≈ 0.71 (chunking/coverage, localized and exhausted).** For ~29% of
   answerable questions the exact answer *chunk* is not in the top-5, though the right *document*
   usually is (recall@5 0.82) — so an LLM that reads the whole retrieved document recovers most.
   Both levers were pushed to their end: the reranker exhausted ranking (F19), and a smaller-chunk
   reindex was tested and **rejected** (Task 4 — evidence@5 fell to 0.615; smaller chunks fragment
   the context the embedder/reranker score on). The residual misses are genuinely-hard tiny spans
   (yp20/38 name/address) — accepted, not an open fixable seam.
2. **Two true retrieval gaps (F11/F14): yp09, yp31.** Short bilingual table rows the dense+lexical
   pair never fetches into the candidate pool (confirmed by the k-sweep). A chunking/embedding
   limit, not ranking. Small and known.
3. **One adversarial no_answer leaks the gate (yp47, "the fourth webinar").** The reranker scores a
   real webinar chunk 0.38 because the chunk genuinely discusses webinars; knowing "there were only
   three" is a *counting* judgement — correctly the LLM's job at WP4, not a retrieval-score job.
4. **Abstention false-abstains on 3 real hits (yp14, yp22, yp45).** The price of catching 10/11
   negatives at threshold 0.26 (a measured knee on a flat plateau). Tunable via
   `retrieval.abstention.min_top_score` if WP4 prefers recall over caution.
5. **162 unparseable files have no content indexed** (OCR/transcription out of scope). Their
   *presence* is now retrievable (`doc_class=filesystem`, Task 3) as clearly-labelled circumstantial
   evidence; their *content* is not. A permanent coverage ceiling for image-only answers.

## What WP4 inherits (the contract the LLM binds to)

- `search_project(project_id, query, top_k, filters)` → `SearchResponse` with `results[]`
  (excerpt + citable source_path/location + per-channel ranks), **`evidence` verdict**
  (`sufficient`, `top_score`, `threshold`), and `warnings`. When evidence is insufficient the gate
  returns the verdict *instead of* passages — the LLM should surface "Not established from accessible
  project documents" (v1 §5.2), not answer.
- `open_source(...)` for exact-span verification; `get_ingestion_status(...)` for coverage health.
- Each result carries `expanded_context` (parent section around the chunk; Task A) when
  `retrieval.context_expansion.enabled` — give the answerer this, not just the top-chunk excerpt,
  to recover answers the chunk alone did not hold.
- `filters={"doc_class": "reference"|"project_record"|"filesystem"}` to scope by material class —
  **use `project_record` for report-drafting** to stop third-party reference material bleeding into
  answers (measured fix for the Task D rp10/rp11 bleed).
- Every query emits a `QueryTrace` (hashed query, retrieved chunk ids, channels, timings) — the
  substrate the WP4 grounding gate (v1 §5.2) will assert against.
- Keep the retriever **warm** (persistent `ProjectRAG` process): cold start is ~31 s of model load,
  paid once; steady-state is ~2.7 s/query.

## Verdict

**The retrieval layer is proven ready for WP4.** All six applicable §6.2 gates pass on the real
corpus, each with a reproducible measurement and a named test; the one correctness-risk gate that
stood open at WP3 (Abstention, F16) is now built and measured. Every residual miss is localized to
an accepted ceiling (chunking/coverage) or to a judgement that belongs to the LLM (counting), **not**
to an un-pulled retrieval lever — the ranking and chunking levers are exhausted and documented. The
system hands WP4 a ranked, citable, trace-backed evidence set **and** an explicit insufficient-
evidence signal, which is exactly what a grounded, abstaining assistant needs.

**Sample-size caveat (unchanged, honest):** n=34–39 scored questions; a single question is ~0.03 on
recall@5 / ~0.026 on evidence@5. The headline margins over baseline are multi-question and outside
that band; the absolute levels (0.82 / 0.71) carry the small-sample caveat. All numbers are this-box
(ROCm WSL, RX 7900 XT); the A01 CUDA gate stays fail-closed on any machine without a visible GPU.

**Not done here, by design (the stop line):** generation, grounding assertions, citation-answer
synthesis, AI-Team tool registration, and the cross-project WP5 proof. WP4 begins there.

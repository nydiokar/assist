# Round-2 handoff brief — retrieval polish + WP4 wiring investigation (NOT wiring)

**For:** the next agent (cold start). **From:** the agent that finished WP0→WP3 and the WP4-prep
round (eval expansion, abstention, filesystem evidence, chunking negative, latency). **Date:** 2026-10-28.

Read this AFTER `AGENTS.md` and the "Read before working" chain it names (A01 > v2 > v1, then
EVAL_LOG, FAILURE_LEDGER, DECISIONS, README, `docs/HANDOFF_WP4_READY.md`). This brief tells you
**what the owner actually wants, what is already true, and exactly what to do and NOT do this round.**
Verify every claim here with `git log` + `pytest` (77 green at HEAD `866607d`), not by trusting prose.

---

## 0. The owner's three goals — these OVERRIDE the spec's priorities where they conflict

1. **Learning is #1.** The owner is learning RAG by building it. *Deliberate, explained decisions and
   honest negatives are the deliverable*, not a high score. "Here is the expensive error and why"
   beats "here is a lucky win." Name techniques and failure modes in professional terms as you go.
2. **Pilot/demo, NOT a service.** This runs on one machine for one viewer. Do **not** spend effort on
   concurrency, p95 SLAs, multi-writer safety, HNSW-across-rebuild determinism, horizontal scale.
   That axis is explicitly out. (The previous round over-invested in latency rigor — don't repeat it.)
3. **Demo to the data/process owner, soon.** The owner is an NGO project manager; he OWNS this corpus
   and already knows the answers. The demo must let **him** ask his own questions and judge usefulness
   himself — not a canned 6-question show. The real test he will likely run: **"can the agent help
   draft the project report?"** The final report was kept as a HOLDOUT for exactly this. So the
   capability that matters is *evidence-gathering per report section* (works for in-motion projects
   too: preliminary/final reports), which is a retrieval-first task — plays to our strength and makes
   abstention essential (never fabricate a section).

**Calibration in one line:** *Google-level process/understanding, demo-level readiness.* When a
decision trades effort for polish, choose the demo-level option and SAY you did and why.

**The push-vs-diminishing-returns decision is already made (owner agreed):** do **one** more targeted
retrieval round, then **STOP chasing retrieval metrics.** Do NOT try to push recall 0.82→0.93. Reason:
the eval set is self-authored, n≈35, no held-out split — pushing the score on *that* set is
overfitting to our own questions (a textbook expensive error). The honest lever is a *better-trusted*
eval and a *better LLM-facing payload*, not a bigger number. The owner's own interrogation is the real
eval.

---

## 1. What is already built and TRUE (do not rebuild, do not re-litigate)

Pipeline (config `configs/yp_qwen4b_gpu_rerank.yaml`, frozen real index `yp_qwen4b_gpu`, project `ypq`,
corpus `yspdb/`):
```
query → Qwen3-Embedding-4B dense (2560-d, ROCm GPU) + SQLite FTS5 lexical
      → RRF fusion (k=60) → per_file_cap=3 dedup
      → cross-encoder rerank (bge-reranker-v2-m3, top_n=12)
      → evidence-sufficiency gate (top-1 rerank_score ≥ 0.26)
      → SearchResponse{results, evidence verdict, trace}
```
Measured (EVAL_LOG 2026-10-28): recall@5 **0.82**, evidence@5 **0.71**, MRR **0.74** (34-scored set);
abstention withholds **10/11** no_answer; warm latency **p50 2.7 s**, VRAM 10.7 GB.

**Settled — do NOT reopen these (each has measured evidence in EVAL_LOG/ledger):**
- Ranking is solved as far as it goes. The k-sweep proved residual misses are chunking/coverage/
  abstention, not ranking. Do not reach for a bigger embedder or re-tune RRF.
- Reranker top_n=12 is a measured peak (wider demotes fusion's strong head). Don't widen it.
- Smaller chunks (500→250) were TESTED and REJECTED (evidence@5 fell). The chunking lever is exhausted.
- Abstention signal = the calibrated cross-encoder score; RRF/Chroma distances are NOT calibrated
  (v1 §4.6) — don't threshold on them.

**The key transferable technique the owner should see you use again:** *localize before you optimize*
— recall-in-pool vs recall-at-k tells you ranking-problem vs retrieval-problem (`scripts/sweep_k.py`).

---

## 2. HARD RULES (breaking any of these is sabotage — from AGENTS.md + the owner)

- **Real corpus NEVER enters git.** `yspdb/`, zips, `local_data/`, `eval/results/` are gitignored —
  keep it that way. Only synthetic fixtures + code + docs are committed.
- **No data egress.** No hosted embedding/LLM/OCR on corpus content without an explicit owner
  decision. (The LLM wiring question in §4 is exactly where this bites — see it.)
- **Read-only on `yspdb/`.** Never modify/rename corpus files.
- **NEVER clobber the frozen index `local_data/yp_qwen4b_gpu/`.** Every experiment goes to a SEPARATE
  `persist_path` + `collection` + `lexical_store.path` (see how `yp_qwen4b_gpu_fs.yaml` /
  `yp_qwen4b_gpu_chunk250.yaml` did it). To reuse embeddings cheaply, COPY
  `local_data/yp_qwen4b_gpu/cache/` into the new index's cache dir before ingest (content chunks hit
  the cache; only new text re-embeds).
- **GPU/WSL incantation (a wrong invocation cost a full day once — ledger F18):**
  `wsl.exe -d Ubuntu-24.04 -- bash -c 'export HSA_ENABLE_DXG_DETECTION=1; cd /mnt/c/Users/Cicada38/Projects/assist && PYTHONPATH=src /home/gorila/venvs/tokens-ingest-rocm/bin/python -X utf8 <cmd>'`
  The env var MUST be in the same command (`wsl -- bash -c` does not source `~/.bashrc`). Device
  string is `"cuda"` even though the card is AMD (ROCm/HIP). `torch.version.hip` set / `cuda` None
  confirms it's really the GPU. **NEVER `wsl --shutdown`/stop/unregister** (a hook blocks it; other
  workloads depend on it). `/dev/kfd` absent is normal, not a failure.
- **Windows Python:** `./.venv/Scripts/python.exe -X utf8` (the `-X utf8` is required for Bulgarian).
  This venv is CPU-only by design; all dense/rerank work runs in the WSL ROCm venv above.
- **`/tmp` differs between Git Bash and WSL.** Put scripts WSL must run inside the repo (e.g.
  `scripts/`), not `/tmp`.
- **Note before committing:** two empty stray files `BG` and `260-char` sit untracked in the repo
  root (prior accidents, not ours). Leave them; don't commit them, don't delete without asking.
- **Process rules:** every new behavior gets a regression test; failures → `FAILURE_LEDGER.md`
  (reproducer·mechanism·fix·test); unspecified design choices → `DECISIONS.md`; spec contradictions →
  a new `AMENDMENT_A0N.md`. Measure before/after, keep negatives. Commit locally at each verified
  checkpoint; do NOT push. Typed/Pydantic/functional/no-`@staticmethod` (global rules).
- **Heavy runs** (any GPU reindex/embed > ~30 min): invoke the `heavy-run-governor` skill, run its G5
  preflight, freeze a G0 contract (hypothesis/baseline/metric/decision-rule) BEFORE compute.

---

## 3. THE WORK THIS ROUND — retrieval polish, in order, each measured, then STOP

Freeze the eval set (`eval/yp_questions_v2.jsonl`, 53 q) and the 34/39/50-scored conventions already
in EVAL_LOG. Measure every change before/after on the frozen reranked pipeline; adopt only on a
measured win outside the ~0.03 (1-question) noise band; record negatives as first-class.

**Task A — Parent/section retrieval (spec §7 "parent-section expansion"; THE highest-value lever).**
evidence@5 plateaus at 0.71 because the exact answer *chunk* is often not top-5 even though the right
*document* is (recall@5 0.82). Instead of reindexing (chunking lever is spent), **return more context
to the LLM**: when a chunk is retrieved, also make its parent section / neighboring chunks available
(via `open_source` expansion or a section-expansion step in the response). This is the standard
production move for exactly this gap and it is query-time, no reindex. Measure a new "answer-span
coverage" proxy: for the scored questions, is the expected substring present in the *expanded* context
of a top-5 hit? Expect this to lift the effective evidence availability toward ~0.85+ honestly.
`src/project_rag/sources.py:open_source` already returns surrounding text — build on it.

**Task B — Cross-language query handling (F11/F14; cheap, high-learning).** yp07/yp09/yp31 fail because
an EN query must hit a short BG table row (and vice versa). Try the standard cheap fix: detect query
language and also search a translated form (or prepend a BG+EN gloss of the query terms) before
embedding/lexical. One experiment. Adopt only if yp07/09/31 move; if not, record the negative. Do NOT
fine-tune or swap the embedder.

**Task C — Owner-drivable evaluation (the REAL professional gap; serves goal #1 and #3).** The current
eval is self-authored — its credibility, not its score, is the bottleneck. Build a mode where the
OWNER types his own questions and the system logs, per question: the retrieved results + citations,
the evidence verdict (sufficient/abstained + top_score), and a slot for his own usefulness rating
(he knows the truth). That logged session IS the honest eval when there is no gold set. This is also
the demo surface's backbone. Keep it dead simple (CLI or a tiny script) — no service.

**Task D — Holdout / report-oriented eval questions.** The owner will probe "can it help draft the
report?" Author ~10–15 **report-section-oriented** questions (e.g. "what evidence exists for the
Results/Indicators/Activities section?") from the HOLDOUT final report, each with the corpus evidence
location, validated against PARSED text via `scripts/eval_yp.py:validate_labels` (raw grep misses
.docx/.xlsx — they are zips; a real past error, ledger F17). These test the capability that matters.
*(This sub-task is the one safely dispatchable to a bounded read-only subagent — see §5.)*

**Task E — Refresh the readiness close-out** (`docs/HANDOFF_WP4_READY.md`) with A–D results and an
updated honest go/no-go.

**STOP criterion for the round:** after A–E, do not keep tuning. If A/B give a measured win, adopt and
move on; if not, record and move on. The target is *trustworthy + interrogable*, not maximal.

---

## 4. THE WP4 WIRING QUESTION — INVESTIGATE AND REPORT, do NOT act (hard stop)

The owner wants the LLM/agent wired the **easiest, most convenient way that actually works** — not
clever, not a framework rebuild. **You must STOP before taking any wiring action.** Produce a short
written recommendation (`docs/WP4_WIRING_OPTIONS.md`) and hand the decision back to the owner. Do NOT
call an LLM, do NOT register tools, do NOT edit the sibling AI-team repo.

**What you already have to build on (verified):** `src/project_rag/ai_team_adapter.py` exposes a clean
`ToolDescriptor` list (`search_project`, `open_source`, `get_ingestion_status`) with JSON schemas and a
`# WP4 SEAM` marker in `register_with_ai_team`. That is the binding point.

**The owner's own words on the options (investigate each for effort/fit, think like an owner):**
- **Option 1 — Ollama local inference.** No egress (keeps the no-egress rule clean). BUT: a weak local
  model could produce FALSE NEGATIVES and invalidate the demo — the owner is explicitly worried about
  this. Investigate: which locally-runnable model is actually strong enough at **Bulgarian + grounded
  citation + abstention** to be a fair test? (It must not confabulate or mangle Cyrillic.) Note VRAM
  budget: the retriever already uses ~10.7 GB of the 21.4 GB card; a co-resident LLM must fit or be
  swapped.
- **Option 2 — Reuse the AI-team harness.** AI-team wraps the SDKs of Codex / Claude Code / opencode.
  It EXPOSES a backend control API; clients integrate as another AI-team impl OR as a tool. So our RAG
  could plug in **either as an MCP tool or as a built-in tool** in the harness — leveraging infra that
  already exists. Investigate how much effort that binding really is.
- **Option 2b — If the full harness is too much:** a simple direct call to the **Claude Code SDK**
  (hosted → triggers the no-egress rule; needs an explicit owner decision on sending corpus text).
- Load the `claude-api` skill before reasoning about any Claude/Anthropic model choice (it exists for
  exactly this; don't answer from memory).

**Deliverable for §4:** a one-page comparison — for each option: effort, does-it-work confidence,
egress/privacy implication, Bulgarian+grounding fitness, and whether our RAG binds as MCP tool vs
built-in tool. End with a recommended path and the open question the owner must answer (which model;
local-vs-hosted egress). **Then stop and report.** The AGENTS.md note on the AI-team seam
(`~/.claude.json` host-global MCP, `claude_driver.py` `_sdk_options` in-process injection,
`permission_mode=bypassPermissions`, tool args not stored in `llm_events`) is a 2026-10-16 read-only
finding — RE-VERIFY it against the live AI-team repo before relying on it, and read THAT repo's
`CLAUDE.md` before running anything there (it has a strict test-cost rule — do not run paid tests).

---

## 5. If you dispatch a subagent

Most of §3 is sequential, context-heavy, and state-mutating (reindex/eval against the one GPU, must not
clobber the frozen index) — keep it in your own warm thread; a cold fan-out will clobber state or
re-litigate settled decisions. **Do NOT fan out Tasks A–C or §4.**

The one safe, bounded, parallelizable unit is **Task D** (draft report-oriented eval questions). If you
dispatch it, the subagent's contract must be: read-only on `yspdb/` and the holdout report; output a
JSONL of candidate questions each with `expect_source_paths` + `expect_substrings` + rationale;
validate every label against PARSED text (not raw grep — .docx/.xlsx are zips) using
`scripts/eval_yp.py:validate_labels`; the owner must be able to verify each answer; touch NO index and
NO frozen file; return the artifact for review, do not append it to the eval set itself.

---

## 6. Definition of done for this round

A–E measured and committed; `pytest` green; every new behavior has a regression test and an
EVAL_LOG/DECISIONS/LEDGER entry; a refreshed go/no-go in `HANDOFF_WP4_READY.md`; and
`docs/WP4_WIRING_OPTIONS.md` with a recommended wiring path and the single decision the owner must make
— **with no wiring action taken.** Work like an owner: deliberate, localize before optimizing, cheapest
correct lever, negatives are results, and stop at demo-level — not Google-level — readiness.

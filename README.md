# Project Assistant — RAG subsystem

Local, project-scoped document ingestion + hybrid (lexical + dense) retrieval with exact
source inspection and query traces. Plugs into AI-Team as read-only tools; retrieval is
independently testable without an LLM. Implements **spec v2**
(`docs/PROJECT_ASSISTANT_RAG_SPEC_v2.md`) as amended by **Amendment A01**
(`docs/AMENDMENT_A01.md`): Qwen3-Embedding-4B dense candidate + reusable fingerprinted stages.

## Status

WP0→WP3 complete; the retrieval layer is **measured on the real corpus and proven ready for WP4**
(wiring an LLM) — see **`docs/HANDOFF_WP4_READY.md`** for the gate-by-gate verdict. WP4 generation
is deliberately NOT built (stop line).

- Real-corpus reranked pipeline (Qwen3-4B dense + FTS5 + RRF + cross-encoder reranker): recall@5
  **0.82**, evidence@5 **0.71**, MRR **0.74** (34-scored v2 set); evidence-sufficiency/abstention
  gate withholds 10/11 no_answer; warm latency p50 2.7 s. See `docs/EVAL_LOG.md`.
- Round 2 (2026-10-29): parent-section context expansion adopted (span@5 0.667→0.718), a
  cross-language glossary tested and rejected (negative), an owner-drivable session (`scripts/ask.py`),
  and a holdout report-section eval that honestly exposes the report-drafting gap (recall@5 0.42).
  WP4 wiring investigated in `docs/WP4_WIRING_OPTIONS.md` — no wiring done (stop line).
- 90 tests pass (offline hash embedder + in-memory store; fixture metrics are *not* a production SLA).
- **GPU path runs in WSL via ROCm, not on the Windows host.** The Windows host has no NVIDIA GPU
  (A01). WSL Ubuntu-24.04 has an AMD Radeon RX 7900 XT (gfx1100) reachable through ROCm 7.2.4;
  the real-corpus Qwen3-4B index was built and evaluated there (see `docs/EVAL_LOG.md` 2026-10-26).
  The A01/A01-era "no GPU in WSL" note was a *driver-state measurement at that time*; the ROCm GPU
  is available now **provided `HSA_ENABLE_DXG_DETECTION=1` is exported in the command** (see WSL
  runbook below — ledger F18). The A01 CUDA gate stays fail-closed on any machine without a visible
  device; `scripts/feasibility_smoke.py` fails visibly rather than falling back to CPU.

## Setup (Windows, PowerShell)

```powershell
cd project-assistant-rag
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe scripts\make_fixtures.py
.\.venv\Scripts\python.exe -m pytest
```

## Runbook (search-only MVP, no LLM)

```powershell
.\.venv\Scripts\python.exe -m project_rag.cli ingest --project demo --root tests\fixtures\synthetic\ngo_project_demo
.\.venv\Scripts\python.exe -m project_rag.cli status --project demo
.\.venv\Scripts\python.exe -m project_rag.cli search --project demo --query "What indicators did the approved project define?" --top-k 5
.\.venv\Scripts\python.exe -m project_rag.cli open   --project demo --doc d --source-path proposal.md --root tests\fixtures\synthetic\ngo_project_demo
```

## Configuration (`configs/retrieval.yaml`)

Single source of provider selection + tuning. Key switches:

| Key | Test/dev default | Production candidate (A01) |
|---|---|---|
| `embedding.provider` | `hash` (offline, deterministic, 2560-dim) | `qwen3` (`Qwen/Qwen3-Embedding-4B`, 2560-dim, CUDA, install `.[qwen3]`) |
| `vector_store.provider` | `memory` (numpy cosine, JSON-persisted) | `chroma` (install `.[chroma]`) |
| `parsing.backend` | `native` (pypdf/python-docx/openpyxl/csv) | `docling` (install `.[docling]`) |
| `retrieval.per_file_cap` | `3` | max chunks from one source in the result window; WP3 2026-10-27 measured evidence@5 peak at 3 (`null` = legacy `top_k//2`) |
| `retrieval.rerank.enabled` | `false` | cross-encoder reranker over the fused pool (spec §7; install `.[rerank]`, pin revision) — staged, see EVAL_LOG 2026-10-27 / ledger F19 |

`e5` remains an optional legacy comparator. Switching `embedding.provider`/model/dim forces a
**new index generation** (ADR-009); the startup compatibility guard refuses writes on a mismatch.
`Config.assert_release_safe()` rejects the `hash`/`memory` fakes and an unpinned Qwen3 revision.
`retrieval.per_file_cap` and `retrieval.rerank` are **query-time only** — no reindex.

### Activating Qwen3-4B (requires CUDA, ~20 GB VRAM, WSL)

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[qwen3]"
.\.venv\Scripts\python.exe scripts\feasibility_smoke.py   # A01 §E gate — must pass first
# then pin the resolved HF revision into configs\retrieval.yaml -> embedding.qwen3.revision
# and set embedding.provider: qwen3, vector_store.provider: chroma
```

### Running the Qwen3-4B index on the WSL ROCm GPU (real corpus)

The real-corpus dense/hybrid pipeline runs in WSL against the ROCm venv. **`HSA_ENABLE_DXG_DETECTION=1`
must be exported in the same command** — a non-interactive `wsl -- bash -c` shell does NOT source
`~/.bashrc`, so without the var ROCm reports `hsa_init Failed` and `torch.cuda.is_available()` is
False (ledger F18, closed). `/dev/kfd` is absent by design under WSL ROCm; do not use it as a
health check. **Do not run `wsl --shutdown` / stop / terminate / unregister** — other workloads
depend on WSL (a hook blocks it).

```powershell
# Verify the GPU is visible (expect: True, 1, "AMD Radeon RX 7900 XT")
wsl.exe -d Ubuntu-24.04 -- bash -c 'export HSA_ENABLE_DXG_DETECTION=1; /home/gorila/venvs/tokens-ingest-rocm/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.get_device_name(0))"'

# Evaluate all channels on the frozen v2 question set (query-embedding only, no re-ingest)
wsl.exe -d Ubuntu-24.04 -- bash -c 'export HSA_ENABLE_DXG_DETECTION=1; cd /mnt/c/Users/Cicada38/Projects/assist && PYTHONPATH=src /home/gorila/venvs/tokens-ingest-rocm/bin/python -X utf8 scripts/eval_yp.py --config configs/yp_qwen4b_gpu.yaml --project ypq --root yspdb --questions eval/yp_questions_v2.jsonl --channels lexical dense lexical+dense --out eval/results/yp_v2_qwen4b_gpu.json'
```

`project_rag.gpu_env.prepare_gpu_env()` sets the var in-process for code paths that import torch,
so Python entry points work; the explicit `export` above is still required for the `wsl -- bash -c`
wrapper itself.

### Owner-drivable retrieval session (demo / honest eval, no LLM)

The corpus owner can interrogate retrieval with his own questions (BG or EN). Per question it shows
the citations, the evidence-sufficiency verdict (sufficient / abstain + top relevance score) and the
parent-section context, then logs it plus a 1–5 usefulness rating to a JSONL session — the honest
eval when there is no gold set (Round-2 Task C). Retriever stays warm for the session.

```powershell
wsl.exe -d Ubuntu-24.04 -- bash -c 'export HSA_ENABLE_DXG_DETECTION=1; cd /mnt/c/Users/Cicada38/Projects/assist && PYTHONPATH=src /home/gorila/venvs/tokens-ingest-rocm/bin/python -X utf8 scripts/ask.py --config configs/yp_qwen4b_gpu_rerank.yaml --project ypq --root yspdb'
```

### Reusable stages (A01 §C)

`stages.py` caches `[1] PARSE` (shared across projects, keyed by content hash + parser config)
and `[2] CHUNK` (per-project lineage, keyed by parsed-artifact + chunker config) under
`local_data/cache/`. An unchanged source is not reparsed; changing the chunker re-chunks without
reparsing; switching the embedding profile re-embeds without touching parse/chunk.

## Architecture (v1 §3.3)

`discovery → [stages: parse-cache → chunk-cache] → embeddings + vector_store (dense) +
lexical_store (FTS5) → retrieval (RRF fusion + dedup) → api (tool contracts) →
ai_team_adapter (WP4 seam)`.
`manifest` (SQLite) holds document state, active index generation, and pinned embedding profile.
`trace` records grounding diagnostics (hashed query, retrieved chunk IDs) with no raw PII.

## Build stop line (waiting on user-provided inputs)

1. **Real NGO corpus** → run the real parsing gate, decide the E5 download, run WP3 on real data.
2. **AI-Team tool contract** → bind `ai_team_adapter.register_with_ai_team` at the `# WP4 SEAM`.

Until then no production gate requiring real data is marked passed, and no unsafe assumption is made.

## What is intentionally NOT built yet

Chat backend, scheduler, agent teams, indicator aggregation engine, knowledge graph, OCR,
cloud hosting (v1 §2.4 / §7). The **LLM/generation path (WP4)** is the stop line: no LLM is called,
no AI-Team tool is registered, no grounded-answer/citation synthesis exists. The retrieval layer is
handed off *ready* for it — `docs/HANDOFF_WP4_READY.md`. (The cross-encoder reranker, once "deferred",
is now built and measured — F19.)

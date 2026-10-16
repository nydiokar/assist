# Project Assistant — RAG subsystem

Local, project-scoped document ingestion + hybrid (lexical + dense) retrieval with exact
source inspection and query traces. Plugs into AI-Team as read-only tools; retrieval is
independently testable without an LLM. Implements **spec v2**
(`docs/PROJECT_ASSISTANT_RAG_SPEC_v2.md`) as amended by **Amendment A01**
(`docs/AMENDMENT_A01.md`): Qwen3-Embedding-4B dense candidate + reusable fingerprinted stages.

## Status

WP0 scaffold + WP1 (parsing/provenance) + WP2 (search-only MVP) + WP3 harness are **built and
green on a synthetic bilingual fixture corpus** using a cheap offline embedder. The system is
**waiting for two inputs** before further gates (see "Build stop line").

- 39/39 tests pass. Fixture Recall@5 = 1.0, MRR = 0.96 (hash embedder — *not* a production SLA).
- **A01 CUDA gate cannot pass on this machine** (no NVIDIA GPU on host or in WSL — measured).
  `scripts/feasibility_smoke.py` fails visibly rather than silently falling back. See
  `docs/AMENDMENT_A01.md` → "Measured environment deviation".

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

`e5` remains an optional legacy comparator. Switching `embedding.provider`/model/dim forces a
**new index generation** (ADR-009); the startup compatibility guard refuses writes on a mismatch.
`Config.assert_release_safe()` rejects the `hash`/`memory` fakes and an unpinned Qwen3 revision.

### Activating Qwen3-4B (requires CUDA, ~20 GB VRAM, WSL)

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[qwen3]"
.\.venv\Scripts\python.exe scripts\feasibility_smoke.py   # A01 §E gate — must pass first
# then pin the resolved HF revision into configs\retrieval.yaml -> embedding.qwen3.revision
# and set embedding.provider: qwen3, vector_store.provider: chroma
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
reranker, cloud hosting (v1 §2.4 / §7). These are deferred/V2 by the spec.

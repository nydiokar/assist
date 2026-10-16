# Project Assistant — RAG subsystem

Local, project-scoped document ingestion + hybrid (lexical + dense) retrieval with exact
source inspection and query traces. Plugs into AI-Team as read-only tools; retrieval is
independently testable without an LLM. Implements **spec v2** (`uploads/PROJECT_ASSISTANT_RAG_SPEC_v2.md`).

## Status

WP0 scaffold + WP1 (parsing/provenance) + WP2 (search-only MVP) + WP3 harness are **built and
green on a synthetic bilingual fixture corpus** using a cheap offline embedder. The system is
**waiting for two inputs** before further gates (see "Build stop line").

- 30/30 tests pass. Fixture Recall@5 = 1.0, MRR = 0.92 (hash embedder — *not* a production SLA).

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

| Key | Test/dev default | Production baseline |
|---|---|---|
| `embedding.provider` | `hash` (offline, deterministic) | `e5` (`intfloat/multilingual-e5-small`, install `.[e5]`) |
| `vector_store.provider` | `memory` (numpy cosine, JSON-persisted) | `chroma` (install `.[chroma]`) |
| `parsing.backend` | `native` (pypdf/python-docx/openpyxl/csv) | `docling` (install `.[docling]`) |

Switching `embedding.provider` or model forces a **new index generation** (ADR-009); the
startup compatibility guard refuses writes on a mismatch. `Config.assert_release_safe()` rejects
the `hash`/`memory` fakes for a release.

## Architecture (v1 §3.3)

`discovery → parsers → chunking → embeddings + vector_store (dense) + lexical_store (FTS5) →
retrieval (RRF fusion + dedup) → api (tool contracts) → ai_team_adapter (WP4 seam)`.
`manifest` (SQLite) holds document state, active index generation, and pinned embedding profile.
`trace` records grounding diagnostics (hashed query, retrieved chunk IDs) with no raw PII.

## Build stop line (waiting on user-provided inputs)

1. **Real NGO corpus** → run the real parsing gate, decide the E5 download, run WP3 on real data.
2. **AI-Team tool contract** → bind `ai_team_adapter.register_with_ai_team` at the `# WP4 SEAM`.

Until then no production gate requiring real data is marked passed, and no unsafe assumption is made.

## What is intentionally NOT built yet

Chat backend, scheduler, agent teams, indicator aggregation engine, knowledge graph, OCR,
reranker, cloud hosting (v1 §2.4 / §7). These are deferred/V2 by the spec.

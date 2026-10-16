# Project Assistant — RAG Engineering Design Specification

**Version:** 2.0 | **Date:** 16 October 2026 | **Supersedes:** v1.0
**Target environment:** Local Windows, Python 3.12, CPU-only (16 cores, no NVIDIA GPU — measured).

> This v2 keeps every contract, boundary, ADR and gate from v1.0 intact. It records the
> **implementation deltas** the implementing agent adopted so the subsystem can be scaffolded,
> built and tested *before the real NGO corpus and AI-Team seam are available*. Every delta below
> is reversible and preserves the v1 architecture. Section numbers map to v1; only changed/added
> items are restated. Read v1.0 for the full rationale that is not repeated here.

---

## v2 CHANGE LOG (reasoned deltas vs v1.0)

Each delta is stated as: **what changed · why · reversibility**. These were adopted without
round-tripping because they do not alter any v1 boundary, contract, or acceptance gate — they
only make the system buildable and testable ahead of data.

### D1 — Pluggable embedder with an offline deterministic fake (NEW)
- **What:** `embeddings.py` defines an `EmbedderProtocol`. Two implementations:
  `HashEmbedder` (deterministic, dependency-free, for cheap/offline validation and CI) and
  `E5Embedder` (the v1 baseline `intfloat/multilingual-e5-small`, loaded only when selected).
  Selected via `configs/retrieval.yaml → embedding.provider`.
- **Why:** User requires validation of embedding+retrieval "on something very cheap and simple
  initially", and the whole stack must prove out before the ~470 MB E5 download or real data
  arrives. ADR-003 fixes E5 as the *production baseline*; the adapter abstraction it already
  mandates makes an interchangeable fake legitimate, not a scope change.
- **Reversible:** Flip `embedding.provider: e5`. The `embedding_profile` (id, revision, dim,
  prefixes, pooling) is recorded per-collection either way; switching providers forces a new
  index generation exactly as ADR-009 requires. The fake is never a valid production choice and
  is gated out of release by a config assertion.

### D2 — Docling is an optional extra; a lightweight native parser set is the default (DELTA on ADR-002)
- **What:** `parsers.py` defines a `ParserProtocol`. Default parsers use `pypdf` (digital PDF),
  `python-docx` (DOCX), `openpyxl` (XLSX), stdlib `csv`, and plain readers (TXT/MD). Docling is an
  optional install (`pip install -e ".[docling]"`) behind the same protocol, activated by
  `parsing.backend: docling`.
- **Why:** Docling drags in `torch` and large model weights — a multi-GB install that would block
  the WP2 "search-only MVP runs now" gate on a CPU-only box before any corpus exists. The native
  parsers cover every v1 in-scope format (§2.3) and preserve the required provenance (page, sheet,
  cell, heading) well enough to validate the pipeline and pass parsing smoke tests today.
- **Reversible:** ADR-002 still holds — Docling remains the intended richer backend; set
  `parsing.backend: docling` once installed and its fidelity is validated on the real corpus
  (v1 §4.2 parsing gate unchanged). Per-format fidelity failures trigger the ADR-002 reconsideration.

### D3 — Vector store abstraction with an in-memory fake (DELTA on ADR-004)
- **What:** `vector_store.py` defines a `VectorStoreProtocol`. `ChromaVectorStore` is the v1
  baseline (`PersistentClient`, explicit cosine HNSW, explicit externally-generated embeddings,
  no default embedding function). `InMemoryVectorStore` (pure numpy cosine) backs fast unit tests
  and offline validation. Selected via `vector_store.provider`.
- **Why:** Lets index-lifecycle, fusion, dedup, deletion-of-stale-ids and project-isolation logic
  be tested deterministically without Chroma's native HNSW nondeterminism or its install, and
  before real data. Chroma remains the default persistent store for any non-test run.
- **Reversible:** Default stays Chroma. The in-memory store is test/dev only and asserted out of
  release configs. Every v1 Chroma rule (§4.6) is implemented in `ChromaVectorStore` and covered
  by an integration test that runs when Chroma is installed.

### D4 — Synthetic bilingual fixture corpus shipped in-repo (NEW, satisfies v1 §3.3 `tests/fixtures/synthetic/`)
- **What:** A generated, PII-free synthetic "NGO project" corpus (proposal with indicators, a
  progress report with target/actual numbers, an XLSX indicator table, a CSV registration list
  with an intentional duplicate, a Bulgarian-language memo, and an unsupported/encrypted-marker
  file). Plus `eval/questions.jsonl` with reviewed expected-evidence spans over these fixtures.
- **Why:** The user asked to validate end-to-end on cheap/simple data now and to not discover
  architectural issues once real data lands. v1 §3.3 already reserves this folder; §6.1 already
  requires 12–20 reviewed questions. These are **test fixtures, not runtime schemas** (v1 §6.1).
- **Reversible:** Fixtures live only under `tests/fixtures/` and `eval/`. Real NGO files never
  enter git (v1 §8). When real data arrives it is ingested into its own `project_id`.

### D5 — Config-driven profiles, startup compatibility guard (TIGHTENS v1 §4.5/§4.6)
- **What:** `configs/retrieval.yaml` is the single source of tuning + provider selection. On every
  index open, the manifest-stored `embedding_profile` + `collection_schema` are compared to the
  live config; a mismatch **refuses writes** (v1 §4.6 rule) and reports a clear error.
- **Why:** Prevents silent vector mixing (ADR-009) and makes the fake↔real swaps safe.
- **Reversible:** It is a guard, not a behavior change.

### D6 — RRF fusion with explicit, recorded parameters (IMPLEMENTS v1 §4.7)
- **What:** Reciprocal Rank Fusion with `k=60` (documented default), top 10–20 per channel,
  return 6–8 after file-diversity dedup. Channel contributions + fusion rank recorded in trace.
- **Why:** v1 §4.7 names RRF "or comparably simple deterministic method"; RRF k=60 is the standard
  deterministic choice and is trivially testable.
- **Reversible:** `retrieval.fusion.k` is config; tune only from evaluated failures (v1 §4.7).

### D7 — Deployment seam deferred to a thin adapter stub (HONORS v1 §3.3 / ADR-001 / Open-decision 3)
- **What:** `ai_team_adapter.py` exposes `search_project`/`open_source`/`get_ingestion_status` as
  plain Python callables with typed signatures and a registration-descriptor list, but does **not**
  import or assume any AI-Team API. A `# WP4 SEAM` marker documents where real registration binds
  once the AI-Team tool contract is inspected (WP0).
- **Why:** v1 forbids guessing the AI-Team API. This keeps WP0→WP4 honest while letting WP2 ship.
- **Reversible:** It is additive; binding happens at WP4.

### D8 — Python/runtime pin (NEW fact)
- Target Python **3.12**; CPU-only. Embedding batch concurrency bounded (default 8 threads,
  `embedding.batch_size: 32`) per v1 §4.5 safeguard 3, tuned to the measured 16-core box.

**Everything not listed here is unchanged from v1.0.** The critical path `WP0→WP1→WP2→WP3→WP4→WP5`,
all ADRs, all acceptance gates (§2.5, §6.2), all security/privacy rules (§8), and the
retrieval-first/generation-second discipline are carried over verbatim.

---

## BUILD STOP LINE (where the agent halts pending user-provided inputs)

The agent builds and self-tests through **WP0 scaffold + WP1 + WP2 + WP3 (on synthetic fixtures)**
using the HashEmbedder + InMemory/Chroma stores. It then **stops** and waits for:

1. The **real NGO corpus** (to run the real parsing gate, the E5 download decision, and WP3 on real data).
2. The **AI-Team tool contract** (to bind WP4 `ai_team_adapter.py`) — v1 Open-decision 3.

Until those arrive, no unsafe assumption is made and no v1 gate that needs real data is marked passed.

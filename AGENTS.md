# AGENTS.md — project-assistant-rag

Read this whole file, then the documents in "Read before working". Do not start work cold.

## What this project is

A **practice / learning build of a RAG system**, done properly and end to end. The owner is
learning the stages of RAG by building them, hitting failures, and finding out what a
successful RAG and its implementation look like. The deliverable is as much the **recorded
process** (prepare → validate → measure → fail → diagnose → fix) as the working system.

Two goals, in order:

1. **RAG as a standalone, measured component.** Ingest a real document corpus, retrieve
   evidence with citations, and prove retrieval quality **without any LLM** (spec v1 §5.1, §6).
2. **Then** wire it into the sibling project `C:\Users\Cicada38\Projects\AI-team` (a gateway that
   already wraps coding-agent backends and is ready to use, not deployed here) so an agent uses
   RAG as tools and becomes a "project assistant". That is WP4, after retrieval is validated.

It is a practice project on a finished, real corpus. There is no production deadline and no
stakeholder SLA. Reproducibility, justification and honest measurement outrank features.

## The corpus: `yspdb/`

`yspdb/` is the document archive of **one completed NGO project** — "Youth Participation"
(Bulgarian: "Младежко участие", acronym **YP**), an **Erasmus+ KA154-YOU** youth participation
project, form ID **KA154-YOU-E2373104**, Call 2024 Round 3, 01/02/2025–31/10/2025, 9 months,
National Agency BG01 (HRDC). The application was **selected and the project was already
executed**. The folder is a mixed dump of: (a) the funding application, (b) evidence and
products of the project's activities (debate, webinars, youth studios, podcast, good-practice
handbook, advocacy, invitations to institutions, communications, sponsors, monitoring &
evaluation), (c) administration/finance. **Content is mostly Bulgarian, with English mixed in.**
It contains personal data (names, contacts, participant feedback) — see "Hard rules".

Measured (2026-10-16, WP0 inventory; re-measure rather than trust if the folder changes):

| Fact | Value |
|---|---|
| Files / size | 337 files, 3.58 GB; longest path 277 chars (**exceeds Windows 260 limit** — use `\\?\` prefix) |
| Text-bearing, in scope | 115 `.docx`, 30 `.xlsx`, 28 `.pdf`, 2 `.txt` |
| Needs a decision | 4 legacy `.doc` (no native parser), 80 `.png` + 53 `.jpg` + 11 `.jpeg` (images, V1 reports coverage gap, no OCR), 1 `.conf` |
| Out of scope (media) | 10 `.mp4` + 1 `.mov` (~3 GB), 2 `.m4a` — no transcription in V1 |
| Heaviest folders | `Communications` 106 files, `04. Сборник с Добри практики` 65, `Покани` 48, `Уебинари` 40 |
| Main proposal | `KA154-YOU-E2373104.pdf` — 65 pages, digital text layer present on every page |

### The duplicate proposal files (resolved)

`KA154-YOU-E2373104.docx`, `...(1).docx`, `...(2).docx` and `.pdf` are the **same application
form**. Verified by text diff: the three DOCX differ by a single stray prefix `мла` on one title
line (`Call 2024 Round 3 YOU`); `(1)` and `(2)` are identical to each other. The PDF (Oct 8 2024,
the submitted export) is the **canonical** version; DOCX copies (Feb 2025) are edited working
copies. Treat as one logical document with several occurrences: dedupe by content hash but keep
every source occurrence (spec v1 §4.1). Do not pick one silently in code.

### The zip files at repo root

`drive-download-20261016T112451Z-1-00{1,2}.zip` are the Google Drive download the `yspdb/` folder
was **extracted from** (280 of 339 zip entries match a path in `yspdb/`; the rest are
unmatched, e.g. renamed during extraction — not yet investigated). They are redundant raw
shards. **Never ingest them, never commit them.** Source of truth for ingestion is `yspdb/`.

## Read before working (in this order)

1. `docs/AMENDMENT_A01.md` — **authoritative** where it differs from v1/v2. Qwen3-Embedding-4B
   profile, CUDA feasibility gate, fingerprinted reusable stages, and the *measured environment
   deviation* (no NVIDIA GPU on this machine).
2. `docs/PROJECT_ASSISTANT_RAG_SPEC_v2.md` — implementation deltas D1–D8 on top of v1; defines the
   **build stop line**.
3. `docs/PROJECT_ASSISTANT_RAG_SPEC_v1.md` — the full spec: contracts (§3.2), data lifecycle (§4),
   query pipeline (§5), evaluation + gates (§6), ADRs (§10), open decisions (§11), work packages
   WP0–WP5 (§9). v2 only restates what it changes; everything else is read from here.
4. `README.md` — setup, runbook, config switches.
5. `configs/retrieval.yaml` — single source of provider selection and tuning.

**Precedence: A01 > v2 > v1.** Anything not changed by a higher document stands.

## Pivot rule (mandatory)

The owner wants the process to be reproducible and every deviation defensible. Where the specs
are silent, **you must design the step's contents** (this is expected, not optional). But:

- Any change that contradicts or reinterprets A01/v2/v1 needs a written entry before it is
  merged: **what changed · why (with measured evidence) · reversibility · which spec section it
  touches**. Put it in a new `docs/AMENDMENT_A0N.md` (next number), never by silently editing the
  spec files. Specs v1/v2 and A01 are frozen records.
- Unspecified-step design choices (e.g. how to normalize a Bulgarian docx table) go in
  `docs/DECISIONS.md` as short dated entries with the evidence that motivated them.
- Failures are first-class output: keep the **failure ledger** (v1 §6.4: reproducer, suspected
  mechanism, fix, counterfactual baseline, regression test) in `docs/FAILURE_LEDGER.md`.
  Diagnose in the v1 §6.4 order; do not reach for rerankers/bigger models before locating the seam.

## Current state (verify with `git log` and `pytest`, not this prose)

- WP0 scaffold, WP1 parsing/provenance, WP2 search-only MVP, WP3 harness: **built and green on a
  synthetic bilingual fixture corpus** (`tests/fixtures/synthetic/`), 39 tests, using the
  offline `HashEmbedder` + in-memory store. Fixture metrics are **not** evidence of real-corpus
  quality.
- **No gate that needs real data has been passed.** The real corpus has not been ingested yet.
- **A01 CUDA gate fails on this machine** (no NVIDIA GPU on host or WSL; 16 CPU cores). Qwen3-4B
  cannot be the active embedder here; the fail is visible by design. Any CPU/other-model
  substitution is a **pivot** and needs an amendment (see Pivot rule). `HashEmbedder` is a test
  fake and is blocked from release by `Config.assert_release_safe()`.
- AI-Team seam (reported by a read-only code exploration on 2026-10-16, not yet re-verified by a
  runtime test): MCP servers are host-global (`~/.claude.json`); `src/backends/claude_driver.py`
  has an in-process SDK MCP injection path (`_sdk_options`); built-in tools are not disableable
  per session without new code (`permission_mode=bypassPermissions` hard-coded); tool args/results
  are not stored in `llm_events`, only tool name/order. Details belong in `docs/integration_seam.md`
  (WP0 output, not yet written).

## Repository map

`src/project_rag/`: `discovery` → `stages` (parse cache → chunk cache) → `embeddings` +
`vector_store` (dense) + `lexical_store` (SQLite FTS5) → `retrieval` (RRF k=60 + dedup) → `api`
(`ingest_project` / `search_project` / `open_source` / `get_ingestion_status`) →
`ai_team_adapter` (WP4 seam, unbound). `manifest` = SQLite state + pinned embedding profile.
`eval/questions.jsonl` = reviewed questions (currently synthetic). `tests/{unit,integration,evaluation}`.
`local_data/` = derived index/cache, gitignored.

## Hard rules

- **Real corpus never enters git or any remote.** `yspdb/`, the zips, `local_data/` and
  `eval/results/` stay out of version control (see `.gitignore`). Only synthetic fixtures are committed.
- **No data egress.** No hosted embedding/LLM/OCR API on corpus content without an explicit
  written decision from the owner (spec v1 §8). Model weights are downloaded to local cache only
  from the pinned official source; never `trust_remote_code` from unvetted repos.
- **Read-only on `yspdb/`.** Never modify, rename or reorganize original files.
- **Retrieval first, generation second.** No LLM in the retrieval path or its evaluation until
  WP3's gate is written up. RAG does not do aggregation/counting (v1 §2.2, §5.3).
- Use `.venv` (`.\.venv\Scripts\python.exe`), dependencies in `pyproject.toml` only.
- Windows: run Python with `-X utf8` when printing Bulgarian text (default cp1252 console
  crashes), and open corpus paths with the `\\?\` long-path prefix.
- Tests here do not call paid services. Run `pytest` on touched modules; the full suite is cheap
  here. (The *AI-team* repo has a different, strict test-cost rule — read its `CLAUDE.md` before
  running anything there.)
- Python rules come from the global instructions: typed, Pydantic models, no `@staticmethod`,
  functional style, `pyproject.toml`.

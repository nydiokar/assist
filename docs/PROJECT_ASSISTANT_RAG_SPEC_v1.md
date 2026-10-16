# Project Assistant — RAG Engineering Design Specification

**Status:** Proposed implementation baseline (not a claim of an implemented or validated system)  
**Version:** 1.0 | **Date:** 16 October 2026  
**Owner:** Project owner / implementation agent  
**Target environment:** Local Windows development; Python retrieval service/module; existing AI-Team application  
**Primary stakeholder:** NGO project manager  
**Document class:** *Engineering Design Specification* (EDS), including a compact product brief, Architecture Decision Records (ADRs), executable work packages, and an acceptance-test plan.

> **Purpose and boundary.** This is not a generic PRD or a design for another agent framework. It specifies a small, reusable, independently testable document-ingestion and retrieval subsystem that plugs into AI-Team. It also specifies how to judge whether the resulting project assistant reliably answers evidence-based questions. Only one completed NGO project is required for the first release; future projects must be onboardable without writing project-specific code.

## 0. Executive decision

**Build:** A local, project-scoped document retrieval service with automatic ingestion, versioned evidence, hybrid search, source opening, and test traces. AI-Team remains the user-facing conversational/orchestration layer.

**Demonstrate:** The manager asks natural-language questions about obligations, planned activities, reports, indicators, and evidence. The assistant returns concise answers grounded in passages it actually retrieved, with source locations and explicit uncertainty. It may compute a numeric result only when underlying facts and definitions are sufficiently established.

**Do not build yet:** Another chat backend, a dedicated workflow scheduler, autonomous agent teams, full indicator monitoring, project portfolio dashboards, a knowledge graph, automated report production, or new RAG-specific agent orchestration.

**Non-negotiable distinction:** RAG can retrieve the definitions and evidence for indicators; it does not by itself guarantee complete aggregation, de-duplication, time-aware progress calculation, or a defensible prioritization recommendation. These are separate capabilities that can be layered on later.

## 1. Why this document, not only a PRD?

A **PRD** establishes the problem, user value, scope, and success criteria. A **technical design** establishes the system boundaries, interfaces, data handling, decisions, and failure controls. An **engineering specification** joins both so a human or coding agent can implement it without silently inventing requirements. This document therefore contains:

- Product contract and explicit exclusions.
- System and data architecture with versioning and provenance.
- Narrow technology decisions with alternatives and migration conditions.
- Implementation sequencing, test fixtures, release gates, and operational playbooks.
- Known unknowns and decision checkpoints, instead of false claims about unseen NGO files or uninspected AI-Team code.

## 2. Product contract and user journeys

### 2.1 Problem

The NGO manager repeatedly searches mixed-format project documents for obligations, activities, dates, evidence, and reported results. Manual lookup is slow and can miss conflicting versions or sources. The prototype must test whether retrieval-backed answers are faster to establish and easier to verify than manual/direct-agent search.

### 2.2 Target user and four initial journeys

| Journey | Example question | V1 required behavior | Boundary |
|---|---|---|---|
| Locate commitments | “What indicators did the approved project define?” | Find relevant definitions, list them with exact references | No human-maintained indicator registry |
| Find reporting obligations | “Which document specifies our reporting deadline and required attachments?” | Retrieve exact text, identify date only if stated, cite | No reminder engine |
| Trace activity evidence | “What evidence exists for the training held in March?” | Find candidate records, explain what they document | Do not equate document mentions with unique completed activities |
| Evaluate progress (limited) | “What is the target and documented actual for indicator X?” | Retrieve both, show source-backed values; calculate only if definitions align | No claim of comprehensive portfolio-wide indicator health |

**Secondary exploratory query:** “How many vegetarians registered?” This requires structured extraction over the *complete relevant registration set* and duplicate/person handling. A vector top-k answer is not a valid count. Treat it as a separate aggregation experiment, not a V1 acceptance gate.

### 2.3 In scope (V1)

1. Ingest one project directory recursively without rearranging original files.
2. Support PDF (digital), DOCX, XLSX, CSV, TXT/MD; report unsupported, encrypted, scanned or corrupted input explicitly. OCR is optional after a documented feasibility check.
3. Maintain machine-readable manifest, parse report, chunks, embedding model information and indexing status.
4. Support incremental add/modify/delete with deterministic identity, clean replacement and recoverable failures.
5. Implement project-scoped lexical + semantic retrieval and exact source inspection.
6. Expose tool contracts to AI-Team without replacing its session manager or agent implementation.
7. Test retrieval without invoking an LLM; evaluate assistant-grounded answers in a separate stage.
8. Preserve project isolation, source provenance and an audit trace for each query.
9. Establish comparison baselines and record measured performance.

### 2.4 Out of scope (V1)

- Automatically generating/sending official NGO reports; notifications and deadline scheduling.
- Automatically certifying financial or contractual compliance or “project on track” status.
- Guaranteed extraction of all numeric indicators from arbitrary spreadsheets.
- Cloud multi-tenant production hosting, enterprise SSO or organization-wide rollout.
- Fine-tuning embedding models, complex multi-agent query planners, knowledge graph, sophisticated reranking.
- Real-time Drive/SharePoint synchronization, unless later explicitly requested.

### 2.5 Core functional acceptance

- Given a valid new project folder, `ingest_project` indexes supported material with no hand-authored business schemas.
- Given a scoped query, `search_project` returns ranked passages plus stable document, page/sheet/section references.
- Given a result, `open_source` returns the original source excerpt/address (not just the vectorized approximation).
- Given a modification or deletion, stale chunks no longer appear after a successful synchronization.
- Given a question whose answer is absent, the assistant reports insufficient evidence rather than inventing a fact.
- Given identical inputs and frozen versions, the index and evaluation can be rebuilt and results compared.

## 3. Context and architecture

### 3.1 Boundaries

```
  NGO directory(ies)                           AI-Team (already exists)
      │                                               ▲
      ▼                                               │
  discovery + hashing                                 │ tool response
      ▼                                               │
  parsing / normalization                             │
      ▼                                               │
  structure-aware chunks       user question -> AI-Team agent
      │                                  │             ▲
      ├──> SQLite manifest + source map   └-> search_project / open_source
      │                                         │
      ├──> SQLite FTS5 (lexical)   <-------------┤
      └──> local embedder -> Chroma <-----------┘
                           │
                      ranked evidence
```

**Ingestion** is an offline/update operation, independent of LLM inference. **Query-time retrieval** consists of embedding the question, searching index(es), merging ranked candidates, applying project filters, and returning source-backed evidence. **Generation** uses AI-Team and is deliberately downstream.

### 3.2 Interfaces (language-neutral contract)

```python
# Pseudocode contracts; implement in Python module or service adapter.
ingest_project(project_id: str, root_path: str, *, dry_run=False) -> IngestReport
search_project(project_id: str, query: str, *, top_k=8,
               filters=None, as_of=None) -> SearchResponse
open_source(project_id: str, document_id: str, *,
            location=None, max_chars=6000) -> SourceResponse
get_ingestion_status(project_id: str) -> StatusReport
```

`search_project` response example (illustrative only):

```json
{
  "query_id": "q_...",
  "project_id": "ngo_project_a",
  "index_generation": "g_...",
  "retriever_version": "hybrid_v1",
  "results": [
    {
      "chunk_id": "ck_...",
      "document_id": "doc_...",
      "source_path": "proposal/project.pdf",
      "source_location": {"page": 8, "heading": "Project indicators"},
      "excerpt": "[actual extracted source text]",
      "rank": 1,
      "retrieval_channels": ["dense", "lexical"],
      "score_details": {"fusion_rank": 1}
    }
  ],
  "warnings": []
}
```

Scores are **ranking evidence**, not probabilities of factual correctness. The tool must not invent a citation or silently omit retrieval failures. Real identifiers and result values are generated at runtime.

### 3.3 Repository scaffold

```text
project-assistant-rag/
  README.md                         # setup + runbook
  pyproject.toml                    # pinned dependency ranges / lockfile
  configs/retrieval.yaml            # versioned, reviewed tuning parameters
  src/project_rag/
    api.py                          # public tool contracts
    discovery.py                    # scan, hashes, deleted-file detection
    parsers.py                      # format-specific adapters
    schema.py                       # document, block, chunk, citation models
    chunking.py                     # structure-aware + table-aware splitting
    embeddings.py                   # explicit local model adapter
    vector_store.py                 # Chroma adapter, no hidden model choice
    lexical_store.py                # SQLite FTS5 adapter
    manifest.py                     # SQLite document state + build versions
    indexing.py                     # idempotent orchestration / reconciliation
    retrieval.py                    # filter, dense/lexical fusion, dedup
    sources.py                      # validated read-only source opens
    trace.py                        # structured diagnostics, no raw PII logs
    ai_team_adapter.py              # thin tool registration only
  tests/fixtures/synthetic/         # public and safe fixture documents
  tests/unit/
  tests/integration/
  tests/evaluation/
  eval/questions.jsonl              # reviewed expected evidence, not generated truth
  eval/results/                     # run artifacts; never raw NGO files in git
  local_data/                       # .gitignored manifest/index/cache
```

**Deployment seam:** Start with an isolated Python package/CLI. If AI-Team's tool runner can import Python, use a thin in-process adapter; otherwise expose a small localhost HTTP tool service. Do not assume the current AI-Team API shape until its actual registration and permission mechanisms are inspected. Avoid changing AI-Team internals beyond the supported tool seam.

## 4. Data lifecycle and indexing strategy

### 4.1 Discovery and inventory are automated, not manual organization

Recursively scan a configured root, preserve relative path and original source, record extension, size, modification timestamp (diagnostic only), SHA-256 content hash, and discovery status. Ignore temp/lock files, known derived outputs, vector index directories, and hidden secrets. Never silently discard unknown formats; record `UNSUPPORTED`.

Identify duplicates by content hash, but **retain each source occurrence and its access/project associations**. Same bytes in two different folders can still have different contextual significance. Never use pathname alone as document identity; detect renames and prevent stale pointers.

**Document state machine:** `DISCOVERED → PARSING → PARSED → INDEXING → READY`, or `FAILED/UNSUPPORTED/QUARANTINED`. On modification, keep last-known-good indexed version until new generation is successfully committed; then replace. Query a consistent active generation.

### 4.2 Parsing and document normalization

Use Docling as the starting adapter for supported formats; pin a tested version. Preserve logical blocks (headings, paragraphs, lists, tables), page locations, worksheet/cell references when available. Validate extraction output against several representative files before scaling across the folder. If a parser fails, retain diagnostics and mark the file inaccessible for retrieval.

Special cases:

- **PDF:** Digital text first; detect likely scans and report `OCR_REQUIRED` rather than silently indexing empty text. Add OCR only when corpus inspection establishes need and permissions.
- **DOCX:** Keep heading hierarchy and tables; watch tracked changes, comments, appendices, and duplicated headers/footers.
- **XLSX:** Preserve sheet names, table headers, row/cell context, units, totals, and formulas vs displayed/cached values. Missing cached formula results must not be treated as zero. A generic flattened paragraph is insufficient for reliable counts.
- **CSV:** Validate encoding, delimiters, header interpretation, empty values; preserve row references.
- **Images / embedded scans:** Report coverage gap; no implicit image understanding in V1.

**Parsing gate:** sample extracted text/tables for each format, especially the exact indicator definition and numeric evidence. Broken extraction is fixed before embedding.

### 4.3 Document, version and chunk identities

Recommended identity levels:

- `project_id`: stable project slug/UUID, supplied by calling application.
- `source_id`: stable logical source reference (manifest-maintained; survives rename when possible).
- `version_id`: hash of source content and parser/config versions.
- `chunk_id`: deterministic hash of `(project_id, source_id, version_id, block_path, chunk_ordinal, chunker_version)`.
- `index_generation`: immutable manifest of active source versions for query traceability.

Never use a naive sequential integer chunk ID across rebuilds. Retain lineage from every chunk to source file plus section/page/sheet/cell. The index is a derivative artifact, not the source of truth.

### 4.4 Chunking policy (initial, not sacred)

**Start with structure-aware chunking**: group paragraphs under their nearest heading; keep short sections intact; split oversized sections at logical boundaries. Keep table headers and units attached to relevant rows; if tables are too large, chunk by row groups with repeated header context. Retain surrounding text for definitions and acronyms.

First experimental values: target ~250–350 model tokens per chunk; maximum comfortably within the embedder's supported context; overlap ~30–50 tokens for free text **only if boundary errors appear**. Table overlap is semantic (header repetition), not blind token duplication. These are candidates to evaluate, not domain truths.

Chunking hazards: splitting a numeric value from its unit/column header; separating an indicator from its target; losing annex headings; blending two projects; overloading chunks with irrelevant boilerplate; treating whole spreadsheet sheets as one blob; token truncation by embedding model.

**Key design:** Chunk records store clean semantic text for embeddings, contextual title/heading, original text/address for citations, and block type. Keep ingestion/embedding context modifications auditable rather than rewriting evidence.

### 4.5 Embeddings — local first

**Proposed baseline:** `intfloat/multilingual-e5-small` via Sentence Transformers, locally on CPU or GPU if available. It produces 384-dimensional embeddings, covers multiple languages, and requires exact asymmetric prefixes: `passage: ` for indexed text and `query: ` for user queries. Keep explicit prefixing inside the embedding adapter and test it. Its model configuration has a 512-position limit, so protect against token truncation. Bulgarian-language retrieval quality is **not assumed**: measure with Bulgarian, English and cross-language questions from the corpus.

Operational safeguards:

1. Pin model identifier and *revision*, pooling/normalization policy, tokenizer version, prefix rules and library versions in an `embedding_profile`.
2. Check model license and trusted download provenance; do not download arbitrary executable model code.
3. Batch embedding with bounded CPU/RAM concurrency; cache by `(content_hash, embedding_profile)`.
4. Embed each unchanged chunk once; new documents add only their changed chunks.
5. Keep dimension, similarity metric and preprocessing consistent at write and query time.
6. Replacing the model requires **new index generation/collection and re-embedding**; do not mix vectors from models that happen to share dimensions.
7. Store embeddings locally; API embeddings are optional only if the local model fails quality or throughput gates and document-processing permission explicitly covers third parties.

An API model may improve language retrieval or lower local maintenance, but entails cost, external data transfer and dependency on availability. Do not pay for embedding generation before validating a local baseline.

### 4.6 Chroma decision and correct usage

**Proposed initial vector store:** Chroma, local Python `PersistentClient` for one developer and one writer. Set an explicit persistent path outside the source folder. Use an explicit collection, explicit externally generated embeddings and explicit HNSW cosine distance; never rely on an implicit default embedding function or metric. Current Chroma documentation identifies default HNSW distance as `l2`, not cosine.

**Initial collection model:** One physical collection for a given embedding/index schema (`rag_chunks_e5small_v1`), with `project_id`, `document_id`, `version_id`, `chunk_id`, section, source type and status as primitive metadata. Always apply project filters at the retrieval boundary. For real multi-user authorization, do **not** treat client-supplied `project_id` as proof of access; derive authorized project scope server-side. A per-project collection is an alternative if policy requires stronger administrative separation, but is not the V1 choice.

**Important Chroma rules:**

- Choose cosine **at collection creation**; validate current installed version's configuration keys. Do not use defaults for distance or embedding function.
- Insert `ids + embeddings + documents + metadatas` explicitly; use batch operations within the installed library's batch limit.
- `upsert` handles same-ID replacements, but does not automatically remove old IDs when a document shrinks or chunk boundaries change. Delete the superseded version's IDs after successful replacement; prove this with a regression test.
- Persist collection schema/config/model ID in the local manifest; verify compatibility on startup; refuse mismatched writes.
- Query with server-enforced `where={"project_id": project_id}` (or equivalent) and inspect documents plus metadata. `where_document` is a text filter, not a substitute for a genuine keyword/BM25 ranker.
- Chroma distances are not calibrated confidence values; compare ranks and labeled relevance.
- Treat local `PersistentClient` as a development/single-process solution. Chroma documentation advises server-backed configuration for production concurrent clients. Do not put one writable persistent store behind uncoordinated independent worker processes.
- Backup/recovery must include index, SQLite manifest and matching generation; rebuild from originals if derivatives are lost.

**Migration conditions:** Move to a server-backed vector store only if concurrent users/writers, operational constraints, or measured retrieval needs justify it. Storage adapter keeps this reversible.

### 4.7 Keyword / hybrid retrieval

Use SQLite FTS5 BM25-like ranking as a lightweight independent lexical channel, indexed from the **same active chunks**. Preserve project scope in the indexed relation and enforce filtering *before* returning results. SQLite's default Unicode tokenization is not Bulgarian stemming; evaluate behavior on inflections and spellings rather than assuming it solves every lexical case.

Retrieve candidates from:

- **Dense search**: semantic/approximate matching (good for paraphrases).
- **Lexical search**: exact names, grant numbers, dates, identifiers, terms and numeric-looking strings.

Fuse ranks with Reciprocal Rank Fusion (RRF) or a comparably simple deterministic method. Start with top 10–20 candidates per channel and return 6–8 after deduplication and diversity across files; **tune only from evaluated failures**. Record channel contributions, fusion rank, matched source and filters. Do not describe all this as 'Chroma hybrid search' unless the installed Chroma version actually provides the exact feature being used; the planned lexical channel is explicitly in SQLite.

### 4.8 Incremental updates and failure safety

`ingest_project` must be idempotent. On each run: discover → compare hashes → parse changed/added files → construct new chunks/embeddings → validate and stage → publish active generation → remove stale derivative records. Re-run with no changes: no new embeddings, identical active membership. Deleting a source from the folder removes it from the *active searchable set*, with optional tombstone/audit metadata retained for diagnosis.

Production-minded protections: dry-run, staging generation, retryable per-file failures, resume after interruption, bounded memory and concurrency, file-access errors reported rather than swallowed, and index rollback/rebuild path. Do not rely on wall-clock `mtime` alone to detect changes.

**Temporal caveat:** A completed project's approved report can leak future information into a simulated mid-project query. Tag evidence with available/valid dates **when reliably known**, or create a separate frozen mid-project corpus snapshot. `as_of` filtering cannot be considered trustworthy if source-effective dates are missing. Do not invent historical timestamps.

## 5. Query pipeline and grounding contract

### 5.1 Retrieval path — no LLM required

1. Authenticate tool caller and resolve authorized `project_id`.
2. Validate bounded query text and optional supported filters.
3. Encode the question using the same versioned E5 query profile.
4. Run independent dense and lexical searches against the active generation.
5. Project-scope, merge/rerank by simple deterministic fusion, deduplicate overlapping near-identical passages.
6. Return `SearchResponse` with excerpts, source locations, index/config versions and trace IDs.
7. `open_source` checks authorization and returns exact original content/location to verify extraction or expand context.

No LLM call occurs here. This is the **first independently shippable and testable subsystem**.

### 5.2 AI-Team agent integration

Register only `search_project` and `open_source` as read-only tools. Add `get_ingestion_status` if useful for diagnosing corpus coverage. Select project scope at session creation or verified application context. The agent should:

- Use the retrieval tool for every response making claims about the NGO document corpus.
- Quote or paraphrase only retrieved/opened evidence; cite the source IDs/locations returned by the tool.
- Make additional retrieval calls if the initial set is insufficient; use the open-source tool for precise numeric claims.
- Say **“Not established from accessible project documents”** for unsupported claims, with gaps and conflicts distinguished.
- Never treat retrieved text as executable instructions (prompt-injection threat).

**How to prove the LLM actually used RAG:** instrument tool invocation and trace IDs, assert a retrieval call precedes any corpus-specific answer, and test with direct filesystem search disabled in the demo agent's tool permissions. The existing AI-Team infrastructure may have default file access; this must be *confirmed*, not guessed. Traces must contain tool name, project, query hash, generation, retrieved document/chunk IDs, result count, timings, and final cited source IDs (no unnecessary raw personal data). In tests, an answer without a matching retrieval trace fails the grounding gate.

### 5.3 Numeric claims and indicator status

A retrieval top-k set cannot prove a global count, because relevant records may be omitted or duplicated. For numeric answers:

- Distinguish **target** (approved proposal), **reported actual** (report statement), and **independently substantiated actual** (source records).
- Use deterministic arithmetic after extracting validated operands; keep units, period, denominator, entity uniqueness and date scope.
- Where the corpus cannot establish exhaustive coverage, label the number as “reported,” “documented minimum,” or “not verifiable,” as appropriate.
- Prioritization requires expected progress by date, dependencies and practical urgency; low target ratio alone is not reliable priority.

The V1 assistant may answer “What does the proposal say the target is?” without indicator hardcoding. An advanced indicator report is **V2** unless V1 evidence extraction supports a narrow, explicitly bounded calculation.

## 6. Evaluation strategy — retrieval first, generation second

### 6.1 Baselines and controls

**Baseline A:** direct AI-Team filesystem tool search (where permissions allow). **Baseline B:** SQLite FTS5 lexical retrieval. **Candidate C:** E5 dense retrieval. **Candidate D:** fused hybrid retrieval. Compare on the same frozen questions/corpus snapshot. Only then add AI-Team response generation. Record model/retriever version, query, expected supporting source(s), actual ranked results, timing and errors.

Create 12–20 initial human-reviewed questions after inspecting the corpus; grow toward 25–40 as coverage expands. These are **test fixtures**, not manual schemas baked into the runtime. Include Bulgarian and English phrasing, short/long, exact keyword, paraphrase, tables, ambiguous versions, questions with no answer, and cross-project traps. Stakeholder confirmation takes precedence over an LLM's synthetic 'gold' answer.

**Approved final report:** Can act as an answer reference; for historical questions, exclude it from candidate retrieval if it was not yet available. It is **not automatically proof** that every individual source datum is correct or complete.

### 6.2 Metrics and gates

| Layer | Measure | How evaluated | Initial acceptance rule |
|---|---|---|---|
| Parsing | coverage and fidelity | extraction status + sampled source-to-text checks | No silently skipped relevant format; critical evidence fields faithfully extracted |
| Indexing | deterministic/idempotent updates | same input twice; modify/delete cases | No duplicate active chunks; stale records absent; model/config mismatch blocked |
| Retrieval | evidence Recall@5; MRR | ranked hits against human-tagged source spans | Beat or match simplest workable baseline on representative set; no regression on critical questions |
| Isolation | cross-project leakage | unauthorized and wrong-project queries | **Zero** cross-project disclosures in tests |
| Grounding | cited claims supported | manual evidence review per claim | No unsupported critical numeric/compliance claims in acceptance set |
| Abstention | absent/ambiguous answers | adversarial no-answer/conflict cases | Explicit no-evidence response, not plausible fabricated answer |
| Efficiency | p50/p95 latency; CPU/RAM; index time | repeatable local benchmark | Measured and acceptable to stakeholder; no premature numeric SLA invented |
| Integration | tool provenance | mocked agent + real instrumented agent | Every corpus answer has corresponding retrieval trace and resolvable citations |

**Release rule:** Blocking correctness/security tests all pass; measured quality is reported with sample size, confidence limitations and examples of failure. If performance does not beat direct search, do not claim the RAG layer provides value. Revise or stop.

### 6.3 Must-have automated tests

- Parsing: DOCX heading; PDF text vs scan; XLSX multi-sheet numeric table/formula cache; CSV accents/Bulgarian; unsupported/encrypted file.
- Chunk boundaries: target and unit in same evidence span; table header repeated; citation points to correct page/sheet.
- Embeddings: model version/prefix/dimension match, length truncation guard, deterministic fixture ranking tolerance.
- Index lifecycle: ingestion twice no duplicates; edited file old content removed; shortened document no orphan chunks; deleted source unavailable; interrupted run recoverable; corrupt index rebuild.
- Retrieval: exact identifier, paraphrase, Bulgarian, cross-language, table evidence, explicit project filter, empty result behavior.
- Security: unauthorized project and path traversal rejected; malicious document instruction not executed; logs do not expose raw participant information.
- Agent: no RAG tool call → no sourced answer; citation identifiers resolve to existing source; disabled filesystem fallback; failures surfaced.
- Temporal: final report cannot appear in a pre-report frozen snapshot.

### 6.4 Failure taxonomy and remediation order

When an answer fails, diagnose in this order:

`source missing/unauthorized → parser failure → chunk boundary failure → embedding/query mismatch → retrieval rank/filter failure → incomplete evidence aggregation → agent reasoning/generation failure → citation mismatch`.

Do **not** reflexively add an LLM reranker, agent loop, larger embedding model, bigger context or new DB before locating the broken seam. Maintain a small failure ledger with reproducer, suspected mechanism, corrective change, counterfactual baseline and regression test.

## 7. Retrieval upgrades — candidate backlog, not default scope

| Technique | Problem it addresses | V1 decision / activation trigger |
|---|---|---|
| Hybrid BM25 + vectors | Exact terms missed by semantic search or paraphrases missed by exact matching | **Implement** simple lexical+dense fusion; benchmark against both alone |
| Metadata prefilter | Wrong project, wrong period, wrong document class | **Implement** project scope now; add temporal/document class filtering only with trusted metadata |
| Parent-section expansion | Answer spans multiple adjacent chunks | **Implement** source-opening tool; use only when queried |
| Heading/table-aware chunks | Units, references, structure lost in flat text | **Implement** from ingestion |
| Query rewriting/decomposition | Broad questions need multiple evidence searches | **Defer** until measured missed multi-hop retrieval |
| Cross-encoder reranking | Correct candidate appears in top 20 but rank is too low | **Defer**; measure language fit and latency before adopting |
| Contextual chunk summaries | Isolated chunks lack context | **Defer**; first use heading prefixes and source expansion; guard against generated false context |
| OCR / image understanding | Real evidence exists only in scans | **Conditional** on corpus coverage audit; benchmark fidelity |
| Indicator-specific aggregation | Need complete counts, unique entities, unit-aware progress | **V2** structured extraction, full scan, dedupe, deterministic calculation |
| Portfolio cross-project Q&A | Several projects are in authorized scope | **V2** only after isolation and second-project onboarding pass |

All augmentation choices must have: named failure, baseline, change, cost/latency effect, measured outcome, rollback path.

## 8. Security, privacy and operating requirements

**Confidentiality first.** NGO documents can contain names, contact details, participant dietary preferences and other personal data. Obtain/confirm authority for the intended processing and do not upload raw files to hosted embedding/LLM providers without an explicit documented decision. Dietary restrictions may be sensitive in context; exclude personal registration records from initial demos if unnecessary. Keep raw corpus, model cache and vector index in access-controlled local paths and out of git; use synthetic fixtures in CI.

**Read-only agent:** RAG tools cannot write/rename/delete original source files. Restrict roots to configured project folders; canonicalize and validate paths to block traversal and symlink escapes. Do not accept arbitrary local paths in `open_source` from the LLM. Treat source documents and extracted text as untrusted data, never instructions. Limit max tool output size and query rate. Log redacted metadata, not full raw text. Define retention/deletion process for derivative index and caches.

**Operations:** Maintain `ingest_report.json`, `status`, query traces, versioned config, reproducible dependency lock, recovery/rebuild command, basic storage backup, and operator guide. Detect empty or partially indexed project and display ingestion health prominently. Require explicit approval before enabling any external network/data egress.

**Local-to-multiuser boundary:** The initial demo is not automatically a secure multi-user production deployment. Production deployment requires verified identity, per-project ACLs, server-backed storage, backups, availability policies and operational ownership. This is a release boundary, not hidden scope expansion.

## 9. Work packages and handoff sequence

### WP0 — Inspect interfaces and corpus (read-only)

**Inputs:** Source folder, AI-Team tool registration surface, available compute; no repo assumptions.  
**Tasks:** Count formats/sizes; identify parsing/PII blockers; sample source structure; inspect exact AI-Team tool contract; capture chosen local model/runtime feasibility.  
**Outputs:** `inventory.json`, `corpus_risks.md`, `integration_seam.md`, initial frozen test questions.  
**Gate:** Supported-format coverage known; no unapproved egress; no edits to original corpus.

### WP1 — Parsing and provenance (no vectors yet)

**Tasks:** Implement discovery, hashing, manifest schema, parser adapters, normalized source blocks, citation locations, per-file failures, parse smoke fixtures.  
**Gate:** Proposal and representative evidence parse faithfully; each extracted block traceable to source; duplicate and unsupported handling tested.

### WP2 — Search-only MVP (no LLM)

**Tasks:** Implement E5 embedding adapter and explicit collection; FTS5; index lifecycle; scoped search; source open; query trace.  
**Gate:** CLI can retrieve exact and paraphrased evidence, with citations, update/re-delete cleanly, without running AI-Team or an LLM.

### WP3 — Retrieval quality study

**Tasks:** Freeze corpus generation; label relevant sources; run lexical, dense, hybrid and optionally direct file search; evaluate quality, latency, costs, dominant errors.  
**Gate:** Written evidence the candidate is useful versus baseline; no project-filter failures; documented unresolved failures.

### WP4 — AI-Team adapter and grounded demo

**Tasks:** Register minimal tools; enforce project context; demonstrate tool trace; ground answers in retrieved text; validate citations and no-answer responses.  
**Gate:** Manager's example questions answered with source references where supported; direct filesystem fallback disabled in demo; critical ungrounded claims blocked.

### WP5 — Second-project and lifecycle proof

**Tasks:** Add a small separate synthetic or consented project without code changes; test project isolation, add/edit/delete, conflicting identically named files and version migration.  
**Gate:** Onboard second project by config + ingestion only; no scope leakage or stale records.

**Critical path:** `WP0 → WP1 → WP2 → WP3 → WP4 → WP5`. Do not start with UI or advanced agent prompts. V1 can stop at WP4 if the second-project demonstration is unavailable, but extensibility cannot be claimed validated until WP5 passes.

## 10. Architecture decisions (explicit and reversible)

| ADR | Decision | Rationale | Failure/trigger for reconsideration |
|---|---|---|---|
| 001 | Retrieval module independent of AI-Team | Isolate retrieval errors and compare without LLM | Tool seam incompatible; create localhost adapter, not rewrite |
| 002 | Local Docling first | Mixed PDFs/Office and provenance support | Fidelity failures in actual corpus → per-format parser |
| 003 | Local multilingual E5-small | No API embedding cost; Bulgarian/English candidate | Measured multilingual Recall@k inadequate → compare alternatives |
| 004 | Chroma PersistentClient, cosine, explicit embeddings | Simple inspectable local vector index | Concurrent writers/users → backed service |
| 005 | SQLite FTS5 lexical channel and manifest | Exact identifiers and controlled lifecycle | Measured relevance/scale limitation → replace adapter |
| 006 | Structure/table-aware chunks | Preserve numeric semantics and source evidence | Regression set shows better segmentation |
| 007 | Hybrid fusion only; reranker deferred | Good discriminating baseline with low complexity | Correct evidence in candidate set but ranked too low |
| 008 | One schema-specific collection, project metadata filter | Reusable onboarding, fewer indexes | Stronger segregation or different model per project required |
| 009 | Re-index immutable new generations on model/parser change | Prevent silent mixing and stale evidence | Never update incompatible vectors in place |
| 010 | No indicator-specific registry in RAG runtime | Generic assistant should discover document definitions | Future deterministic indicator engine becomes separate module |

## 11. Open decisions — must be resolved by evidence, not guessed

1. What file types and proportions are really present? Are PDFs mostly scanned? Are participant lists sensitive?
2. Are Bulgarian and English mixed in sources and questions? E5 language suitability requires testing.
3. Which AI-Team tool API and sandbox boundaries actually exist? Can the agent's direct file tools be disabled per session?
4. Is this to remain a personal local demo or be exposed to an NGO user across machines? Hosting/security differs materially.
5. Can project managers provide 10–20 accurate expected evidence locations, or must an evaluator inspect source files manually?
6. Are source effective dates available, or must the first release restrict itself to current/full-history questions?
7. What actual model hardware/memory are available on the development machine? Benchmark instead of assuming CPU/GPU throughput.
8. Is document ingestion one-shot or periodically triggered? Default V1 is manual `sync`; folder watching is a later convenience.

Unanswered questions **must not block WP1's general ingestion scaffold** if no unsafe assumptions are required. Any question affecting data disclosure, scope isolation, or factual correctness must be answered before the corresponding release gate.

## 12. Definition of Done and owner review

A project is **not done** because Chroma returned search results or the chatbot produced a plausible answer. Done means:

- Reproducible setup and documented tool contracts exist.
- Supported corpus files are parsed with reportable coverage/failures and resolvable citations.
- Embedding/collection/chunker versions are pinned and incompatible configurations rejected.
- Incremental sync is idempotent; removed/changed files cannot leak stale evidence.
- Project-scoped lexical+dense retrieval works without LLM and is compared against alternatives.
- Independent retrieval metrics and labeled examples are recorded, including error taxonomy.
- AI-Team calls retrieval tools verifiably and can abstain; retrieval provenance matches citations.
- Privacy and path controls pass; no unapproved remote processing occurs.
- Second-project separation is demonstrated before making a multi-project capability claim.
- Stakeholder value is assessed on representative real questions, not on a single showcase answer.

### Owner-level adversarial review

**Measured object:** retrieval quality, grounding reliability, traceability, and incremental lifecycle — not model eloquence.  
**Alternative mechanism:** direct agent file search and keyword search, both benchmarked.  
**Causal seam:** verify parse correctness before indexing, retrieval before generation, and source evidence before numeric conclusions.  
**Invalidation:** failed evidence recall, unsafe project leakage, stale chunks, or no improvement over simpler methods.  
**Smallest next action:** WP0 read-only inventory + AI-Team tool seam inspection, then parse a single approved proposal and one evidence file **without writing embeddings yet**.

## 13. Reference material and factual checkpoints

References are to official vendor/model documentation consulted on 16 October 2026. Pin package versions during implementation and revalidate exact APIs against the installed packages.

1. **Chroma clients / persistence / production recommendation:** https://docs.trychroma.com/reference/python
2. **Chroma collection configuration, HNSW distance/parameters, explicit embedding functions:** https://docs.trychroma.com/docs/collections/configure
3. **Chroma metadata filtering:** https://docs.trychroma.com/docs/querying-collections/metadata-filtering
4. **Chroma collection APIs (upsert/delete/query):** https://docs.trychroma.com/reference/typescript/
5. **Docling supported formats, tables and OCR:** https://docling-project.github.io/docling/usage/supported_formats/
6. **Docling DocumentConverter:** https://docling-project.github.io/docling/reference/document_converter/
7. **E5-small model card (prefix rules, multilingual support):** https://huggingface.co/intfloat/multilingual-e5-small
8. **E5-small config (hidden dimension and 512-position limit):** https://huggingface.co/intfloat/multilingual-e5-small/blob/main/config.json
9. **Sentence Transformers retrieve-and-rerank pattern:** https://github.com/huggingface/sentence-transformers/tree/main/examples/sentence_transformer/applications/retrieve_rerank

---

**Specification discipline:** All numeric sizes, thresholds and parameter values above are **initial hypotheses unless directly defined by a referenced vendor/model contract**. The repository work must turn them into measured, versioned choices. This document does not imply access to or an audit of the user's NGO files or AI-Team implementation.

# Amendment A01 — GPU-Aware Qwen3 Embeddings and Reusable Indexing Stages

**Applies to:** PROJECT_ASSISTANT_RAG_SPEC v1.0 / v2.0. A01 is authoritative wherever it differs.
**Status:** Approved design direction; retrieval quality + hardware feasibility unverified until measured.

This file records (1) the binding decisions adopted into the codebase and (2) the **measured
environment deviation** found on this machine, reported per directive J rather than silently
substituted.

---

## A01 implementation state in this repo

| A01 section | Requirement | Status in code |
|---|---|---|
| A1, D | Qwen3-Embedding-4B, 2560-dim, CUDA, FP16 infer / FP32 storage, query-prompt (no E5 prefix) | `Qwen3Embedder` + `embedding_profile_qwen3` config implemented |
| A4 | 0.6B + BGE-M3 as optional comparators, not auto-fallback | documented; not wired into critical path |
| A5, G | Chroma dense + SQLite FTS5 lexical, deterministic RRF, renamed collection `rag_chunks_qwen3_4b_2560_v1` | collection renamed; stores unchanged |
| A6, C | Separate rerunnable stages with fingerprint keys; cached intermediate artifacts | **`stages.py` artifact cache** (discover/parse/chunk) implemented |
| C | chunk IDs from source/version/block lineage, not position/UUID | already held from v2; retained |
| D | profile pinned alongside vectors; fail-closed on mismatch | manifest compatibility guard retained + dim/profile check |
| E | WSL/CUDA feasibility smoke before bulk embedding | **`scripts/feasibility_smoke.py`** implemented; fail-visible |
| F | chunk target 350–700, cap 2048, token counting + truncation detection, distinct `retrieval_text` | implemented in `chunking.py` + `schema.Chunk.retrieval_text` |
| H | staged gates, regression cases | tests added: profile mismatch, no-prefix leak, stage-cache reuse, truncation |

## Measured environment deviation (directive J)

**Observed, not assumed:**

- Windows host: `nvidia-smi` absent → no NVIDIA GPU on host.
- WSL: `Ubuntu-24.04` present but was **Stopped**; when started, `nvidia-smi` → `NO_CUDA_IN_WSL`.
- Earlier host probe: `nvidia-smi` unavailable; `os.cpu_count()` = 16; CPU-only.

**Consequence:** The A01 §E CUDA feasibility gate **cannot pass on this machine**. Per §A2 and
§E ("return a visible failure if still infeasible") and §J ("report deviations … rather than
silently substituting"), the pipeline:

1. Implements the full Qwen3-4B adapter + profile + CUDA gate.
2. Runs the feasibility smoke, which **fails visibly** at the CUDA/VRAM check — no silent CPU
   fallback, no silent swap to 0.6B inside a 4B profile, no silent dimension change.
3. Continues to use the offline deterministic `HashEmbedder` as the **test/dev** embedder only,
   which is not a production profile and is rejected by `Config.assert_release_safe()`. This is a
   test fake, not a substitution of the production embedding model.

**To activate Qwen3-4B** on a real CUDA box (~20 GB VRAM, WSL): set `embedding.provider: qwen3`,
install `pip install -e ".[qwen3]"`, run `scripts/feasibility_smoke.py`, pin the resolved HF
revision into `configs/retrieval.yaml → embedding.qwen3.revision`, then ingest. Switching to a
Qwen3 profile from any prior profile forces a new index generation (ADR-009) — the compatibility
guard refuses mixing.

## Stage-cache contract (A01 §C)

`stages.py` persists cached artifacts under `local_data/cache/<project_id>/`:

- `[1] PARSE` key = `sha256(source_content_hash + parser_version + parser_config)`
- `[2] CHUNK` key = `sha256(parsed_artifact_id + chunker_version + chunk_config)`

Re-running with an unchanged source reuses the parse+chunk artifacts (no reparse/rechunk).
Changing the chunker config invalidates only `[2]` and downstream; changing the parser config
invalidates `[1]`+`[2]`. Embedding `[3]` is keyed by `chunk_content_hash + embedding_profile_id`
so a model/profile change re-embeds without reparsing. The acceptance trace
(parse → chunk → profile → Chroma → retrieved passage → source address) is preserved end-to-end.

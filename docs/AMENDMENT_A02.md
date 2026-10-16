# Amendment A02 — CPU execution of the Qwen3 profile (supersedes A01 §E gate on this machine)

**Applies to:** A01 §A2/§E/§J. **Status:** adopted 2026-10-17 on measured evidence; quality still to be measured.
**Touches:** `configs/yp_qwen06.yaml`, `configs/yp_qwen4b.yaml`; no code contract changed
(`Qwen3Embedder` already accepted `device: cpu`; only `device: cuda` fails closed).

## What changes
A01 required CUDA and failed visibly without it. It still does (default `device: cuda`). A02 adds
**explicit, recorded CPU profiles** — `device: cpu`, `inference_dtype: bfloat16` — selected by
config only, never by silent fallback. The model, revision, dimension, prompts and FP32 vector
storage stay as in A01. `inference_dtype`/`device` are recorded in the profile for drift diagnosis.

## Why (measured, `scripts/embed_bench.py`, real chunks, Ryzen 7 7800X3D 8c/16t, 31 GB RAM, no GPU)

| Model | dtype | tokens/chunk (own tokenizer) | tokens/s | Full index estimate |
|---|---|---|---|---|
| multilingual-e5-small | fp32 | 537 (24 chunks, 15 > 512-token limit) | 9,194 | 0.04 h |
| Qwen3-Embedding-0.6B | fp32 | 876 | 459 | 1.2 h (fp32 only; bf16 not yet benchmarked) |
| Qwen3-Embedding-4B | bf16 | 824 | 132 (6-chunk sample) | 3.9 h |

- The corpus is small (≈2.3k chunks), so a one-time ~4 h CPU embed of the A01 model is feasible,
  and it is **resumable** (`embedding_cache.py` commits per batch of 64) and **never repeated for
  unchanged chunks** (ledger F13, fixed).
- Cyrillic costs ~1.6× more tokens in Qwen's tokenizer than in E5's; chunk size must be set in the
  model's own tokens (A01 §F), so chunk sets differ per model (see Confound).

## Confound to remember
Dense runs differ in model **and** chunk boundaries (chunk target is in each model's tokens:
E5 300, Qwen 500). Differences between runs cannot be attributed to the model alone. Lexical-only
results are reported on every chunk set for reference.

## Reversibility
Delete the per-config `local_data/<name>/` directory and the profile is gone; the CUDA path is
untouched. If a GPU becomes available, flip `device: cuda`; the profile key is unchanged, so
A01 §D mismatch protection applies only to model/revision/dim/prompt, not to device.

## Pilot gate before the multi-hour run (heavy-run discipline)
1. 0.6B first (≈ tens of minutes): proves the dense pipeline, query prompt, Chroma, fusion end-to-end.
2. 4B (~4 h) is launched only after the 0.6B run shows the pipeline is healthy (finite vectors,
   unit norm, Recall@5 not below lexical on the exact-match questions). Abort criteria: any
   non-finite vector, RSS > 24 GB, throughput < 60 tokens/s.

## Pilot finding (2026-10-17): bf16 vectors are not exactly unit-norm
The A01 §D output check aborted the first 0.6B run on batch 1 (norms 0.998). Measured: bf16 vs
fp32 cosine = 0.9998–0.9999, so the vectors are faithful; only the norm drifts. The adapter now
re-normalizes in FP32 before validation (`test_qwen3_lowprecision_norm.py`). The health gate worked
as designed: a bad run died in seconds, not hours.

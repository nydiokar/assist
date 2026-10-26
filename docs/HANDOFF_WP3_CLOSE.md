# Handoff prompt — WP3 gate close + eval expansion

Paste this entire prompt to Opus (or your strongest available model) as a single message.

---

## Context you must read before touching anything

You are working on a RAG project at `C:\Users\Cicada38\Projects\assist`.
The project has a strict process discipline — read these files in order before making any changes:

1. `docs/AGENTS.md` — project identity, corpus description, hard rules, current state summary
2. `docs/AMENDMENT_A01.md` — authoritative deviations from the base spec
3. `docs/AMENDMENT_A02.md` — CPU execution of Qwen3, pilot gate outcomes
4. `docs/EVAL_LOG.md` — all prior eval runs (append-only; never edit existing entries)
5. `docs/FAILURE_LEDGER.md` — all known failures and their status
6. `docs/PROJECT_ASSISTANT_RAG_SPEC_v1.md` §6, §9 — gates, metrics, WP3 definition

**Precedence: A02 > A01 > spec v2 > spec v1.**

---

## What has already been done (verify with `git log --oneline -20`, do not assume)

- WP0–WP3 scaffold green on synthetic fixtures, 53 tests passing
- Real corpus (`yspdb/`, 172 text files) ingested with Qwen3-4B GPU embedder into index `yp_qwen4b_gpu`
  (config: `configs/yp_qwen4b_gpu.yaml`, data: `local_data/yp_qwen4b_gpu/`, project_id: `ypq`)
- Eval was run against 18+10 = 28 reviewed questions; results in:
  - `eval/results/yp_qwen4b_gpu_main.json`
  - `eval/results/yp_qwen4b_gpu_oos.json`
- A new expanded question set has been written to `eval/yp_questions_v2.jsonl`
  (39 questions, 34 scoreable, all marked `reviewed: true`, all grounded from real corpus text)

---

## Your tasks — in this exact order

### Task 0: Read the files listed above

Do not skip this. The rules are non-negotiable and the spec's gate definitions are the
acceptance criteria for everything below.

### Task 1: Validate the new question file labels

Run the label validator that the eval script uses to verify that every `expect_substring`
actually appears in the parsed text of the `expect_source_path` file.

```
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe -X utf8 scripts\eval_yp.py `
  --config configs\yp_qwen4b_gpu.yaml `
  --project ypq `
  --root yspdb `
  --questions eval\yp_questions_v2.jsonl `
  --channels lexical `
  --out eval\results\yp_v2_lexical_validate.json
```

If the validator reports label errors (substring not found in source text), fix them:
- If the substring is genuinely in the file but the path is wrong → fix the path
- If the substring is not in any parsed text → remove it from expect_substrings or replace
  with a substring that IS in the parsed text for that source
- Record any such fix in `docs/DECISIONS.md` with the date and reason

**Do not delete questions just because they are hard.** Only remove a question if its
expected answer is provably absent from every parsed chunk in the corpus.

### Task 2: Run the full eval on the new question set

Run all three channel sets:

```
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe -X utf8 scripts\eval_yp.py `
  --config configs\yp_qwen4b_gpu.yaml `
  --project ypq `
  --root yspdb `
  --questions eval\yp_questions_v2.jsonl `
  --channels lexical dense "lexical+dense" `
  --out eval\results\yp_v2_qwen4b_gpu.json
```

If Qwen3-4B requires CUDA and this machine has no GPU: the dense channel will fail
with the CUDA gate error (expected; do not suppress). Run lexical only on this machine:

```
  --channels lexical
```

and note in the log that dense + hybrid require the GPU box where the index was built.

### Task 3: Update the eval log

Append a new entry to `docs/EVAL_LOG.md` (never edit existing entries) that records:
- Date, corpus snapshot, config, question set (`yp_questions_v2.jsonl`, n=39, 34 scored)
- Results table: Recall@5 / evidence@5 / MRR for each channel actually run
- Per-question misses by ID and kind
- What it teaches vs. prior runs (compare to `yp_qwen4b_gpu_main.json` + `_oos.json`)

### Task 4: Update the failure ledger

For every failing question in the new run, check `docs/FAILURE_LEDGER.md`:
- If the failure maps to an existing open entry → update status/evidence in that entry
- If it is a new failure pattern → add a new entry (next available ID, same format)
- If a previously open failure is now **fixed** by the new question set (i.e., passes) →
  mark status **Fixed** with evidence

Do not add entries for `no_answer` questions — they are expected to have no retrieval hits
(abstention is a separate open decision tracked as F16).

### Task 5: Write the WP3 gate record

WP3 gate (spec v1 §9): *"Written evidence the candidate is useful versus baseline;
no project-filter failures; documented unresolved failures."*

Create or update `docs/EVAL_LOG.md` with a clearly marked `## WP3 Gate` section that states:

- Whether hybrid Recall@5 beats or matches the lexical baseline on the full v2 question set
- Whether any project-filter failures exist (wrong project's docs returned — see yp14 isolation)
- A disposition for each open failure: "accept at WP3 and track" vs "must fix before WP3"
- A plain statement: **WP3 PASSED** or **WP3 BLOCKED** with reason

The gate does NOT require all failures to be fixed. It requires them to be documented and
dispositioned, and the system to be demonstrably better than direct text search on the
scored questions.

### Task 6: Write the abstention decision (F16)

The no_answer questions (yp15, yp16, yp19, yp36, yp39) need a policy.
In `docs/DECISIONS.md` write a dated entry for:

- What the current behavior is (always returns top-k regardless of relevance)
- The two candidate policies: (a) score threshold cutoff, (b) evidence sufficiency check
  by the LLM at WP4
- Which one you are deferring to WP4 and why (keep it brief, this is a WP3 decision log)

### Task 7: Retire the old unreviewed question file

`eval/questions.jsonl` contains 15 questions against a synthetic Green Villages corpus
that no longer exists. It should not be present alongside the real corpus questions.
Move it to `eval/archive/questions_synthetic_gv.jsonl` and note in `docs/DECISIONS.md`.

### Task 8: Commit

Commit everything with a message that says what changed and that WP3 is closed (or blocked
if the gate was not passed). Follow the repo's existing commit style.

---

## Hard rules you must follow

- **Windows only.** Use `.\.venv\Scripts\python.exe`, PowerShell syntax.
- **No GPU on this machine.** If Qwen3-4B dense fails the CUDA gate, do not suppress it —
  run lexical only and note it.
- **Real corpus never enters git.** `yspdb/`, `local_data/`, `eval/results/` are gitignored.
  Never commit them.
- **No LLM in the retrieval path.** Evaluation is purely retrieval metrics, no generation.
- **Spec freeze.** `docs/PROJECT_ASSISTANT_RAG_SPEC_v1.md`, `_v2.md`, `AMENDMENT_A01.md`,
  `AMENDMENT_A02.md` are frozen records — never edit them. New decisions go in new amendment
  files or `docs/DECISIONS.md`.
- **Pivot rule.** Any change that contradicts or reinterprets A01/A02/v1/v2 needs a new
  `docs/AMENDMENT_A0N.md` with: what changed · why (with measured evidence) · reversibility
  · which spec section it touches.
- **Failure ledger discipline.** Diagnose in the order: source → parser → chunk boundary →
  embedding/query → retrieval rank/filter. Do not reach for a larger model or reranker before
  locating the broken seam.

---

## What "done" looks like

At the end of this task:
- `eval/yp_questions_v2.jsonl` has passed label validation (all substrings verified)
- `eval/results/yp_v2_qwen4b_gpu.json` (or lexical-only variant) exists with real numbers
- `docs/EVAL_LOG.md` has a new entry with those numbers and a WP3 gate section
- `docs/FAILURE_LEDGER.md` is up to date with all new failures and dispositions
- `docs/DECISIONS.md` has the abstention decision and the synthetic-file retirement note
- `eval/archive/questions_synthetic_gv.jsonl` exists
- One clean commit captures all of this

After that, WP4 (AI-Team adapter) can begin.

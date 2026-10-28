# Decisions log

Short, dated, unspecified-step design choices (AGENTS.md "Pivot rule"). Spec-contradicting
changes go in `docs/AMENDMENT_A0N.md`, not here. Failures go in `docs/FAILURE_LEDGER.md`.

## 2026-10-28 — Round-2 Task A/B/C design choices (parent-section expansion · cross-lang · owner eval)

**Task A — parent-section context expansion (ADOPTED, on in prod config).** Spec v1 §7 names
"parent-section expansion" but not its mechanism. Chose to reuse the already path-safe
`open_source`: re-read the retrieved chunk's source filtered to its coarsest provenance anchor
(heading → sheet → page) and return the sibling blocks of that section as `expanded_context`.
Query-time, additive (the verbatim `excerpt` is never replaced), off by default. Rationale: the
evidence@5 miss is chunk-granularity (localized by the k-sweep + the rejected smaller-chunk
experiment), and the production move for that is more context to the answerer, not a reindex.
Measured span@5 0.667→0.718 (+yp28/yp33); honest limit recorded (cannot recover boilerplate-stripped
IDs or questions whose doc isn't retrieved). On in `yp_qwen4b_gpu_rerank.yaml` for WP4/demo.

**Task B — cross-language query-expansion glossary (REJECTED, off by default).** Built but measured
NEGATIVE (EVAL_LOG 2026-10-28): the general EN↔BG domain glossary misses word-sense (query
"attended" vs header `постигнати`=achieved) and over-triggers on common phrases (`Национална среща`
floods invitation letters), net −1 evidence@5, yp09 unmoved. Kept in tree default-off as the
reproducible record. Not adopted — tuning it to pass yp09 would be fitting to n=1, which the owner
explicitly said to avoid. Honest fix (index-time bilingual header enrichment) is out of the
no-reindex scope this round.

**Task C — owner-drivable retrieval session (`scripts/ask.py`).** The automated eval is
self-authored; its credibility is the bottleneck, not its score. Built a dead-simple warm-retriever
CLI (no service, no LLM) where the corpus OWNER types his own questions and the tool logs, per
question, the citations + evidence verdict + top relevance score + his own 1–5 usefulness rating to
a JSONL session. That logged session is the honest eval when there is no gold set, and is the
backbone of the demo surface. Retriever kept warm for the session (cold start ~30 s paid once).
Design is unspecified by the specs (a demo/eval surface), recorded here; reversible (a standalone
script, touches no index). Log-row schema pinned by `tests/unit/test_ask_session_log.py`.

## 2026-10-28 — Expand the v2 eval set where it was too thin to tune (WP4 prep, Task 1)

**Motivation.** The frozen v2 set (39q) is numeric-heavy (17) and too thin on the kinds that gate
WP4 to tune against: only 5 `no_answer`, 2 `date`, 1 `conflict`, 1 `cross_language`. An abstention
threshold (Task 2) cannot be calibrated against 5 negatives without overfitting — exactly the
objection raised in the 2026-10-26 abstention-deferral note. So the set is grown *before* building
abstention. **No existing question was edited or deleted** (the 39 stay frozen, labels unchanged);
11 were appended (yp41–yp51), lifting the set to 50 (now: `no_answer` 11, `date` 5, `conflict` 3).

**New questions and their verified labels** (all validated by `scripts/eval_yp.py:validate_labels`
against *parsed* corpus text — not raw files; the whole set now passes, 50/50):

- **date (3), each a distinct fact from yp11/yp24:**
  - yp41 — official opening day of the National Meeting → article, `На 18 юни`.
  - yp42 — end of the validity period of civil contract 05-YP-2025 → contract, `30.06.2026`.
  - yp43 — National-Meeting dates *as stated in the civil contract* → contract, `17-22 юни 2025`
    (same fact as yp24 but a different source document, to test cross-document date agreement).
- **conflict (2), each a genuine two-source documentary tension, distinct source pair from yp12:**
  - yp44 — National-Meeting dates: the official timetable says `04-07.07.2025` (the *planned*
    Erasmus activity window) while the civil contract and all post-event material say
    `17-22 юни 2025` (when it *actually* ran). Source pair = timetable + contract (yp12 used
    timetable + programme).
  - yp45 — the project's identifier: the application form carries Form ID `KA154-YOU-E2373104`
    while the civil contract cites grant number `2024-3-BG01-KA154-YOU-000283706`. Both are
    "the number" depending on the document; an answerer must distinguish them.
    *Incidental correction of the record:* the 2026-10-26 note said `grep -rl 000283706 yspdb`
    returns nothing and the grant number "is simply not in this corpus." That was a **raw-file**
    grep; `.docx` is a zip, so raw grep cannot see its text. The string **is** present in the
    parsed text of `Administration YP/05-YP-2025_Tsveti.docx` (and 11 other files). yp30 is left
    exactly as-is (it asks for the Form ID, which is correct); yp45 is the question that uses the
    now-confirmed grant number. Lesson reinforced: validate labels against *parsed* text, never raw.
- **no_answer (6), adversarial near-misses — plausible for this corpus but the specific answer is
  genuinely absent (absence verified by parsed-text grep across all 169 READY files, 0 hits for the
  operative term):**
  - yp46 — "budget for catering (`кетъринг`)": the budget has food/`храноден` lines but no catering
    line; `кетъринг` = 0 files.
  - yp47 — "when was the *fourth* webinar": only three webinars exist (yp06/yp31); `четвърти
    уебинар` = 0.
  - yp48 — "which hotel accommodated the participants": no hotel is named for the meeting
    (`Хотел ` appears only in an unrelated survey-responses sheet).
  - yp49 — "prize fund for competition winners": no prize/award exists; `наградн*` = 0.
  - yp50 — "results of the participant satisfaction survey": there is входящо/изходящо измерване
    but no satisfaction survey; `анкета за удовлетвореност` / its results = 0.
  - yp51 — "registration fee participants paid": the event was Erasmus-funded and free; all
    fee terms (`такса за участие`, `регистрационна такса`, `participation fee`) = 0.
  These are deliberately *harder* than the original 5 (weather/drones/vegetarians/buses/meals):
  each shares vocabulary and context with real corpus content, so a retriever WILL return
  plausible-looking chunks — the right test for an abstention signal that must reject on evidence
  *relevance*, not on topic familiarity.

**Reversibility.** Additive to one fixture file; the 39 frozen labels are untouched. Tracked
under ledger F16 (abstention) as the dataset the Task-2 threshold is tuned and reported on.

## 2026-10-28 — Abstention is a RETRIEVAL-side gate on the reranker score (resolves the 2026-10-26 deferral)

**Supersedes the 2026-10-26 "defer to WP4, policy (b)" decision.** That note deferred abstention
to the LLM because (i) WP3 was retrieval-only and (ii) n=5 negatives could not calibrate a
threshold without overfitting. Both objections are now removed: the task is explicitly a *WP4
precondition* (build a retrieval-side evidence-sufficiency verdict before wiring the LLM), and the
negative set was grown to 11 first (entry above). So the gate is built now, at the retrieval layer,
and the earlier "score-threshold is unjustified tuning" concern is answered by *measuring* the
threshold rather than guessing it.

**What was built (unspecified-step design; spec names the gate in §6.2 but not its mechanism):**
- The signal is the **top-1 cross-encoder `rerank_score`** — a calibrated 0–1 relevance already
  produced by the reranker stage. The uncalibrated alternatives the spec distrusts (Chroma
  distance, RRF fusion score; v1 §4.6) were measured and do not separate relevant from irrelevant,
  so they are only a weak fallback (`signal: fusion_score`) and the gate warns when forced onto it.
- Threshold **0.26**, chosen by a sweep (EVAL_LOG 2026-10-28), on a flat 0.26–0.32 plateau:
  abstains 10/11 no_answer, 3 real-hit false abstains. Not a round-number guess — the knee.
- `SearchResponse.evidence` (an `EvidenceVerdict`) is **always attached**. `suppress_on_insufficient`
  decides behavior: true → return the verdict *instead of* passages (the spec's "explicit
  no-evidence response", used by the WP4/prod config); false → keep passages, verdict advisory
  (eval mode + retrieval-first discipline, v1 §2.2 — the retriever flags, it does not decide truth).
- The gate is **off by default** (`AbstentionConfig.enabled=False`) so it changes nothing unless a
  config opts in; the reranked prod config `yp_qwen4b_gpu_rerank.yaml` opts in with suppression on.

**Why retrieval-side and not purely the LLM's job.** The 2026-10-26 note was right that the final
abstention *statement* is a generation behavior — but a retrieval-side gate is the honest place to
catch the easy negatives (topic-plausible, evidence-irrelevant) deterministically and cheaply,
before any token is spent, and to hand the LLM an explicit "insufficient evidence" signal instead
of plausible-but-wrong chunks to confabulate over. The subtle residual (yp47: "there was no fourth
webinar" needs counting) is correctly left to the LLM. This is a division of labor, not a
relitigation of retrieval-first.

**Reversibility:** `retrieval.abstention.enabled: false` restores exact prior behavior; the verdict
is additive to the response contract (default `enabled=False`, `sufficient=True`).

## 2026-10-28 — Filesystem-presence evidence for unparseable files (owner-requested; Task 3)

**Problem.** 162 of 334 corpus files are unparseable (80 png, 53 jpg, 11 jpeg, 10 mp4, 4 legacy
.doc, …) and OCR/transcription are out of scope. Discovery already walks them (`supported=False`)
but their **filenames and folder paths are discarded** — yet those carry Bulgarian event/topic
names (`Уебинари/2. Младежко участие на практика`, `Communications/Видеа - Декларация_`,
`…/Качете снимки (File responses)`) that are the only trace an activity or artefact exists. Owner
asked (explicitly NOT OCR) to make that trace retrievable as circumstantial evidence.

**Design (unspecified by the specs — a new doc_class).** `src/project_rag/filesystem.py` builds one
synthetic chunk **per folder that contains ≥1 unparseable file** (37 folders in this corpus):
- granularity is folder, not file — most image names are opaque hashes that would only add noise;
  the signal is the folder name + that it is non-empty. Parseable-only folders are skipped (already
  covered by real content chunks).
- each chunk is tagged `doc_class="filesystem"` (new, filterable; allowlisted in `api.search_project`
  and in the stores' generic doc_class metadata filter) and its text opens with an explicit
  `[FILESYSTEM PRESENCE — NOT DOCUMENT CONTENT]` banner + "contents are NOT indexed (no OCR, no
  transcription)" — so the future LLM can **never** mistake a filename for confirmed content.
- text lists folder path, parseable/unparseable counts with an extension breakdown, and up to 12
  representative filenames.
- ids are deterministic (v1 §4.3) and the version id hashes the folder inventory, so re-ingest is
  idempotent and a changed folder forces a new chunk.
- gated by `classification.filesystem_evidence` (**off by default**); it is in the pipeline
  fingerprint, so toggling it forces a new index generation (ADR-009). Synthesized from the
  discovery walk (no parse/chunk cache) because it is cheap and derived purely from the inventory.

**Measured (real corpus, reranked+abstain pipeline, separate index `yp_qwen4b_gpu_fs` so the frozen
index is untouched; EVAL_LOG 2026-10-28).**
- *It helps the intended questions.* 3 new `filesystem`-kind eval questions (yp52 photos folder,
  yp53 declaration videos, yp54 webinar-2 files) all retrieve the correct folder in top-5 (yp52
  rr=1.0, yp53 rr=1.0, yp54 rr=0.5). With a `doc_class: filesystem` filter the right folder is
  top-1 for photo/video/webinar queries that previously returned nothing citable.
- *It does not pollute normal retrieval.* On the 39 content questions the fs index scores
  Recall@5 0.769 / evidence@5 0.641 / MRR 0.714 vs the no-fs index 0.795 / 0.667 / 0.719 — a single
  question moved (yp21, exact, a borderline rank-5 hit). Probed: yp21's evidence chunk is **absent
  from both the dense and lexical top-16 and from the rerank window on the fs index, and no
  filesystem chunk appears anywhere in yp21's pool** — so the move is the known fresh-collection
  Chroma HNSW rebuild variance (v2 §D3), not fs chunks competing. The 37 fs chunks never entered a
  content question's result window. Net: the feature adds a new, honestly-labelled evidence class
  with no demonstrated cost to content retrieval.

**Validation note.** `filesystem`-kind questions reference a *folder* (the synthetic chunk's
source_path), not a parseable file, so `eval_yp.py:validate_labels` skips them — their labels are
correct by construction (the folder exists in discovery; the substrings are generated banner/count
text).

**Reversibility:** `classification.filesystem_evidence: false` (the default) removes the chunks
entirely on the next ingest; the frozen `yp_qwen4b_gpu` index never had them.

## 2026-10-28 — Chunking stays at target_tokens=500 (smaller tested, rejected; Task 4)

Pre-registered experiment (heavy-run-governor G0): `target_tokens 500→250` to lift evidence@5 by
isolating short exact-span answers. Measured on a separate reindexed generation (11,223 chunks,
GPU re-embed): recall@5 0.795→0.744, evidence@5 0.667→0.615, MRR flat — **every metric down or
flat**. Smaller chunks fragment the context the embedder and cross-encoder use to score relevance
(EVAL_LOG 2026-10-28). Decision rule was fixed before the run (adopt only if evidence@5 improves);
it did not, so **the 500-token chunking is kept** and the negative is recorded. `overlap_tokens` was
NOT exercised (it is unimplemented in `chunking.py` — a known limitation, not a tested lever). The
`chunk250` config + result JSON are kept as the reproducible record; the heavy derived index data
is regenerable from the config and is not retained. No spec change (chunking params are explicitly
"initial, not sacred", v1 §4.4); this is the measured tuning the spec asks for.

## 2026-10-26 — v2 question-label fixes (WP3, Task 1)

The eval validator (`scripts/eval_yp.py:validate_labels`, which checks every `expect_substring`
against the parsed text of its `expect_source_path`) rejected 14 labels in the first author draft
of `eval/yp_questions_v2.jsonl`. Diagnosed each in source→substring order; no question deleted.

- **Path errors (substring present, path wrong)** — corrected to the real `yspdb/` location:
  - yp20, yp21: `Покани/ПОКАНА-АДЕЛА-БОЗМАРОВА.docx` → `Покани/Адела Бозмарова/ПОКАНА-АДЕЛА-БОЗМАРОВА.docx`
  - yp24, yp25: bare filename → `Communications/Статии/Поглед към Национална среща „Младежко участие“ в Ямбол.docx`
  - yp26: bare filename → `Communications/Прессъобщения/ПРЕССЪОБЩЕНИЕ-Декларация.docx`
  - yp27, yp28: `АНАЛИЗ_ММП.docx` → `Проучване някакво___/АНАЛИЗ_ММП.docx`
  - yp29, yp38: `05-YP-2025_Tsveti.docx` → `Administration YP/05-YP-2025_Tsveti.docx`
  - yp31: `Уебинари/Уебинар 2.docx` (does not exist) → `Communications/Постове/Уебинар 2.docx` (contains `Младежкото участие на практика`)
  - All corrected substrings verified present in the parsed text of the corrected source.
- **Absent substring (fabricated value)** — yp30 expected `2024-3-BG01-KA154-YOU-000283706`.
  `grep -rl 000283706 yspdb` returns nothing; the string is in NO parsed corpus file. The
  proposal's actual identifier is the application Form ID `KA154-YOU-E2373104` (present on every
  page). Query reworded to "application form ID" and the substring replaced with
  `KA154-YOU-E2373104`. The grant-agreement number is simply not in this corpus.

After fixes the validator passes all 39 questions. Tracked as ledger **F17** (labels authored
from assumption, not from parsed text — the exact anti-pattern the validator exists to catch).

## 2026-10-26 — Abstention policy for `no_answer` questions (F16) — DEFERRED to WP4

**Current behavior (measured, v2 run 2026-10-26):** retrieval always returns top-k regardless of
relevance. All 5 `no_answer` questions (yp15, yp16, yp19, yp36, yp39 — weather, drones, vegetarians,
buses, vegetarian meals: none in the corpus) returned a full top-5 of plausible-but-irrelevant
chunks. There is no "no evidence" signal. These are excluded from the recall denominator so they
do not distort retrieval metrics, but that is a measurement convenience, not a policy.

**Two candidate policies considered:**
- **(a) Score-threshold cutoff** — abstain when the top fused score (or per-channel best) is below
  a calibrated threshold. Pure-retrieval, no LLM, testable now. Risk: BM25/cosine scores are not
  calibrated across query types or languages; a single threshold that abstains correctly on the 5
  no-answer questions without suppressing the hard-but-answerable ones (yp06/yp07/yp09) is unlikely
  on n=5 negatives — would overfit. Choosing the threshold needs many more negative examples.
- **(b) Evidence-sufficiency check by the LLM at WP4** — the agent inspects the retrieved passages
  and states "no supporting evidence found" rather than answering. Matches the spec's own
  abstention gate (§6.2: "Explicit no-evidence response, not plausible fabricated answer") and the
  retrieval-first/generation-second rule (the retriever is not supposed to decide truth; v1 §2.2).

**Decision:** Defer to **WP4**, policy **(b)**. Reasons, kept brief for a WP3 log:
1. WP3 is retrieval-only; abstention is fundamentally a generation-layer behavior (the retriever
   returning candidates and the answerer declining to answer are different concerns).
2. n=5 negatives cannot calibrate a score threshold without overfitting; a threshold added now
   would be unjustified tuning (AGENTS.md: every deviation defensible with evidence).
3. The spec's abstention acceptance criterion is written against the *answer*, not the ranked list.
Until WP4, `no_answer` questions stay out of the recall denominator and are tracked under F16.
A score-threshold experiment (a) remains a fallback if WP4 evidence-sufficiency proves insufficient.

## 2026-10-26 — Retire the synthetic question file (WP3, Task 7)

`eval/questions.jsonl` held 15 questions against a synthetic "Green Villages" fixture corpus that
is not the real corpus and is not ingested in any real-corpus index. Keeping it beside the real
`yp_questions*.jsonl` sets invites running it against the wrong index. Moved to
`eval/archive/questions_synthetic_gv.jsonl` (kept, not deleted — it is still the fixture for the
synthetic unit/integration tests). Active real-corpus question sets are `eval/yp_questions.jsonl`
(original 16), `eval/yp_questions_oos.jsonl` (out-of-sample 10), and `eval/yp_questions_v2.jsonl`
(expanded 39).

## 2026-10-26 — GPU env is set by code, not by the shell (ROCm-on-WSL)

Evidence: dense/hybrid eval was reported "GPU down" (F18). Real cause: ROCm 7.2.4 under WSL needs
`HSA_ENABLE_DXG_DETECTION=1`; it was exported only in `~/.bashrc`, which non-interactive
`wsl -- bash -c` shells never read, so `rocminfo` → `hsa_init Failed` and
`torch.cuda.is_available()` → False. Same shell with the var: gfx1100 (RX 7900 XT) visible, torch
True. Decision: `project_rag.gpu_env.prepare_gpu_env()` (called before `import torch` in the Qwen3
embedder and `scripts/feasibility_smoke.py`) sets the var when running in WSL with `/dev/dxg`;
`describe_gpu_env()` is appended to the CUDA-gate error so the failure names its own cause. The
gate stays fail-closed (no CPU fallback). Reversible: delete the two calls. Touches A01 §E only as
diagnostics; no behavior change when the GPU is truly absent.

## 2026-10-27 — Retrieval ranking tuning (WP3→WP4 readiness; all evidence in EVAL_LOG 2026-10-27)

Context: the k-sweep localized the loss to *ranking* (answer in pool 0.82–0.94, shown at @5 only
0.59–0.71), not retrieval. Three tuning choices followed, each measured on the frozen v2 set:

- **candidates_per_channel 16→32: REJECTED.** Hybrid evidence@5 regressed 0.676→0.647 — extra
  candidates are distractors a rank-only fusion can float above the true chunk. Kept at 16.
- **serve top_k 8→10/12: NOT ADOPTED.** No change in the production pipeline (the file-diversity
  dedup flattens the gain raw RRF showed at @10).
- **per_file_cap = 3: ADOPTED** (`retrieval.per_file_cap`, default 3, query-time, reversible).
  Swept {∞,3,2,1}; evidence@5 peaks at 3 (0.706 vs 0.676 uncapped; 2→0.618, 1→0.588). Mechanism:
  caps how many chunks of one source file may occupy the result window so sibling chunks cannot
  crowd out the file's evidence chunk (yp28/yp35). Moved yp33/yp35 to full hit, nothing regressed.
  Reversible: set `per_file_cap: null` to restore the legacy `top_k//2` behavior. Design choice
  under spec §4.7 (file-diversity dedup existed; this parameterizes its cap), not a spec change.

- **Cross-encoder reranker: STAGED, not enabled, not measured.** Implemented (`rerank.py` +
  `retrieval.rerank`, off by default) because the k-sweep met the spec §7 precondition
  ("candidate in top-20 but ranked too low"). Could not be measured — HF CDN blocked both model
  downloads (F19). Default-off so it changes nothing until a box with model access enables and
  measures it. This is the next real lever and the one most likely to move evidence@5 further;
  it is deliberately left justified-but-unverified rather than claimed to work.

- **Cross-encoder reranker: ENABLED and MEASURED (supersedes the "STAGED" note above), 2026-10-28.**
  HF CDN reachable this session; `BAAI/bge-reranker-v2-m3` pinned (rev `953dc6f…`) and run on GPU.
  Swept `rerank.top_n` {8,10,12,15,20}: peak at **12** — recall@5 0.7647→0.8235, evidence@5 holds
  0.7059, MRR 0.6186→0.7363. Adopted top_n=12 in `configs/yp_qwen4b_gpu_rerank.yaml` and as the
  `RerankConfig` default. **Why 12, not more:** top_n≥15 lets the cross-encoder demote fusion's
  strong short-chunk head (yp02/yp21/yp33 lose evidence), eroding evidence@5. A reranker is a
  corrective over the mis-ranked tail, not a replacement for the confident head — scope it narrow.
  The reranker is a query-time stage (no reindex), offline (local HF cache, `trust_remote_code=False`),
  pinned revision, regression-tested (`tests/unit/test_rerank.py`, fake scorer, no download).
  Reversible: `retrieval.rerank.enabled: false` restores the per_file_cap-only pipeline.
  Design choice under spec §7 (reranker preconditioned on the measured k-sweep), not a spec change.

# Failure ledger (v1 §6.4)

Order of diagnosis: source → parser → chunk boundary → embedding/query → retrieval rank/filter →
aggregation → generation → citation. Each entry: reproducer · mechanism · fix · regression test.

| ID | Layer | Reproducer (real corpus) | Mechanism | Status |
|---|---|---|---|---|
| F1 | source/discovery | `yspdb` contains 277-char paths | Windows MAX_PATH; `rglob`/`open` fail | **Fixed** — `os.walk` with `\\?\` prefix; `test_discovery_longpath.py` |
| F2 | parser (DOCX) | `KA154_Annex_Timetable…docx`, `Покани/Приложение 2…docx` | python-docx repeats merged cells; first row always used as header; tables appended after all paragraphs | **Fixed** — body-order walk, dedupe by `_tc`, header only if table-like; `test_docx_tables.py`. Fidelity 114/115 within band |
| F3 | parser (PDF) | `Main_Full-Report_Youth-Research-Varna-2025…pdf` → 29,596 blocks | pypdf emits whitespace-only lines between words; splitter treated them as paragraph breaks | **Fixed** — strict blank rule for PDF; test in `test_docx_tables.py` (130 blocks now) |
| F4 | source | proposal existed as PDF + 3 DOCX (3-char diff) | exact hash misses near-duplicates | **Mitigated by owner** (copies deleted 2026-10-17); generic near-dup grouping still open. `Copy of Overview_Debate 2025.xlsx` is a live near-duplicate (older version without the achieved column) |
| F5 | chunker | `АНКЕТА - ММС_МП (отговори) (2).xlsx` rows up to 30k words; 38 chunks truncated | row-group chunking ignored size | **Fixed** — oversized blocks split at sentence/word boundaries, table groups token-bounded (`structure_v3`); 0 truncated; `test_chunk_oversized.py` |
| F6 | chunker | Cyrillic text | `_approx_tokens` = 1.3 tok/word, wrong for Cyrillic with real tokenizers | **Open** — must use the chosen model's tokenizer (A01 §F already requires) |
| F7 | parser (XLSX) | `Overview_Debate 2025.xlsx` etc. (title rows, blank row 1) | header assumed = row 1 | **Fixed** — header detection in top 30 rows, loose rows kept; all formulas had cached values (0 uncached); `test_xlsx_headers.py`. Residual: side-by-side tables in one sheet leak as `I2=…` cells (see `Budgete_YP_25.xlsx`) |
| F8 | parser (PDF) | 3 image-only PDFs, 144 images, 4 `.doc` | no OCR/legacy parser | Reported as `OCR_REQUIRED`/`UNSUPPORTED`; decision open |
| F9 | chunker/source | yp02, yp03: budget/participant/organisation facts are in the application's summary pages | every page of the DOCX repeats form boilerplate (`Call 2024 Round 3 YOU`, `Form ID … Deadline`, `BG`, `n / 65`); it dilutes chunks and BM25 | **Open** — strip repeated lines per document (frequency > N pages) before chunking; measure |
| F10 | retrieval (lexical) | yp06 `уебинара` vs corpus `Уебинари` | FTS5 unicode61 has no Bulgarian stemming (v1 §4.7 predicted) | **Open** — dense channel / prefix queries / stemmer; decide from dense results |
| F11 | retrieval (lexical) | yp09, yp12, yp07 English questions over Bulgarian tables | no cross-language match in a lexical channel | **Open** — expected; this is what the dense channel must fix |
| F12 | retrieval ranking | yp11, yp13: right document found, wrong chunk ranks first | OR-of-all-terms BM25 lets generic words (среща, закон) outweigh the specific token; long chunks | **Open** — compare dense/hybrid before touching |
| F13 | indexing | any 1-file change | `ingest` re-embedded ALL staged chunks into the new generation (A01 §C stage [3] cache missing) | **Fixed** — `embedding_cache.py`, key (sha256(embed_text), profile key), batch-committed so interrupted runs resume; `test_embedding_cache.py` |
| F14 | retrieval (dense) | yp07, yp09: English question, answer is a short numeric row in a Bulgarian sheet (`Overview_Debate 2025.xlsx`/`Резултати`) | **Unresolved.** Hypothesis "row groups blend topics" tested (20→3 rows/chunk, E5): no effect. Remaining candidates: short rows lose to long prose; header text `брой заложени`/`Брой постигнати` has no English anchor; query language | **Open** — next test: prepend sheet/column glossary or run question in Bulgarian to separate language from granularity |
| F15 | eval | `conflict` kind (yp12) counts as a hit if ONE of two required sources is found | metric too lenient for multi-source questions | **Open** — add all-sources-in-top-5 metric |
| F16 | eval | `no_answer` (yp15, yp16) unscored | no abstention policy; every channel returns top-k | **Open** — decide score-threshold / evidence-sufficiency rule, then score |

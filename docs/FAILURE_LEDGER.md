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
| F13 | indexing | any 1-file change | `IndexingService.ingest` re-embeds ALL staged chunks into the new generation; the A01 §C embedding cache `[3]` is not implemented | **Open, blocks a CPU Qwen3-4B run** — implement cache keyed by `(chunk text hash, embedding_profile)` first |

# Failure ledger (v1 §6.4)

Order of diagnosis: source → parser → chunk boundary → embedding/query → retrieval rank/filter →
aggregation → generation → citation. Each entry: reproducer · mechanism · fix · regression test.

| ID | Layer | Reproducer (real corpus) | Mechanism | Status |
|---|---|---|---|---|
| F1 | source/discovery | `yspdb` contains 277-char paths | Windows MAX_PATH; `rglob`/`open` fail | **Fixed** — `os.walk` with `\\?\` prefix; `test_discovery_longpath.py` |
| F2 | parser (DOCX) | `KA154_Annex_Timetable…docx`, `Покани/Приложение 2…docx` | python-docx repeats merged cells; first row always used as header; tables appended after all paragraphs | **Fixed** — body-order walk, dedupe by `_tc`, header only if table-like; `test_docx_tables.py`. Fidelity 114/115 within band |
| F3 | parser (PDF) | `Main_Full-Report_Youth-Research-Varna-2025…pdf` → 29,596 blocks | pypdf emits whitespace-only lines between words; splitter treated them as paragraph breaks | **Fixed** — strict blank rule for PDF; test in `test_docx_tables.py` (130 blocks now) |
| F4 | source | proposal exists as PDF + 3 DOCX (3-char diff) | exact hash misses near-duplicates → same fact returned up to 4× in top-k | **Open** — needs near-dup grouping (normalized-text hash) + retrieval dedup by group |
| F5 | chunker | `АНКЕТА - ММС_МП (отговори) (2).xlsx` rows up to 30k words; 38 chunks truncated | row-group chunking ignores size; text beyond cap not embedded | **Open** |
| F6 | chunker | Cyrillic text | `_approx_tokens` = 1.3 tok/word, wrong for Cyrillic with real tokenizers | **Open** — must use the chosen model's tokenizer (A01 §F already requires) |
| F7 | parser (XLSX) | e.g. `Budgete_YP_25.xlsx`, `Overview_Debate 2025.xlsx` | header = row 1 only; title rows/multi-table sheets/formulas unverified | **Open** — sample before deciding |
| F8 | parser (PDF) | 3 image-only PDFs, 144 images, 4 `.doc` | no OCR/legacy parser | Reported as `OCR_REQUIRED`/`UNSUPPORTED`; decision open |

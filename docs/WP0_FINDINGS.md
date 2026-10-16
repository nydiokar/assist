# WP0 findings — corpus inventory and parsing gate (2026-10-17)

Reproduce: `$env:PYTHONPATH="src"; .\.venv\Scripts\python.exe -X utf8 scripts\wp0_inventory.py --root yspdb`
(writes `local_data/wp0/inventory.json`, gitignored: it contains real file names).

## Inventory (337 files, 3.58 GB)

| Scope | Files | Size | Decision |
|---|---|---|---|
| text (docx 115, xlsx 30, pdf 28, txt 2) | 175 | 208 MB | ingest |
| image (png/jpg/jpeg) | 144 | 168 MB | report as coverage gap (v1 §4.2), no OCR |
| media (mp4/mov/m4a) | 13 | 3.2 GB | out of scope, no transcription |
| legacy `.doc` | 4 | 5.7 MB | no native parser → report; decision pending (convert or skip) |
| other (`.conf`) | 1 | 126 B | report |

Exact-hash duplicates: 3. Near-duplicates (3 DOCX copies of the proposal, differing by 3 chars)
are **not** caught by hashing — see failure ledger F4.
After exact dedupe the text corpus yields ~1,510 chunks (approx token counter, current chunker),
~5.2 M characters. This is small: embedding cost is bounded (see A02 candidate in ledger).

## Parsing gate result (native backend)

| Format | Result |
|---|---|
| DOCX | 114/115 within 0.85–1.3 of raw XML text length (median 0.995). One outlier 1.38 = header repeated per row, not loss. |
| PDF | 25 parsed; 3 `OCR_REQUIRED` correctly detected (attendance list, MentiMeter export, flyer — all image-only). |
| XLSX | 30 parsed; first row assumed header; sheets not yet sampled for title rows/formulas. **Not passed** — needs sampling. |
| `.doc`, images | not parsed (see above). |

Gate status: **DOCX and PDF text pass; XLSX not yet verified; OCR/legacy decisions open.**
Source-to-text spot checks of the proposal's indicator section against the PDF are still to do.

## Corpus risks

- **PII:** participant feedback/registration xlsx (emails, names, phone), municipality survey
  (`АНКЕТА…`) with emails/phones, attendance-list PDF, contact lists (`Покани/`, `Спонсори/`),
  administration contracts (`Administration YP/`). Retrieval output and traces must not echo them
  raw; decide per folder whether to exclude from the index (owner decision, open).
- **Bulgarian + English mixed;** lexical FTS5 has no Bulgarian stemming (v1 §4.7) — expect
  inflection misses; measure.
- **Survey rows are huge** (up to ~30k words per row): current row-group chunking truncates them.
- **Reference material vs project record:** `Youth Participation Resources/` (laws, strategies,
  guides, ~98 MB) is third-party reading material, not project evidence. Retrieval must be able to
  tell "what the project says" from "what a cited law says" (candidate metadata: `collection` =
  project_record | reference). Open design decision.
- **Temporal:** source dates exist only as file mtimes (diagnostic only, v1 §4.1). No `as_of`.

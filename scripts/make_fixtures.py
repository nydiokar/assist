"""Generate a PII-free synthetic bilingual NGO project fixture corpus (v2 §D4).

Creates tests/fixtures/synthetic/ngo_project_demo/ with:
  proposal.md         - project proposal with indicator definitions (EN)
  progress_report.md  - report with target/actual numbers (EN)
  indicators.xlsx     - indicator table (sheet with target/actual, one None cached cell)
  registrations.csv   - participant list with a duplicate person + dietary column (synthetic)
  memo_bg.md          - Bulgarian-language memo
  scanned_placeholder.pdf - a (text-light) PDF to exercise OCR_REQUIRED detection path

All content is invented; no real personal data. Run: python scripts/make_fixtures.py
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "synthetic" / "ngo_project_demo"

PROPOSAL = """# Green Villages Project Proposal

## Project summary
The Green Villages project runs from January 2025 to December 2025. It supports three rural
communities with training and equipment grants under grant number GV-2025-017.

## Project indicators
The approved project defines the following indicators:

- Indicator 1: Number of people trained in sustainable farming. Target: 120 people.
- Indicator 2: Number of training sessions delivered. Target: 8 sessions.
- Indicator 3: Number of community gardens established. Target: 3 gardens.

## Reporting obligations
The grantee must submit an interim report by 15 July 2025 and a final report by 31 January 2026.
Required attachments: participant lists, photographs, and a financial summary.
"""

PROGRESS = """# Green Villages Interim Progress Report

## Reporting period
This interim report covers January to June 2025.

## Training activity
A sustainable farming training was held in March 2025 in the village of Breza. Evidence of the
March training includes the signed attendance sheet and five photographs.

## Indicator progress
- Indicator 1 (people trained): reported actual 74 people against a target of 120.
- Indicator 2 (training sessions): reported actual 5 sessions against a target of 8.
- Indicator 3 (community gardens): reported actual 1 garden against a target of 3.
"""

MEMO_BG = """# Вътрешна бележка

Обучението по устойчиво земеделие се проведе през март 2025 г. в село Бреза. Присъстваха
74 участници. Крайният отчет трябва да се подаде до 31 януари 2026 г.

Регистрираните вегетарианци получиха специално меню на обяд.
"""

REGISTRATIONS = (
    "participant_id,name,village,session,dietary\n"
    "P001,Ivan Petrov,Breza,March,none\n"
    "P002,Maria Ivanova,Breza,March,vegetarian\n"
    "P003,Georgi Dimitrov,Breza,March,vegetarian\n"
    "P004,Elena Koleva,Dabrava,April,none\n"
    "P002,Maria Ivanova,Breza,March,vegetarian\n"  # intentional duplicate person row
    "P005,Petar Stoyanov,Dabrava,April,vegan\n"
)


def _write_xlsx(path: Path) -> None:
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "Indicators"
    ws.append(["indicator", "description", "target", "reported_actual", "unit"])
    ws.append([1, "People trained", 120, 74, "people"])
    ws.append([2, "Training sessions", 8, 5, "sessions"])
    ws.append([3, "Community gardens", 3, None, "gardens"])  # None cached value, not zero
    wb.save(path)


def _write_scanned_pdf(path: Path) -> None:
    """Minimal valid PDF with no extractable text (simulates a scan → OCR_REQUIRED)."""
    content = (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]>>endobj\n"
        b"xref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n"
        b"0000000052 00000 n \n0000000101 00000 n \n"
        b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n164\n%%EOF\n"
    )
    path.write_bytes(content)


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "proposal.md").write_text(PROPOSAL, encoding="utf-8")
    (ROOT / "progress_report.md").write_text(PROGRESS, encoding="utf-8")
    (ROOT / "memo_bg.md").write_text(MEMO_BG, encoding="utf-8")
    (ROOT / "registrations.csv").write_text(REGISTRATIONS, encoding="utf-8")
    _write_xlsx(ROOT / "indicators.xlsx")
    _write_scanned_pdf(ROOT / "scanned_placeholder.pdf")
    print(f"fixtures written to {ROOT}")


if __name__ == "__main__":
    main()

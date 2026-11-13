"""E1 step 3 (S5 task 3): compare eval_yp.py outputs of several embedder configs to the reference.

Metrics per config: recall@5, evidence@5, MRR, abstention accuracy, each also as a question
count. Verdict per config (spec §6.8.1): within band when no primary metric (recall@5,
evidence@5, abstention accuracy) is more than `--band` questions below the reference.
Per-question diffs list question ids only (no corpus text), so the output can be committed.

  uv run python scripts/e1_compare.py --ref ref=local_data/yp_all/e1_ref.json \
      q8=local_data/yp_all/e1_q8.json q6=... q4=... --channels lexical+dense --md out.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic import BaseModel

from project_rag.services.eval import EvalReport


class Row(BaseModel):
    label: str
    recall_hits: int
    evidence_hits: int
    abstain_correct: int
    n_answerable: int
    n_abstain_scored: int
    recall_at_5: float
    evidence_at_5: float
    mrr: float
    abstention_accuracy: float


def _row(label: str, rep: EvalReport) -> Row:
    scored = [q for q in rep.per_question if q.kind != "no_answer"]
    recall = sum(q.hit_at_5 for q in scored)
    evidence = sum(q.evidence_hit_at_5 for q in scored)
    total = rep.noanswer_total + rep.answerable_total
    correct = rep.noanswer_abstained + (rep.answerable_total - rep.answerable_abstained)
    return Row(
        label=label,
        recall_hits=recall,
        evidence_hits=evidence,
        abstain_correct=correct,
        n_answerable=len(scored),
        n_abstain_scored=total,
        recall_at_5=rep.recall_at_5,
        evidence_at_5=rep.evidence_recall_at_5,
        mrr=rep.mrr,
        abstention_accuracy=round(correct / total, 4) if total else 0.0,
    )


def _load(spec: str, channels: str) -> tuple[str, EvalReport]:
    label, path = spec.split("=", 1)
    data = json.loads(Path(path).read_text("utf-8"))
    return label, EvalReport.model_validate(data[channels])


def _diffs(ref: EvalReport, other: EvalReport) -> list[str]:
    by_id = {q.id: q for q in ref.per_question}
    out: list[str] = []
    for q in other.per_question:
        r = by_id.get(q.id)
        if r is None:
            continue
        changes = [
            f"{name} {'+' if new else '-'}"
            for name, old, new in (
                ("hit", r.hit_at_5, q.hit_at_5),
                ("evidence", r.evidence_hit_at_5, q.evidence_hit_at_5),
                ("abstain", r.abstained, q.abstained),
            )
            if old != new
        ]
        if changes:
            out.append(f"{q.id} ({q.kind}): " + ", ".join(changes))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="label=path of the reference eval JSON")
    ap.add_argument("others", nargs="+", help="label=path eval JSONs to compare")
    ap.add_argument("--channels", default="lexical+dense")
    ap.add_argument("--band", type=int, default=2, help="noise band in questions (spec §6.8.1)")
    ap.add_argument("--md", required=True)
    args = ap.parse_args()

    ref_label, ref = _load(args.ref, args.channels)
    ref_row = _row(ref_label, ref)
    lines = [
        "| config | recall@5 | evidence@5 | MRR | abstention acc. | Δ recall | Δ evidence | Δ abstention | verdict |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    def fmt(r: Row, d: tuple[int, int, int] | None, verdict: str) -> str:
        ds = ["—", "—", "—"] if d is None else [f"{x:+d}" for x in d]
        return (
            f"| {r.label} | {r.recall_at_5:.3f} ({r.recall_hits}/{r.n_answerable}) "
            f"| {r.evidence_at_5:.3f} ({r.evidence_hits}/{r.n_answerable}) | {r.mrr:.3f} "
            f"| {r.abstention_accuracy:.3f} ({r.abstain_correct}/{r.n_abstain_scored}) "
            f"| {ds[0]} | {ds[1]} | {ds[2]} | {verdict} |"
        )

    lines.append(fmt(ref_row, None, "reference"))
    diff_blocks: list[str] = []
    for spec in args.others:
        label, rep = _load(spec, args.channels)
        row = _row(label, rep)
        d = (
            row.recall_hits - ref_row.recall_hits,
            row.evidence_hits - ref_row.evidence_hits,
            row.abstain_correct - ref_row.abstain_correct,
        )
        ok = all(x >= -args.band for x in d)
        lines.append(fmt(row, d, f"within ±{args.band}" if ok else f"OUT (> {args.band} below)"))
        diffs = _diffs(ref, rep)
        diff_blocks.append(f"- **{label} vs {ref_label}**: " + ("; ".join(diffs) or "identical"))
    text = "\n".join(lines) + "\n\nPer-question differences:\n\n" + "\n".join(diff_blocks) + "\n"
    Path(args.md).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

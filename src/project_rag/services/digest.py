"""Weekly operator digest (spec §6.9, A16). No LLM: aggregate feedback signals, turn every
`rescue` into a tier-F candidate (query + found source as the label), list stale gold questions,
show coverage changes from the run reports. An empty digest is skipped (None)."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Sequence
from datetime import datetime

from ..domain.models import FeedbackSignal, Frozen, RunReport

_FAILED = ("abstain", "empty", "degraded", "user_dissatisfied")


class FCandidate(Frozen):
    project_id: str
    query: str
    source_label: str  # the source the agent found the answer in
    at: str


class Digest(Frozen):
    since: str
    counts: dict[str, int]
    f_candidates: list[FCandidate]
    recurring_failed: list[tuple[str, int]]  # (normalised query, count), top 5
    stale_gold: list[str]
    coverage: dict[str, tuple[float | None, float]]  # project -> (previous, latest)
    text: str


def _norm(q: str) -> str:
    return " ".join(q.lower().split())


def build_digest(
    signals: Sequence[FeedbackSignal],
    since: datetime,
    *,
    runs_by_project: dict[str, list[RunReport]] | None = None,
    stale_gold: Sequence[str] = (),
) -> Digest | None:
    counts = Counter(s.kind for s in signals)
    f = [
        FCandidate(
            project_id=s.project_id, query=s.query, source_label=s.detail, at=s.at.isoformat()
        )
        for s in signals
        if s.kind == "rescue" and s.query and s.detail
    ]
    failed = Counter(_norm(s.query) for s in signals if s.kind in _FAILED and s.query)
    recurring = [(q, n) for q, n in failed.most_common(5) if n > 1]
    coverage: dict[str, tuple[float | None, float]] = {}
    for pid, runs in (runs_by_project or {}).items():
        published = [r for r in runs if r.published]
        if published:
            prev = published[1].coverage_pct if len(published) > 1 else None
            coverage[pid] = (prev, published[0].coverage_pct)
    moved = {p: c for p, c in coverage.items() if c[0] is None or abs(c[1] - c[0]) >= 0.1}
    if not signals and not stale_gold and not moved:
        return None  # nothing to say: skip the digest
    lines = [f"assist weekly digest since {since.date().isoformat()}"]
    lines.append("signals: " + (", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "none"))
    for i, c in enumerate(f, start=1):
        qid = hashlib.sha256(c.query.encode()).hexdigest()[:6]
        lines.append(
            f"F{i} [{c.project_id}] {c.query!r} -> {c.source_label}  (approve: G-{qid} / drop)"
        )
    for q, n in recurring:
        lines.append(f"recurring miss x{n}: {q!r}")
    if stale_gold:
        lines.append("stale gold questions (source changed): " + ", ".join(stale_gold))
    for pid, (prev, latest) in sorted(moved.items()):
        lines.append(f"coverage {pid}: {'-' if prev is None else f'{prev:.1f}'} -> {latest:.1f} %")
    return Digest(
        since=since.isoformat(),
        counts=dict(counts),
        f_candidates=f,
        recurring_failed=recurring,
        stale_gold=list(stale_gold),
        coverage=coverage,
        text="\n".join(lines),
    )

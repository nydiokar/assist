"""Tier-S smoke pieces (spec §6.8.2/§6.8.3): deterministic stratified chunk sampling, label
staleness, smoke verdicts, and the one operator message. Pure."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

from ..domain.models import ChunkView, Frozen


class SmokeResult(Frozen):
    project_id: str
    n_questions: int
    recall_at_5: float
    probes_total: int
    probes_answered: int  # a no-answer probe that got a confident answer is a failure
    alert: bool
    reasons: list[str]


def _stratum(c: ChunkView) -> tuple[str, str]:
    folder = c.source_path.split("/", 1)[0] if "/" in c.source_path else ""
    ext = c.source_path.rsplit(".", 1)[-1].lower() if "." in c.source_path else ""
    return folder, ext


def stratified_sample(chunks: Sequence[ChunkView], k: int, seed: str) -> list[ChunkView]:
    """Up to k chunks, round-robin over (top folder, file type) strata; within a stratum the
    order is a seeded hash, so the sample is reproducible and needs no random state (R5)."""

    def key(c: ChunkView) -> str:
        return hashlib.sha256(f"{seed}|{c.chunk_id}".encode()).hexdigest()

    strata: dict[tuple[str, str], list[ChunkView]] = {}
    for c in chunks:
        if c.text.strip():
            strata.setdefault(_stratum(c), []).append(c)
    queues = [sorted(v, key=key) for _, v in sorted(strata.items())]
    picked: list[ChunkView] = []
    seen_sources: set[str] = set()
    while len(picked) < k and any(queues):
        for q in queues:
            if not q or len(picked) >= k:
                continue
            # Prefer a chunk from a file not sampled yet, so questions spread over documents.
            idx = next((i for i, c in enumerate(q) if c.source_path not in seen_sources), 0)
            c = q.pop(idx)
            picked.append(c)
            seen_sources.add(c.source_path)
    return picked


def stale_questions(
    labelled: Mapping[str, Mapping[str, str]], current_hashes: Mapping[str, str]
) -> list[str]:
    """Question ids whose labelled source hash changed or whose source was deleted (§6.8.4).
    `labelled` = {question_id: {source_path: content_hash_at_labelling}}."""
    stale: list[str] = []
    for qid, sources in labelled.items():
        if any(current_hashes.get(path) != h for path, h in sources.items()):
            stale.append(qid)
    return sorted(stale)


def smoke_verdict(
    project_id: str,
    hits: Sequence[bool],
    probes_answered: Sequence[bool],
    *,
    min_recall: float = 0.5,
) -> SmokeResult:
    n = len(hits)
    recall = round(sum(hits) / n, 4) if n else 0.0
    answered = sum(probes_answered)
    reasons: list[str] = []
    if n and recall < min_recall:
        reasons.append(f"smoke recall@5 {recall:.2f} < {min_recall:.2f}")
    if answered:
        reasons.append(f"{answered} no-answer probe(s) answered")
    return SmokeResult(
        project_id=project_id,
        n_questions=n,
        recall_at_5=recall,
        probes_total=len(probes_answered),
        probes_answered=answered,
        alert=bool(reasons),
        reasons=reasons,
    )


def new_project_message(
    display_name: str,
    n_docs: int,
    coverage_pct: float,
    n_no_text: int,
    smoke: SmokeResult,
) -> str:
    """The one operator message after a new project's first ingest (§6.8.3)."""
    ok = smoke.probes_total - smoke.probes_answered
    msg = (
        f"Project {display_name} indexed: {n_docs} docs, coverage {coverage_pct:.0f} %, "
        f"{n_no_text} no-text images, smoke recall@5 {smoke.recall_at_5:.2f}, "
        f"no-answer probes {ok}/{smoke.probes_total} OK"
    )
    return ("ALERT: " + msg + " (" + "; ".join(smoke.reasons) + ")") if smoke.alert else msg

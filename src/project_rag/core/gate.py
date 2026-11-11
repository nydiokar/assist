"""Tolerance gate (spec §6.8.1, A18). Pure: metrics in, decision out; the floor never moves.

- Gated only when the pipeline fingerprint changed (parser, OCR, chunker, model, quant, fusion,
  rerank, thresholds, prompt). Data-only ingests are never gated.
- Pass = no primary metric drops by more than the noise band (+/- `band_questions` questions of
  that metric's own n) vs the last promoted baseline, AND every metric is at or above the floor.
- On a pass the candidate becomes the new baseline; the floor stays where the owner set it.
"""

from __future__ import annotations

import math

from ..domain.models import Frozen

_EPS = 1e-9
PRIMARY = ("recall_at_5", "evidence_at_5", "abstention_accuracy")


class GateMetrics(Frozen):
    recall_at_5: float
    evidence_at_5: float
    abstention_accuracy: float
    n_answerable: int  # denominator of recall/evidence
    n_total: int  # denominator of abstention accuracy


class GatePolicy(Frozen):
    band_questions: int = 2
    floor: GateMetrics | None = None  # proposed by the agent from E1, confirmed by the owner


class GateDecision(Frozen):
    gated: bool
    passed: bool
    reasons: list[str]
    baseline: GateMetrics  # the baseline after this decision (promoted on a pass)
    floor: GateMetrics | None  # always the policy's floor, unchanged


def _n(metric: str, m: GateMetrics) -> int:
    return m.n_total if metric == "abstention_accuracy" else m.n_answerable


def gate(
    candidate: GateMetrics, baseline: GateMetrics, policy: GatePolicy, *, pipeline_changed: bool
) -> GateDecision:
    if not pipeline_changed:
        return GateDecision(
            gated=False,
            passed=True,
            reasons=["data-only change: not gated (smoke checks and coverage alerts only)"],
            baseline=baseline,
            floor=policy.floor,
        )
    reasons: list[str] = []
    for metric in PRIMARY:
        new = float(getattr(candidate, metric))
        old = float(getattr(baseline, metric))
        allowed = policy.band_questions / max(1, _n(metric, candidate))
        if old - new > allowed + _EPS:
            reasons.append(
                f"{metric} dropped {old - new:.4f} > band {allowed:.4f} ({old:.4f}->{new:.4f})"
            )
        if policy.floor is not None:
            floor = float(getattr(policy.floor, metric))
            if new + _EPS < floor:
                reasons.append(f"{metric} {new:.4f} below floor {floor:.4f}")
    passed = not reasons
    return GateDecision(
        gated=True,
        passed=passed,
        reasons=reasons or ["within band and above floor"],
        baseline=candidate if passed else baseline,
        floor=policy.floor,
    )


def propose_floor(chosen: GateMetrics, questions_below: int = 3) -> GateMetrics:
    """E1 rule: floor = chosen-quant score minus 3 questions per metric (owner confirms once)."""

    def down(value: float, n: int) -> float:
        # Round DOWN: a score exactly `questions_below` questions lower must sit on the floor.
        return max(0.0, math.floor((value - questions_below / max(1, n)) * 1e4 + _EPS) / 1e4)

    return chosen.model_copy(
        update={
            "recall_at_5": down(chosen.recall_at_5, chosen.n_answerable),
            "evidence_at_5": down(chosen.evidence_at_5, chosen.n_answerable),
            "abstention_accuracy": down(chosen.abstention_accuracy, chosen.n_total),
        }
    )

"""S7 exit checks (offline): the tolerance gate (hypothesis over metric deltas), the weekly digest
from fixture signals (rescue -> F candidate with its source label), stale-label detection, and
the tier-S generator with the fixture writer on the synthetic corpus."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from hypothesis import given
from hypothesis import strategies as st

from project_rag.adapters.writer_fixture import FixtureQuestionWriter
from project_rag.core.gate import GateMetrics, GatePolicy, gate, propose_floor
from project_rag.core.smoke import stale_questions, stratified_sample
from project_rag.domain.config import Config, with_overrides
from project_rag.domain.models import ChunkView, FeedbackSignal, RunReport
from project_rag.services.digest import build_digest
from project_rag.services.smoke import TierSService
from project_rag.wiring import build
from tests.conftest import FIXTURE_CORPUS

N_ANS, N_TOT = 36, 47
BASE = GateMetrics(
    recall_at_5=0.82,
    evidence_at_5=0.71,
    abstention_accuracy=0.85,
    n_answerable=N_ANS,
    n_total=N_TOT,
)


def _with(**deltas: float) -> GateMetrics:
    return BASE.model_copy(update={k: getattr(BASE, k) + v for k, v in deltas.items()})


q = st.integers(min_value=-6, max_value=6)  # change measured in questions


@given(dr=q, de=q, da=q)
def test_gate_band_rule(dr: int, de: int, da: int) -> None:
    cand = _with(recall_at_5=dr / N_ANS, evidence_at_5=de / N_ANS, abstention_accuracy=da / N_TOT)
    d = gate(cand, BASE, GatePolicy(band_questions=2), pipeline_changed=True)
    within = min(dr, de, da) >= -2
    assert d.gated and d.passed == within
    assert d.baseline == (cand if within else BASE)  # promoted only on a pass


@given(dr=q, de=q, da=q)
def test_data_only_change_is_never_gated(dr: int, de: int, da: int) -> None:
    cand = _with(recall_at_5=dr / N_ANS, evidence_at_5=de / N_ANS, abstention_accuracy=da / N_TOT)
    floor = propose_floor(BASE)
    d = gate(cand, BASE, GatePolicy(floor=floor), pipeline_changed=False)
    assert not d.gated and d.passed and d.baseline == BASE and d.floor == floor


@given(steps=st.lists(st.integers(min_value=1, max_value=2), min_size=1, max_size=10))
def test_floor_stops_slow_drift_and_never_moves(steps: list[int]) -> None:
    floor = propose_floor(BASE)  # = baseline - 3 questions per metric
    policy = GatePolicy(floor=floor)
    baseline = BASE
    lost = 0
    for s in steps:  # repeated small drops, each inside the band
        lost += s
        cand = _with(recall_at_5=-lost / N_ANS)
        d = gate(cand, baseline, policy, pipeline_changed=True)
        assert d.floor == floor  # never moves automatically
        if lost > 3:
            assert not d.passed and any("below floor" in r for r in d.reasons)
            break
        assert d.passed
        baseline = d.baseline


def test_floor_proposal_is_three_questions_down() -> None:
    f = propose_floor(BASE)
    assert abs((BASE.recall_at_5 - f.recall_at_5) - 3 / N_ANS) < 1e-3
    assert abs((BASE.abstention_accuracy - f.abstention_accuracy) - 3 / N_TOT) < 1e-3


# ---- digest ----

T0 = datetime(2026, 11, 2, tzinfo=UTC)


def _sig(kind: str, query: str, detail: str = "", hours: int = 1) -> FeedbackSignal:
    return FeedbackSignal(
        kind=kind, project_id="yp", at=T0 + timedelta(hours=hours), query=query, detail=detail
    )


def test_digest_renders_and_rescue_becomes_f_candidate() -> None:
    signals = [
        _sig("abstain", "Колко участници имаше дебатът?"),
        _sig("abstain", "колко участници   имаше дебатът?", hours=2),
        _sig("empty", "catering budget"),
        _sig("rescue", "Who signed the partnership agreement?", "Partners/agreement_signed.pdf"),
        _sig("user_dissatisfied", "webinar dates"),
    ]
    run = RunReport(
        run_id="r",
        project_id="yp",
        started_at="t",
        finished_at="t",
        published=True,
        coverage_pct=91.5,
    )
    d = build_digest(signals, T0, runs_by_project={"yp": [run]}, stale_gold=["yp14"])
    assert d is not None
    assert d.counts == {"abstain": 2, "empty": 1, "rescue": 1, "user_dissatisfied": 1}
    (fc,) = d.f_candidates
    assert fc.query == "Who signed the partnership agreement?"
    assert fc.source_label == "Partners/agreement_signed.pdf"
    assert d.recurring_failed == [("колко участници имаше дебатът?", 2)]
    assert "F1 [yp]" in d.text and "stale gold questions" in d.text and "coverage yp" in d.text


def test_empty_digest_is_skipped() -> None:
    assert build_digest([], T0) is None


def test_stale_labels() -> None:
    labelled = {"g1": {"a.docx": "h1"}, "g2": {"b.xlsx": "h2"}, "g3": {"c.pdf": "h3"}}
    assert stale_questions(labelled, {"a.docx": "h1", "b.xlsx": "CHANGED"}) == ["g2", "g3"]


def test_stratified_sample_spreads_and_is_reproducible() -> None:
    chunks = [
        ChunkView(chunk_id=f"c{i}", source_path=f"{d}/f{i % 3}.{e}", text="t")
        for i, (d, e) in enumerate([("A", "md"), ("A", "pdf"), ("B", "md")] * 6)
    ]
    s1 = stratified_sample(chunks, 6, "seed")
    assert s1 == stratified_sample(chunks, 6, "seed")
    assert {c.source_path.split("/")[0] for c in s1} == {"A", "B"}
    assert len({c.chunk_id for c in s1}) == 6


# ---- tier S on the fixture corpus ----


def test_tier_s_with_fixture_writer(tmp_path: Path) -> None:
    cfg = with_overrides(
        Config(),
        {
            "lexical_store.path": str(tmp_path / "s.sqlite3"),
            "vector_store.persist_path": "",
            "retrieval.abstention.enabled": True,
            "retrieval.abstention.signal": "fusion_score",
            "retrieval.abstention.min_top_score": 0.0325,
        },
    )
    s = build(cfg)
    s.ingest.ingest("demo", str(FIXTURE_CORPUS))
    svc = TierSService(
        FixtureQuestionWriter(),
        s.lexical,
        s.catalog,
        lambda pid, qtext: s.search.search_project(pid, qtext, top_k=5),
        s.ingest.status,
        k=8,
    )
    run = svc.run("demo", "Demo")
    assert len([q for q in run.questions if q.expect_source_path]) == 8
    assert len([q for q in run.questions if q.expect_source_path is None]) == 3
    assert run.result.recall_at_5 >= 0.5  # known-item questions over-reward lexical overlap
    assert run.message.startswith(("Project Demo indexed:", "ALERT: Project Demo indexed:"))
    assert "smoke recall@5" in run.message and "no-answer probes" in run.message
    s.close()

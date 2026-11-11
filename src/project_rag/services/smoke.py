"""Tier-S smoke eval (spec §6.8.2/§6.8.3, A17): sample K chunks per project (stratified by
folder and file type), have the QuestionWriter write one known-item question per chunk (label =
the chunk's source), add no-answer probes, score them through the production search, and render
the one operator message. A health floor, not a quality claim."""

from __future__ import annotations

from collections.abc import Callable

from ..core.smoke import SmokeResult, new_project_message, smoke_verdict, stratified_sample
from ..domain.errors import BudgetExceeded
from ..domain.models import Frozen, SearchResponse, StatusReport
from ..ports.catalog import Catalog
from ..ports.lexical import LexicalIndex
from ..ports.question_writer import QuestionWriter


class SmokeQuestion(Frozen):
    question: str
    expect_source_path: str | None  # None for a no-answer probe


class SmokeRun(Frozen):
    project_id: str
    questions: list[SmokeQuestion]
    result: SmokeResult
    message: str


Search = Callable[[str, str], SearchResponse]  # (project_id, query) -> response


class TierSService:
    def __init__(
        self,
        writer: QuestionWriter,
        lexical: LexicalIndex,
        catalog: Catalog,
        search: Search,
        status: Callable[[str], StatusReport],
        *,
        k: int = 20,
        probes: int = 3,
        seed: str = "tier-s-v1",
    ) -> None:
        self._writer = writer
        self._lexical = lexical
        self._catalog = catalog
        self._search = search
        self._status = status
        self._k = k
        self._probes = probes
        self._seed = seed

    def run(self, project_id: str, display_name: str) -> SmokeRun:
        gen = self._catalog.active_generation(project_id)
        if gen is None:
            raise ValueError(f"project {project_id} has no active generation")
        chunks = stratified_sample(self._lexical.chunk_views(project_id, gen), self._k, self._seed)
        questions: list[SmokeQuestion] = []
        notes: list[str] = []
        try:
            for c in chunks:
                text = self._writer.question_for(c.text, c.source_path)
                if text:  # a refused/empty answer gives no question, never an invented one
                    questions.append(SmokeQuestion(question=text, expect_source_path=c.source_path))
            summary = ", ".join(sorted({c.source_path.split("/", 1)[0] for c in chunks}))
            questions += [
                SmokeQuestion(question=q, expect_source_path=None)
                for q in self._writer.no_answer_probes(summary, self._probes)
                if q
            ]
        except BudgetExceeded as e:  # score what was written before the cap
            notes.append(str(e))
        hits: list[bool] = []
        probes_answered: list[bool] = []
        for q in questions:
            resp = self._search(project_id, q.question)
            if q.expect_source_path is None:
                ev = resp.evidence
                answered = bool(resp.results) and ev.enabled and ev.sufficient
                probes_answered.append(answered)
            else:
                hits.append(any(r.source_path == q.expect_source_path for r in resp.results[:5]))
        result = smoke_verdict(project_id, hits, probes_answered)
        st = self._status(project_id)
        n_docs = st.n_documents
        ready = st.states.get("READY", 0)
        coverage = 100.0 * ready / n_docs if n_docs else 0.0
        message = new_project_message(
            display_name, n_docs, coverage, st.states.get("NO_TEXT", 0), result
        )
        if notes:
            message += " [" + "; ".join(notes) + "]"
        return SmokeRun(project_id=project_id, questions=questions, result=result, message=message)

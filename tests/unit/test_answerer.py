"""Grounding contract (WP4). Pure, offline — NO paid call.

Exercises the retrieval-first decision path (abstention short-circuit) and the grounding-prompt
assembly with a stubbed `call_llm`, so the whole decision logic is covered without any LLM call
(AGENTS.md: tests call no paid service). The live turn runs in the agent session that loads the
stdio MCP server (`project_rag.mcp_server`), not here.
"""

from __future__ import annotations

from project_rag.answerer import (
    ABSTENTION_LINE,
    SYSTEM_PROMPT,
    answer_from_search,
    build_grounding_prompt,
)
from project_rag.schema import (
    EvidenceVerdict,
    RetrievedResult,
    SearchResponse,
    SourceLocation,
)


def _sufficient_response() -> SearchResponse:
    return SearchResponse(
        query_id="q", project_id="ypq", index_generation="g",
        evidence=EvidenceVerdict(enabled=True, sufficient=True, signal="rerank_score",
                                 top_score=0.91, threshold=0.26),
        results=[RetrievedResult(
            chunk_id="c1", document_id="d1", source_path="proposal.docx",
            source_location=SourceLocation(heading="Бюджет"),
            excerpt="общ бюджет 49 697,00 EUR",
            expanded_context="Раздел Бюджет: общ бюджет 49 697,00 EUR разпределен по дейности.",
            rank=1, retrieval_channels=["dense", "lexical"],
            score_details={"rerank_score": 0.91})],
    )


def _suppressed_response() -> SearchResponse:
    # Gate abstained: passages withheld (api.search_project sets results=[] on insufficient).
    return SearchResponse(
        query_id="q", project_id="ypq", index_generation="g",
        evidence=EvidenceVerdict(enabled=True, sufficient=False, signal="rerank_score",
                                 top_score=0.004, threshold=0.26),
        results=[])


def test_abstains_pre_llm_when_gate_suppressed():
    """No passages -> fixed abstention line, no LLM consulted (call_llm not even required)."""
    ans = answer_from_search("колко е таксата за участие?", _suppressed_response())
    assert ans.abstained_pre_llm is True
    assert ans.answer == ABSTENTION_LINE
    assert ans.citations == []
    assert ans.evidence_sufficient is False


def test_grounds_on_passages_with_injected_call():
    calls: list[tuple[str, str]] = []

    def fake_call(system: str, user: str) -> str:
        calls.append((system, user))
        return "Общият бюджет е 49 697,00 EUR [#1] (proposal.docx)."

    ans = answer_from_search("какъв е общият бюджет?", _sufficient_response(),
                             call_llm=fake_call)
    assert ans.abstained_pre_llm is False
    assert "49 697,00" in ans.answer
    assert ans.citations == ["proposal.docx"]
    assert ans.evidence_sufficient is True
    # the LLM was called exactly once, with the grounding system prompt
    assert len(calls) == 1
    assert calls[0][0] == SYSTEM_PROMPT


def test_prompt_uses_expanded_context_and_cites_sources():
    prompt = build_grounding_prompt("какъв е общият бюджет?", _sufficient_response())
    # parent-section context (Task A) is preferred over the bare excerpt when present
    assert "разпределен по дейности" in prompt
    assert "[#1]" in prompt
    assert "proposal.docx" in prompt
    assert "§Бюджет" in prompt           # heading provenance surfaced
    assert ABSTENTION_LINE in prompt     # the abstain instruction is carried into the user turn


def test_prompt_falls_back_to_excerpt_without_expanded_context():
    res = _sufficient_response()
    res.results[0].expanded_context = ""   # no parent section attached
    prompt = build_grounding_prompt("q", res)
    assert "общ бюджет 49 697,00 EUR" in prompt


def test_prompt_warns_when_passages_shown_below_threshold():
    """Report-drafting profile (suppress off): passages present but gate said insufficient —
    the prompt must tell the model the signal is weak and to abstain unless a passage states it."""
    res = _sufficient_response()
    res.results[0].score_details["rerank_score"] = 0.05
    res.evidence = EvidenceVerdict(enabled=True, sufficient=False, signal="rerank_score",
                                   top_score=0.05, threshold=0.26)
    prompt = build_grounding_prompt("q", res)
    assert "BELOW" in prompt
    assert "abstain" in prompt.lower()

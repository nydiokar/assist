"""Grounding contract for the WP4 answerer (prompt + abstention logic). NO network, NO API key.

The LLM is wired via the standalone stdio MCP server (`project_rag.mcp_server`), registered in a
project's `.mcp.json` and loaded by Claude Code / AI-Team at session start — on the owner's existing
plan, not the Anthropic API. This module holds the *grounding contract* that travels into that
session: the fixed abstention line (v1 §5.2), the system/grounding prompt, and the retrieval-first
decision helper. These are pure and unit-tested offline; the agent harness runs the actual turn.

(Superseded 2026-10-29: both an earlier direct-Anthropic-API path and a prototype in-process SDK
helper were retracted in favour of the stdio-MCP-server wiring — see docs/DECISIONS.md.
`GROUNDING_GUIDANCE` in ai_team_adapter carries the same rules into the tool descriptions.)
"""

from __future__ import annotations

from dataclasses import dataclass

from ..domain.models import RetrievedResult, SearchResponse

# v1 §5.2 — the exact abstention statement the retriever's verdict maps to.
ABSTENTION_LINE = "Not established from accessible project documents."

SYSTEM_PROMPT = (
    "You are a project-assistant that answers ONLY from the retrieved source passages supplied in "
    "the user message. The corpus is the document archive of one completed Erasmus+ youth project; "
    "it is mostly Bulgarian with some English. Rules, in order of priority:\n"
    "1. GROUND EVERYTHING. Every factual claim must be supported by one of the numbered passages. "
    "After each claim, cite the passage it came from as [#n] (its number) and name the source file.\n"
    "2. ABSTAIN rather than guess. If the passages do not contain the answer, reply exactly: "
    f'"{ABSTENTION_LINE}" and nothing else. Do NOT answer from your own prior knowledge, and do NOT '
    "infer facts the passages do not state. A plausible-looking passage that does not actually "
    "contain the asked fact is NOT sufficient — abstain.\n"
    "3. Do NOT count, aggregate, or compute totals across passages as if authoritative — retrieval "
    "returns evidence, not a complete set. If asked 'how many' or for a total, report only what a "
    "passage explicitly states and flag that the set may be incomplete.\n"
    "4. Preserve Bulgarian text exactly (Cyrillic intact); answer in the language of the question.\n"
    "5. Be concise. Lead with the answer, then the citation(s)."
)


@dataclass(frozen=True)
class GroundedAnswer:
    """Result of answering one question. `abstained_pre_llm=True` means the retrieval gate decided
    and no LLM call was made (no egress, no cost)."""

    question: str
    answer: str
    abstained_pre_llm: bool
    citations: list[str]  # source_path strings of the passages offered to the LLM
    evidence_sufficient: bool
    evidence_top_score: float | None


def _format_passage(n: int, r: RetrievedResult) -> str:
    loc = r.source_location
    where: list[str] = []
    if loc.page is not None:
        where.append(f"p.{loc.page}")
    if loc.sheet:
        where.append(f"sheet={loc.sheet}")
    if loc.heading:
        where.append(f"§{loc.heading}")
    where_s = (" (" + ", ".join(where) + ")") if where else ""
    # Prefer the parent-section context (Task A) when present — it carries answers the top chunk
    # alone often did not (span@5 0.718 vs 0.667). Fall back to the verbatim excerpt.
    body = (
        r.expanded_context if r.expanded_context and r.expanded_context != r.excerpt else r.excerpt
    )
    return f"[#{n}] source: {r.source_path}{where_s}\n{body.strip()}"


def build_grounding_prompt(question: str, res: SearchResponse) -> str:
    """Pure: assemble the user-message prompt from the retrieved passages. Network-free."""
    parts = [f"Question: {question}", "", "Retrieved source passages:"]
    for i, r in enumerate(res.results, start=1):
        parts.append("")
        parts.append(_format_passage(i, r))
    ev = res.evidence
    parts.append("")
    if ev.enabled and ev.top_score is not None:
        if ev.sufficient:
            parts.append(
                f"(retrieval evidence signal: top relevance {ev.top_score:.3f}, "
                f"threshold {ev.threshold}; passages cleared the sufficiency gate.)"
            )
        else:
            # Report-drafting profile: passages are shown despite a weak gate signal so you can
            # judge them, but be EXTRA cautious — relevance is low, so abstain unless a passage
            # genuinely contains the asked fact.
            parts.append(
                f"(retrieval evidence signal: top relevance {ev.top_score:.3f} is BELOW the "
                f"{ev.threshold} sufficiency threshold. The passages may be only loosely related. "
                f"Ground only on a passage that explicitly states the answer; otherwise abstain.)"
            )
    parts.append("")
    parts.append(
        "Answer the question using ONLY the passages above. Cite each claim as [#n] with the file "
        f'name. If they do not contain the answer, reply exactly "{ABSTENTION_LINE}".'
    )
    return "\n".join(parts)


def answer_from_search(question: str, res: SearchResponse, *, call_llm=None) -> GroundedAnswer:
    """Turn a SearchResponse into a grounded answer, retrieval-first.

    The real path is the AI-Team SDK-MCP session (the agent calls the RAG tools itself); this helper
    is the pure decision contract used in tests and non-agent harnesses. `call_llm(system, user)
    -> str` is injected so the whole decision path (including the abstention short-circuit) runs
    without any paid call. When the retriever already abstained (empty results), no LLM is consulted.
    """
    citations = [r.source_path for r in res.results]
    ev = res.evidence

    # Retrieval already decided: the gate suppressed passages as insufficient. Emit the abstention
    # line without consulting the LLM. (v1 §5.2 / §2.2; never fabricate a section with no evidence.)
    if not res.results:
        return GroundedAnswer(
            question=question,
            answer=ABSTENTION_LINE,
            abstained_pre_llm=True,
            citations=[],
            evidence_sufficient=ev.sufficient if ev.enabled else False,
            evidence_top_score=ev.top_score,
        )

    if call_llm is None:  # pragma: no cover - guarded; the agent harness supplies the turn
        raise ValueError("call_llm is required when passages are present")

    user = build_grounding_prompt(question, res)
    answer = call_llm(SYSTEM_PROMPT, user)
    return GroundedAnswer(
        question=question,
        answer=answer,
        abstained_pre_llm=False,
        citations=citations,
        evidence_sufficient=ev.sufficient if ev.enabled else True,
        evidence_top_score=ev.top_score,
    )

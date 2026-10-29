"""WP4 grounded answerer (Option 2b, owner egress decision 2026-10-29).

Retrieval-first by construction (v1 §2.2): `search_project` runs and the evidence-sufficiency gate
decides BEFORE any LLM call. When the gate abstains (passages suppressed), the answer is the fixed
"Not established…" line (v1 §5.2) and NO token is spent. Otherwise the retrieved passages + the
evidence verdict are formatted into a grounding prompt and a single hosted-Claude Messages call
produces a citation-bearing answer. The LLM only grounds on what the retriever handed it; it never
re-queries the corpus and is told to abstain rather than draw on its own prior.

Egress: retrieved Bulgarian passages go to Anthropic (api.anthropic.com). Authorized by the written
owner decision in docs/DECISIONS.md (2026-10-29), per the v1 §8 no-egress rule.

The prompt builder (`build_grounding_prompt`) and the abstention short-circuit (`answer_from_search`)
are pure / network-free so they are unit-tested without a paid call (AGENTS.md: tests call no paid
service). Only `_call_claude` touches the SDK and the network.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from .schema import RetrievedResult, SearchResponse

# v1 §5.2 — the exact abstention statement the retriever's verdict maps to.
ABSTENTION_LINE = "Not established from accessible project documents."

DEFAULT_MODEL = "claude-opus-4-8"
DEFAULT_MAX_TOKENS = 4096

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
    citations: list[str]          # source_path strings of the passages offered to the model
    evidence_sufficient: bool
    evidence_top_score: float | None
    model: str | None             # None when abstained pre-LLM


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
    body = r.expanded_context if r.expanded_context and r.expanded_context != r.excerpt else r.excerpt
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
        parts.append(
            f"(retrieval evidence signal: top relevance {ev.top_score:.3f}, "
            f"threshold {ev.threshold}; passages cleared the sufficiency gate.)"
        )
    parts.append("")
    parts.append(
        "Answer the question using ONLY the passages above. Cite each claim as [#n] with the file "
        f'name. If they do not contain the answer, reply exactly "{ABSTENTION_LINE}".'
    )
    return "\n".join(parts)


def answer_from_search(question: str, res: SearchResponse, *,
                       call_claude=None, model: str = DEFAULT_MODEL) -> GroundedAnswer:
    """Turn a SearchResponse into a grounded answer, retrieval-first.

    `call_claude(system, user, model) -> str` is injected so tests can run the whole decision path
    (including the abstention short-circuit) without a paid call. The real caller passes the SDK
    wrapper `_call_claude`.
    """
    citations = [r.source_path for r in res.results]
    ev = res.evidence

    # Retrieval already decided: the gate suppressed passages as insufficient. Do not spend a
    # token — emit the abstention line. (v1 §5.2 / §2.2; the owner's false-negative worry cuts both
    # ways — never fabricate a section when the retriever found no evidence.)
    if not res.results:
        return GroundedAnswer(
            question=question, answer=ABSTENTION_LINE, abstained_pre_llm=True,
            citations=[], evidence_sufficient=ev.sufficient if ev.enabled else False,
            evidence_top_score=ev.top_score, model=None,
        )

    if call_claude is None:  # pragma: no cover - guarded; real runs inject _call_claude
        raise ValueError("call_claude is required when passages are present")

    user = build_grounding_prompt(question, res)
    answer = call_claude(SYSTEM_PROMPT, user, model)
    return GroundedAnswer(
        question=question, answer=answer, abstained_pre_llm=False,
        citations=citations, evidence_sufficient=ev.sufficient if ev.enabled else True,
        evidence_top_score=ev.top_score, model=model,
    )


def _call_claude(system: str, user: str, model: str,
                 *, max_tokens: int = DEFAULT_MAX_TOKENS) -> str:
    """Single hosted-Claude Messages call (streamed). Imports the SDK lazily so the module loads
    without the `answer` extra installed. Streaming + adaptive thinking per the claude-api skill;
    `claude-opus-4-8` uses adaptive thinking (no budget_tokens)."""
    import anthropic  # noqa: PLC0415  (optional dep; only needed for a real egress call)

    client = anthropic.Anthropic()  # ANTHROPIC_API_KEY from env; never hardcoded
    with client.messages.stream(
        model=model,
        max_tokens=max_tokens,
        thinking={"type": "adaptive"},
        system=system,
        messages=[{"role": "user", "content": user}],
    ) as stream:
        msg = stream.get_final_message()
    return "".join(b.text for b in msg.content if b.type == "text").strip()


def stream_claude(system: str, user: str, model: str,
                  *, max_tokens: int = DEFAULT_MAX_TOKENS) -> Iterator[str]:
    """Token iterator for the interactive CLI — same call, yields text deltas as they arrive."""
    import anthropic  # noqa: PLC0415

    client = anthropic.Anthropic()
    with client.messages.stream(
        model=model,
        max_tokens=max_tokens,
        thinking={"type": "adaptive"},
        system=system,
        messages=[{"role": "user", "content": user}],
    ) as stream:
        yield from stream.text_stream

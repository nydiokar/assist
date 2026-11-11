"""Parent-section / neighbour-chunk context expansion (spec v1 §7 "parent-section expansion").

Query-time, no reindex. The k-sweep (WP3, EVAL_LOG 2026-10-27) and the F19 reranker run localized
the residual evidence@5 miss to CHUNK GRANULARITY, not ranking: for ~29% of answerable questions
the right *document* is in the top-5 but the answer *chunk* is not (recall@5 0.82 vs evidence@5
0.71). Smaller chunks were tested and rejected (Task 4) because they fragment the context the
embedder/reranker score on. The production move for exactly this gap is to hand the answerer MORE
context around each retrieved chunk instead of reindexing: the parent section the chunk came from.

Mechanism (reuses the already path-safe `open_source`): a retrieved chunk carries its
`source_location` (heading for free text, sheet for tables, page for PDFs). Re-reading the source
filtered to that same location returns the sibling blocks of the chunk's section — i.e. the parent
section. No new store, no new embeddings, no new query. The chunk's own `excerpt` (verbatim
citation text) is never lost; the expansion is additive context for grounding.

Nothing here decides truth or ranks; it only widens the evidence window (retrieval-first, v1 §2.2).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .core.expand import merge_expansion, parent_location
from .schema import RetrievedResult

if TYPE_CHECKING:
    from .api import ProjectRAG


def expand_result_context(
    rag: ProjectRAG, project_id: str, result: RetrievedResult, *, max_chars: int = 4000
) -> str:
    """Return the parent-section text for one retrieved result (chunk excerpt if none exists).

    Pure read via `open_source` (path-safe, server-side root). Falls back to the verbatim chunk
    excerpt when the chunk has no coarser location anchor or the re-read yields nothing — so the
    result is never *less* than what the caller already had.
    """
    parent = parent_location(result.source_location)
    if parent is None:
        return result.excerpt
    resp = rag.open_source(
        project_id,
        result.document_id,
        source_path=result.source_path,
        location=parent,
        max_chars=max_chars,
    )
    return merge_expansion(resp.excerpt, result.excerpt, max_chars)


def attach_expanded_context(
    rag: ProjectRAG, project_id: str, results: list[RetrievedResult], *, max_chars: int = 4000
) -> list[RetrievedResult]:
    """Return copies of the served results with `expanded_context` populated.

    Re-reads are cached per source_path inside `open_source`'s parser registry call; we cap the
    count by operating only on the already-served (top_k) window, so cost is bounded.
    """
    return [
        r.model_copy(
            update={
                "expanded_context": expand_result_context(rag, project_id, r, max_chars=max_chars)
            }
        )
        for r in results
    ]

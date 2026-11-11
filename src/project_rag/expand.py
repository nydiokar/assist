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

from .schema import RetrievedResult, SourceLocation

if TYPE_CHECKING:
    from .api import ProjectRAG


def _parent_location(loc: SourceLocation) -> SourceLocation | None:
    """The coarsest provenance key that defines a chunk's "parent section".

    Priority matches how chunking groups blocks (chunking.py): free text is grouped under a
    heading; tables under a sheet; PDF prose falls back to the page. Returns None when the chunk
    has no coarser anchor than itself (then expansion is a no-op and we keep the chunk excerpt).
    """
    if loc.heading:
        return SourceLocation(heading=loc.heading)
    if loc.sheet:
        return SourceLocation(sheet=loc.sheet)
    if loc.page is not None:
        return SourceLocation(page=loc.page)
    return None


def expand_result_context(
    rag: ProjectRAG, project_id: str, result: RetrievedResult, *, max_chars: int = 4000
) -> str:
    """Return the parent-section text for one retrieved result (chunk excerpt if none exists).

    Pure read via `open_source` (path-safe, server-side root). Falls back to the verbatim chunk
    excerpt when the chunk has no coarser location anchor or the re-read yields nothing — so the
    result is never *less* than what the caller already had.
    """
    parent = _parent_location(result.source_location)
    if parent is None:
        return result.excerpt
    resp = rag.open_source(
        project_id,
        result.document_id,
        source_path=result.source_path,
        location=parent,
        max_chars=max_chars,
    )
    text = resp.excerpt.strip()
    if not text:
        return result.excerpt
    # Guarantee the chunk's own text is present even if location filtering dropped it (e.g. a
    # heading match that excludes a trailing block the chunk spanned).
    if result.excerpt and result.excerpt.strip() not in text:
        text = (text + "\n" + result.excerpt.strip())[:max_chars]
    return text


def attach_expanded_context(
    rag: ProjectRAG, project_id: str, results: list[RetrievedResult], *, max_chars: int = 4000
) -> None:
    """Populate `RetrievedResult.expanded_context` in place for each served result.

    Re-reads are cached per source_path inside `open_source`'s parser registry call; we cap the
    count by operating only on the already-served (top_k) window, so cost is bounded.
    """
    for r in results:
        r.expanded_context = expand_result_context(rag, project_id, r, max_chars=max_chars)

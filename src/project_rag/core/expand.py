"""Parent-section context expansion, pure parts (spec v1 §7; F12/F9 granularity gap).

The service re-reads the source at `parent_location(...)` through the path-safe open_source and
passes the text to `merge_expansion`, which guarantees the result is never less than the excerpt.
"""

from __future__ import annotations

from ..domain.models import SourceLocation


def parent_location(loc: SourceLocation) -> SourceLocation | None:
    """The coarsest provenance key that defines a chunk's "parent section".

    Priority matches how chunking groups blocks: free text under a heading; tables under a sheet;
    PDF prose falls back to the page. None when there is no coarser anchor than the chunk itself.
    """
    if loc.heading:
        return SourceLocation(heading=loc.heading)
    if loc.sheet:
        return SourceLocation(sheet=loc.sheet)
    if loc.page is not None:
        return SourceLocation(page=loc.page)
    return None


def merge_expansion(reread: str, excerpt: str, max_chars: int) -> str:
    """Parent-section text, falling back to the excerpt; the excerpt is always contained."""
    text = reread.strip()
    if not text:
        return excerpt
    if excerpt and excerpt.strip() not in text:
        text = (text + "\n" + excerpt.strip())[:max_chars]
    return text

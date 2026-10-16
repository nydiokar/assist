"""Validated read-only source opens (v1 §5.1 step 7, §8 security).

open_source returns the original extracted text for a document/location. Paths are never
accepted from the caller/LLM directly: the document_id is resolved against the manifest's
recorded source within the configured project root, then canonicalized and checked to be
inside the root (blocks traversal / symlink escape).
"""

from __future__ import annotations

from pathlib import Path

from .parsers import ParserRegistry
from .schema import SourceLocation, SourceResponse


class PathTraversalError(Exception):
    pass


def _resolve_within_root(root: Path, source_path: str) -> Path:
    root_resolved = root.resolve()
    candidate = (root_resolved / source_path).resolve()
    if root_resolved != candidate and root_resolved not in candidate.parents:
        raise PathTraversalError(f"path escapes project root: {source_path}")
    return candidate


def open_source(
    *,
    project_id: str,
    document_id: str,
    root_path: str,
    source_path: str,
    registry: ParserRegistry,
    location: SourceLocation | None = None,
    max_chars: int = 6000,
) -> SourceResponse:
    root = Path(root_path)
    try:
        abs_path = _resolve_within_root(root, source_path)
    except PathTraversalError as e:
        return SourceResponse(project_id=project_id, document_id=document_id,
                              source_path=source_path, location=location, excerpt="",
                              warnings=[str(e)])
    if not abs_path.is_file():
        return SourceResponse(project_id=project_id, document_id=document_id,
                              source_path=source_path, location=location, excerpt="",
                              warnings=["source file not found"])

    doc = registry.parse(str(abs_path), source_path, abs_path.suffix.lower())
    blocks = doc.blocks
    if location is not None:
        filtered = [b for b in blocks if _matches(b.location, location)]
        if filtered:
            blocks = filtered
    text = "\n".join(b.text for b in blocks)
    truncated = len(text) > max_chars
    return SourceResponse(
        project_id=project_id, document_id=document_id, source_path=source_path,
        location=location, excerpt=text[:max_chars], truncated=truncated,
        warnings=list(doc.diagnostics),
    )


def _matches(block_loc: SourceLocation, want: SourceLocation) -> bool:
    if want.page is not None and block_loc.page != want.page:
        return False
    if want.sheet is not None and block_loc.sheet != want.sheet:
        return False
    if want.heading is not None and block_loc.heading != want.heading:
        return False
    return True

"""Pure ingest policies (S4): exclusion filters, the OCR keep rule, and the run summary."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from fnmatch import fnmatchcase

from ..domain.config import IngestConfig, OcrConfig
from ..domain.models import DiscoveredFile, DocState, FailedDoc, FileReport

_FAILING = {DocState.FAILED.value, DocState.NO_TEXT.value, DocState.QUARANTINED.value}


def exclusion_reason(f: DiscoveredFile, cfg: IngestConfig) -> str | None:
    """Why a discovered file is out of scope, or None. Mirrors the rclone filters (D-9)."""
    name = f.source_path.rsplit("/", 1)[-1]
    if f.source_path in cfg.exclude_paths:
        return "excluded by curation list (ingest.exclude_paths)"
    if f.ext in {e.lower() for e in cfg.exclude_extensions}:
        return f"excluded extension {f.ext}"
    for pattern in cfg.exclude_name_globs:
        if fnmatchcase(name, pattern):
            return f"excluded name pattern {pattern}"
    if f.size > cfg.max_file_bytes:
        return f"over size cap ({f.size} > {cfg.max_file_bytes} bytes)"
    return None


_COPY_NAME = re.compile(r"^(copy of |копие на |копие )|\(\d+\)(\.[^.]*)?$", re.IGNORECASE)


def _canonical_rank(
    path: str, reference_prefixes: tuple[str, ...]
) -> tuple[int, int, int, int, str]:
    """Lower is preferred: not a "Copy of"/"(1)" name, not third-party reference material (so a
    project-record copy keeps its classification), shallow, short, then lexicographic."""
    name = path.rsplit("/", 1)[-1]
    is_copy = 1 if _COPY_NAME.search(name) else 0
    is_reference = 1 if reference_prefixes and path.startswith(reference_prefixes) else 0
    return (is_copy, is_reference, path.count("/"), len(path), path)


def exact_duplicates(
    files: Sequence[DiscoveredFile], reference_prefixes: tuple[str, ...] = ()
) -> dict[str, str]:
    """{duplicate path: canonical path} for files sharing a SHA-256 within one project.

    Deterministic and independent of listing order. Every occurrence stays recorded (v1 §4.1);
    only the canonical one is parsed, chunked and embedded."""
    groups: dict[str, list[str]] = {}
    for f in files:
        groups.setdefault(f.content_hash, []).append(f.source_path)
    out: dict[str, str] = {}
    for paths in groups.values():
        if len(paths) < 2:
            continue
        canonical = min(paths, key=lambda p: _canonical_rank(p, reference_prefixes))
        out.update({p: canonical for p in paths if p != canonical})
    return out


def ocr_keep(text: str, mean_conf: float, cfg: OcrConfig) -> bool:
    """Keep OCR'd image text only with enough characters at enough mean confidence."""
    return len(text.strip()) >= cfg.min_chars and mean_conf >= cfg.min_mean_conf


def embed_abort_reason(failed: int, needing: int, cfg: IngestConfig) -> str | None:
    """A systemic embedding failure aborts the run instead of publishing a hollow index."""
    if needing <= 0 or failed < cfg.abort_min_failures:
        return None
    if failed / needing > cfg.max_embed_failure_fraction:
        return (
            f"embedding failed for {failed}/{needing} documents "
            f"(> {cfg.max_embed_failure_fraction:.0%}); nothing published"
        )
    return None


def counts_by_state(files: Iterable[FileReport]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for f in files:
        counts[str(f.state)] = counts.get(str(f.state), 0) + 1
    return dict(sorted(counts.items()))


def coverage_pct(files: Sequence[FileReport]) -> float:
    """READY documents as a share of distinct in-scope documents (excluded files and DUPLICATE
    occurrences never count; a duplicate's content is covered by its canonical copy)."""
    distinct = [f for f in files if f.state != DocState.DUPLICATE]
    if not distinct:
        return 0.0
    ready = sum(1 for f in distinct if f.state == DocState.READY)
    return round(100.0 * ready / len(distinct), 1)


def failed_docs(files: Iterable[FileReport]) -> list[FailedDoc]:
    return [
        FailedDoc(source_path=f.source_path, state=str(f.state), reason="; ".join(f.diagnostics))
        for f in files
        if str(f.state) in _FAILING
    ]


def newly_failed(current: Sequence[FailedDoc], prior_states: Mapping[str, str]) -> list[FailedDoc]:
    """Failures that were not already failing in the prior generation."""
    return [d for d in current if prior_states.get(d.source_path) not in _FAILING]

"""Pure ingest policies (S4): exclusion filters, the OCR keep rule, and the run summary."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from fnmatch import fnmatchcase

from ..domain.config import IngestConfig, OcrConfig
from ..domain.models import DiscoveredFile, DocState, FailedDoc, FileReport

_FAILING = {DocState.FAILED.value, DocState.NO_TEXT.value, DocState.QUARANTINED.value}


def exclusion_reason(f: DiscoveredFile, cfg: IngestConfig) -> str | None:
    """Why a discovered file is out of scope, or None. Mirrors the rclone filters (D-9)."""
    name = f.source_path.rsplit("/", 1)[-1]
    if f.ext in {e.lower() for e in cfg.exclude_extensions}:
        return f"excluded extension {f.ext}"
    for pattern in cfg.exclude_name_globs:
        if fnmatchcase(name, pattern):
            return f"excluded name pattern {pattern}"
    if f.size > cfg.max_file_bytes:
        return f"over size cap ({f.size} > {cfg.max_file_bytes} bytes)"
    return None


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
    """READY documents as a share of in-scope documents (excluded files never count)."""
    if not files:
        return 0.0
    ready = sum(1 for f in files if f.state == DocState.READY)
    return round(100.0 * ready / len(files), 1)


def failed_docs(files: Iterable[FileReport]) -> list[FailedDoc]:
    return [
        FailedDoc(source_path=f.source_path, state=str(f.state), reason="; ".join(f.diagnostics))
        for f in files
        if str(f.state) in _FAILING
    ]


def newly_failed(current: Sequence[FailedDoc], prior_states: Mapping[str, str]) -> list[FailedDoc]:
    """Failures that were not already failing in the prior generation."""
    return [d for d in current if prior_states.get(d.source_path) not in _FAILING]

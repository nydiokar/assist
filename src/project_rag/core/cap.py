"""Per-file diversity cap over a fused ranking (WP3 2026-10-27; evidence@5 peaks at 3)."""

from __future__ import annotations

from collections.abc import Sequence

from .fusion import FusedCandidate


def apply_per_file_cap(
    ranking: Sequence[FusedCandidate], per_file: int, window: int
) -> list[FusedCandidate]:
    """Pick up to `window` candidates with at most `per_file` from one source path.

    Over-cap candidates go to an overflow list (order preserved) and refill the window only if it
    would otherwise be short, so the result is never shorter than min(window, len(ranking)).
    """
    picked: list[FusedCandidate] = []
    seen_files: dict[str, int] = {}
    overflow: list[FusedCandidate] = []
    for c in ranking:
        path = c.payload.source_path
        if seen_files.get(path, 0) >= per_file:
            overflow.append(c)
            continue
        picked.append(c)
        seen_files[path] = seen_files.get(path, 0) + 1
        if len(picked) >= window:
            break
    for c in overflow:
        if len(picked) >= window:
            break
        picked.append(c)
    return picked[:window]

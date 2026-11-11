"""Classify discovered sources against the prior generation (v1 §4.8 reconciliation)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ..domain.models import DiscoveredFile, Frozen


class ChangeSet(Frozen):
    """Partition of the union of prior and discovered source paths."""

    added: tuple[str, ...] = ()
    modified: tuple[str, ...] = ()
    unchanged: tuple[str, ...] = ()
    deleted: tuple[str, ...] = ()


def diff_corpus(
    prior_hashes: Mapping[str, tuple[str, str]], discovered: Sequence[DiscoveredFile]
) -> ChangeSet:
    """`prior_hashes` is {source_path: (content_hash, source_id)} of the active generation.

    Discovered order is preserved within each class; `deleted` is sorted for determinism.
    """
    added: list[str] = []
    modified: list[str] = []
    unchanged: list[str] = []
    for d in discovered:
        prior = prior_hashes.get(d.source_path)
        if prior is None:
            added.append(d.source_path)
        elif prior[0] != d.content_hash:
            modified.append(d.source_path)
        else:
            unchanged.append(d.source_path)
    seen = {d.source_path for d in discovered}
    deleted = sorted(p for p in prior_hashes if p not in seen)
    return ChangeSet(
        added=tuple(added),
        modified=tuple(modified),
        unchanged=tuple(unchanged),
        deleted=tuple(deleted),
    )

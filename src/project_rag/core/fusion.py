"""Reciprocal-rank fusion of retrieval channels (v1 §4.7/§5.1). Pure and deterministic."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ..domain.models import ChannelHit, Frozen, HitPayload


class FusedCandidate(Frozen):
    """A chunk after fusion: its payload, per-channel 1-based ranks and RRF score."""

    chunk_id: str
    payload: HitPayload
    ranks: dict[str, int]
    fused: float


def rrf(rank: int, k: int) -> float:
    return 1.0 / (k + rank)


def rrf_fuse(rankings: Mapping[str, Sequence[ChannelHit]], k: int) -> list[FusedCandidate]:
    """Fuse ranked channels by RRF; sort by (-score, chunk_id) for a stable tie-break.

    `rankings` maps channel name to its hits in rank order. A chunk's payload comes from the
    first channel (in mapping order) that returned it.
    """
    ranks: dict[str, dict[str, int]] = {}
    payloads: dict[str, HitPayload] = {}
    for channel, hits in rankings.items():
        for rank, hit in enumerate(hits, start=1):
            ranks.setdefault(hit.chunk_id, {})[channel] = rank
            payloads.setdefault(hit.chunk_id, hit.payload)
    fused = [
        FusedCandidate(
            chunk_id=cid,
            payload=payloads[cid],
            ranks=r,
            fused=sum(rrf(x, k) for x in r.values()),
        )
        for cid, r in ranks.items()
    ]
    return sorted(fused, key=lambda c: (-c.fused, c.chunk_id))

"""Unit of work over the shared store (S2): stage + verify + publish atomically."""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Protocol


class UnitOfWork(Protocol):
    def transaction(self) -> AbstractContextManager[None]:
        """Writes inside commit together or not at all; nested calls join the outer one.
        Indexes outside the store (e.g. Chroma) are not covered: the caller's two-phase
        protocol and the startup GC handle them."""
        ...

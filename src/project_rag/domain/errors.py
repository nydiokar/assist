"""Port error types (spec A03 §15.3). Adapters translate third-party exceptions into these;
services decide what to do with them (degrade, isolate, fail) — rule R6."""

from __future__ import annotations


class PortError(Exception):
    """Base for every error a port may raise."""


class ProviderUnavailable(PortError):
    """A model provider (embedder/reranker) cannot be reached."""


class ProviderTimeout(PortError):
    """A model provider did not answer within its deadline."""


class ProviderContractViolation(PortError):
    """A provider answered with something the port contract forbids (wrong dim, NaN, not normed)."""


class ParserCrashed(PortError):
    """A parser failed unexpectedly on one document (caught per document by the ingest service)."""


class OcrFailed(PortError):
    """The OCR engine failed on one document."""


class PathOutsideRoot(PortError):
    """A requested path resolves outside the project root (traversal or symlink escape)."""


class StoreError(PortError):
    """A lexical/vector/catalog store operation failed."""


class ProfileMismatch(StoreError):
    """The live embedding profile/collection differs from the one pinned for the project."""


class BudgetExceeded(PortError):
    """A paid adapter reached its configured spend cap (tier-S question writer, D-7)."""

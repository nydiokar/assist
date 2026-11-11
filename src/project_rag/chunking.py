"""Compatibility shim (S1 move in progress): moved to `project_rag.core.chunking`."""

from .core.chunking import *  # noqa: F403
from .core.chunking import CHUNKER_VERSION, _approx_tokens  # noqa: F401

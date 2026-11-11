"""Compatibility shim (S1 move in progress; removed at the end of S1)."""

from .adapters.parse_native import *  # noqa: F403
from .adapters.parse_native import _split_paragraphs  # noqa: F401
from .adapters.parse_registry import ParserRegistry  # noqa: F401

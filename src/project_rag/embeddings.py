"""Compatibility shim (S1 move in progress; removed at the end of S1)."""

from .adapters.embed_hash import HashEmbedder  # noqa: F401
from .adapters.embed_torch import E5Embedder, Qwen3Embedder  # noqa: F401
from .ports.embedder import Embedder as EmbedderProtocol  # noqa: F401
from .wiring import build_embedder  # noqa: F401

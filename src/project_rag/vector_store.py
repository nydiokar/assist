"""Compatibility shim (S1 move in progress; removed at the end of S1)."""

from .adapters.vector_memory import InMemoryVectorStore  # noqa: F401
from .ports.vector import VectorIndex as VectorStoreProtocol  # noqa: F401
from .wiring import build_vector_store  # noqa: F401

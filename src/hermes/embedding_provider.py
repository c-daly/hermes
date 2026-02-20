"""Pluggable embedding providers for Hermes.

Defines a protocol for embedding providers and a default implementation
using sentence-transformers. The active provider is selected via env vars:

    EMBEDDING_PROVIDER  (default: "sentence-transformers")
    EMBEDDING_MODEL     (default: "all-MiniLM-L6-v2")
"""

from __future__ import annotations

import logging
import os
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""

    @property
    def dimension(self) -> int: ...

    @property
    def model_name(self) -> str: ...

    async def embed(self, text: str) -> list[float]: ...


class SentenceTransformerProvider:
    """Default provider using sentence-transformers."""

    def __init__(self, model: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading sentence-transformers model: %s", self._model_name)
            self._model = SentenceTransformer(self._model_name)
            logger.info("Embedding model loaded (dim=%d)", self.dimension)
        return self._model

    @property
    def dimension(self) -> int:
        model = self._load()
        dim: int = model.get_sentence_embedding_dimension()
        return dim

    @property
    def model_name(self) -> str:
        return self._model_name

    async def embed(self, text: str) -> list[float]:
        model = self._load()
        vec = model.encode(text)
        return vec.tolist()


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------
_provider: EmbeddingProvider | None = None


def get_embedding_provider() -> EmbeddingProvider:
    """Return the configured embedding provider (lazy singleton)."""
    global _provider
    if _provider is not None:
        return _provider

    backend = os.environ.get("EMBEDDING_PROVIDER", "sentence-transformers")
    model = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    if backend == "sentence-transformers":
        _provider = SentenceTransformerProvider(model=model)
    else:
        raise ValueError(f"Unknown EMBEDDING_PROVIDER: {backend!r}")

    return _provider

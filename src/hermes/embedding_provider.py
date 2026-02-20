"""Pluggable embedding providers for Hermes.

Defines a protocol for embedding providers and implementations using
sentence-transformers (local) or OpenAI (API). The active provider is
selected via env vars:

    EMBEDDING_PROVIDER  (default: auto-detect — "openai" if API key present,
                         else "sentence-transformers")
    EMBEDDING_MODEL     (default depends on provider)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Protocol, runtime_checkable

import httpx
from logos_config import get_env_value

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
    """Local provider using sentence-transformers."""

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
        vec = await asyncio.to_thread(model.encode, text)
        return vec.tolist()


class OpenAIEmbeddingProvider:
    """OpenAI Embeddings API provider."""

    # Default dimensions per model. text-embedding-3-small/large support the
    # `dimensions` param; ada-002 is fixed at 1536.
    _MODEL_DEFAULTS: dict[str, int] = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        base_url: str = "https://api.openai.com/v1",
        dimensions: int | None = None,
    ) -> None:
        self._api_key = api_key
        self._model_name = model
        self._base_url = base_url.rstrip("/")
        # If caller specifies dimensions, use it; otherwise use native default.
        self._dimension = dimensions or self._MODEL_DEFAULTS.get(model, 1536)
        self._timeout = httpx.Timeout(30.0)
        logger.info(
            "OpenAI embedding provider: model=%s, dim=%d", model, self._dimension
        )

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    async def embed(self, text: str) -> list[float]:
        payload: dict[str, Any] = {
            "model": self._model_name,
            "input": text,
        }
        # Only pass dimensions for models that support it (3-small, 3-large).
        if self._model_name != "text-embedding-ada-002":
            payload["dimensions"] = self._dimension

        url = f"{self._base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()

        data = response.json()
        return data["data"][0]["embedding"]


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------
_provider: EmbeddingProvider | None = None


def _detect_backend() -> str:
    """Auto-detect the best available backend."""
    explicit = get_env_value("EMBEDDING_PROVIDER")
    if explicit:
        return explicit.strip().lower()
    # Prefer OpenAI if an API key is available.
    has_key = get_env_value("HERMES_LLM_API_KEY") or get_env_value("OPENAI_API_KEY")
    if has_key:
        return "openai"
    return "sentence-transformers"


def get_embedding_provider() -> EmbeddingProvider:
    """Return the configured embedding provider (lazy singleton)."""
    global _provider
    if _provider is not None:
        return _provider

    backend = _detect_backend()

    if backend == "openai":
        api_key = get_env_value("HERMES_LLM_API_KEY") or get_env_value("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OpenAI embedding provider selected but no API key found "
                "in HERMES_LLM_API_KEY or OPENAI_API_KEY."
            )
        model = (
            get_env_value("EMBEDDING_MODEL", default="text-embedding-3-small")
            or "text-embedding-3-small"
        )
        base_url = (
            get_env_value("HERMES_LLM_BASE_URL", default="https://api.openai.com/v1")
            or "https://api.openai.com/v1"
        )
        dim_str = get_env_value("LOGOS_EMBEDDING_DIM")
        dimensions = int(dim_str) if dim_str else None
        _provider = OpenAIEmbeddingProvider(
            api_key=api_key,
            model=model,
            base_url=base_url,
            dimensions=dimensions,
        )
    elif backend == "sentence-transformers":
        model = (
            get_env_value("EMBEDDING_MODEL", default="all-MiniLM-L6-v2")
            or "all-MiniLM-L6-v2"
        )
        _provider = SentenceTransformerProvider(model=model)
    else:
        raise ValueError(f"Unknown EMBEDDING_PROVIDER: {backend!r}")

    return _provider

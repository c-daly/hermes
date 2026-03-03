"""Fetches ontology type lists from Sophia with in-memory caching.

Falls back to None when Sophia is unreachable, allowing callers to
use hardcoded types as a default.
"""

from __future__ import annotations

import logging
import time

import httpx
from logos_config import get_env_value

logger = logging.getLogger(__name__)

_DEFAULT_TTL_SECONDS = 300  # 5 minutes


class _TypeCache:
    """Simple in-memory cache with TTL for type lists."""

    def __init__(self, ttl_seconds: float = _DEFAULT_TTL_SECONDS) -> None:
        self._ttl = ttl_seconds
        self._data: dict[str, tuple[float, list[dict]]] = {}

    def get(self, key: str) -> list[dict] | None:
        entry = self._data.get(key)
        if entry is None:
            return None
        ts, value = entry
        if time.monotonic() - ts > self._ttl:
            del self._data[key]
            return None
        return value

    def set(self, key: str, value: list[dict]) -> None:
        self._data[key] = (time.monotonic(), value)


def get_sophia_url() -> str:
    """Build Sophia base URL from env config."""
    sophia_host = get_env_value("SOPHIA_HOST", default="localhost") or "localhost"
    sophia_port = get_env_value("SOPHIA_PORT", default="8080") or "8080"
    return f"http://{sophia_host}:{sophia_port}"


# Module-level shared cache instance
_shared_cache = _TypeCache()


async def fetch_type_list(
    sophia_url: str,
    *,
    _cache: _TypeCache | None = None,
) -> list[dict] | None:
    """Fetch current node types from Sophia.

    Args:
        sophia_url: Base URL of the Sophia service (e.g. "http://localhost:8080").
        _cache: Optional cache override for testing.

    Returns:
        List of {"name": ..., "description": ...} dicts, or None on failure.
    """
    cache = _cache if _cache is not None else _shared_cache
    cache_key = "node_types"

    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            response = await client.get(f"{sophia_url}/api/ontology/node-types")
        if response.status_code != 200:
            logger.warning(
                "Sophia returned %d for node types, falling back to defaults",
                response.status_code,
            )
            return None
        data = response.json()
        types: list[dict] = data.get("types", [])
        cache.set(cache_key, types)
        return types
    except Exception as e:
        logger.warning("Failed to fetch node types from Sophia: %s", e)
        return None


async def fetch_edge_type_list(
    sophia_url: str,
    *,
    _cache: _TypeCache | None = None,
) -> list[dict] | None:
    """Fetch current edge types from Sophia.

    Args:
        sophia_url: Base URL of the Sophia service (e.g. "http://localhost:8080").
        _cache: Optional cache override for testing.

    Returns:
        List of {"name": ..., "description": ...} dicts, or None on failure.
    """
    cache = _cache if _cache is not None else _shared_cache
    cache_key = "edge_types"

    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            response = await client.get(f"{sophia_url}/api/ontology/edge-types")
        if response.status_code != 200:
            logger.warning(
                "Sophia returned %d for edge types, falling back to defaults",
                response.status_code,
            )
            return None
        data = response.json()
        types: list[dict] = data.get("types", [])
        cache.set(cache_key, types)
        return types
    except Exception as e:
        logger.warning("Failed to fetch edge types from Sophia: %s", e)
        return None

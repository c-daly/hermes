"""Node-type lists for extraction prompts, from the Redis-backed TypeRegistry.

Historically this fetched ``GET {sophia}/api/ontology/node-types`` — an
endpoint that never existed sophia-side, so every caller silently fell back
to hardcoded defaults (hermes#146). Types now come from the TypeRegistry
that ``hermes.main`` wires at startup (sophia publishes the snapshot to
``logos:ontology:types``; the registry stays live via pub/sub).

Falls back to None when no registry is wired or it holds no types,
allowing callers to use hardcoded types as a default.
"""

from __future__ import annotations

import logging
import time

from logos_config import get_env_value

from hermes.type_registry import get_active_registry

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
    """Current node types from the active TypeRegistry.

    Args:
        sophia_url: Unused; retained for caller compatibility (the legacy
            HTTP endpoint never existed — hermes#146).
        _cache: Optional cache override for testing.

    Returns:
        List of {"name": ..., "description": ...} dicts, or None when no
        registry is wired or it holds no types (callers fall back to
        defaults).
    """
    cache = _cache if _cache is not None else _shared_cache
    cache_key = "node_types"

    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    registry = get_active_registry()
    if registry is None:
        logger.debug("No active TypeRegistry; using fallback types")
        return None
    names = registry.get_type_names()
    if not names:
        logger.debug("TypeRegistry holds no types; using fallback types")
        return None
    types = [
        {
            "name": name,
            "description": (registry.get_type(name) or {}).get("description", ""),
        }
        for name in names
    ]
    cache.set(cache_key, types)
    return types

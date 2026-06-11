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

from hermes.type_registry import get_active_registry

logger = logging.getLogger(__name__)

_DEFAULT_TTL_SECONDS = 300  # 5 minutes


class _TypeCache:
    """In-memory TTL cache for type lists, keyed by registry generation.

    An entry is valid only while (a) its TTL has not elapsed and (b) the
    registry has not reloaded since the entry was stored — a live reload
    (pub/sub ``proposal_processed``) bumps the registry generation, which
    invalidates the entry immediately rather than after a full TTL window.
    """

    def __init__(self, ttl_seconds: float = _DEFAULT_TTL_SECONDS) -> None:
        self._ttl = ttl_seconds
        self._data: dict[str, tuple[float, int, list[dict]]] = {}

    def get(self, key: str, generation: int) -> list[dict] | None:
        entry = self._data.get(key)
        if entry is None:
            return None
        ts, gen, value = entry
        if gen != generation or time.monotonic() - ts > self._ttl:
            del self._data[key]
            return None
        return value

    def set(self, key: str, generation: int, value: list[dict]) -> None:
        self._data[key] = (time.monotonic(), generation, value)


# Module-level shared cache instance
_shared_cache = _TypeCache()


async def fetch_type_list(
    *,
    _cache: _TypeCache | None = None,
) -> list[dict] | None:
    """Current node types from the active TypeRegistry.

    Args:
        _cache: Optional cache override for testing.

    Returns:
        List of {"name": ..., "description": ...} dicts, or None when no
        registry is wired or it holds no types (callers fall back to
        defaults).
    """
    cache = _cache if _cache is not None else _shared_cache
    cache_key = "node_types"

    registry = get_active_registry()
    if registry is None:
        logger.debug("No active TypeRegistry; using fallback types")
        return None

    generation = registry.generation
    cached = cache.get(cache_key, generation)
    if cached is not None:
        return cached

    names = registry.get_type_names()
    if not names:
        logger.debug("TypeRegistry holds no types; using fallback types")
        return None
    types = []
    for name in names:
        type_info = registry.get_type(name)
        if type_info is None:
            # Deleted concurrently between get_type_names() and here.
            continue
        types.append({"name": name, "description": type_info.get("description", "")})
    if not types:
        logger.debug("TypeRegistry holds no types; using fallback types")
        return None
    cache.set(cache_key, generation, types)
    return types

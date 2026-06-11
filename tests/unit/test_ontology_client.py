"""Tests for ontology_client — node types from the Redis-backed TypeRegistry."""

import asyncio

import pytest

from hermes.ontology_client import fetch_type_list, _TypeCache
from hermes.type_registry import TypeRegistry, set_active_registry


class _FakeRedis:
    """Stub redis carrying a (mutable) snapshot for TypeRegistry."""

    def __init__(self, snapshot: str | None) -> None:
        self._snapshot = snapshot

    def get(self, key: str) -> str | None:
        return self._snapshot

    def set_snapshot(self, snapshot: str | None) -> None:
        self._snapshot = snapshot


def _registry_with(snapshot_json: str | None) -> TypeRegistry:
    return TypeRegistry(_FakeRedis(snapshot_json))


class TestFetchTypeList:
    @pytest.mark.asyncio
    async def test_returns_types_from_active_registry(self):
        set_active_registry(
            _registry_with(
                '{"location": {"description": "a geographic place"},'
                ' "object": {"description": "a physical object"}}'
            )
        )
        try:
            result = await fetch_type_list(_cache=_TypeCache())
        finally:
            set_active_registry(None)

        assert result == [
            {"name": "location", "description": "a geographic place"},
            {"name": "object", "description": "a physical object"},
        ]

    @pytest.mark.asyncio
    async def test_no_active_registry_returns_none(self):
        set_active_registry(None)
        result = await fetch_type_list(_cache=_TypeCache())
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_registry_returns_none(self):
        set_active_registry(_registry_with(None))
        try:
            result = await fetch_type_list(_cache=_TypeCache())
        finally:
            set_active_registry(None)
        assert result is None

    @pytest.mark.asyncio
    async def test_missing_description_defaults_to_empty(self):
        set_active_registry(_registry_with('{"creature": {}}'))
        try:
            result = await fetch_type_list(_cache=_TypeCache())
        finally:
            set_active_registry(None)
        assert result == [{"name": "creature", "description": ""}]

    @pytest.mark.asyncio
    async def test_type_deleted_between_listing_and_read_is_skipped(self):
        """A type deleted concurrently between get_type_names() and
        get_type() must be filtered out, not returned as a phantom."""

        class _RacyRegistry:
            generation = 0

            def get_type_names(self) -> list[str]:
                return ["ghost", "location"]

            def get_type(self, name: str) -> dict | None:
                if name == "ghost":
                    return None  # deleted mid-flight
                return {"description": "a place"}

        set_active_registry(_RacyRegistry())  # type: ignore[arg-type]
        try:
            result = await fetch_type_list(_cache=_TypeCache())
        finally:
            set_active_registry(None)
        assert result == [{"name": "location", "description": "a place"}]


class _CountingRegistry:
    """Duck-typed registry counting reads — the cache seam under test."""

    def __init__(self) -> None:
        self.reads = 0
        self.generation = 0

    def get_type_names(self) -> list[str]:
        self.reads += 1
        return ["location"]

    def get_type(self, name: str) -> dict | None:
        return {"description": "a place"}


class TestTypeCache:
    @pytest.mark.asyncio
    async def test_cache_returns_value_within_ttl(self):
        cache = _TypeCache(ttl_seconds=60)
        registry = _CountingRegistry()
        set_active_registry(registry)  # type: ignore[arg-type]
        try:
            result1 = await fetch_type_list(_cache=cache)
            result2 = await fetch_type_list(_cache=cache)
        finally:
            set_active_registry(None)

        assert result1 == result2 == [{"name": "location", "description": "a place"}]
        # Only one registry read — second call used the cache
        assert registry.reads == 1

    @pytest.mark.asyncio
    async def test_cache_expires_after_ttl(self):
        cache = _TypeCache(ttl_seconds=0.01)  # 10ms TTL
        registry = _CountingRegistry()
        set_active_registry(registry)  # type: ignore[arg-type]
        try:
            await fetch_type_list(_cache=cache)
            await asyncio.sleep(0.02)  # Wait for TTL to expire
            await fetch_type_list(_cache=cache)
        finally:
            set_active_registry(None)

        # Two registry reads — cache expired between calls
        assert registry.reads == 2

    @pytest.mark.asyncio
    async def test_registry_reload_invalidates_cache_within_ttl(self):
        """A live reload (proposal_processed) must invalidate the cache
        immediately — callers see the new types, not a stale TTL window."""
        cache = _TypeCache(ttl_seconds=60)
        fake_redis = _FakeRedis('{"location": {"description": "a place"}}')
        registry = TypeRegistry(fake_redis)
        set_active_registry(registry)
        try:
            first = await fetch_type_list(_cache=cache)
            fake_redis.set_snapshot(
                '{"location": {"description": "a place"},'
                ' "vehicle": {"description": "a conveyance"}}'
            )
            registry.on_proposal_processed({})  # pub/sub reload
            second = await fetch_type_list(_cache=cache)
        finally:
            set_active_registry(None)

        assert first == [{"name": "location", "description": "a place"}]
        assert second == [
            {"name": "location", "description": "a place"},
            {"name": "vehicle", "description": "a conveyance"},
        ]

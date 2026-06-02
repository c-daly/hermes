"""Opportunistic, non-blocking Sophia context retrieval (#154 Stage 1).

`_get_sophia_context` must:
- return cached context if present (opportunistic), else [] on a miss,
- NEVER make a synchronous Sophia HTTP call (no httpx.AsyncClient), and
- enqueue the proposal for background ingestion fire-and-forget.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from hermes import main


@pytest.mark.asyncio
class TestOpportunisticContext:
    async def _run(self, monkeypatch, cached):
        cache = MagicMock()
        cache.available = True
        cache.get_context.return_value = cached
        monkeypatch.setattr(main, "_get_context_cache", lambda: cache)

        # A synchronous Sophia call would instantiate httpx.AsyncClient; assert it never does.
        sync_client = MagicMock()
        monkeypatch.setattr(main.httpx, "AsyncClient", sync_client)

        build = AsyncMock(
            return_value={
                "proposal_id": "p",
                "source_service": "hermes",
                "proposed_nodes": [],
                "proposed_edges": [],
            }
        )
        monkeypatch.setattr(main._proposal_builder, "build", build)

        result = await main._get_sophia_context(
            "hello", "req-1", {"conversation_id": "c1"}
        )
        # let the fire-and-forget background task run
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        return result, cache, sync_client, build

    async def test_cache_hit_returns_cached_no_sync_call(self, monkeypatch):
        cached = [{"node_uuid": "u1", "name": "x"}]
        result, cache, sync_client, build = await self._run(monkeypatch, cached)
        assert result == cached
        assert sync_client.call_count == 0, "must not make a synchronous Sophia call"
        cache.enqueue_proposal.assert_called_once()

    async def test_cache_miss_returns_empty_no_sync_call(self, monkeypatch):
        result, cache, sync_client, build = await self._run(monkeypatch, [])
        assert result == []
        assert sync_client.call_count == 0, "must not make a synchronous Sophia call"
        cache.enqueue_proposal.assert_called_once()

"""Tests for ontology_client — fetches type lists from Sophia with caching."""

import time

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from hermes.ontology_client import fetch_type_list, fetch_edge_type_list, _TypeCache


class TestFetchTypeList:
    @pytest.mark.asyncio
    async def test_successful_fetch_returns_parsed_types(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "types": [
                {"name": "location", "description": "a geographic place"},
                {"name": "object", "description": "a physical object"},
            ]
        }

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch("hermes.ontology_client.httpx.AsyncClient", return_value=mock_client):
            result = await fetch_type_list("http://localhost:8080", _cache=_TypeCache())

        assert result is not None
        assert len(result) == 2
        assert result[0] == {"name": "location", "description": "a geographic place"}
        assert result[1] == {"name": "object", "description": "a physical object"}

    @pytest.mark.asyncio
    async def test_fetch_failure_returns_none(self):
        import httpx

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))

        with patch("hermes.ontology_client.httpx.AsyncClient", return_value=mock_client):
            result = await fetch_type_list("http://localhost:8080", _cache=_TypeCache())

        assert result is None

    @pytest.mark.asyncio
    async def test_non_200_returns_none(self):
        mock_response = MagicMock()
        mock_response.status_code = 404

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch("hermes.ontology_client.httpx.AsyncClient", return_value=mock_client):
            result = await fetch_type_list("http://localhost:8080", _cache=_TypeCache())

        assert result is None


class TestFetchEdgeTypeList:
    @pytest.mark.asyncio
    async def test_successful_fetch_returns_parsed_edge_types(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "types": [
                {"name": "LOCATED_IN", "description": "spatial containment"},
                {"name": "PART_OF", "description": "part-whole relation"},
            ]
        }

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch("hermes.ontology_client.httpx.AsyncClient", return_value=mock_client):
            result = await fetch_edge_type_list("http://localhost:8080", _cache=_TypeCache())

        assert result is not None
        assert len(result) == 2
        assert result[0]["name"] == "LOCATED_IN"

    @pytest.mark.asyncio
    async def test_fetch_failure_returns_none(self):
        import httpx

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))

        with patch("hermes.ontology_client.httpx.AsyncClient", return_value=mock_client):
            result = await fetch_edge_type_list("http://localhost:8080", _cache=_TypeCache())

        assert result is None


class TestTypeCache:
    @pytest.mark.asyncio
    async def test_cache_returns_value_within_ttl(self):
        cache = _TypeCache(ttl_seconds=60)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "types": [{"name": "location", "description": "a place"}]
        }

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch("hermes.ontology_client.httpx.AsyncClient", return_value=mock_client):
            result1 = await fetch_type_list("http://localhost:8080", _cache=cache)
            result2 = await fetch_type_list("http://localhost:8080", _cache=cache)

        assert result1 == result2
        # Only one HTTP call — second used cache
        assert mock_client.get.await_count == 1

    @pytest.mark.asyncio
    async def test_cache_expires_after_ttl(self):
        cache = _TypeCache(ttl_seconds=0.01)  # 10ms TTL

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "types": [{"name": "location", "description": "a place"}]
        }

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch("hermes.ontology_client.httpx.AsyncClient", return_value=mock_client):
            await fetch_type_list("http://localhost:8080", _cache=cache)
            time.sleep(0.02)  # Wait for TTL to expire
            await fetch_type_list("http://localhost:8080", _cache=cache)

        # Two HTTP calls — cache expired
        assert mock_client.get.await_count == 2

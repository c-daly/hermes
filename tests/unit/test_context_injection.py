"""Tests for context injection — the loop-closing flow."""

import httpx
import pytest
from unittest.mock import AsyncMock, patch, MagicMock


@pytest.mark.asyncio
class TestContextInjection:
    async def test_build_context_message(self):
        from hermes.main import _build_context_message

        context = [
            {"name": "Paris", "type": "GPE", "properties": {}, "score": 0.1},
            {"name": "France", "type": "GPE", "properties": {}, "score": 0.2},
        ]
        msg = _build_context_message(context)
        assert msg["role"] == "system"
        assert "Paris" in msg["content"]
        assert "France" in msg["content"]

    async def test_build_context_message_empty(self):
        from hermes.main import _build_context_message

        assert _build_context_message([]) is None

    async def test_build_context_message_with_properties(self):
        from hermes.main import _build_context_message

        context = [
            {
                "name": "Paris",
                "type": "location",
                "properties": {"population": "2M", "country": "France"},
            },
        ]
        msg = _build_context_message(context)
        assert msg is not None
        assert "population=2M" in msg["content"]
        assert "country=France" in msg["content"]

    async def test_build_context_message_filters_internal_properties(self):
        from hermes.main import _build_context_message

        context = [
            {
                "name": "Paris",
                "type": "location",
                "properties": {
                    "population": "2M",
                    "source": "ingestion",
                    "derivation": "observed",
                    "confidence": 0.9,
                    "raw_text": "some text",
                    "created_at": "2024-01-01",
                    "updated_at": "2024-01-01",
                },
            },
        ]
        msg = _build_context_message(context)
        assert msg is not None
        assert "population=2M" in msg["content"]
        # Internal properties should be excluded
        assert "source=" not in msg["content"]
        assert "derivation=" not in msg["content"]
        assert "raw_text=" not in msg["content"]

    async def test_get_sophia_context_returns_empty_on_failure(self):
        """When Sophia is unavailable, _get_sophia_context should return empty list."""
        from hermes.main import _get_sophia_context

        with (
            patch("hermes.main._proposal_builder") as mock_builder,
            patch("hermes.main.httpx.AsyncClient") as mock_async_client,
            patch("hermes.main.get_env_value") as mock_env,
        ):
            mock_builder.build = AsyncMock(
                return_value={
                    "proposal_id": "p1",
                    "proposed_nodes": [],
                    "document_embedding": None,
                    "source_service": "hermes",
                    "metadata": {},
                }
            )
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.ConnectError("Connection refused")
            mock_async_client.return_value.__aenter__ = AsyncMock(
                return_value=mock_client
            )
            mock_async_client.return_value.__aexit__ = AsyncMock(return_value=False)

            def env_side_effect(key, default=None):
                mapping = {
                    "SOPHIA_HOST": "localhost",
                    "SOPHIA_PORT": "47000",
                    "SOPHIA_API_KEY": "test-token",
                    "SOPHIA_API_TOKEN": None,
                }
                return mapping.get(key, default)

            mock_env.side_effect = env_side_effect

            context = await _get_sophia_context("Hello", "req-1", {})

        assert context == []

    async def test_get_sophia_context_returns_empty_without_token(self):
        """Without SOPHIA_API_TOKEN, context retrieval is disabled."""
        from hermes.main import _get_sophia_context

        with patch("hermes.main.get_env_value", return_value=None):
            context = await _get_sophia_context("Hello", "req-1", {})

        assert context == []

    async def test_get_sophia_context_returns_relevant_context(self):
        """When Sophia responds 201, we extract the relevant_context list."""
        from hermes.main import _get_sophia_context

        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {
            "relevant_context": [
                {"name": "Paris", "type": "location", "properties": {}, "score": 0.1},
            ],
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with (
            patch("hermes.main._proposal_builder") as mock_builder,
            patch("hermes.main.httpx.AsyncClient") as mock_async_client,
            patch("hermes.main.get_env_value") as mock_env,
        ):
            mock_builder.build = AsyncMock(
                return_value={
                    "proposal_id": "p1",
                    "proposed_nodes": [],
                    "document_embedding": None,
                    "source_service": "hermes",
                    "metadata": {},
                }
            )
            mock_async_client.return_value.__aenter__ = AsyncMock(
                return_value=mock_client
            )
            mock_async_client.return_value.__aexit__ = AsyncMock(return_value=False)

            def env_side_effect(key, default=None):
                mapping = {
                    "SOPHIA_HOST": "localhost",
                    "SOPHIA_PORT": "47000",
                    "SOPHIA_API_KEY": "test-token",
                    "SOPHIA_API_TOKEN": None,
                }
                return mapping.get(key, default)

            mock_env.side_effect = env_side_effect

            context = await _get_sophia_context("Tell me about Paris", "req-2", {})

        assert len(context) == 1
        assert context[0]["name"] == "Paris"

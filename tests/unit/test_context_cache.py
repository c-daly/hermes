"""Tests for ContextCache — Redis-backed context cache and proposal queue."""

import json

from unittest.mock import MagicMock

from hermes.context_cache import ContextCache


class TestContextCache:
    def test_get_returns_empty_when_no_context(self):
        mock_redis = MagicMock()
        mock_redis.get.return_value = None

        cache = ContextCache.__new__(ContextCache)
        cache._redis = mock_redis
        cache._available = True

        result = cache.get_context("conv-1")
        assert result == []

    def test_get_returns_cached_context(self):
        context = [{"node_uuid": "n1", "name": "robot"}]
        mock_redis = MagicMock()
        mock_redis.get.return_value = json.dumps(context).encode()

        cache = ContextCache.__new__(ContextCache)
        cache._redis = mock_redis
        cache._available = True

        result = cache.get_context("conv-1")
        assert len(result) == 1
        assert result[0]["name"] == "robot"

    def test_get_context_uses_correct_key(self):
        mock_redis = MagicMock()
        mock_redis.get.return_value = None

        cache = ContextCache.__new__(ContextCache)
        cache._redis = mock_redis
        cache._available = True

        cache.get_context("conv-42")
        mock_redis.get.assert_called_once_with("sophia:context:conv-42")

    def test_enqueue_proposal(self):
        mock_redis = MagicMock()

        cache = ContextCache.__new__(ContextCache)
        cache._redis = mock_redis
        cache._available = True

        cache.enqueue_proposal({"raw_text": "hello"}, conversation_id="conv-1")
        mock_redis.lpush.assert_called_once()

        # Verify the queued message structure
        args = mock_redis.lpush.call_args
        assert args[0][0] == "sophia:proposals:pending"
        payload = json.loads(args[0][1])
        assert payload["payload"] == {"raw_text": "hello"}
        assert payload["conversation_id"] == "conv-1"
        assert payload["attempts"] == 0
        assert payload["id"].startswith("pq-")
        assert "created_at" in payload

    def test_graceful_fallback_when_redis_unavailable(self):
        cache = ContextCache.__new__(ContextCache)
        cache._redis = None
        cache._available = False

        assert cache.get_context("conv-1") == []
        cache.enqueue_proposal({"raw_text": "hello"})  # should not raise

    def test_get_context_handles_redis_exception(self):
        mock_redis = MagicMock()
        mock_redis.get.side_effect = Exception("connection lost")

        cache = ContextCache.__new__(ContextCache)
        cache._redis = mock_redis
        cache._available = True

        result = cache.get_context("conv-1")
        assert result == []

    def test_enqueue_handles_redis_exception(self):
        mock_redis = MagicMock()
        mock_redis.lpush.side_effect = Exception("connection lost")

        cache = ContextCache.__new__(ContextCache)
        cache._redis = mock_redis
        cache._available = True

        # Should not raise
        cache.enqueue_proposal({"raw_text": "hello"})

    def test_available_property(self):
        cache = ContextCache.__new__(ContextCache)
        cache._available = True
        assert cache.available is True

        cache._available = False
        assert cache.available is False

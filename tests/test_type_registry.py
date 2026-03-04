"""Tests for hermes.type_registry.TypeRegistry."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from hermes.type_registry import TypeRegistry


class TestTypeRegistryInit:
    """Tests for TypeRegistry initialization from Redis snapshot."""

    def test_loads_types_from_redis_on_init(self):
        """TypeRegistry reads logos:ontology:types from Redis on init."""
        mock_redis = MagicMock()
        mock_redis.get.return_value = json.dumps(
            {
                "person": {"uuid": "t1", "member_count": 10},
                "location": {"uuid": "t2", "member_count": 5},
            }
        )

        registry = TypeRegistry(mock_redis)

        mock_redis.get.assert_called_with("logos:ontology:types")
        assert registry.get_type_names() == sorted(["person", "location"])

    def test_empty_registry_when_no_snapshot(self):
        """TypeRegistry starts empty if no Redis snapshot exists."""
        mock_redis = MagicMock()
        mock_redis.get.return_value = None

        registry = TypeRegistry(mock_redis)

        assert registry.get_type_names() == []

    def test_get_type_returns_type_dict(self):
        """get_type() returns the type properties dict."""
        mock_redis = MagicMock()
        mock_redis.get.return_value = json.dumps(
            {
                "person": {"uuid": "t1", "member_count": 10},
            }
        )

        registry = TypeRegistry(mock_redis)

        result = registry.get_type("person")
        assert result == {"uuid": "t1", "member_count": 10}

    def test_get_type_returns_none_for_unknown(self):
        """get_type() returns None for unknown types."""
        mock_redis = MagicMock()
        mock_redis.get.return_value = json.dumps({})

        registry = TypeRegistry(mock_redis)

        assert registry.get_type("unknown") is None

    def test_format_for_prompt(self):
        """format_for_prompt() returns formatted type list string."""
        mock_redis = MagicMock()
        mock_redis.get.return_value = json.dumps(
            {
                "person": {"uuid": "t1", "member_count": 10},
                "location": {"uuid": "t2", "member_count": 5},
            }
        )

        registry = TypeRegistry(mock_redis)
        prompt = registry.format_for_prompt()

        assert "person" in prompt
        assert "location" in prompt


class TestTypeRegistryUpdate:
    """Tests for TypeRegistry updates via events."""

    def test_update_from_event(self):
        """on_proposal_processed reloads types from Redis."""
        mock_redis = MagicMock()
        # First call (init): empty snapshot
        # Second call (after event): updated snapshot with new type
        mock_redis.get.side_effect = [
            json.dumps({}),
            json.dumps({"vehicle": {"uuid": "t3", "member_count": 1}}),
        ]

        registry = TypeRegistry(mock_redis)
        assert registry.get_type_names() == []

        event = {
            "event_type": "proposal_processed",
            "source": "sophia",
            "timestamp": "2026-03-04T00:00:00Z",
            "payload": {
                "new_types": ["vehicle"],
                "updated_types": [],
            },
        }
        registry.on_proposal_processed(event)

        assert mock_redis.get.call_count == 2
        assert registry.get_type_names() == ["vehicle"]
        assert registry.get_type("vehicle") == {"uuid": "t3", "member_count": 1}

    def test_reload_clears_types_when_key_missing(self):
        """on_proposal_processed clears types when Redis key is absent."""
        mock_redis = MagicMock()
        # First call (init): has types
        # Second call (after event): key deleted
        mock_redis.get.side_effect = [
            json.dumps({"person": {"uuid": "t1", "member_count": 10}}),
            None,
        ]

        registry = TypeRegistry(mock_redis)
        assert registry.get_type_names() == ["person"]

        registry.on_proposal_processed({"payload": {}})

        assert registry.get_type_names() == []

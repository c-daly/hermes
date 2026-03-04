"""TypeRegistry — live ontology type cache backed by Redis + pub/sub.

Reads the full type list from Redis on boot. Subscribes to
logos:sophia:proposal_processed events to stay in sync as Sophia
evolves the ontology.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)


class TypeRegistry:
    """Thread-safe registry of ontology types, synced from Redis."""

    REDIS_KEY = "logos:ontology:types"

    def __init__(self, redis_client: Any) -> None:
        self._redis = redis_client
        self._lock = threading.Lock()
        self._types: dict[str, dict] = {}
        self._load_from_redis()

    def _load_from_redis(self) -> None:
        """Load type snapshot from Redis.

        The entire read-parse-assign sequence is performed under the lock
        to prevent TOCTOU races where a slower concurrent reload could
        overwrite the registry with a stale snapshot.
        """
        try:
            with self._lock:
                raw = self._redis.get(self.REDIS_KEY)
                if raw is not None:
                    data = json.loads(raw)
                    if not isinstance(data, dict):
                        logger.error(
                            "TypeRegistry: invalid snapshot type %s (expected dict)",
                            type(data).__name__,
                        )
                        self._types = {}
                    else:
                        self._types = data
                        logger.info(
                            "TypeRegistry loaded %d types from Redis", len(self._types)
                        )
                else:
                    self._types = {}
                    logger.info("TypeRegistry: no snapshot in Redis, cleared to empty")
        except Exception:
            logger.exception("TypeRegistry: failed to load from Redis")

    def get_type_names(self) -> list[str]:
        """Return sorted list of known type names."""
        with self._lock:
            return sorted(self._types.keys())

    def get_type(self, name: str) -> dict | None:
        """Return type properties dict, or None if unknown."""
        with self._lock:
            t = self._types.get(name)
            return dict(t) if t is not None else None

    def format_for_prompt(self) -> str:
        """Format type list for injection into NER prompt."""
        with self._lock:
            if not self._types:
                return "No ontology types available."
            lines = []
            for name in sorted(self._types):
                info = self._types[name]
                count = info.get("member_count", 0)
                lines.append(f"- {name} ({count} members)")
            return "Known entity types:\n" + "\n".join(lines)

    def on_proposal_processed(self, event: dict) -> None:
        """Handle proposal_processed event — reload types from Redis.

        This is the EventBus callback. Rather than incrementally updating,
        we re-read the full snapshot from Redis. This is simple and
        guarantees consistency even if events are missed.
        """
        logger.info("TypeRegistry: proposal_processed event, reloading types")
        self._load_from_redis()

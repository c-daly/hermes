"""Redis-backed cache for Sophia context and proposal queue.

Provides a non-blocking path for the /llm endpoint: read cached context
from a prior Sophia processing turn instead of calling Sophia synchronously,
and enqueue new proposals for background processing.
"""

import json
import logging
import uuid
from datetime import UTC, datetime
from typing import Optional

import redis

logger = logging.getLogger(__name__)


class ContextCache:
    """Redis-backed cache for Sophia context and proposal queue."""

    QUEUE_KEY = "sophia:proposals:pending"
    CONTEXT_PREFIX = "sophia:context:"

    def __init__(self, redis_url: str):
        try:
            self._redis: Optional[redis.Redis] = redis.from_url(redis_url)
            self._redis.ping()
            self._available = True
            logger.info("Context cache connected to Redis")
        except Exception as e:
            logger.warning(f"Redis unavailable for context cache: {e}")
            self._redis = None
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def get_context(self, conversation_id: str) -> list[dict]:
        """Return cached Sophia context for *conversation_id*, or ``[]``."""
        if not self._available or not self._redis:
            return []
        try:
            data = self._redis.get(f"{self.CONTEXT_PREFIX}{conversation_id}")
            if data:
                return json.loads(data)
        except Exception as e:
            logger.debug(f"Context cache read failed: {e}")
        return []

    def enqueue_proposal(
        self, proposal: dict, conversation_id: str | None = None
    ) -> None:
        """Push a proposal onto the Redis queue for background Sophia processing."""
        if not self._available or not self._redis:
            return
        try:
            message = json.dumps(
                {
                    "id": f"pq-{datetime.now(UTC).timestamp()}",
                    "payload": proposal,
                    "conversation_id": conversation_id,
                    "attempts": 0,
                    "created_at": datetime.now(UTC).isoformat(),
                }
            )
            self._redis.lpush(self.QUEUE_KEY, message)
        except Exception as e:
            logger.warning(f"Failed to enqueue proposal: {e}")

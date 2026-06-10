"""RelationRegistry — seeds the predicate vocabulary from Redis + pub/sub.

The edge-axis analog of TypeRegistry (hermes#137, H4). Reads the
descriptive-relation snapshot Sophia publishes to ``logos:ontology:relations``
(sophia#190) and seeds the match-before-mint ``PredicateVocabulary`` on boot,
re-seeding on every ``logos:sophia:proposal_processed`` event so newly-minted
relations propagate to every Hermes worker. Redis is the shared substrate, so
this makes match-before-mint cross-process; the EventBus reload keeps it
fresh.

Unlike TypeRegistry this holds no state of its own — the vocabulary is the
store. Seeding is additive (``PredicateVocabulary.seed`` uses setdefault), so
a reload never drops a predicate minted in-process that has not yet
round-tripped back through Sophia's next snapshot.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from hermes.predicate_resolver import PredicateVocabulary

logger = logging.getLogger(__name__)


class RelationRegistry:
    """Syncs the descriptive-relation vocabulary from Redis into the
    match-before-mint vocabulary."""

    REDIS_KEY = "logos:ontology:relations"

    def __init__(self, redis_client: Any, vocabulary: PredicateVocabulary) -> None:
        self._redis = redis_client
        self._vocabulary = vocabulary
        self._load_and_seed()

    def _load_and_seed(self) -> None:
        """Read the relation snapshot from Redis and seed the vocabulary.

        Fail-soft: a missing key, malformed snapshot, or Redis error leaves
        the vocabulary as-is (extraction still works with whatever it has).
        """
        try:
            raw = self._redis.get(self.REDIS_KEY)
        except Exception:
            logger.exception("RelationRegistry: failed to read %s", self.REDIS_KEY)
            return
        if raw is None:
            logger.info("RelationRegistry: no relation snapshot in Redis")
            return
        try:
            data = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            logger.exception("RelationRegistry: malformed relation snapshot")
            return
        if not isinstance(data, dict):
            logger.error(
                "RelationRegistry: invalid snapshot type %s (expected dict)",
                type(data).__name__,
            )
            return
        # seed() filters reserved typing relations and dedups by canonical key
        self._vocabulary.seed(data.keys())
        logger.info(
            "RelationRegistry seeded %d relations from Redis", len(data)
        )

    def on_proposal_processed(self, event: dict) -> None:
        """EventBus callback — re-read the snapshot and re-seed (additive)."""
        logger.info("RelationRegistry: proposal_processed event, reloading relations")
        self._load_and_seed()

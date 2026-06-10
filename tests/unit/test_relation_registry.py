"""Tests for RelationRegistry (hermes#137, H4).

Mirrors TypeRegistry: reads the descriptive-relation snapshot Sophia
publishes to logos:ontology:relations (sophia#190) and seeds the
match-before-mint PredicateVocabulary, on boot and on each
proposal_processed event. Fail-soft.
"""

import json
from unittest.mock import MagicMock

from hermes.predicate_resolver import PredicateVocabulary
from hermes.relation_registry import RelationRegistry


def _redis_with(snapshot):
    r = MagicMock()
    r.get.return_value = json.dumps(snapshot) if snapshot is not None else None
    return r


def test_seeds_vocabulary_from_snapshot_on_boot():
    vocab = PredicateVocabulary()
    redis = _redis_with(
        {"LOCATED_IN": {"edge_count": 12}, "PRODUCES": {"edge_count": 5}}
    )
    RelationRegistry(redis, vocab)
    assert vocab.known() == {"LOCATED_IN", "PRODUCES"}
    # reads the relation key, not the type key
    assert redis.get.call_args[0][0] == "logos:ontology:relations"


def test_seeded_vocabulary_matches_inflected_variant():
    vocab = PredicateVocabulary()
    RelationRegistry(_redis_with({"LOCATED_IN": {"edge_count": 3}}), vocab)
    r = vocab.resolve("locates in")
    assert r.status == "matched" and r.relation == "LOCATED_IN"


def test_missing_snapshot_leaves_vocabulary_empty_no_error():
    vocab = PredicateVocabulary()
    RelationRegistry(_redis_with(None), vocab)
    assert vocab.known() == set()


def test_redis_failure_is_swallowed():
    vocab = PredicateVocabulary()
    redis = MagicMock()
    redis.get.side_effect = RuntimeError("redis down")
    RelationRegistry(redis, vocab)  # must not raise
    assert vocab.known() == set()


def test_non_dict_snapshot_is_ignored():
    vocab = PredicateVocabulary()
    redis = MagicMock()
    redis.get.return_value = json.dumps(["LOCATED_IN"])  # wrong shape
    RelationRegistry(redis, vocab)
    assert vocab.known() == set()


def test_reload_on_event_seeds_new_relations():
    vocab = PredicateVocabulary()
    redis = _redis_with({"LOCATED_IN": {"edge_count": 1}})
    reg = RelationRegistry(redis, vocab)
    # Sophia publishes a new relation; next snapshot includes it
    redis.get.return_value = json.dumps(
        {"LOCATED_IN": {"edge_count": 1}, "ORBITS_AROUND": {"edge_count": 2}}
    )
    reg.on_proposal_processed({"event_type": "proposal_processed"})
    assert "ORBITS_AROUND" in vocab.known()


def test_reload_preserves_in_process_mints():
    # a predicate minted in-process (not yet in any snapshot) must survive a
    # reload -- seed() is additive (the mint round-trips via Sophia later)
    vocab = PredicateVocabulary()
    reg = RelationRegistry(_redis_with({"LOCATED_IN": {"edge_count": 1}}), vocab)
    vocab.resolve("TELEPORTS")  # in-process mint
    reg.on_proposal_processed({})
    assert "TELEPORTS" in vocab.known()


def test_reserved_relation_in_snapshot_never_seeded():
    # defense in depth: even if a reserved relation leaks into the snapshot,
    # the vocabulary never accepts it
    vocab = PredicateVocabulary()
    RelationRegistry(_redis_with({"IS_A": {"edge_count": 9}}), vocab)
    assert "IS_A" not in vocab.known()

"""H2 wiring: the OpenAI/combined relation parse path resolves predicates
and attaches provenance (hermes#132).

Exercises OpenAIRelationExtractor._parse_response (the shared choke point
for the openai AND combined backends) against a fresh, isolated
vocabulary so match-before-mint is observable end to end.
"""

import json

import pytest

import hermes.predicate_resolver as pr
from hermes.relation_extractor import OpenAIRelationExtractor


@pytest.fixture
def fresh_vocab(monkeypatch):
    """Swap the process singleton for an isolated, empty vocabulary."""
    vocab = pr.PredicateVocabulary()
    monkeypatch.setattr(pr, "_VOCABULARY", vocab)
    monkeypatch.setattr(
        "hermes.relation_extractor.get_predicate_vocabulary", lambda: vocab
    )
    return vocab


def _payload(*relations):
    return json.dumps({"relations": list(relations)})


def test_provenance_attached_on_mint(fresh_vocab):
    content = _payload(
        {"source_name": "A", "target_name": "B", "relation": "orbits around"}
    )
    out = OpenAIRelationExtractor._parse_response(content, {"A", "B"})
    assert len(out) == 1
    prov = out[0]["properties"]["predicate"]
    assert prov["status"] == "minted"
    assert prov["raw"] == "orbits around"
    assert prov["canonical"] == "ORBIT_AROUND"
    assert out[0]["relation"] == "ORBITS_AROUND"  # readable surface, not stem


def test_inflected_variant_matches_earlier_mint(fresh_vocab):
    fresh_vocab.seed(["LOCATED_IN"])
    content = _payload(
        {"source_name": "A", "target_name": "B", "relation": "locates in"}
    )
    out = OpenAIRelationExtractor._parse_response(content, {"A", "B"})
    assert out[0]["relation"] == "LOCATED_IN"
    assert out[0]["properties"]["predicate"]["status"] == "matched"


def test_two_variants_in_one_batch_collapse_to_one_edge(fresh_vocab):
    # PRODUCES and PRODUCED between the same pair -> same canonical -> the
    # (src,tgt,relation) dedup now collapses them into a single edge
    content = _payload(
        {"source_name": "A", "target_name": "B", "relation": "PRODUCES"},
        {"source_name": "A", "target_name": "B", "relation": "PRODUced"},
    )
    out = OpenAIRelationExtractor._parse_response(content, {"A", "B"})
    assert len(out) == 1
    assert out[0]["relation"] == "PRODUCES"

"""Tests for catalog-aware prefer-existing predicate resolution (hermes#132, H2).

Match-before-mint for descriptive relations: a candidate predicate is
checked against the known vocabulary by its H1 canonical key before a new
one is minted. The STORED relation is a readable surface form (first-seen
per canonical class); minting is the exception. IS_A / INSTANCE_OF /
SUBTYPE_OF bypass the path untouched. Every resolution carries provenance.
"""

import pytest

from hermes.predicate_resolver import (
    PredicateVocabulary,
    index_vocabulary,
    resolve_predicate,
)


class TestIndexVocabulary:
    def test_groups_by_canonical_keeping_min_surface(self):
        idx = index_vocabulary(["LOCATED_IN", "LOCATES_IN", "PRODUCES"])
        # LOCATED_IN / LOCATES_IN share canonical LOCAT_IN -> one entry, the
        # lexicographically-min surface is the representative (deterministic)
        assert idx["LOCAT_IN"] == "LOCATED_IN"
        assert idx["PRODUC"] == "PRODUCES"

    def test_reserved_predicates_excluded_from_index(self):
        idx = index_vocabulary(["IS_A", "INSTANCE_OF", "PART_OF"])
        assert "IS_A" not in idx and "INSTANCE_OF" not in idx
        assert idx  # PART_OF survives


class TestResolveAgainstFixedIndex:
    def _idx(self):
        return index_vocabulary(["LOCATED_IN", "PRODUCES", "PART_OF"])

    def test_exact_canonical_match_reuses_existing_surface(self):
        r = resolve_predicate("locates in", self._idx())
        assert r.relation == "LOCATED_IN"
        assert r.status == "matched"
        assert r.canonical == "LOCAT_IN"
        assert r.raw == "locates in"

    def test_inflected_variant_matches(self):
        assert resolve_predicate("PRODUCED", self._idx()).relation == "PRODUCES"
        assert resolve_predicate("PRODUCING", self._idx()).status == "matched"

    def test_unknown_predicate_mints_readable_surface(self):
        r = resolve_predicate("ORBITS_AROUND", self._idx())
        assert r.relation == "ORBITS_AROUND"  # surface, not the ORBIT_AROUND stem
        assert r.status == "minted"
        assert r.canonical == "ORBIT_AROUND"

    def test_reserved_relation_passes_through(self):
        r = resolve_predicate("IS_A", self._idx())
        assert r.relation == "IS_A" and r.status == "reserved"

    def test_typing_relations_never_match_descriptive_vocab(self):
        # SUBTYPE_OF must not be folded into PART_OF or anything else
        r = resolve_predicate("SUBTYPE_OF", self._idx())
        assert r.status == "reserved" and r.relation == "SUBTYPE_OF"

    def test_polarity_predicate_mints_separately(self):
        idx = index_vocabulary(["REFERS_TO"])
        r = resolve_predicate("DOES_NOT_REFER_TO", idx)
        assert r.status == "minted"
        assert r.relation == "DOES_NOT_REFER_TO"

    def test_empty_predicate_is_dropped(self):
        r = resolve_predicate("   ", self._idx())
        assert r.relation == "" and r.status == "empty"


class TestPredicateVocabularyAccumulation:
    def test_first_occurrence_mints_then_variants_match(self):
        vocab = PredicateVocabulary()
        first = vocab.resolve("LOCATED_IN")
        assert first.status == "minted"
        # a later inflected variant in the same process matches the mint
        second = vocab.resolve("locates in")
        assert second.status == "matched"
        assert second.relation == "LOCATED_IN"

    def test_seed_injects_known_vocabulary(self):
        vocab = PredicateVocabulary()
        vocab.seed(["PART_OF", "PRODUCES"])
        r = vocab.resolve("PRODUCED")
        assert r.status == "matched" and r.relation == "PRODUCES"

    def test_reserved_never_added_to_vocabulary(self):
        vocab = PredicateVocabulary()
        vocab.resolve("IS_A")
        assert "IS_A" not in vocab.known()
        assert vocab.resolve("INSTANCE_OF").status == "reserved"

    def test_known_returns_current_surfaces(self):
        vocab = PredicateVocabulary()
        vocab.resolve("ORBITS")
        vocab.resolve("PART_OF")
        assert vocab.known() == {"ORBITS", "PART_OF"}

    def test_accumulation_is_order_stable_on_surface_choice(self):
        # whichever surface is seen FIRST for a canonical class wins and sticks
        vocab = PredicateVocabulary()
        vocab.resolve("LOCATES_IN")  # mints this surface
        r = vocab.resolve("LOCATED_IN")  # same canonical -> matches the mint
        assert r.relation == "LOCATES_IN" and r.status == "matched"

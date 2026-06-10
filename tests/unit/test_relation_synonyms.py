"""Fail-closed contract tests for the relation synonym-collapse pass (#133, H3).

The LLM proposes descriptive-relation synonym groups (HAULS/DRAGS/CARRIES ->
CARRIES); this module validates server-side. The hard rules (epic #131):
no group may contain a reserved typing relation; members must come from the
submitted candidates (no hallucinated predicates); a "group" of one is not a
collapse; the canonical label is H1-normalized. Output is a PROPOSAL.
"""

import json

import pytest

from hermes.relation_synonyms import (
    build_synonym_messages,
    parse_synonym_response,
)


def _content(*groups):
    return json.dumps({"groups": list(groups)})


CANDIDATES = ["HAULS", "DRAGS", "CARRIES", "PRODUCES", "MAKES", "IS_A", "PART_OF"]


class TestParseHappyPath:
    def test_valid_group_kept_and_canonicalized(self):
        out = parse_synonym_response(
            _content({"canonical": "carries", "members": ["HAULS", "DRAGS", "CARRIES"]}),
            CANDIDATES,
        )
        assert len(out) == 1
        g = out[0]
        assert g.canonical == "CARRIES"  # normalized surface
        assert set(g.members) == {"HAULS", "DRAGS", "CARRIES"}
        assert 0.0 <= g.confidence <= 1.0

    def test_confidence_passed_through_and_clamped(self):
        out = parse_synonym_response(
            _content(
                {"canonical": "MAKES", "members": ["PRODUCES", "MAKES"], "confidence": 2.0}
            ),
            CANDIDATES,
        )
        assert out[0].confidence == 1.0


class TestFailClosed:
    def test_group_with_reserved_relation_is_dropped(self):
        out = parse_synonym_response(
            _content({"canonical": "IS_A", "members": ["IS_A", "PART_OF"]}),
            CANDIDATES,
        )
        assert out == []

    def test_reserved_member_anywhere_drops_the_group(self):
        out = parse_synonym_response(
            _content({"canonical": "CARRIES", "members": ["CARRIES", "INSTANCE_OF"]}),
            CANDIDATES + ["INSTANCE_OF"],
        )
        assert out == []

    def test_hallucinated_member_drops_the_group(self):
        out = parse_synonym_response(
            _content({"canonical": "CARRIES", "members": ["CARRIES", "TELEPORTS"]}),
            CANDIDATES,
        )
        assert out == []

    def test_singleton_group_is_not_a_collapse(self):
        out = parse_synonym_response(
            _content({"canonical": "CARRIES", "members": ["CARRIES"]}),
            CANDIDATES,
        )
        assert out == []

    def test_members_dedup_by_canonical_then_singleton_dropped(self):
        # CARRIES + CARRY are the same canonical -> effectively one member
        out = parse_synonym_response(
            _content({"canonical": "CARRIES", "members": ["CARRIES", "CARRY"]}),
            CANDIDATES + ["CARRY"],
        )
        assert out == []

    def test_malformed_json_returns_empty(self):
        assert parse_synonym_response("not json", CANDIDATES) == []

    def test_missing_groups_key_returns_empty(self):
        assert parse_synonym_response(json.dumps({"x": 1}), CANDIDATES) == []

    def test_canonical_not_among_members_is_repaired_to_a_member(self):
        # if the LLM names a canonical that isn't one of the members, keep the
        # group but the canonical must be a real member's surface
        out = parse_synonym_response(
            _content({"canonical": "TRANSPORTS", "members": ["HAULS", "CARRIES"]}),
            CANDIDATES,
        )
        assert len(out) == 1
        assert out[0].canonical in {"HAULS", "CARRIES"}


class TestPromptConstruction:
    def test_messages_list_the_candidates_and_forbid_is_a(self):
        msgs = build_synonym_messages(["HAULS", "CARRIES"], context="cargo domain")
        blob = " ".join(m["content"] for m in msgs)
        assert "HAULS" in blob and "CARRIES" in blob
        assert "IS_A" in blob  # the prompt must state the exclusion
        assert "cargo domain" in blob

    def test_reserved_candidates_filtered_from_the_prompt(self):
        msgs = build_synonym_messages(["CARRIES", "IS_A", "SUBTYPE_OF"])
        user = [m for m in msgs if m["role"] == "user"][0]["content"]
        # IS_A/SUBTYPE_OF must not be offered as groupable candidates
        assert "SUBTYPE_OF" not in user.split("Candidates:")[-1]

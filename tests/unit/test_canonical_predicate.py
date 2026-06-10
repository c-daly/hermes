"""Frozen golden-table tests for canonicalize_predicate (hermes#130, H1).

The edge-axis analog of test_canonical.py: ONE predicate canonicalizer lives
in hermes.canonical, the deterministic foundation under match-before-mint
(#132) and synonym-collapse (#133). It folds morphological variants of the
same descriptive relation onto one canonical UPPER_SNAKE key so the open
relation vocabulary (2,244 distinct, 62.6% df=1 on the live graph,
logos-experiments#25) stops sprawling at the source.

FROZEN table = contract. Convergence is the property under test: every
inflection of a relation must reach the SAME key, so two surface forms that
mean the same relation compare equal. The key is a stem (ACQUIR), not
necessarily pretty English -- consistency, not prose, is the job.

Hard constraints encoded below:
  * polarity markers (NOT/NO/NEVER/...) are NEVER folded onto the positive
    form -- DOES_NOT_REFER_TO must not collapse to REFERS_TO;
  * IS_A / INSTANCE_OF / SUBTYPE_OF are out of scope (callers exclude them),
    asserted here as fixed points so a regression is visible;
  * idempotent: canonicalize_predicate(canonicalize_predicate(x)) == itself.
"""

import pytest

from hermes.canonical import canonicalize_predicate

# FROZEN golden table -- (raw_predicate, canonical_key)
GOLDEN: list[tuple[str, str]] = [
    # separator / case normalization
    ("located in", "LOCAT_IN"),
    ("Located-In", "LOCAT_IN"),
    ("LOCATED_IN", "LOCAT_IN"),
    # present-3sg / base / past / gerund all converge (the core property)
    ("ACQUIRE", "ACQUIR"),
    ("ACQUIRES", "ACQUIR"),
    ("ACQUIRED", "ACQUIR"),
    ("ACQUIRING", "ACQUIR"),
    ("PRODUCES", "PRODUC"),
    ("PRODUCED", "PRODUC"),
    ("PRODUCING", "PRODUC"),
    # multi-token: every content token folds, preposition kept
    ("ACTS_IN", "ACT_IN"),
    ("ACT_IN", "ACT_IN"),
    ("LOCATES_IN", "LOCAT_IN"),
    ("BINDS_TO", "BIND_TO"),
    ("CATALYZES", "CATALYZ"),
    ("CATALYZE", "CATALYZ"),
    # -ies / -ied / -y all converge (review #134: -ied past tense)
    ("CARRIES", "CARRY"),
    ("CARRY", "CARRY"),
    ("CARRIED", "CARRY"),
    ("STUDIES", "STUDY"),
    ("STUDIED", "STUDY"),
    ("STUDY", "STUDY"),
    # short-token and consonant-cluster guards: no over-strip
    ("IS", "IS"),
    ("HAS", "HAS"),  # -AS guard
    ("USES", "USE"),  # -S strip; -E guard (len<5) keeps the E
    # -SS / -US / -IS suffix guard (not plurals)
    ("ASSESS", "ASSESS"),
    ("FOCUS", "FOCUS"),
    ("BASIS", "BASIS"),
    # negation folds per-token (the NOT/NEVER token is preserved, so it can
    # never collide with the positive form) -- review #134
    ("DOES_NOT_REFER_TO", "DOE_NOT_REFER_TO"),
    ("NOT_PART_OF", "NOT_PART_OF"),
    ("NEVER_OCCURS_WITH", "NEVER_OCCUR_WITH"),
    # typing relations are fixed points (out of scope, but must not mangle)
    ("IS_A", "IS_A"),
    ("INSTANCE_OF", "INSTANCE_OF"),
    ("SUBTYPE_OF", "SUBTYPE_OF"),
    # empty / whitespace
    ("", ""),
    ("   ", ""),
]


@pytest.mark.parametrize("raw,expected", GOLDEN)
def test_golden(raw, expected):
    assert canonicalize_predicate(raw) == expected


@pytest.mark.parametrize("raw,_", GOLDEN)
def test_idempotent(raw, _):
    once = canonicalize_predicate(raw)
    assert canonicalize_predicate(once) == once


class TestProperties:
    def test_present_forms_converge(self):
        # base and 3sg of a long-enough verb reach the same key
        assert canonicalize_predicate("USES") == canonicalize_predicate("USE")
        assert canonicalize_predicate("PRODUCES") == canonicalize_predicate("PRODUCE")

    def test_y_verb_past_tense_converges(self):
        # review #134: -IED past tense must not strand a bare -I
        key = canonicalize_predicate("CARRY")
        assert (
            key
            == canonicalize_predicate("CARRIES")
            == canonicalize_predicate("CARRIED")
        )
        assert canonicalize_predicate("APPLIED") == canonicalize_predicate("APPLIES")

    def test_negation_folds_but_never_collides_with_positive(self):
        # negated predicates fold (so their own inflections converge) but the
        # preserved polarity token keeps them distinct from the positive form
        assert canonicalize_predicate("does not produces") == canonicalize_predicate(
            "does not produced"
        )
        assert canonicalize_predicate("does not produce") != canonicalize_predicate(
            "produces"
        )

    def test_non_string_input_returns_empty(self):
        assert canonicalize_predicate(None) == ""  # type: ignore[arg-type]

    def test_separators_unified(self):
        forms = ["LOCATED IN", "located-in", "Located_In", "  located   in  "]
        keys = {canonicalize_predicate(f) for f in forms}
        assert len(keys) == 1

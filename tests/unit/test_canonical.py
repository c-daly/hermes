"""Frozen golden-table tests for the canonical name normalizer.

ONE canonicalize() lives in Hermes (hermes.canonical) -- normalization is
Hermes' job (2026-06-05). Sophia never normalizes: every name entering the
graph is already canonical (canonical-at-the-boundary), and the offline
naming-driven-typing harness imports this exact symbol in-process (path A).
This golden table is FROZEN: every (input -> output) pair below is a
contract. If a change here is intentional, it is a language-layer contract
change -- update the table, the defnoun list, and the boundary consumers in
the same PR.

Behavior under test (SPEC s5.4 as amended 2026-06-05):
NFKC -> strip -> collapse whitespace -> lowercase -> drop leading a/an/the ->
singularize HEAD NOUN ONLY via the shared word-level core (inflect; keep-list
+ -ss/-is/-us suffix guard + defnoun irregulars for the -ie/-ies class) ->
strip curated trailing fillers {type, category, kind, class, group, entity}
only when not the whole string.
canonicalize is idempotent: canonicalize(canonicalize(x)) == canonicalize(x).
"""

import pytest

from hermes.canonical import canonicalize, singularize_word

# FROZEN golden table -- the contract. (raw_input, canonical_output)
GOLDEN: list[tuple[str, str]] = [
    # plural -> singular head noun
    ("vehicle", "vehicle"),
    ("vehicles", "vehicle"),
    ("mammal", "mammal"),
    ("mammals", "mammal"),
    ("process", "process"),
    ("processes", "process"),
    ("bus", "bus"),
    ("buses", "bus"),
    # inflect over-strips these -- keep-list / suffix guard must protect them
    ("species", "species"),
    ("analysis", "analysis"),
    ("class", "class"),
    # realm roots are stable fixed points
    ("entity", "entity"),
    ("concept", "concept"),
    ("node", "node"),
    # -ie/-ies class: defnoun irregulars (inflect's unknown--ies default is -y;
    # these are the pinned exceptions -- grow the defnoun list, never fork rules)
    ("selfie", "selfie"),
    ("selfies", "selfie"),
    ("smoothie", "smoothie"),
    ("smoothies", "smoothie"),
    ("hoodies", "hoodie"),
    ("foodies", "foodie"),
    ("rottie", "rottie"),
    ("rotties", "rottie"),
    ("hippies", "hippie"),
    # unknown -ies defaults to -y (inflect) -- pinned so the convention is explicit
    ("townies", "towny"),
    # dictionary -ies words inflect already knows
    ("cookies", "cookie"),
    ("movies", "movie"),
    ("ponies", "pony"),
    ("cities", "city"),
    # Latin -ix/-ices: pinned defnoun (inflect default gives "matrice")
    ("matrix", "matrix"),
    ("matrices", "matrix"),
    # case / whitespace / unicode normalization
    ("  Vehicle  ", "vehicle"),
    ("VEHICLES", "vehicle"),
    ("vehicle\xa0\xa0type", "vehicle"),  # NBSP collapses; trailing "type" stripped
    # leading article stripping
    ("the vehicle", "vehicle"),
    ("a mammal", "mammal"),
    ("an analysis", "analysis"),
    # trailing filler stripping (only when not the whole string)
    ("vehicle type", "vehicle"),
    ("vehicle types", "vehicle"),
    ("mammal category", "mammal"),
    ("process group", "process"),
    # whole-string filler is NOT stripped (would empty the name)
    ("type", "type"),
    ("category", "category"),
    ("group", "group"),
]


@pytest.mark.parametrize("raw,expected", GOLDEN, ids=[g[0] for g in GOLDEN])
def test_canonicalize_golden_table(raw: str, expected: str) -> None:
    """Every frozen (input -> output) pair is a hard contract."""
    assert canonicalize(raw) == expected


@pytest.mark.parametrize("raw,expected", GOLDEN, ids=[g[0] for g in GOLDEN])
def test_canonicalize_is_idempotent(raw: str, expected: str) -> None:
    """canonicalize(canonicalize(x)) == canonicalize(x) for every golden input."""
    once = canonicalize(raw)
    assert canonicalize(once) == once
    assert once == expected


def test_vehicle_and_vehicles_collide() -> None:
    """The harness relies on this collision for semantic reuse_collapses."""
    assert canonicalize("vehicle") == canonicalize("vehicles")


def test_ie_ies_defnoun_class_collides() -> None:
    """Pinned -ie nouns: plural and singular collide (the rottie/rotty fix)."""
    for singular, plural in [
        ("selfie", "selfies"),
        ("smoothie", "smoothies"),
        ("rottie", "rotties"),
        ("hippie", "hippies"),
    ]:
        assert canonicalize(plural) == canonicalize(singular) == singular


def test_singularize_word_core_is_shared() -> None:
    """The word-level core is exported for name_normalizer (one engine, no forks)."""
    assert singularize_word("vehicles") == "vehicle"
    assert singularize_word("rotties") == "rottie"
    assert singularize_word("analysis") == "analysis"


def test_empty_and_whitespace_only_return_empty() -> None:
    """Degenerate inputs normalize to empty string, never raise."""
    assert canonicalize("") == ""
    assert canonicalize("   ") == ""
    assert canonicalize("\xa0") == ""

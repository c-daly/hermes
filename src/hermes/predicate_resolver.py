"""Catalog-aware prefer-existing predicate resolution (hermes#132, H2).

Match-before-mint for descriptive relations -- the edge-axis analog of
node-side rollup match-before-mint. A candidate relation label is checked
against the known vocabulary by its H1 canonical key (hermes.canonical):

  * reserved typing relations (IS_A / INSTANCE_OF / SUBTYPE_OF) pass through
    untouched -- they are structural, never consolidated;
  * if the candidate's canonical key matches a known predicate, REUSE that
    predicate's existing surface form (matched);
  * otherwise MINT: store the candidate's readable surface form (not the
    stem) as a new predicate (minted).

The canonical key does the matching; the stored label is a real surface
form, so the graph stays readable while the vocabulary stops sprawling.
Every resolution carries provenance (status + raw + canonical) for audit.

`resolve_predicate(raw, index)` is pure (testable). `PredicateVocabulary`
is the stateful wrapper extraction uses: mints accumulate in-process, and
`seed()` is the injection point for the persisted vocabulary (the
ontology-backed source / hermes#122 batch surface). H3 (#133) will add a
synonym-similarity step between match and mint.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from hermes.canonical import (
    RESERVED_PREDICATES,
    canonicalize_predicate,
    normalize_predicate_surface,
)


@dataclass(frozen=True)
class Resolution:
    relation: str  # the chosen stored label ("" if dropped)
    status: Literal["matched", "minted", "reserved", "empty"]
    canonical: str  # the H1 match key
    raw: str  # the original surface as received


def index_vocabulary(predicates: Iterable[str]) -> dict[str, str]:
    """canonical key -> representative surface, over known descriptive
    predicates. Reserved typing relations are excluded. On a canonical
    collision the lexicographically-min surface wins (deterministic)."""
    index: dict[str, str] = {}
    for p in predicates:
        surface = normalize_predicate_surface(p)
        if not surface or surface in RESERVED_PREDICATES:
            continue
        key = canonicalize_predicate(surface)
        existing = index.get(key)
        if existing is None or surface < existing:
            index[key] = surface
    return index


def resolve_predicate(raw: str, index: dict[str, str]) -> Resolution:
    """Resolve one candidate predicate against a fixed vocabulary index.

    Pure: does not mutate `index`. See module docstring for the policy.
    """
    surface = normalize_predicate_surface(raw)
    if not surface:
        return Resolution("", "empty", "", raw)
    if surface in RESERVED_PREDICATES:
        return Resolution(surface, "reserved", surface, raw)
    key = canonicalize_predicate(surface)
    existing = index.get(key)
    if existing is not None:
        return Resolution(existing, "matched", key, raw)
    return Resolution(surface, "minted", key, raw)


class PredicateVocabulary:
    """Stateful, thread-safe match-before-mint vocabulary.

    Resolving an unknown descriptive predicate mints it (the surface is
    added to the index); later inflected variants in the same process then
    match it. `seed()` injects a persisted vocabulary up front.
    """

    def __init__(self) -> None:
        self._index: dict[str, str] = {}
        self._lock = threading.Lock()

    def seed(self, predicates: Iterable[str]) -> None:
        # Normalize/canonicalize outside the lock (CPU-bound); hold the lock
        # only for the dict merge, to avoid contending with resolve().
        incoming = index_vocabulary(predicates)
        with self._lock:
            for key, surface in incoming.items():
                # seed never overwrites an existing mint for the same key
                self._index.setdefault(key, surface)

    def resolve(self, raw: str) -> Resolution:
        surface = normalize_predicate_surface(raw)
        if not surface:
            return Resolution("", "empty", "", raw)
        if surface in RESERVED_PREDICATES:
            return Resolution(surface, "reserved", surface, raw)
        key = canonicalize_predicate(surface)
        with self._lock:
            existing = self._index.get(key)
            if existing is not None:
                return Resolution(existing, "matched", key, raw)
            self._index[key] = surface
            return Resolution(surface, "minted", key, raw)

    def known(self) -> set[str]:
        with self._lock:
            return set(self._index.values())


# Process-local singleton used by the extractors. Eagerly initialized (the
# object is lightweight and has no import-time side effects) so there is no
# lazy check-then-set race if it is reached from a worker thread before
# relation_extractor's import-time seed runs. Seed it from the persisted
# vocabulary at startup (the ontology-injection hook) so match-before-mint
# spans processes; until then it accumulates within a process.
_VOCABULARY = PredicateVocabulary()


def get_predicate_vocabulary() -> PredicateVocabulary:
    return _VOCABULARY


# Graph usage counts (surface -> edge_count) published by RelationRegistry
# from Sophia's relation snapshot. Consumed by the H5 known-relations clause
# to rank the advertised vocabulary window by usage instead of slicing it
# alphabetically. Replaced wholesale on every snapshot load (atomic rebind;
# readers only .get on the current dict); {} until the first snapshot lands.
_RELATION_COUNTS: dict[str, int] = {}


def set_relation_counts(counts: dict[str, int]) -> None:
    """Replace the published usage-count map (surface -> edge_count)."""
    global _RELATION_COUNTS
    _RELATION_COUNTS = dict(counts)


def get_relation_counts() -> dict[str, int]:
    """Usage counts for known relations; {} before the first snapshot."""
    return _RELATION_COUNTS

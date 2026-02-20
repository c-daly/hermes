"""Pluggable relation extraction for Hermes.

Defines a protocol for relation extractors and a default implementation
using spaCy dependency parsing. The active extractor is selected via:

    RELATION_EXTRACTOR  (default: "spacy")
"""

from __future__ import annotations

import logging
import os
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# Map common verb/prep patterns to canonical relation labels.
_VERB_TO_RELATION: dict[str, str] = {
    "work": "WORKS_AT",
    "employ": "WORKS_AT",
    "locate": "LOCATED_IN",
    "base": "LOCATED_IN",
    "develop": "DEVELOPS",
    "build": "DEVELOPS",
    "create": "DEVELOPS",
    "research": "RESEARCHES",
    "found": "FOUNDED",
    "lead": "LEADS",
    "manage": "LEADS",
    "collaborate": "COLLABORATES_WITH",
    "partner": "COLLABORATES_WITH",
}

# Relations that are inherently symmetric.
_SYMMETRIC_RELATIONS: frozenset[str] = frozenset({
    "COLLABORATES_WITH",
})


@runtime_checkable
class RelationExtractor(Protocol):
    """Protocol for relation extractors."""

    async def extract(
        self, text: str, entities: list[dict]
    ) -> list[dict]: ...


class SpacyRelationExtractor:
    """Extract relations via spaCy dependency parsing.

    For each sentence containing 2+ recognized entities, walks the dependency
    tree between entity pairs to find a connecting verb or preposition.  The
    verb lemma is mapped to a canonical relation label when possible.
    """

    def __init__(self) -> None:
        self._nlp: Any = None

    def _get_nlp(self) -> Any:
        if self._nlp is None:
            from hermes.services import get_spacy_model

            self._nlp = get_spacy_model()
        return self._nlp

    async def extract(
        self, text: str, entities: list[dict]
    ) -> list[dict]:
        """Extract relations between *entities* found in *text*.

        Args:
            text: The original document text.
            entities: List of entity dicts (must have ``name`` and
                optionally ``start``/``end`` character offsets from NER).

        Returns:
            List of proposed edge dicts with keys ``source_name``,
            ``target_name``, ``relation``, ``confidence``, ``bidirectional``,
            and ``properties``.
        """
        if len(entities) < 2:
            return []

        nlp = self._get_nlp()
        doc = nlp(text)

        # Build a map of entity name -> character spans for matching.
        entity_names = {e["name"] for e in entities}

        relations: list[dict] = []

        for sent in doc.sents:
            # Find entity spans that fall within this sentence.
            sent_entities: list[Any] = []
            for ent in sent.ents:
                if ent.text in entity_names:
                    sent_entities.append(ent)

            if len(sent_entities) < 2:
                continue

            # For each ordered pair, extract the connecting relation.
            for i, src_ent in enumerate(sent_entities):
                for tgt_ent in sent_entities[i + 1 :]:
                    rel = self._extract_relation(doc, src_ent, tgt_ent)
                    if rel:
                        relations.append(rel)

        # Deduplicate by (source, target, relation).
        seen: set[tuple[str, str, str]] = set()
        deduped: list[dict] = []
        for r in relations:
            key = (r["source_name"], r["target_name"], r["relation"])
            if key not in seen:
                seen.add(key)
                deduped.append(r)
        return deduped

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_relation(
        self, doc: Any, src_ent: Any, tgt_ent: Any
    ) -> dict | None:
        """Walk the dep tree between two entity spans and find a relation."""
        # Collect tokens between the two entities.
        if src_ent.start < tgt_ent.start:
            start, end = src_ent.end, tgt_ent.start
        else:
            start, end = tgt_ent.end, src_ent.start

        # Gather verbs and preps between the entities.
        verbs: list[Any] = []
        preps: list[str] = []
        for tok in doc[max(0, start):min(len(doc), end)]:
            if tok.pos_ == "VERB":
                verbs.append(tok)
            elif tok.pos_ == "ADP":
                preps.append(tok.lemma_)

        # Also check the syntactic head of the target for a governing verb.
        if not verbs:
            head = tgt_ent.root.head
            if head.pos_ == "VERB":
                verbs.append(head)

        if not verbs and not preps:
            # Fallback: try the root of the source entity's head chain.
            head = src_ent.root.head
            if head.pos_ == "VERB":
                verbs.append(head)

        if not verbs and not preps:
            return None

        # Build the raw phrase and canonical relation.
        if verbs:
            lemma = verbs[0].lemma_.lower()
            raw_phrase = " ".join(
                tok.text for tok in doc[src_ent.start : tgt_ent.end]
            )
            relation = _VERB_TO_RELATION.get(lemma, lemma.upper())
        else:
            raw_phrase = " ".join(
                tok.text for tok in doc[src_ent.start : tgt_ent.end]
            )
            relation = preps[0].upper() if preps else "RELATED_TO"

        bidirectional = relation in _SYMMETRIC_RELATIONS
        # Simple heuristic confidence: verbs are more reliable than preps.
        confidence = 0.8 if verbs else 0.5

        return {
            "source_name": src_ent.text,
            "target_name": tgt_ent.text,
            "relation": relation,
            "confidence": confidence,
            "bidirectional": bidirectional,
            "properties": {
                "raw_phrase": raw_phrase,
            },
        }


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------
_extractor: RelationExtractor | None = None


def get_relation_extractor() -> RelationExtractor:
    """Return the configured relation extractor (lazy singleton)."""
    global _extractor
    if _extractor is not None:
        return _extractor

    backend = os.environ.get("RELATION_EXTRACTOR", "spacy")

    if backend == "spacy":
        _extractor = SpacyRelationExtractor()
    else:
        raise ValueError(f"Unknown RELATION_EXTRACTOR: {backend!r}")

    return _extractor

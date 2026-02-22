"""Pluggable relation extraction for Hermes.

Defines a protocol for relation extractors and implementations using
spaCy dependency parsing (local) or OpenAI (API). The active extractor
is selected via env vars:

    RELATION_EXTRACTOR  (default: auto-detect — "openai" if API key
                         present, else "spacy")
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Protocol, runtime_checkable

from logos_config import get_env_value

logger = logging.getLogger(__name__)

# Map common verb/prep patterns to canonical relation labels.
# Rationale: spaCy dependency parsing yields verb lemmas between entity pairs.
# This lookup normalises high-frequency verbs into a small set of canonical
# relation labels that align with the LOGOS ontology graph schema, keeping the
# knowledge graph consistent regardless of surface-form variation (e.g.
# "employ" and "work" both map to WORKS_AT).
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
_SYMMETRIC_RELATIONS: frozenset[str] = frozenset(
    {
        "COLLABORATES_WITH",
    }
)


@runtime_checkable
class RelationExtractor(Protocol):
    """Protocol for relation extractors."""

    async def extract(self, text: str, entities: list[dict]) -> list[dict]: ...


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

    async def extract(self, text: str, entities: list[dict]) -> list[dict]:
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

        # Build a normalised set of entity names for case-insensitive matching.
        # NER output and spaCy entities may differ in casing (e.g. "OpenAI" vs
        # "openai"), so we compare lowered forms to avoid missed relations.
        entity_names_lower = {e["name"].lower() for e in entities}

        relations: list[dict] = []

        for sent in doc.sents:
            # Find entity spans that fall within this sentence.
            sent_entities: list[Any] = []
            for ent in sent.ents:
                if ent.text.lower() in entity_names_lower:
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

    def _extract_relation(self, doc: Any, src_ent: Any, tgt_ent: Any) -> dict | None:
        """Walk the dep tree between two entity spans and find a relation."""
        # Collect tokens between the two entities.
        if src_ent.start < tgt_ent.start:
            start, end = src_ent.end, tgt_ent.start
        else:
            start, end = tgt_ent.end, src_ent.start

        # Gather verbs and preps between the entities.
        verbs: list[Any] = []
        preps: list[str] = []
        for tok in doc[max(0, start) : min(len(doc), end)]:
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
            raw_phrase = " ".join(tok.text for tok in doc[src_ent.start : tgt_ent.end])
            relation = _VERB_TO_RELATION.get(lemma, lemma.upper())
        else:
            raw_phrase = " ".join(tok.text for tok in doc[src_ent.start : tgt_ent.end])
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


class OpenAIRelationExtractor:
    """Extract relations using OpenAI chat completions.

    Sends the text and pre-extracted entities to the LLM and asks it to
    identify semantic relations between them.  Reuses ``hermes.llm`` for
    auth and config.
    """

    name: str = "openai"

    _SYSTEM_PROMPT = (
        "You are a relation extraction system for the LOGOS knowledge graph. "
        "Given input text and a list of entities already extracted from that "
        "text, identify meaningful semantic relations between pairs of entities.\n\n"
        "For each relation, provide:\n"
        '  - "source_name": the entity where the relation originates\n'
        '  - "target_name": the entity the relation points to\n'
        '  - "relation": a short UPPER_SNAKE_CASE label (e.g. LOCATED_IN, '
        "PART_OF, FOUNDED_BY, CAPITAL_OF, MEMBER_OF, WORKS_AT, CREATED_BY, "
        "GOVERNED_BY, PLAYS, ORIGINATED_IN)\n"
        '  - "confidence": float 0-1 indicating how confident the relation is\n'
        '  - "bidirectional": true if the relation is inherently symmetric\n\n'
        "Rules:\n"
        "- Only extract relations that are clearly supported by the text\n"
        "- Use specific, meaningful relation labels — not generic ones like "
        "RELATED_TO\n"
        "- source_name and target_name must exactly match entity names from "
        "the provided list\n"
        "- If no clear relations exist, return an empty array\n\n"
        'Return a JSON object with a single key "relations" containing an '
        "array of relation objects.\n"
        "Return ONLY valid JSON, no other text."
    )

    async def extract(self, text: str, entities: list[dict]) -> list[dict]:
        if len(entities) < 2:
            return []

        from hermes.llm import generate_completion

        entity_list = ", ".join(
            f"{e['name']} ({e.get('type', 'entity')})" for e in entities
        )
        user_msg = (
            f"Text: {text}\n\n"
            f"Entities: {entity_list}\n\n"
            "Extract all meaningful relations between these entities."
        )

        try:
            result = await generate_completion(
                messages=[
                    {"role": "system", "content": self._SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=2048,
                metadata={"scenario": "relation_extraction"},
            )
        except Exception as e:
            logger.warning("OpenAI relation extraction failed: %s", e, exc_info=True)
            return []

        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        entity_names = {e["name"] for e in entities}
        return self._parse_response(content, entity_names)

    @staticmethod
    def _parse_response(content: str, entity_names: set[str]) -> list[dict]:
        """Parse the LLM JSON response into edge dicts."""
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"```(?:json)?\s*(.*?)```", content, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(1))
                except json.JSONDecodeError:
                    logger.warning("Failed to parse OpenAI relation response")
                    return []
            else:
                logger.warning("Failed to parse OpenAI relation response")
                return []

        raw_relations = data.get("relations", [])
        relations: list[dict] = []
        seen: set[tuple[str, str, str]] = set()

        for rel in raw_relations:
            src = rel.get("source_name", "")
            tgt = rel.get("target_name", "")
            relation = rel.get("relation", "RELATED_TO")

            if not src or not tgt:
                continue
            if src not in entity_names or tgt not in entity_names:
                continue

            # Normalize relation label
            relation = relation.upper().replace(" ", "_")

            key = (src, tgt, relation)
            if key in seen:
                continue
            seen.add(key)

            confidence = rel.get("confidence", 0.7)
            if not isinstance(confidence, (int, float)):
                confidence = 0.7
            confidence = max(0.0, min(1.0, float(confidence)))

            relations.append(
                {
                    "source_name": src,
                    "target_name": tgt,
                    "relation": relation,
                    "confidence": confidence,
                    "bidirectional": bool(rel.get("bidirectional", False)),
                    "properties": {},
                }
            )

        return relations


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------
_extractor: RelationExtractor | None = None


def _detect_backend() -> str:
    """Auto-detect the best available relation extraction backend."""
    explicit = get_env_value("RELATION_EXTRACTOR")
    if explicit:
        return explicit.strip().lower()
    has_key = get_env_value("HERMES_LLM_API_KEY") or get_env_value("OPENAI_API_KEY")
    if has_key:
        return "openai"
    return "spacy"


def get_relation_extractor() -> RelationExtractor:
    """Return the configured relation extractor (lazy singleton)."""
    global _extractor
    if _extractor is not None:
        return _extractor

    backend = _detect_backend()

    if backend == "openai":
        _extractor = OpenAIRelationExtractor()
        logger.info("Relation extractor: openai")
    elif backend == "spacy":
        _extractor = SpacyRelationExtractor()
        logger.info("Relation extractor: spacy")
    else:
        raise ValueError(f"Unknown RELATION_EXTRACTOR: {backend!r}")

    return _extractor

"""Combined NER + relation extraction in a single LLM call.

Reduces the proposal-building pipeline from 3 sequential round trips to 2
by merging entity extraction and relation extraction into one prompt.
The active provider is selected via env vars:

    NER_PROVIDER=combined  /  RELATION_EXTRACTOR=combined
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any


from hermes.ner_provider import ONTOLOGY_TYPES, OpenAINERProvider
from hermes.ontology_client import fetch_type_list, get_sophia_url
from hermes.relation_extractor import OpenAIRelationExtractor

logger = logging.getLogger(__name__)

# Singleton instance shared between NER and RE factory functions.
_combined_instance: OpenAICombinedExtractor | None = None


class OpenAICombinedExtractor:
    """Single-call extractor for both entities and relations.

    Satisfies both :class:`~hermes.ner_provider.NERProvider` and
    :class:`~hermes.relation_extractor.RelationExtractor` protocols so it
    can be used as a drop-in replacement for either.
    """

    name: str = "combined"

    def _build_system_prompt(
        self,
        type_list: list[dict] | None = None,
    ) -> str:
        """Build the system prompt, optionally using dynamic type lists."""
        if type_list is not None:
            entity_types_section = "\n".join(
                f"- {t['name']}: {t.get('description', '')}" for t in type_list
            )
            entity_types_section += "\nIf none of these types fit, use 'object'."
        else:
            entity_types_section = "\n".join(
                f"- {t}: {desc}" for t, desc in ONTOLOGY_TYPES.items()
            )

        prompt = (
            "You are a combined named-entity recognition and relation extraction "
            "system for the LOGOS robotics ontology.\n\n"
            "Given input text, extract ALL named entities and ALL meaningful "
            "semantic relations between them in a SINGLE pass.\n\n"
            "## Entity Types\n"
            f"{entity_types_section}\n\n"
        )

        prompt += (
            "## Output Format\n"
            "Return a JSON object with two keys:\n\n"
            "```json\n"
            "{\n"
            '  "entities": [\n'
            '    {"name": "...", "type": "...", "start": 0, "end": 5, "value": null, "unit": null}\n'
            "  ],\n"
            '  "relations": [\n'
            "    {\n"
            '      "source_name": "...",\n'
            '      "target_name": "...",\n'
            '      "relation": "UPPER_SNAKE_CASE",\n'
            '      "confidence": 0.9,\n'
            '      "bidirectional": false\n'
            "    }\n"
            "  ]\n"
            "}\n"
            "```\n\n"
            "## Rules\n"
            "- Extract each entity individually; never combine multiple entities "
            "with conjunctions like 'and', 'or', or commas into one name\n"
            "- Entity names must match the text exactly\n"
            "- start/end are character offsets into the input text\n"
            "- Entity type must be one of the types listed above\n"
            "- Prefer specific, contentful noun phrases; do NOT extract bare "
            "adjectives, pronouns, sentence fragments, or generic standalone nouns "
            "(e.g. 'structure', 'system', 'thing', 'result'); use the canonical "
            "noun-phrase surface form\n"
            "- Temporal entities (type 'state': dates/times): set \"value\" to a "
            "normalized form, ISO-8601 where possible ('1943'->'1943', "
            "'March 1898'->'1898-03', 'the 1950s'->'195X'); \"unit\" is null\n"
            "- Quantitative entities (type 'data': measurements/quantities): set "
            '"value" to the numeric magnitude and "unit" to the unit '
            "('10 km'->10/'km'; '5 mg'->5/'mg'); plain counts: \"unit\" is null\n"
            '- All other entities: "value" and "unit" are null\n'
            "- Relation source_name and target_name must exactly match an "
            "extracted entity name\n"
            "- Use specific UPPER_SNAKE_CASE relation labels (e.g. LOCATED_IN, "
            "PART_OF, WORKS_AT, FOUNDED_BY, CAPITAL_OF)\n"
            "- Only extract relations clearly supported by the text\n"
            "- confidence is a float 0\u20131\n"
            "- If no entities or relations are found, return empty arrays\n\n"
        )

        prompt += self._known_relations_section()
        prompt += "Return ONLY valid JSON, no other text."

        return prompt

    @staticmethod
    def _known_relations_section() -> str:
        """Closed-vocabulary clause for the RE step (H5, hermes#140).

        Injects the current match-before-mint predicate vocabulary so the
        model reuses an existing relation instead of minting a near-duplicate
        -- the source-side lever for relation over-generation (the df=1
        problem), validated by the NER/RE bake-off (logos-experiments#38).

        Fail-soft: an empty or unavailable vocabulary yields no clause, so the
        prompt degrades to the open form rather than raising. Bounded by
        ``RE_VOCAB_CAP`` (default 150) because the live vocabulary can hold
        thousands of surfaces; the cap keeps prompt tokens in check and, as
        consolidation shrinks the vocabulary below it, the full clean set is
        injected automatically.
        """
        try:
            from hermes.predicate_resolver import get_predicate_vocabulary

            known = get_predicate_vocabulary().known()
        except Exception:
            logger.warning("H5: predicate vocabulary unavailable; using open RE prompt")
            return ""

        if not known:
            return ""

        cap = int(os.getenv("RE_VOCAB_CAP", "150"))
        vocab = sorted(known)[:cap]
        return (
            "## Known Relations\n"
            "Prefer an existing predicate from this list when one fits the "
            "meaning; only mint a NEW UPPER_SNAKE_CASE label when none of "
            "these applies:\n"
            f"{', '.join(vocab)}\n\n"
        )

    async def extract_entities_and_relations(
        self, text: str
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Extract entities and relations in a single LLM call.

        Returns:
            Tuple of (entities, relations) where each is a list of dicts.
        """
        from hermes.llm import generate_completion

        sophia_url = get_sophia_url()
        type_list = await fetch_type_list(sophia_url)
        system_prompt = self._build_system_prompt(type_list)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ]

        try:
            result = await generate_completion(
                messages=messages,
                temperature=0.0,
                max_tokens=4096,
                metadata={"scenario": "combined_ner_re"},
            )
        except Exception:
            logger.warning("Combined NER+RE extraction failed, falling back to empty")
            return [], []

        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")

        valid_types: set[str] | None = None
        if type_list is not None:
            valid_types = {t["name"] for t in type_list}

        return self._parse_combined_response(content, text, valid_types=valid_types)

    # -- Protocol compat: NERProvider ------------------------------------

    async def extract_entities(self, text: str) -> list[dict]:
        """NERProvider protocol -- extracts entities only."""
        entities, _ = await self.extract_entities_and_relations(text)
        return entities

    # -- Protocol compat: RelationExtractor ------------------------------

    async def extract(self, text: str, entities: list[dict]) -> list[dict]:
        """RelationExtractor protocol -- extracts relations only.

        Note: *entities* is accepted for protocol compat but is ignored;
        this method re-extracts entities internally so it can be used
        standalone.  The combined pipeline in ProposalBuilder calls
        ``extract_entities_and_relations`` directly to avoid the double call.
        """
        _, relations = await self.extract_entities_and_relations(text)
        return relations

    # -- Parsing ---------------------------------------------------------

    @staticmethod
    def _parse_combined_response(
        content: str,
        original_text: str,
        *,
        valid_types: set[str] | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Parse the combined JSON response into (entities, relations)."""
        data = _extract_json(content)
        if data is None:
            return [], []

        entities = OpenAINERProvider._parse_response(
            json.dumps({"entities": data.get("entities", [])}),
            original_text,
            valid_types=valid_types,
        )

        entity_names = {e["name"] for e in entities}
        relations = OpenAIRelationExtractor._parse_response(
            json.dumps({"relations": data.get("relations", [])}),
            entity_names,
        )

        return entities, relations


def _extract_json(content: str) -> dict[str, Any] | None:
    """Try to parse JSON from raw content or markdown fences."""
    try:
        return json.loads(content)  # type: ignore[no-any-return]
    except json.JSONDecodeError:
        pass

    match = re.search(r"```(?:json)?\s*(.*?)```", content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))  # type: ignore[no-any-return]
        except json.JSONDecodeError:
            pass

    logger.warning("Failed to parse combined NER+RE response")
    return None


def get_combined_instance() -> OpenAICombinedExtractor:
    """Return the shared combined extractor singleton."""
    global _combined_instance
    if _combined_instance is None:
        _combined_instance = OpenAICombinedExtractor()
    return _combined_instance

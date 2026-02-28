"""Combined NER + relation extraction in a single LLM call.

Reduces the proposal-building pipeline from 3 sequential round trips to 2
by merging entity extraction and relation extraction into one prompt.
The active provider is selected via env vars:

    NER_PROVIDER=combined  /  RELATION_EXTRACTOR=combined
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from hermes.ner_provider import ONTOLOGY_TYPES, OpenAINERProvider
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

    _SYSTEM_PROMPT = (
        "You are a combined named-entity recognition and relation extraction "
        "system for the LOGOS robotics ontology.\n\n"
        "Given input text, extract ALL named entities and ALL meaningful "
        "semantic relations between them in a SINGLE pass.\n\n"
        "## Entity Types\n"
        + "\n".join(f"- {t}: {desc}" for t, desc in ONTOLOGY_TYPES.items())
        + "\n\n"
        "## Output Format\n"
        "Return a JSON object with two keys:\n\n"
        "```json\n"
        "{\n"
        '  "entities": [\n'
        '    {"name": "...", "type": "...", "start": 0, "end": 5}\n'
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
        "- Entity names must match the text exactly\n"
        "- start/end are character offsets into the input text\n"
        "- Entity type must be one of the types listed above\n"
        "- Relation source_name and target_name must exactly match an "
        "extracted entity name\n"
        "- Use specific UPPER_SNAKE_CASE relation labels (e.g. LOCATED_IN, "
        "PART_OF, WORKS_AT, FOUNDED_BY, CAPITAL_OF)\n"
        "- Only extract relations clearly supported by the text\n"
        "- confidence is a float 0\u20131\n"
        "- If no entities or relations are found, return empty arrays\n\n"
        "Return ONLY valid JSON, no other text."
    )

    async def extract_entities_and_relations(
        self, text: str
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Extract entities and relations in a single LLM call.

        Returns:
            Tuple of (entities, relations) where each is a list of dicts.
        """
        from hermes.llm import generate_completion

        messages = [
            {"role": "system", "content": self._SYSTEM_PROMPT},
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

        return self._parse_combined_response(content, text)

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
        content: str, original_text: str
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Parse the combined JSON response into (entities, relations)."""
        data = _extract_json(content)
        if data is None:
            return [], []

        entities = OpenAINERProvider._parse_response(
            json.dumps({"entities": data.get("entities", [])}),
            original_text,
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

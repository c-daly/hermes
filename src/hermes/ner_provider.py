"""Pluggable NER providers for Hermes.

Defines a protocol for named-entity recognition and implementations using
spaCy (local) or OpenAI (API). The active provider is selected via env vars:

    NER_PROVIDER  (default: auto-detect — "openai" if API key present,
                   else "spacy")
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Protocol, runtime_checkable

from logos_config import get_env_value

logger = logging.getLogger(__name__)

# LOGOS ontology types that Hermes can propose, with descriptions for the
# LLM prompt so it understands what each means.
ONTOLOGY_TYPES: dict[str, str] = {
    "entity": "a general entity that doesn't fit a more specific type",
    "location": "a geographic or spatial place (city, room, region)",
    "object": "a physical object or artifact",
    "agent": "a robotic physical entity that can act in the world",
    "process": "an ongoing process or event",
    "action": "a discrete action or activity",
    "concept": "an abstract concept, idea, or category",
    "state": "a state, condition, or temporal reference (dates, times)",
    "data": "a data value, measurement, or quantity",
    "workspace": "a defined work area or environment",
    "zone": "a bounded spatial region within a workspace",
    "goal": "a desired outcome or objective",
    "plan": "a structured plan or strategy",
    "capability": "an ability or skill",
}

# Map spaCy NER labels to ontology types (fallback classification).
SPACY_TO_ONTOLOGY: dict[str, str] = {
    "GPE": "location",
    "LOC": "location",
    "FAC": "object",
    "ORG": "entity",
    "PERSON": "entity",
    "NORP": "entity",
    "PRODUCT": "object",
    "EVENT": "process",
    "WORK_OF_ART": "entity",
    "LAW": "concept",
    "LANGUAGE": "concept",
    "DATE": "state",
    "TIME": "state",
    "QUANTITY": "data",
    "CARDINAL": "data",
    "ORDINAL": "data",
    "MONEY": "data",
    "PERCENT": "data",
}


@runtime_checkable
class NERProvider(Protocol):
    """Protocol for named-entity recognition providers."""

    async def extract_entities(self, text: str) -> list[dict]: ...


class SpacyNERProvider:
    """Local provider using spaCy NER pipeline."""

    def __init__(self) -> None:
        self._nlp: Any = None

    def _get_nlp(self) -> Any:
        if self._nlp is None:
            from hermes.services import get_spacy_model

            self._nlp = get_spacy_model()
        return self._nlp

    async def extract_entities(self, text: str) -> list[dict]:
        nlp = self._get_nlp()
        doc = nlp(text)
        return [
            {
                "name": ent.text,
                "type": SPACY_TO_ONTOLOGY.get(ent.label_, "entity"),
                "start": ent.start_char,
                "end": ent.end_char,
            }
            for ent in doc.ents
        ]


class OpenAINERProvider:
    """NER provider using OpenAI chat completions.

    Reuses hermes.llm.generate_completion for auth, base URL, and model
    config — no raw httpx calls needed.
    """

    _SYSTEM_PROMPT = (
        "You are a named-entity recognition system for the LOGOS robotics "
        "ontology. Given input text, extract all named entities and classify "
        "each with exactly one of the following types:\n\n"
        + "\n".join(f"- {t}: {desc}" for t, desc in ONTOLOGY_TYPES.items())
        + "\n\n"
        'Return a JSON object with a single key "entities" containing an '
        "array of objects. Each object must have:\n"
        '  - "name": the entity text as it appears in the input\n'
        '  - "type": one of the types listed above\n'
        '  - "start": character offset where the entity starts in the input\n'
        '  - "end": character offset where the entity ends in the input\n\n'
        'If no entities are found, return {"entities": []}.\n'
        "Return ONLY valid JSON, no other text."
    )

    async def extract_entities(self, text: str) -> list[dict]:
        from hermes.llm import generate_completion

        messages = [
            {"role": "system", "content": self._SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]

        try:
            result = await generate_completion(
                messages=messages,
                temperature=0.0,
                max_tokens=2048,
                metadata={"scenario": "ner_extraction"},
            )
        except Exception:
            logger.warning("OpenAI NER extraction failed, falling back to empty")
            return []

        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")

        return self._parse_response(content, text)

    @staticmethod
    def _parse_response(content: str, original_text: str) -> list[dict]:
        """Parse the LLM JSON response into entity dicts."""
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code fences
            match = re.search(r"```(?:json)?\s*(.*?)```", content, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(1))
                except json.JSONDecodeError:
                    logger.warning("Failed to parse OpenAI NER response")
                    return []
            else:
                logger.warning("Failed to parse OpenAI NER response")
                return []

        raw_entities = data.get("entities", [])
        entities: list[dict] = []
        for ent in raw_entities:
            name = ent.get("name", "")
            ent_type = ent.get("type", "entity")
            if not name:
                continue
            # Validate type is in our vocabulary
            if ent_type not in ONTOLOGY_TYPES:
                ent_type = "entity"
            # Use provided offsets, or find them in the text
            start = ent.get("start")
            end = ent.get("end")
            if start is None or end is None:
                idx = original_text.find(name)
                if idx >= 0:
                    start = idx
                    end = idx + len(name)
                else:
                    start = 0
                    end = len(name)
            entities.append(
                {
                    "name": name,
                    "type": ent_type,
                    "start": start,
                    "end": end,
                }
            )
        return entities


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------
_ner_provider: NERProvider | None = None


def _detect_backend() -> str:
    """Auto-detect the best available NER backend."""
    explicit = get_env_value("NER_PROVIDER")
    if explicit:
        return explicit.strip().lower()
    # Prefer OpenAI if an API key is available.
    has_key = get_env_value("HERMES_LLM_API_KEY") or get_env_value("OPENAI_API_KEY")
    if has_key:
        return "openai"
    return "spacy"


def get_ner_provider() -> NERProvider:
    """Return the configured NER provider (lazy singleton)."""
    global _ner_provider
    if _ner_provider is not None:
        return _ner_provider

    backend = _detect_backend()

    if backend == "openai":
        _ner_provider = OpenAINERProvider()
        logger.info("NER provider: openai")
    elif backend == "spacy":
        _ner_provider = SpacyNERProvider()
        logger.info("NER provider: spacy")
    else:
        raise ValueError(f"Unknown NER_PROVIDER: {backend!r}")

    return _ner_provider

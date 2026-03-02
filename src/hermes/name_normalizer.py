"""Post-extraction entity name normalization.

Cleans entity names via lowercasing, singularization, and deduplication
before they leave Hermes. No extra LLM call — pure string processing.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Words ending in these suffixes should not have trailing 's' stripped.
# Covers common English words that end in 's' but are not plurals.
_NO_STRIP_SUFFIXES = frozenset({
    "ss",      # grass, lass, class
    "us",      # bus, campus, virus
    "is",      # analysis, basis, diabetes
    "sis",     # thesis, crisis
    "ous",     # dangerous, famous
    "ics",     # physics, mathematics
    "ies",     # handled separately by _ies rule
})


_VOWELS = frozenset("aeiou")


def _singularize(word: str) -> str:
    """Lightweight English singularization using suffix rules.

    Handles common plural patterns without a full NLP library.
    Expects a lowercased input. Returns the word unchanged if it
    doesn't match a known plural pattern.
    """
    if len(word) <= 2:
        return word

    # -ies → -ie (rotties → rottie, puppies → puppie)
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "ie"

    # -sses → -ss (grasses → grass)
    if word.endswith("sses"):
        return word[:-2]

    # -ches → -ch (watches → watch)
    if word.endswith("ches"):
        return word[:-2]

    # -shes → -sh (bushes → bush)
    if word.endswith("shes"):
        return word[:-2]

    # -xes → -x (boxes → box)
    if word.endswith("xes"):
        return word[:-2]

    # -zes → -z (quizzes → quiz)
    if word.endswith("zes") and len(word) > 4:
        return word[:-2]

    # Check exclusion suffixes before stripping bare -s
    for suffix in _NO_STRIP_SUFFIXES:
        if word.endswith(suffix):
            return word

    # -s after a consonant → strip (dogs → dog, checkups → checkup)
    # Don't strip -s after vowels to avoid mangling non-plurals (diabetes)
    if word.endswith("s") and len(word) > 2 and word[-2] not in _VOWELS:
        return word[:-1]

    return word


def normalize_entities(
    entities: list[dict], text: str
) -> list[dict]:
    """Normalize entity names: lowercase, singularize, deduplicate.

    Args:
        entities: List of entity dicts with name, type, start, end fields.
        text: Original text (unused currently, reserved for future context).

    Returns:
        Normalized and deduplicated entity list.
    """
    if not entities:
        return []

    # Phase 1: normalize names
    normalized: list[dict] = []
    for ent in entities:
        name = ent.get("name", "")
        if not name:
            continue

        norm_name = _singularize(name.lower())

        normalized.append({
            "name": norm_name,
            "type": ent.get("type", "entity"),
            "start": ent.get("start", 0),
            "end": ent.get("end", 0),
        })

    # Phase 2: deduplicate — keep the entity with the longer span
    seen: dict[str, int] = {}  # normalized name -> index in result
    result: list[dict] = []

    for ent in normalized:
        name = ent["name"]
        span_len = ent["end"] - ent["start"]

        if name in seen:
            existing_idx = seen[name]
            existing_span = result[existing_idx]["end"] - result[existing_idx]["start"]
            if span_len > existing_span:
                result[existing_idx] = ent
        else:
            seen[name] = len(result)
            result.append(ent)

    return result

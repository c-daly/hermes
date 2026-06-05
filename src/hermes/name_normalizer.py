"""Post-extraction entity name normalization.

Cleans entity names via lowercasing, singularization, and deduplication
before they leave Hermes.  Word-level singularization is delegated to the
shared inflect-based core in ``hermes.canonical`` -- one engine for entity,
edge, and type names.
"""

from __future__ import annotations

import logging

from hermes.canonical import singularize_word

logger = logging.getLogger(__name__)


def _lemmatize_word(word: str) -> str:
    """Singularize a single word via the shared core (hermes.canonical).

    One engine for entity, edge, AND type names (2026-06-05) -- no WordNet
    fork. Words of length <= 2 are kept as-is.
    """
    if len(word) <= 2:
        return word
    return singularize_word(word)


def _strip_possessive(word: str) -> str:
    """Strip English possessive suffixes ('s / ’s)."""
    if word.endswith("’s") or word.endswith("'s"):
        return word[:-2]
    # Plural possessive: dogs' -> dogs (then lemmatized to dog)
    if word.endswith("’") or word.endswith("'"):
        return word[:-1]
    return word


def _lemmatize_name(name: str, original_name: str | None = None) -> str:
    """Lemmatize each word in a (possibly multi-word) entity name.

    Words that were capitalized in the original text are assumed to be
    proper nouns and are lowercased but not lemmatized (e.g. "United
    States" stays "united states", not "united state").
    """
    words = name.split()
    if not words:
        return name
    orig_words = original_name.split() if original_name else []
    result: list[str] = []
    for i, w in enumerate(words):
        w = _strip_possessive(w)
        # If the original word was capitalized and this is a multi-word
        # entity, treat it as a proper noun — lowercase only, no lemma.
        if len(words) > 1 and i < len(orig_words) and orig_words[i][0:1].isupper():
            result.append(w)
        else:
            result.append(_lemmatize_word(w))
    return " ".join(result)


def normalize_entities(entities: list[dict], text: str) -> list[dict]:
    """Normalize entity names: lowercase, lemmatize, deduplicate.

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

        norm_name = _lemmatize_name(name.lower(), original_name=name)

        normalized.append(
            {
                "name": norm_name,
                "type": ent.get("type", "entity"),
                "start": ent.get("start", 0),
                "end": ent.get("end", 0),
            }
        )

    # Phase 2: deduplicate -- keep the entity with the longer span
    seen: dict[str, int] = {}
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

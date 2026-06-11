"""Post-extraction entity name normalization.

Cleans entity names via lowercasing, singularization, leading-determiner
stripping, junk rejection (bare pronouns, preposition-led phrases), and
deduplication before they leave Hermes.  Word-level singularization is
delegated to the shared inflect-based core in ``hermes.canonical`` -- one
engine for entity, edge, and type names.
"""

from __future__ import annotations

import logging

from hermes.canonical import singularize_word

logger = logging.getLogger(__name__)

# Leading tokens stripped from names: determiners and possessive pronouns
# carry no referent of their own ("the morning light" -> "morning light").
# A name that strips to nothing is dropped.
_DETERMINERS = {
    "a",
    "an",
    "the",
    "this",
    "that",
    "these",
    "those",
    "my",
    "your",
    "his",
    "her",
    "its",
    "our",
    "their",
    "some",
    "any",
    "each",
    "every",
    "no",
}

# A name that is nothing but a pronoun has no cross-document referent.
_PRONOUNS = {
    "i",
    "me",
    "my",
    "mine",
    "myself",
    "we",
    "us",
    "our",
    "ours",
    "ourselves",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "he",
    "him",
    "his",
    "himself",
    "she",
    "her",
    "hers",
    "herself",
    "it",
    "its",
    "itself",
    "they",
    "them",
    "their",
    "theirs",
    "themselves",
    "this",
    "that",
    "these",
    "those",
    "who",
    "whom",
    "whose",
    "which",
    "what",
    "something",
    "anything",
    "nothing",
    "everything",
    "someone",
    "anyone",
    "everyone",
    "somebody",
    "anybody",
    "nobody",
    "everybody",
    "one",
    "none",
}

# A multi-word name led by a preposition is an adverbial or measure phrase,
# not an entity ("over two hundred miles per hour"). "up"/"down"/"out"/"off"
# are deliberately absent: they lead legitimate compounds ("down payment").
_PREPOSITIONS = {
    "about",
    "above",
    "across",
    "after",
    "against",
    "along",
    "amid",
    "among",
    "around",
    "at",
    "before",
    "behind",
    "below",
    "beneath",
    "beside",
    "besides",
    "between",
    "beyond",
    "by",
    "despite",
    "during",
    "except",
    "for",
    "from",
    "in",
    "inside",
    "into",
    "near",
    "of",
    "on",
    "onto",
    "outside",
    "over",
    "past",
    "per",
    "since",
    "through",
    "throughout",
    "till",
    "to",
    "toward",
    "towards",
    "under",
    "underneath",
    "until",
    "unto",
    "upon",
    "via",
    "with",
    "within",
    "without",
}


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


def clean_entity_name(name: str, original_name: str | None = None) -> str:
    """Full name-cleaning pipeline: strip leading determiners, then lemmatize.

    Expects ``name`` already lowercased (``original_name`` carries the raw
    casing for the proper-noun heuristic in ``_lemmatize_name``). Returns ""
    when nothing survives the strip (caller drops the entity).
    """
    words = name.split()
    orig_words = original_name.split() if original_name else []
    while words and words[0] in _DETERMINERS:
        words.pop(0)
        if orig_words:
            orig_words.pop(0)
    if not words:
        return ""
    stripped_orig = " ".join(orig_words) if orig_words else None
    return _lemmatize_name(" ".join(words), original_name=stripped_orig)


def is_junk_entity_name(name: str) -> bool:
    """True for cleaned names with no usable referent.

    Bare pronouns ("it", "they") refer only within their source document;
    preposition-led multi-word names ("over two hundred mile per hour") are
    adverbial/measure phrases, not entities.
    """
    words = name.split()
    if not words:
        return True
    if len(words) == 1:
        return words[0] in _PRONOUNS
    return words[0] in _PREPOSITIONS


def normalize_entities(entities: list[dict], text: str) -> list[dict]:
    """Normalize entity names: lowercase, strip determiners, lemmatize,
    reject junk, deduplicate.

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

        norm_name = clean_entity_name(name.lower(), original_name=name)
        if not norm_name or is_junk_entity_name(norm_name):
            logger.debug("Dropping junk entity name: %r", name)
            continue

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

"""Post-extraction entity name normalization.

Cleans entity names via lowercasing, lemmatization, and deduplication
before they leave Hermes.  Uses NLTK WordNet for dictionary words and
falls back to suffix rules for informal/slang terms not in WordNet.
"""

from __future__ import annotations

import logging

import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

logger = logging.getLogger(__name__)

# Ensure WordNet corpus is available (downloads once, ~3 MB).
# In production, pre-bake the corpus into the container image.
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet", quiet=True)

_wnl = WordNetLemmatizer()

# ---------------------------------------------------------------------------
# Suffix-rule fallback for words not in WordNet
# ---------------------------------------------------------------------------
_NO_STRIP_SUFFIXES = frozenset(
    {
        "ss",  # grass, lass, class
        "us",  # bus, campus, virus
        "is",  # analysis, basis, diabetes
        "sis",  # thesis, crisis
        "ous",  # dangerous, famous
        "ics",  # physics, mathematics
        "ies",  # handled separately by _ies rule
    }
)

_VOWELS = frozenset("aeiou")


def _singularize_fallback(word: str) -> str:
    """Suffix-rule singularization for words not in WordNet.

    Handles informal/slang plurals that the WordNet lemmatizer
    returns unchanged (e.g. "rotties" -> "rottie").
    Expects a lowercased input.
    """
    if len(word) <= 2:
        return word

    # -ies -> -ie (rotties -> rottie)
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "ie"

    # -sses -> -ss (grasses -> grass)
    if word.endswith("sses"):
        return word[:-2]

    # -ches -> -ch, -shes -> -sh
    if word.endswith("ches") or word.endswith("shes"):
        return word[:-2]

    # -xes -> -x (boxes -> box)
    if word.endswith("xes"):
        return word[:-2]

    # -zes -> -z (fizzes -> fizz)
    if word.endswith("zes") and len(word) > 4:
        return word[:-2]

    # Check exclusion suffixes before stripping bare -s
    for suffix in _NO_STRIP_SUFFIXES:
        if word.endswith(suffix):
            return word

    # -s after a consonant -> strip (dogs -> dog)
    if word.endswith("s") and len(word) > 2 and word[-2] not in _VOWELS:
        return word[:-1]

    return word


def _is_wordnet_lemma(word: str) -> bool:
    """Check if *word* is itself a lemma form in WordNet (not just morphologically related)."""
    try:
        for ss in wordnet.synsets(word, pos="n"):
            if word in [lemma.name() for lemma in ss.lemmas()]:
                return True
    except LookupError:
        pass
    return False


def _lemmatize_word(word: str) -> str:
    """Lemmatize a single word as a noun, with suffix-rule fallback.

    Guards against two classes of error:
    1. Invariant nouns that WordNet incorrectly treats as plurals
       (e.g. "species" -> "specie").  Detected by checking whether
       the *original* word is itself a WordNet lemma.
    2. Informal/slang words absent from WordNet (e.g. "rotties").
       Handled by suffix-rule fallback.
    """
    if len(word) <= 2:
        return word
    try:
        lemma = _wnl.lemmatize(word, pos="n")
    except LookupError:
        return _singularize_fallback(word)
    if lemma == word:
        # WordNet didn't change it -- if it recognises the word as a
        # noun it's already correct; only apply suffix rules for words
        # WordNet doesn't know at all (e.g. "rotties").
        if _is_wordnet_lemma(word):
            return word
        return _singularize_fallback(word)
    # Lemmatizer changed the word.  But the original may itself be a
    # valid lemma that WordNet incorrectly maps elsewhere
    # (e.g. "species" -> "specie", "corps" -> "corp").
    if _is_wordnet_lemma(word):
        return word
    return str(lemma)


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

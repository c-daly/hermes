"""Canonical name normalization -- the single implementation, Hermes-internal.

Normalization is Hermes' job (2026-06-05): ONE canonicalize() lives here and
nowhere else. Sophia never normalizes -- the boundary contract is that every
name entering the graph is already canonical, so the cognitive core compares
names by exact match. The offline naming-driven-typing harness imports this
module in-process (catalog-injection path A). Out-of-process callers that
genuinely need normalization (migrations, apollo, audits) get a batch
POST /normalize endpoint (deferred -- documented in the experiment plan),
never a shared SDK symbol.

Two exports:
  singularize_word(word) -- the word-level core (inflect + keep-list +
    suffix guard + defnoun irregulars). name_normalizer.py uses this for
    entity/edge names, so types are NOT normalized differently than edges.
  canonicalize(name) -- the type/cluster-name flavor: NFKC -> strip ->
    collapse whitespace -> lowercase -> drop leading a/an/the -> singularize
    HEAD NOUN ONLY -> strip curated trailing fillers, only when not the
    whole string. Idempotent.
"""

import re
import unicodedata

import inflect

# Single shared inflect engine (thread-safe for read-only singular_noun calls).
_INFLECT = inflect.engine()

# The -ie/-ies class inflect guesses wrong (-ies -> -y for unknown nouns).
# Pinned irregulars: grow THIS list (and the golden table) -- never fork rules.
_IE_IRREGULARS = ("selfie", "smoothie", "hoodie", "foodie", "rottie", "hippie")
for _ie in _IE_IRREGULARS:
    _INFLECT.defnoun(_ie, _ie + "s")

# Latin -ix/-ices: inflect mis-singularizes (matrices -> matrice); pinned.
_INFLECT.defnoun("matrix", "matrices")

# Leading articles dropped before head-noun singularization.
_LEADING_ARTICLES = {"a", "an", "the"}

# Curated trailing fillers stripped only when they are NOT the whole name.
_TRAILING_FILLERS = {"type", "category", "kind", "class", "group", "entity"}

# Words inflect mis-singularizes (over-strips); kept verbatim. Includes the
# realm roots, structural keywords, and the SPEC golden cases.
_SINGULAR_KEEP = {
    "entity",
    "concept",
    "process",
    "node",
    "species",
    "class",
    "bus",
    "analysis",
}

# Any word ending in one of these is already singular for our purposes;
# inflect would wrongly strip the trailing s (e.g. -ss class/glass,
# -is analysis/crisis, -us bus/corpus/status).
_SINGULAR_SUFFIX_GUARD = ("ss", "is", "us")

_WHITESPACE_RE = re.compile(r"\s+")


def singularize_word(word: str) -> str:
    """Singularize a single already-lowercased token, conservatively.

    The ONE word-level singularization core: inflect.singular_noun returns
    False when the word is already singular; in that case (and for guarded
    words) the input is kept unchanged. Consumed by canonicalize() here and
    by name_normalizer._lemmatize_word (entity/edge flavor).
    """
    if word in _SINGULAR_KEEP or word.endswith(_SINGULAR_SUFFIX_GUARD):
        return word
    singular = _INFLECT.singular_noun(word)
    return singular if isinstance(singular, str) else word


def canonicalize(name: str) -> str:
    """Return the canonical lowercase singular form of *name*.

    Idempotent. Empty / whitespace-only input returns the empty string.
    """
    # NFKC -> strip -> collapse whitespace -> lowercase.
    text = unicodedata.normalize("NFKC", name)
    text = _WHITESPACE_RE.sub(" ", text).strip().lower()
    if not text:
        return ""

    tokens = text.split(" ")

    # Drop a single leading article (keep it if it is the whole string).
    if len(tokens) > 1 and tokens[0] in _LEADING_ARTICLES:
        tokens = tokens[1:]

    # Singularize the HEAD NOUN ONLY (the last token). Modifiers are untouched.
    tokens[-1] = singularize_word(tokens[-1])

    # Strip a single trailing curated filler, but never empty the name.
    if len(tokens) > 1 and tokens[-1] in _TRAILING_FILLERS:
        tokens = tokens[:-1]

    return " ".join(tokens)

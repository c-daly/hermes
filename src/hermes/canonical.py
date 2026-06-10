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
del _ie

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

# --- predicate (descriptive-relation) canonicalization (hermes#130, H1) -----
# The edge-axis analog of canonicalize(): folds morphological variants of a
# descriptive relation onto one UPPER_SNAKE key so the open relation
# vocabulary stops sprawling. Applied only to descriptive relations; the
# reserved typing relations below are structural and never touched. Public
# (consumed by predicate_resolver / relation_synonyms) -- the boundary is
# explicit, not a private import.
RESERVED_PREDICATES = frozenset({"IS_A", "INSTANCE_OF", "SUBTYPE_OF"})

# Tokens ending in these are not plurals/3sg -- never strip the trailing S
# (ALIAS, BASIS, FOCUS, CHAOS, MASS). Same family as _SINGULAR_SUFFIX_GUARD,
# uppercased for the predicate token space.
_PREDICATE_NO_S_STRIP = ("SS", "US", "IS", "AS", "OS")

_PREDICATE_SEP_RE = re.compile(r"[\s\-_]+")


def normalize_predicate_surface(raw: str) -> str:
    """Readable UPPER_SNAKE surface form of a relation, no morphological fold.

    NFKC -> strip -> unify separators (space/hyphen/underscore) -> upper.
    This is the form STORED as the relation label; canonicalize_predicate()
    derives the match KEY from the same normalization. Non-string / empty in
    -> "" (the single chokepoint, so every predicate caller -- the resolver
    and synonym pass included -- is robust to a non-string LLM value).
    """
    if not isinstance(raw, str):
        return ""
    text = unicodedata.normalize("NFKC", raw).strip()
    if not text:
        return ""
    return "_".join(t for t in _PREDICATE_SEP_RE.split(text.upper()) if t)


def _fold_predicate_token(token: str) -> str:
    """Crude, deterministic, convergent stem for one UPPER-case token.

    Strips one inflectional suffix (-ING / -IES / -IED / -ED / -S) then a
    trailing stem -E, so base/3sg/past/gerund of a long-enough verb reach
    the SAME key (ACQUIRE/ACQUIRES/ACQUIRED/ACQUIRING -> ACQUIR). The -IED
    past tense of -y verbs maps to -Y to match -IES (CARRY/CARRIES/CARRIED
    -> CARRY), without which -y past tenses would diverge (review #134).
    Length guards keep short tokens intact (RED, BED, IN); the
    -SS/-US/-IS/-AS/-OS guard keeps non-plurals. The key is a stem, not
    necessarily prose -- convergence is the contract (see golden table).
    """
    t = token
    if len(t) >= 6 and t.endswith("ING"):
        t = t[:-3]
    elif len(t) >= 5 and t.endswith("IES"):
        t = t[:-3] + "Y"
    elif len(t) >= 5 and t.endswith("IED"):
        # past tense of a -y verb (carried/studied) -> the -y stem, so it
        # converges with -IES/-Y rather than stranding a bare -I.
        t = t[:-3] + "Y"
    elif len(t) >= 5 and t.endswith("ED"):
        t = t[:-2]
    elif len(t) >= 4 and t.endswith("S") and not t.endswith(_PREDICATE_NO_S_STRIP):
        t = t[:-1]
    if len(t) >= 4 and t.endswith("E"):
        t = t[:-1]
    return t


def canonicalize_predicate(raw: str) -> str:
    """Return the canonical UPPER_SNAKE key for a descriptive relation.

    Idempotent. Non-string / empty / whitespace-only input returns "".
    Reserved typing relations (IS_A / INSTANCE_OF / SUBTYPE_OF) are returned
    unchanged -- structural, never consolidated (callers exclude them too;
    defense in depth).

    Negation is safe WITHOUT special-casing: folding is per-token and never
    removes a polarity token (NOT/NEVER/...), so a negated predicate always
    keeps that token and can never collide with its positive form
    (DOES_NOT_REFER_TO -> DOE_NOT_REFER_TO vs REFERS_TO -> REFER_TO). Folding
    them is therefore both correct AND lets negated variants converge
    (NOT_PRODUCES / NOT_PRODUCED -> NOT_PRODUC).
    """
    joined = normalize_predicate_surface(raw)
    if not joined:
        return ""
    tokens = joined.split("_")

    if joined in RESERVED_PREDICATES:
        return joined

    return "_".join(_fold_predicate_token(t) for t in tokens)


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

    # Strip trailing curated fillers -- looped, so stacked fillers
    # ("vehicle type group") canonicalize in one pass -- but never empty the
    # name: "type group" stops at "type".
    while len(tokens) > 1 and tokens[-1] in _TRAILING_FILLERS:
        tokens = tokens[:-1]
        # Each stripped filler exposes a new head noun -- singularize it too,
        # or "vehicles type" -> "vehicles" breaks canonicality + idempotence.
        tokens[-1] = singularize_word(tokens[-1])

    return " ".join(tokens)

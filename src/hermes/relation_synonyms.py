"""Descriptive-relation synonym-collapse pass (hermes#133, H3).

The relation-axis analog of /name-cluster: the LLM detects DESCRIPTIVE
relation synonyms (HAULS/DRAGS/CARRIES -> CARRIES) -- the linguistic-
boundary work the H1 canonicalizer structurally cannot do. The LLM is a
codec: it proposes synonym groupings + a canonical name; it does not decide
placement or touch structure. This module owns the contract + fail-closed
server-side validation only. The endpoint (main.py) calls the LLM; the
graph is never mutated here -- output is a PROPOSAL routed through the
existing propose->assert / review path.

Hard rules (epic #131, enforced here, not trusted to the model):
  * no group may contain a reserved typing relation (IS_A/INSTANCE_OF/
    SUBTYPE_OF) -- those are structural and off-limits to consolidation;
  * members must be among the submitted candidates (no hallucinations);
  * members are deduped by H1 canonical key; a group that collapses to one
    distinct relation is not a synonym group;
  * the canonical label is an H1-normalized surface and must be a real
    member.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from hermes.canonical import (
    _RESERVED_PREDICATES,
    canonicalize_predicate,
    normalize_predicate_surface,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SynonymGroup:
    canonical: str
    members: list[str]
    confidence: float


_SYSTEM_PROMPT = (
    "You are a relation-vocabulary synonym detector for the LOGOS knowledge "
    "graph. Given a list of DESCRIPTIVE relation labels, group together the "
    "ones that mean the SAME relation (e.g. HAULS, DRAGS, CARRIES) and give "
    "each group one canonical UPPER_SNAKE_CASE label chosen from its "
    "members.\n\n"
    "Hard rules:\n"
    "- NEVER group the structural typing relations IS_A, INSTANCE_OF, or "
    "SUBTYPE_OF with anything -- they are off-limits and must not appear in "
    "any group.\n"
    "- Only group labels that are TRUE synonyms (interchangeable in meaning "
    "and direction); when unsure, leave a label out.\n"
    "- Never invent labels: every member must come from the provided list.\n"
    "- A group needs at least two distinct relations; do not emit singletons.\n\n"
    'Return ONLY valid JSON: {"groups": [{"canonical": "...", "members": '
    '["...", "..."], "confidence": 0.0-1.0}]}.'
)


def build_synonym_messages(
    predicates: list[str], context: str | None = None
) -> list[dict[str, str]]:
    """Build the chat messages for the synonym pass.

    Reserved typing relations are filtered out of the candidate list before
    the model ever sees them (defense in depth -- the prompt also forbids
    grouping them).
    """
    candidates = [
        p
        for p in predicates
        if normalize_predicate_surface(p) not in _RESERVED_PREDICATES
        and normalize_predicate_surface(p)
    ]
    ctx = f"Domain context: {context}\n\n" if context else ""
    user = f"{ctx}Candidates: {', '.join(candidates)}\n\nGroup the synonyms."
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def _coerce_json(content: str) -> dict | None:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"```(?:json)?\s*(.*?)```", content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                return None
    return None


def parse_synonym_response(
    content: str, candidates: list[str]
) -> list[SynonymGroup]:
    """Validate the LLM response into safe synonym groups. Fail closed:
    anything malformed or rule-violating drops the offending group."""
    data = _coerce_json(content)
    if not isinstance(data, dict):
        return []
    raw_groups = data.get("groups")
    if not isinstance(raw_groups, list):
        return []

    # candidate surfaces by canonical key (what the members may reference)
    allowed_surface = {
        normalize_predicate_surface(c)
        for c in candidates
        if normalize_predicate_surface(c)
    }

    out: list[SynonymGroup] = []
    for grp in raw_groups:
        if not isinstance(grp, dict):
            continue
        raw_members = grp.get("members")
        if not isinstance(raw_members, list):
            continue

        members_surface: list[str] = []
        seen_canon: set[str] = set()
        bad = False
        for m in raw_members:
            if not isinstance(m, str):
                bad = True
                break
            surface = normalize_predicate_surface(m)
            if not surface or surface not in allowed_surface:
                bad = True  # hallucinated or empty member
                break
            if surface in _RESERVED_PREDICATES:
                bad = True  # reserved relation anywhere kills the group
                break
            canon = canonicalize_predicate(surface)
            if canon not in seen_canon:
                seen_canon.add(canon)
                members_surface.append(surface)
        if bad:
            continue
        if len(seen_canon) < 2:
            continue  # not a real collapse

        # canonical label: the LLM's pick if it is a real member, else the
        # first member (deterministic) -- never a hallucinated/reserved name
        proposed = normalize_predicate_surface(str(grp.get("canonical", "")))
        canonical = (
            proposed
            if proposed in members_surface and proposed not in _RESERVED_PREDICATES
            else members_surface[0]
        )

        confidence = grp.get("confidence", 0.7)
        if not isinstance(confidence, (int, float)):
            confidence = 0.7
        confidence = max(0.0, min(1.0, float(confidence)))

        out.append(SynonymGroup(canonical, members_surface, confidence))
    return out

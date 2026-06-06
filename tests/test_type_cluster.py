"""Tests for the v2 /type-cluster endpoint -- as-designed contract (#127).

Hermes NAMES the cluster and flags the OUTLIERS; it does NOT partition into
subgroups, propose an IS_A chain, or decide reuse/mint (the placement cascade
does all of that from the name). The model reasons over member NAMES and
returns outlier NAMES; Hermes maps those back to input ids over the known
member set -- the model never handles an id. Every test monkeypatches
generate_completion; no live LLM calls.
"""

from __future__ import annotations

import json

import hermes.main as m
from fastapi.testclient import TestClient

client = TestClient(m.app)


def _make_completion(content: str):
    async def fake_completion(messages, temperature=0.0, max_tokens=512, **kwargs):
        # The pin must reach the wire, and the prompt must carry member NAMES.
        assert temperature == 0.0
        fake_completion.last_messages = messages  # type: ignore[attr-defined]
        return {"choices": [{"message": {"content": content}}]}

    return fake_completion


def _members(*pairs):
    return [{"id": i, "name": n} for i, n in pairs]


def _post(members, request_id="t::0"):
    return client.post(
        "/type-cluster", json={"members": members, "request_id": request_id}
    )


# --------------------------------------------------------------------------
# Naming
# --------------------------------------------------------------------------


def test_names_the_cluster_and_canonicalizes(monkeypatch):
    monkeypatch.setattr(
        m, "generate_completion", _make_completion(json.dumps({"name": "Vehicles"}))
    )
    resp = _post(_members(("i1", "boat"), ("i2", "car")))
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "vehicle"  # canonicalized lowercase singular
    assert body["residual_ids"] == []
    assert body["raw_partition_ok"] is True
    assert body["request_id"] == "t::0"


def test_no_outliers_key_means_no_residuals(monkeypatch):
    monkeypatch.setattr(
        m, "generate_completion", _make_completion(json.dumps({"name": "mammal"}))
    )
    body = _post(_members(("i1", "cat"), ("i2", "dog"))).json()
    assert body["residual_ids"] == []
    assert body["raw_partition_ok"] is True


# --------------------------------------------------------------------------
# Outlier name -> id mapping (the model never echoes ids, #127)
# --------------------------------------------------------------------------


def test_outlier_name_maps_to_member_id(monkeypatch):
    monkeypatch.setattr(
        m,
        "generate_completion",
        _make_completion(json.dumps({"name": "carbon", "outliers": ["diamond ring"]})),
    )
    body = _post(
        _members(("u-graphite", "graphite"), ("u-ring", "diamond ring"))
    ).json()
    assert body["residual_ids"] == ["u-ring"]
    assert body["raw_partition_ok"] is True


def test_outlier_match_is_case_and_space_normalized(monkeypatch):
    monkeypatch.setattr(
        m,
        "generate_completion",
        _make_completion(json.dumps({"name": "star", "outliers": ["  THE   Sun "]})),
    )
    body = _post(_members(("u-vega", "vega"), ("u-sun", "the sun"))).json()
    assert body["residual_ids"] == ["u-sun"]
    assert body["raw_partition_ok"] is True


def test_hallucinated_outlier_name_is_dropped_and_flags_raw_partition(monkeypatch):
    monkeypatch.setattr(
        m,
        "generate_completion",
        _make_completion(
            json.dumps({"name": "star", "outliers": ["a name not in the cluster"]})
        ),
    )
    body = _post(_members(("u-vega", "vega"), ("u-sun", "the sun"))).json()
    assert body["residual_ids"] == []  # unmatched outlier dropped
    assert body["raw_partition_ok"] is False  # but the miss is surfaced


def test_duplicate_member_name_claims_one_unclaimed_id(monkeypatch):
    monkeypatch.setattr(
        m,
        "generate_completion",
        _make_completion(json.dumps({"name": "metal", "outliers": ["mercury"]})),
    )
    body = _post(
        _members(("u-m1", "mercury"), ("u-m2", "mercury"), ("u-iron", "iron"))
    ).json()
    assert len(body["residual_ids"]) == 1
    assert body["residual_ids"][0] in {"u-m1", "u-m2"}


# --------------------------------------------------------------------------
# over_specified ceiling signal (computed on the raw name)
# --------------------------------------------------------------------------


def test_over_specified_name_is_flagged(monkeypatch):
    monkeypatch.setattr(
        m,
        "generate_completion",
        _make_completion(json.dumps({"name": "carbon and its allotropes"})),
    )
    body = _post(_members(("i1", "graphite"))).json()
    assert body["over_specified"] is True


def test_clean_name_not_over_specified(monkeypatch):
    monkeypatch.setattr(
        m, "generate_completion", _make_completion(json.dumps({"name": "carbon"}))
    )
    body = _post(_members(("i1", "graphite"))).json()
    assert body["over_specified"] is False


# --------------------------------------------------------------------------
# The model is never asked to handle ids (the #127 contract guarantee)
# --------------------------------------------------------------------------


def test_prompt_carries_names_not_ids(monkeypatch):
    fake = _make_completion(json.dumps({"name": "vehicle"}))
    monkeypatch.setattr(m, "generate_completion", fake)
    _post(_members(("uuid-aaaaaaaa", "boat"), ("uuid-bbbbbbbb", "car")))
    prompt = " ".join(msg["content"] for msg in fake.last_messages)
    assert "boat" in prompt and "car" in prompt
    assert "uuid-aaaaaaaa" not in prompt and "uuid-bbbbbbbb" not in prompt


# --------------------------------------------------------------------------
# Fail-closed paths -> 502
# --------------------------------------------------------------------------


def test_unparseable_json_is_502(monkeypatch):
    monkeypatch.setattr(m, "generate_completion", _make_completion("not json {{"))
    assert _post(_members(("i1", "x"))).status_code == 502


def test_non_object_json_is_502(monkeypatch):
    monkeypatch.setattr(m, "generate_completion", _make_completion(json.dumps([1, 2])))
    assert _post(_members(("i1", "x"))).status_code == 502


def test_missing_name_is_502(monkeypatch):
    monkeypatch.setattr(
        m, "generate_completion", _make_completion(json.dumps({"outliers": []}))
    )
    assert _post(_members(("i1", "x"))).status_code == 502


def test_blank_name_is_502(monkeypatch):
    monkeypatch.setattr(
        m, "generate_completion", _make_completion(json.dumps({"name": "   "}))
    )
    assert _post(_members(("i1", "x"))).status_code == 502


def test_no_choices_is_502(monkeypatch):
    async def empty(messages, temperature=0.0, max_tokens=512, **kwargs):
        return {"choices": []}

    monkeypatch.setattr(m, "generate_completion", empty)
    assert _post(_members(("i1", "x"))).status_code == 502


def test_empty_members_rejected_by_contract():
    # min_length=1 on the request model: an empty cluster is a 422, not a call.
    assert client.post("/type-cluster", json={"members": []}).status_code == 422


# --------------------------------------------------------------------------
# parent: null => reuse `name`; a string => mint `name` under that existing
# type. Canonicalized, then resolved closed-world: structural roots
# (`node`/`root`) and -- with a catalog -- unpublished names coerce to None;
# remaining placement validity is left to the cascade.
# --------------------------------------------------------------------------


def test_parent_null_passes_through_as_reuse(monkeypatch):
    monkeypatch.setattr(
        m,
        "generate_completion",
        _make_completion(json.dumps({"name": "vehicle", "parent": None})),
    )
    body = _post(_members(("i1", "boat"), ("i2", "car"))).json()
    assert body["name"] == "vehicle"
    assert body["parent"] is None  # reuse: no new parent edge target


def test_parent_string_passes_through_canonicalized(monkeypatch):
    monkeypatch.setattr(
        m,
        "generate_completion",
        _make_completion(json.dumps({"name": "sedan", "parent": "Vehicles"})),
    )
    body = _post(_members(("i1", "a sedan"),)).json()
    assert body["name"] == "sedan"
    assert body["parent"] == "vehicle"  # canonicalized existing-parent name


def test_missing_parent_key_is_reuse(monkeypatch):
    monkeypatch.setattr(
        m, "generate_completion", _make_completion(json.dumps({"name": "star"}))
    )
    body = _post(_members(("i1", "vega"),)).json()
    assert body["parent"] is None


# --------------------------------------------------------------------------
# Catalog construction: domain roots (entity/concept/process) are ordinary
# catalog citizens -- aliased like every published type, valid both as a
# `parent` and as an honest `name` reuse. Only the structural scaffolding
# above them (`node`, `root`) and the unminted `cognition` are excluded.
# --------------------------------------------------------------------------


_CATALOG_TYPES = {
    "entity": {"uuid": "u-entity", "root": "entity", "chain": []},
    "concept": {"uuid": "u-concept", "root": "concept", "chain": []},
    "process": {"uuid": "u-process", "root": "process", "chain": []},
    "vehicle": {"uuid": "u-vehicle", "root": "entity", "chain": ["entity"]},
    # Structural / unminted: must never reach the catalog.
    "node": {"uuid": "u-node", "root": "", "chain": []},
    "root": {"uuid": "u-root", "root": "", "chain": []},
    "cognition": {"uuid": "u-cognition", "root": "", "chain": []},
}


class _FakeRegistry:
    def get_type_names(self):
        return list(_CATALOG_TYPES)

    def get_type(self, name):
        return dict(_CATALOG_TYPES[name])


def test_domain_roots_are_ordinary_catalog_entries(monkeypatch):
    monkeypatch.setattr(m, "_type_registry", _FakeRegistry())
    block, alias_to_uuid, _published, catalog_names = m._build_catalog_context()
    entry_lines = [ln for ln in block.splitlines() if ln.lstrip().startswith("[t_")]
    for root in ("entity", "concept", "process"):
        assert any(f"] {root} (" in ln for ln in entry_lines)  # aliased entry
        assert root in catalog_names
    assert {"u-entity", "u-concept", "u-process"} <= set(alias_to_uuid.values())
    assert "GRAFT-ONLY" not in block  # no special-casing block anywhere


def test_catalog_never_lists_node_root_or_cognition(monkeypatch):
    monkeypatch.setattr(m, "_type_registry", _FakeRegistry())
    _block, alias_to_uuid, published, catalog_names = m._build_catalog_context()
    assert {"node", "root", "cognition"}.isdisjoint(catalog_names)
    assert {"u-node", "u-root", "u-cognition"}.isdisjoint(published)
    assert {"u-node", "u-root", "u-cognition"}.isdisjoint(set(alias_to_uuid.values()))


def test_prompt_lists_roots_without_graft_only_block(monkeypatch):
    monkeypatch.setattr(m, "_type_registry", _FakeRegistry())
    fake = _make_completion(json.dumps({"name": "vehicle"}))
    monkeypatch.setattr(m, "generate_completion", fake)
    _post(_members(("i1", "boat")))
    system_prompt = fake.last_messages[0]["content"]
    assert "GRAFT-ONLY" not in system_prompt
    assert "] entity (" in system_prompt  # roots listed as ordinary entries


# --------------------------------------------------------------------------
# parent resolution is fail-closed: `node`/`root` (structural, never in the
# catalog) coerce to None with a logged warning; with a catalog present, a
# parent that does not resolve to a published name coerces too (closed-world).
# --------------------------------------------------------------------------


def _spy_warnings(monkeypatch):
    logged: list[str] = []
    monkeypatch.setattr(
        m.logger, "warning", lambda msg, *args, **kw: logged.append(msg % args)
    )
    return logged


def test_parent_node_is_coerced_and_logged(monkeypatch):
    logged = _spy_warnings(monkeypatch)
    monkeypatch.setattr(
        m,
        "generate_completion",
        _make_completion(json.dumps({"name": "sensor", "parent": "node"})),
    )
    body = _post(_members(("i1", "lidar"))).json()
    assert body["parent"] is None  # structural root never passes through
    assert any("bad_root_coerce" in entry for entry in logged)


def test_parent_root_is_coerced_and_logged(monkeypatch):
    logged = _spy_warnings(monkeypatch)
    monkeypatch.setattr(
        m,
        "generate_completion",
        _make_completion(json.dumps({"name": "sensor", "parent": "Root"})),
    )
    body = _post(_members(("i1", "lidar"))).json()
    assert body["parent"] is None
    assert any("bad_root_coerce" in entry for entry in logged)


def test_unresolvable_parent_is_coerced_when_catalog_present(monkeypatch):
    monkeypatch.setattr(m, "_type_registry", _FakeRegistry())
    monkeypatch.setattr(
        m,
        "generate_completion",
        _make_completion(json.dumps({"name": "sedan", "parent": "starship"})),
    )
    body = _post(_members(("i1", "a sedan"))).json()
    assert body["name"] == "sedan"
    assert body["parent"] is None  # closed-world: not a published name


def test_domain_root_is_a_valid_parent(monkeypatch):
    monkeypatch.setattr(m, "_type_registry", _FakeRegistry())
    monkeypatch.setattr(
        m,
        "generate_completion",
        _make_completion(json.dumps({"name": "sedan", "parent": "Entity"})),
    )
    body = _post(_members(("i1", "a sedan"))).json()
    assert body["parent"] == "entity"  # ordinary catalog citizen


def test_domain_root_is_a_valid_name_reuse(monkeypatch):
    monkeypatch.setattr(m, "_type_registry", _FakeRegistry())
    monkeypatch.setattr(
        m,
        "generate_completion",
        _make_completion(json.dumps({"name": "process", "parent": None})),
    )
    body = _post(_members(("i1", "fermentation"))).json()
    assert body["name"] == "process"
    assert body["parent"] is None

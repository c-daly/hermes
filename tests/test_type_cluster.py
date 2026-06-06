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
    async def fake_completion(messages, temperature=0.0, max_tokens=512):
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
    async def empty(messages, temperature=0.0, max_tokens=512):
        return {"choices": []}

    monkeypatch.setattr(m, "generate_completion", empty)
    assert _post(_members(("i1", "x"))).status_code == 502


def test_empty_members_rejected_by_contract():
    # min_length=1 on the request model: an empty cluster is a 422, not a call.
    assert client.post("/type-cluster", json={"members": []}).status_code == 422


# --------------------------------------------------------------------------
# parent: null => reuse `name`; a string => mint `name` under that existing
# type. Passed through canonicalized; placement validity is the cascade's job.
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

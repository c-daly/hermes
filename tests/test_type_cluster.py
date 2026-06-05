"""Tests for the v2 /type-cluster endpoint (naming-driven typing experiment T4).

Every test monkeypatches generate_completion -- no live LLM calls. The endpoint
contract and server-side validation are exercised in isolation; where a test
needs a catalog it injects an in-process stub TypeRegistry (experiment path A).
The placement cascade lives in later tasks.
"""

from __future__ import annotations

import json
from typing import Any

import hermes.main as m
from fastapi.testclient import TestClient


def _make_completion(content: str):
    """Build a fake generate_completion returning *content* as the LLM message."""

    async def fake_completion(messages, temperature=0.0, max_tokens=512):
        return {"choices": [{"message": {"content": content}}]}

    return fake_completion


class _StubTypeRegistry:
    """Minimal in-process TypeRegistry stub (experiment path A)."""

    def __init__(self, types: dict[str, dict[str, Any]]) -> None:
        self._types = types

    def get_type_names(self) -> list[str]:
        return sorted(self._types)

    def get_type(self, name: str) -> dict[str, Any] | None:
        info = self._types.get(name)
        return dict(info) if info is not None else None


def test_type_cluster_happy_path_partitions_ids(monkeypatch):
    """A well-formed two-group response round-trips, partitions all ids, and is clean."""
    content = json.dumps(
        {
            "groups": [
                {
                    "assign_to": "NEW",
                    "name": "Vehicles",
                    "chain": ["Vehicle", "Entity"],
                    "member_ids": ["i1", "i2"],
                    "confidence": 0.9,
                    "description": "wheeled",
                },
                {
                    "assign_to": "NEW",
                    "name": "Mammal",
                    "chain": ["mammal", "animal", "entity"],
                    "member_ids": ["i3"],
                },
            ],
            "residual_ids": [],
        }
    )
    monkeypatch.setattr(m, "generate_completion", _make_completion(content))
    client = TestClient(m.app)
    resp = client.post(
        "/type-cluster",
        json={
            "members": [
                {"id": "i1", "name": "car"},
                {"id": "i2", "name": "truck"},
                {"id": "i3", "name": "dog"},
            ],
            "request_id": "req-1",
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["request_id"] == "req-1"
    assert body["raw_partition_ok"] is True
    assert body["residual_ids"] == []
    groups = body["groups"]
    assert len(groups) == 2
    # name + chain entries are canonicalized (lowercase singular); chain[0]==name.
    g0 = groups[0]
    assert g0["name"] == "vehicle"
    assert g0["chain"][0] == "vehicle"
    assert g0["chain"][-1] == "entity"
    assert g0["member_ids"] == ["i1", "i2"]
    assert g0["over_specified"] is False
    # total partition: every input id is claimed exactly once across groups+residual.
    claimed = [mid for g in groups for mid in g["member_ids"]] + body["residual_ids"]
    assert sorted(claimed) == ["i1", "i2", "i3"]


def test_type_cluster_empty_members_is_422():
    """An empty cluster is a request-validation 422, not a fabricated typing."""
    client = TestClient(m.app)
    resp = client.post("/type-cluster", json={"members": []})
    assert resp.status_code == 422, resp.text


def test_type_cluster_non_dict_llm_is_502(monkeypatch):
    """A non-object JSON payload from the LLM surfaces as a 502 (bad upstream shape)."""
    monkeypatch.setattr(
        m, "generate_completion", _make_completion(json.dumps(["not", "a", "dict"]))
    )
    client = TestClient(m.app)
    resp = client.post("/type-cluster", json={"members": [{"id": "i1", "name": "x"}]})
    assert resp.status_code == 502, resp.text


def test_type_cluster_truncated_member_ids_is_502(monkeypatch):
    """A clipped member_ids array (truncated tail) => raw_partition_ok False => 502."""
    # Clip the closing brackets so the first-{/last-} fallback of _extract_json
    # also fails (no closing brace remains anywhere in the payload).
    full = json.dumps({"groups": [{"name": "vehicle", "member_ids": ["i1", "i2"]}]})
    truncated = full[:-4]
    monkeypatch.setattr(m, "generate_completion", _make_completion(truncated))
    client = TestClient(m.app)
    resp = client.post("/type-cluster", json={"members": [{"id": "i1", "name": "x"}]})
    assert resp.status_code == 502, resp.text


def test_type_cluster_unclaimed_ids_become_residual_and_flag_raw(monkeypatch):
    """Ids the LLM omits land in residual_ids; raw_partition_ok is False (not total)."""
    content = json.dumps(
        {
            "groups": [
                {
                    "assign_to": "NEW",
                    "name": "vehicle",
                    "chain": ["vehicle", "entity"],
                    "member_ids": ["i1"],
                }
            ],
            "residual_ids": [],
        }
    )
    monkeypatch.setattr(m, "generate_completion", _make_completion(content))
    client = TestClient(m.app)
    resp = client.post(
        "/type-cluster",
        json={"members": [{"id": "i1", "name": "car"}, {"id": "i2", "name": "truck"}]},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["residual_ids"] == ["i2"]
    assert body["raw_partition_ok"] is False


def test_type_cluster_hallucinated_and_duplicate_ids_dropped(monkeypatch):
    """Unknown ids are dropped; a re-claimed id goes to its first group (first-claim wins)."""
    content = json.dumps(
        {
            "groups": [
                {
                    "assign_to": "NEW",
                    "name": "vehicle",
                    "chain": ["vehicle", "entity"],
                    "member_ids": ["i1", "i1", "ghost"],
                },
                {
                    "assign_to": "NEW",
                    "name": "animal",
                    "chain": ["animal", "entity"],
                    "member_ids": ["i1", "i2"],
                },
            ],
            "residual_ids": [],
        }
    )
    monkeypatch.setattr(m, "generate_completion", _make_completion(content))
    client = TestClient(m.app)
    resp = client.post(
        "/type-cluster",
        json={"members": [{"id": "i1", "name": "car"}, {"id": "i2", "name": "dog"}]},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    groups = body["groups"]
    assert groups[0]["member_ids"] == ["i1"]  # dedup + first-claim, ghost dropped
    assert groups[1]["member_ids"] == ["i2"]  # i1 already claimed by group 0
    assert body["raw_partition_ok"] is False  # raw output was not disjoint


def test_type_cluster_chain_root_appended_when_missing(monkeypatch):
    """A chain not terminating in a realm root gets a deterministic root appended."""
    content = json.dumps(
        {
            "groups": [
                {
                    "assign_to": "NEW",
                    "name": "sedan",
                    "chain": ["sedan", "vehicle"],
                    "member_ids": ["i1"],
                }
            ],
            "residual_ids": [],
        }
    )
    monkeypatch.setattr(m, "generate_completion", _make_completion(content))
    client = TestClient(m.app)
    resp = client.post("/type-cluster", json={"members": [{"id": "i1", "name": "car"}]})
    assert resp.status_code == 200, resp.text
    chain = resp.json()["groups"][0]["chain"]
    assert chain[0] == "sedan"
    assert chain[-1] in {"entity", "concept", "process"}


def test_type_cluster_chain_sliced_from_existing_name_occurrence(monkeypatch):
    """A canon name already mid-chain slices the chain from that occurrence.

    Naive prepend would mint a non-consecutive duplicate
    (vehicle > car > vehicle > ...); slicing drops the more-specific
    prefix and keeps the general tail intact.
    """
    content = json.dumps(
        {
            "groups": [
                {
                    "assign_to": "NEW",
                    "name": "vehicle",
                    "chain": ["car", "vehicle", "machine", "entity"],
                    "member_ids": ["i1"],
                }
            ],
            "residual_ids": [],
        }
    )
    monkeypatch.setattr(m, "generate_completion", _make_completion(content))
    client = TestClient(m.app)
    resp = client.post("/type-cluster", json={"members": [{"id": "i1", "name": "car"}]})
    assert resp.status_code == 200, resp.text
    chain = resp.json()["groups"][0]["chain"]
    assert chain == ["vehicle", "machine", "entity"]


def test_type_cluster_over_specified_flag(monkeypatch):
    """A conjunction / too-many-words RAW name sets over_specified on the group."""
    content = json.dumps(
        {
            "groups": [
                {
                    "assign_to": "NEW",
                    "name": "cars and trucks",
                    "chain": ["vehicle", "entity"],
                    "member_ids": ["i1"],
                }
            ],
            "residual_ids": [],
        }
    )
    monkeypatch.setattr(m, "generate_completion", _make_completion(content))
    client = TestClient(m.app)
    resp = client.post("/type-cluster", json={"members": [{"id": "i1", "name": "car"}]})
    assert resp.status_code == 200, resp.text
    assert resp.json()["groups"][0]["over_specified"] is True


def test_type_cluster_all_groups_invalid_is_502(monkeypatch):
    """Every group missing name/member_ids => nothing usable => 502."""
    content = json.dumps(
        {"groups": [{"name": "", "member_ids": []}], "residual_ids": []}
    )
    monkeypatch.setattr(m, "generate_completion", _make_completion(content))
    client = TestClient(m.app)
    resp = client.post("/type-cluster", json={"members": [{"id": "i1", "name": "x"}]})
    assert resp.status_code == 502, resp.text


def test_type_cluster_assign_to_unresolvable_coerced_new(monkeypatch):
    """A bracketed alias that does not resolve against the catalog coerces to NEW."""
    content = json.dumps(
        {
            "groups": [
                {
                    "assign_to": "[nope-uuid]",
                    "name": "vehicle",
                    "chain": ["vehicle", "entity"],
                    "member_ids": ["i1"],
                }
            ],
            "residual_ids": [],
        }
    )
    monkeypatch.setattr(m, "generate_completion", _make_completion(content))
    client = TestClient(m.app)
    resp = client.post("/type-cluster", json={"members": [{"id": "i1", "name": "car"}]})
    assert resp.status_code == 200, resp.text
    assert resp.json()["groups"][0]["assign_to"] == "NEW"


def test_canonicalize_name_lowercases_and_singularizes():
    """The shared canonicalize impl is idempotent and collapses vehicle/vehicles."""
    assert m.canonicalize("Vehicles") == "vehicle"
    assert m.canonicalize("vehicle") == "vehicle"
    assert m.canonicalize(m.canonicalize("Vehicles")) == "vehicle"
    assert m.canonicalize("process") == "process"


def test_type_cluster_duplicate_member_ids_is_422():
    """Duplicate member ids break the total-partition contract => 422."""
    client = TestClient(m.app)
    resp = client.post(
        "/type-cluster",
        json={
            "members": [
                {"id": "i1", "name": "car"},
                {"id": "i1", "name": "truck"},
            ]
        },
    )
    assert resp.status_code == 422, resp.text


def test_type_cluster_alias_assign_resolves_against_stub_catalog(monkeypatch):
    """A bracketed [t_xxxx] alias resolves to the published uuid it aliases."""
    stub = _StubTypeRegistry(
        {
            "vehicle": {
                "uuid": "type_vehicle_aaaa1111",
                "root": "entity",
                "chain": ["vehicle", "entity"],
            }
        }
    )
    monkeypatch.setattr(m, "_type_registry", stub)
    content = json.dumps(
        {
            "groups": [
                {
                    "assign_to": "[t_0000]",
                    "name": "vehicle",
                    "chain": ["vehicle", "entity"],
                    "member_ids": ["i1"],
                }
            ],
            "residual_ids": [],
        }
    )
    monkeypatch.setattr(m, "generate_completion", _make_completion(content))
    client = TestClient(m.app)
    resp = client.post("/type-cluster", json={"members": [{"id": "i1", "name": "car"}]})
    assert resp.status_code == 200, resp.text
    assert resp.json()["groups"][0]["assign_to"] == "type_vehicle_aaaa1111"


def test_type_cluster_protected_root_assign_coerced_new(monkeypatch):
    """assign_to resolving to a realm root uuid coerces to NEW (graft-only roots)."""
    stub = _StubTypeRegistry(
        {
            "entity": {"uuid": "type_entity_root0001", "is_root": True},
            "vehicle": {"uuid": "type_vehicle_aaaa1111", "root": "entity"},
        }
    )
    monkeypatch.setattr(m, "_type_registry", stub)
    content = json.dumps(
        {
            "groups": [
                {
                    "assign_to": "type_entity_root0001",
                    "name": "vehicle",
                    "chain": ["vehicle", "entity"],
                    "member_ids": ["i1"],
                }
            ],
            "residual_ids": [],
        }
    )
    monkeypatch.setattr(m, "generate_completion", _make_completion(content))
    client = TestClient(m.app)
    resp = client.post("/type-cluster", json={"members": [{"id": "i1", "name": "car"}]})
    assert resp.status_code == 200, resp.text
    assert resp.json()["groups"][0]["assign_to"] == "NEW"


def test_type_cluster_homonym_groups_not_merged(monkeypatch):
    """Same canonical name under DIFFERENT proposed roots stays two groups."""
    content = json.dumps(
        {
            "groups": [
                {
                    "assign_to": "NEW",
                    "name": "bank",
                    "chain": ["bank", "entity"],
                    "member_ids": ["i1"],
                },
                {
                    "assign_to": "NEW",
                    "name": "bank",
                    "chain": ["bank", "concept"],
                    "member_ids": ["i2"],
                },
            ],
            "residual_ids": [],
        }
    )
    monkeypatch.setattr(m, "generate_completion", _make_completion(content))
    client = TestClient(m.app)
    resp = client.post(
        "/type-cluster",
        json={
            "members": [
                {"id": "i1", "name": "river bank"},
                {"id": "i2", "name": "trust"},
            ]
        },
    )
    assert resp.status_code == 200, resp.text
    groups = resp.json()["groups"]
    assert len(groups) == 2
    assert {g["chain"][-1] for g in groups} == {"entity", "concept"}


def test_type_cluster_same_name_same_root_groups_merged(monkeypatch):
    """Same canonical name AND same proposed root merge: union ids, best chain."""
    content = json.dumps(
        {
            "groups": [
                {
                    "assign_to": "NEW",
                    "name": "vehicle",
                    "chain": ["vehicle", "entity"],
                    "member_ids": ["i1"],
                    "confidence": 0.6,
                },
                {
                    "assign_to": "NEW",
                    "name": "Vehicles",
                    "chain": ["vehicle", "conveyance", "entity"],
                    "member_ids": ["i2"],
                    "confidence": 0.9,
                },
            ],
            "residual_ids": [],
        }
    )
    monkeypatch.setattr(m, "generate_completion", _make_completion(content))
    client = TestClient(m.app)
    resp = client.post(
        "/type-cluster",
        json={"members": [{"id": "i1", "name": "car"}, {"id": "i2", "name": "truck"}]},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    groups = body["groups"]
    assert len(groups) == 1
    assert groups[0]["member_ids"] == ["i1", "i2"]
    assert groups[0]["confidence"] == 0.9
    assert groups[0]["chain"] == ["vehicle", "conveyance", "entity"]
    assert body["residual_ids"] == []
    assert body["raw_partition_ok"] is True


def test_type_cluster_catalog_block_in_system_prompt(monkeypatch):
    """The catalog block (aliases + GRAFT-ONLY roots) lands in the SYSTEM prompt."""
    stub = _StubTypeRegistry(
        {"vehicle": {"uuid": "type_vehicle_aaaa1111", "root": "entity"}}
    )
    monkeypatch.setattr(m, "_type_registry", stub)
    captured: dict[str, Any] = {}

    async def fake_completion(messages, temperature=0.0, max_tokens=512):
        captured["messages"] = messages
        captured["temperature"] = temperature
        content = json.dumps(
            {
                "groups": [
                    {
                        "assign_to": "NEW",
                        "name": "vehicle",
                        "chain": ["vehicle", "entity"],
                        "member_ids": ["i1"],
                    }
                ],
                "residual_ids": [],
            }
        )
        return {"choices": [{"message": {"content": content}}]}

    monkeypatch.setattr(m, "generate_completion", fake_completion)
    client = TestClient(m.app)
    resp = client.post("/type-cluster", json={"members": [{"id": "i1", "name": "car"}]})
    assert resp.status_code == 200, resp.text
    system = captured["messages"][0]
    assert system["role"] == "system"
    assert "PUBLISHED TYPE CATALOG" in system["content"]
    assert "[t_0000] vehicle" in system["content"]
    assert "GRAFT-ONLY ROOTS" in system["content"]
    assert captured["temperature"] == 0.0

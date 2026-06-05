"""Tests for the /name-cluster endpoint (Sophia emergence #505)."""

from __future__ import annotations

import hermes.main as m
from fastapi.testclient import TestClient


def test_name_cluster_names_the_bind(monkeypatch):
    async def fake_completion(messages, temperature=0.0, max_tokens=128):
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"label": "Concept", '
                        '"description": "abstract ideas", "confidence": 0.82}'
                    }
                }
            ]
        }

    monkeypatch.setattr(m, "generate_completion", fake_completion)

    client = TestClient(m.app)
    resp = client.post(
        "/name-cluster",
        json={
            "members": [
                {
                    "name": "derivative",
                    "type": "entity",
                    "hermes_type_hint": "concept",
                    "neighbors": [
                        {
                            "relation": "DEFINED_AS",
                            "neighbor_name": "limit",
                            "neighbor_type": "entity",
                        }
                    ],
                },
                {"name": "integral", "type": "entity", "hermes_type_hint": "concept"},
            ],
            "candidates": ["object", "location", "concept"],
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["label"] == "concept"  # normalized to lowercase
    assert 0.0 <= body["confidence"] <= 1.0


def test_name_cluster_rejects_empty_members():
    """An empty cluster is a 422 (validation), not a fabricated label (greptile P1)."""
    client = TestClient(m.app)
    resp = client.post(
        "/name-cluster",
        json={"members": [], "candidates": ["object", "concept"]},
    )
    assert resp.status_code == 422, resp.text


def test_name_cluster_clamps_out_of_range_confidence(monkeypatch):
    """A model confidence outside [0,1] is clamped, not passed through (greptile P2)."""

    async def fake_completion(messages, temperature=0.0, max_tokens=128):
        return {
            "choices": [
                {"message": {"content": '{"label": "thing", "confidence": 1.7}'}}
            ]
        }

    monkeypatch.setattr(m, "generate_completion", fake_completion)
    client = TestClient(m.app)
    resp = client.post("/name-cluster", json={"members": [{"name": "x"}]})
    assert resp.status_code == 200, resp.text
    assert resp.json()["confidence"] == 1.0


def test_name_cluster_malformed_llm_response_is_502(monkeypatch):
    """Unparseable / label-less LLM output surfaces as 502, not an unhandled 500
    (gemini high / greptile P2)."""

    async def no_label(messages, temperature=0.0, max_tokens=128):
        return {"choices": [{"message": {"content": '{"name": "oops"}'}}]}

    monkeypatch.setattr(m, "generate_completion", no_label)
    client = TestClient(m.app)
    resp = client.post("/name-cluster", json={"members": [{"name": "x"}]})
    assert resp.status_code == 502, resp.text


def test_name_cluster_maps_removed_names_to_ids(monkeypatch):
    """Outliers Hermes flags by name come back as the caller's member ids (#504)."""

    async def fake_completion(messages, temperature=0.0, max_tokens=128):
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"label": "tree", "confidence": 0.9, '
                        '"removed": ["Tusk", "amber sap"]}'
                    }
                }
            ]
        }

    monkeypatch.setattr(m, "generate_completion", fake_completion)
    client = TestClient(m.app)
    resp = client.post(
        "/name-cluster",
        json={
            "members": [
                {"name": "oak", "id": "n1"},
                {"name": "pine", "id": "n2"},
                {"name": "tusk", "id": "n3"},  # case-insensitive match to "Tusk"
                {"name": "amber sap", "id": "n4"},
            ],
            "candidates": [],
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["label"] == "tree"
    # Names map back to ids, case-insensitively; coherent majority is untouched.
    assert set(body["removed"]) == {"n3", "n4"}


def test_name_cluster_removed_defaults_empty(monkeypatch):
    """A response with no 'removed' key yields an empty list, not an error (#504)."""

    async def fake_completion(messages, temperature=0.0, max_tokens=512):
        return {
            "choices": [
                {"message": {"content": '{"label": "tree", "confidence": 0.9}'}}
            ]
        }

    monkeypatch.setattr(m, "generate_completion", fake_completion)
    client = TestClient(m.app)
    resp = client.post(
        "/name-cluster",
        json={"members": [{"name": "oak", "id": "n1"}]},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["removed"] == []


def test_name_cluster_malformed_removed_degrades_not_500(monkeypatch):
    """A non-list 'removed' (here an int) must not crash the mapping (greptile):
    the label is still valid, so degrade to no outliers rather than a 500."""

    async def fake_completion(messages, temperature=0.0, max_tokens=512):
        # `removed` is a bare int -> not iterable. Pre-fix this raised a 500.
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"label": "tree", "confidence": 0.9, '
                        '"removed": 7}'
                    }
                }
            ]
        }

    monkeypatch.setattr(m, "generate_completion", fake_completion)
    client = TestClient(m.app)
    resp = client.post(
        "/name-cluster",
        json={"members": [{"name": "oak", "id": "n1"}]},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["label"] == "tree"
    assert body["removed"] == []


def test_name_cluster_suggests_graft_parent(monkeypatch):
    """On a coined-new label, Hermes returns a `parent` drawn from the
    candidates so Sophia can graft the new type under it (concept/process
    population)."""

    async def fake_completion(messages, temperature=0.0, max_tokens=512):
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"label": "calculus", "confidence": 0.9, '
                        '"parent": "concept"}'
                    }
                }
            ]
        }

    monkeypatch.setattr(m, "generate_completion", fake_completion)
    client = TestClient(m.app)
    resp = client.post(
        "/name-cluster",
        json={
            "members": [{"name": "derivative", "id": "n1"}],
            "candidates": ["object", "concept"],
        },
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["parent"] == "concept"


def test_name_cluster_rejects_unknown_parent(monkeypatch):
    """A `parent` not among the supplied candidates degrades to None
    (closed-world), so Sophia never grafts onto a dangling target."""

    async def fake_completion(messages, temperature=0.0, max_tokens=512):
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"label": "calculus", "parent": "mathematics"}'
                    }
                }
            ]
        }

    monkeypatch.setattr(m, "generate_completion", fake_completion)
    client = TestClient(m.app)
    resp = client.post(
        "/name-cluster",
        json={"members": [{"name": "x", "id": "n1"}], "candidates": ["concept"]},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["parent"] is None

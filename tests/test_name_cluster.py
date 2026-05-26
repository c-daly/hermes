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

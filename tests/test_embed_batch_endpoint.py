"""Tests for POST /embed_text_batch — the rate-safe bulk embedding path.

One provider call (OpenAI batch) for many texts instead of N single
/embed_text round-trips. generate_embeddings_batch is monkeypatched; no
live provider calls.
"""

from __future__ import annotations

import hermes.main as m
from fastapi.testclient import TestClient

client = TestClient(m.app)


def _fake_batch(n_dim=4):
    async def fake(texts, model_name="default", **kwargs):
        return [
            {
                "embedding": [float(i)] * n_dim,
                "dimension": n_dim,
                "model": "test-model",
                "embedding_id": f"id-{i}",
            }
            for i, _ in enumerate(texts)
        ]

    return fake


def test_batch_embeds_in_order(monkeypatch):
    fake = _fake_batch()
    monkeypatch.setattr(m, "generate_embeddings_batch", fake)
    resp = client.post("/embed_text_batch", json={"texts": ["CARRIES", "HAULS", "DRAGS"]})
    assert resp.status_code == 200
    embeddings = resp.json()["embeddings"]
    assert len(embeddings) == 3
    assert embeddings[0]["embedding"] == [0.0, 0.0, 0.0, 0.0]
    assert embeddings[1]["embedding"] == [1.0, 1.0, 1.0, 1.0]
    assert embeddings[2]["dimension"] == 4


def test_single_provider_call_for_the_whole_batch(monkeypatch):
    calls = {"n": 0}

    async def fake(texts, model_name="default", **kwargs):
        calls["n"] += 1
        return [
            {"embedding": [0.0], "dimension": 1, "model": "m", "embedding_id": "x"}
            for _ in texts
        ]

    monkeypatch.setattr(m, "generate_embeddings_batch", fake)
    client.post("/embed_text_batch", json={"texts": ["A", "B", "C", "D", "E"]})
    assert calls["n"] == 1  # one batch call, not five


def test_empty_texts_returns_empty(monkeypatch):
    monkeypatch.setattr(m, "generate_embeddings_batch", _fake_batch())
    resp = client.post("/embed_text_batch", json={"texts": []})
    assert resp.status_code == 200
    assert resp.json()["embeddings"] == []


def test_blank_entry_rejected(monkeypatch):
    monkeypatch.setattr(m, "generate_embeddings_batch", _fake_batch())
    resp = client.post("/embed_text_batch", json={"texts": ["CARRIES", "  "]})
    assert resp.status_code == 400


def test_oversize_batch_rejected(monkeypatch):
    monkeypatch.setattr(m, "generate_embeddings_batch", _fake_batch())
    resp = client.post(
        "/embed_text_batch", json={"texts": ["x"] * (m.MAX_EMBED_BATCH + 1)}
    )
    assert resp.status_code == 400


def test_model_passed_through(monkeypatch):
    captured = {}

    async def fake(texts, model_name="default", **kwargs):
        captured["model_name"] = model_name
        return [
            {"embedding": [0.0], "dimension": 1, "model": "m", "embedding_id": "x"}
            for _ in texts
        ]

    monkeypatch.setattr(m, "generate_embeddings_batch", fake)
    client.post(
        "/embed_text_batch", json={"texts": ["A", "B"], "model": "text-embedding-3-large"}
    )
    assert captured["model_name"] == "text-embedding-3-large"

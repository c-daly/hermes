"""Tests for the /relation-synonyms endpoint (hermes#133, H3).

The relation-axis naming pass: the LLM proposes descriptive-relation synonym
groups; the endpoint validates fail-closed (epic #131: no reserved typing
relation in any group). Every test monkeypatches generate_completion; no
live LLM calls. Parse/validation rules are covered in
tests/unit/test_relation_synonyms.py -- here we cover the HTTP contract.
"""

from __future__ import annotations

import json

import hermes.main as m
from fastapi.testclient import TestClient

client = TestClient(m.app)


def _make_completion(content: str):
    async def fake_completion(messages, temperature=0.0, max_tokens=1024, **kwargs):
        assert temperature == 0.0
        fake_completion.last_messages = messages  # type: ignore[attr-defined]
        return {"choices": [{"message": {"content": content}}]}

    return fake_completion


def _post(predicates, context=None):
    body = {"predicates": predicates, "request_id": "r::0"}
    if context is not None:
        body["context"] = context
    return client.post("/relation-synonyms", json=body)


def test_groups_returned_and_canonicalized(monkeypatch):
    content = json.dumps(
        {"groups": [{"canonical": "carries", "members": ["HAULS", "DRAGS", "CARRIES"]}]}
    )
    monkeypatch.setattr(m, "generate_completion", _make_completion(content))
    resp = _post(["HAULS", "DRAGS", "CARRIES", "PRODUCES"])
    assert resp.status_code == 200
    groups = resp.json()["groups"]
    assert len(groups) == 1
    assert groups[0]["canonical"] == "CARRIES"
    assert set(groups[0]["members"]) == {"HAULS", "DRAGS", "CARRIES"}


def test_reserved_relation_group_rejected(monkeypatch):
    content = json.dumps(
        {"groups": [{"canonical": "IS_A", "members": ["IS_A", "SUBTYPE_OF"]}]}
    )
    monkeypatch.setattr(m, "generate_completion", _make_completion(content))
    resp = _post(["IS_A", "SUBTYPE_OF", "PART_OF", "CARRIES"])
    assert resp.status_code == 200
    assert resp.json()["groups"] == []


def test_fewer_than_two_predicates_short_circuits(monkeypatch):
    # must not even call the LLM
    called = {"n": 0}

    async def boom(*a, **k):
        called["n"] += 1
        return {"choices": [{"message": {"content": "{}"}}]}

    monkeypatch.setattr(m, "generate_completion", boom)
    resp = _post(["CARRIES"])
    assert resp.status_code == 200
    assert resp.json()["groups"] == []
    assert called["n"] == 0


def test_reserved_predicates_filtered_from_prompt(monkeypatch):
    fake = _make_completion(json.dumps({"groups": []}))
    monkeypatch.setattr(m, "generate_completion", fake)
    _post(["CARRIES", "HAULS", "IS_A"], context="logistics")
    user = [msg for msg in fake.last_messages if msg["role"] == "user"][0]["content"]
    assert "IS_A" not in user.split("Candidates:")[-1]
    assert "logistics" in user


def test_llm_not_configured_returns_503(monkeypatch):
    from hermes.llm import LLMProviderNotConfiguredError

    async def not_configured(*a, **k):
        raise LLMProviderNotConfiguredError("no key")

    monkeypatch.setattr(m, "generate_completion", not_configured)
    resp = _post(["HAULS", "CARRIES"])
    assert resp.status_code == 503

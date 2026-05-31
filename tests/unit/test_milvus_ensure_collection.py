"""ensure_collection fast-path / dim-resolution guarantees (PR #108).

Two correctness properties pinned here (no live Milvus required — the pymilvus
``Collection``/``utility`` symbols are monkeypatched):

* The cached fast path returns a freshly-fetched server-side ``Collection``
  reference, not the stale Python-side snapshot.
* The provider dimension is resolved exactly once per ``ensure_collection``
  call, so the mismatch check and the created FieldSchema can never disagree.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from hermes import milvus_client as mc


class _FakeCollection:
    """Minimal stand-in exposing the schema shape ensure_collection reads."""

    def __init__(self, name: str, schema: Any = None, dim: int = 1536) -> None:
        self.name = name
        self._dim = dim
        self.schema = SimpleNamespace(
            fields=[SimpleNamespace(name="embedding", params={"dim": dim})]
        )

    def create_index(self, *args: Any, **kwargs: Any) -> None:
        pass


@pytest.fixture(autouse=True)
def _reset_milvus_globals() -> Any:
    """Isolate module-level connection/cache state between tests."""
    saved = (mc._milvus_connected, mc._milvus_collection, mc._collection_name)
    mc._milvus_connected = True
    mc._milvus_collection = None
    mc._collection_name = "unit_embeddings"
    yield
    (mc._milvus_connected, mc._milvus_collection, mc._collection_name) = saved


def test_fast_path_returns_fresh_server_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cached collection at the right dim is re-fetched, not returned as-is."""
    monkeypatch.setattr(mc, "get_embedding_dimension", lambda: 1536)

    constructed: list[_FakeCollection] = []

    def _make(name: str, schema: Any = None) -> _FakeCollection:
        col = _FakeCollection(name, schema, dim=1536)
        constructed.append(col)
        return col

    monkeypatch.setattr(mc, "Collection", _make)

    # Seed the cache with a stale snapshot that already matches the wanted dim.
    stale = _FakeCollection("unit_embeddings", dim=1536)
    mc._milvus_collection = stale

    result = mc.ensure_collection()

    # Fast path hit: must NOT hand back the cached snapshot object.
    assert result is not stale
    assert result is constructed[-1]
    assert mc._milvus_collection is result


def test_dimension_resolved_once_per_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On the create path the provider dim is read once and reused for the schema."""
    calls = {"n": 0}

    def _dim() -> int:
        calls["n"] += 1
        return 1536

    monkeypatch.setattr(mc, "get_embedding_dimension", _dim)
    monkeypatch.setattr(mc.utility, "has_collection", lambda name: False)

    captured: dict[str, Any] = {}

    def _make(name: str, schema: Any = None) -> _FakeCollection:
        # The created FieldSchema dim must equal the resolved dim.
        for field in schema.fields:
            if field.name == "embedding":
                captured["dim"] = field.dim
        return _FakeCollection(name, schema, dim=captured.get("dim", 1536))

    monkeypatch.setattr(mc, "Collection", _make)

    result = mc.ensure_collection()

    assert result is not None
    assert calls["n"] == 1
    assert captured["dim"] == 1536

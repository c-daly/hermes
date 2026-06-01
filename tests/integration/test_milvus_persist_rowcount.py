"""persist_embedding must actually write rows: persisting N embeddings grows the
collection by exactly N (logos#535 hermes-side acceptance, hermes#111).

This guards the keystone regression — "extracted 18 entities, persisted 0
embeddings" — where a stale-dim / wrongly-shaped insert made every
``persist_embedding`` silently fail. A unit test with a mocked collection cannot
catch that; the row count has to be read back from a real Milvus.

Integration test — needs a reachable Milvus (dev stack on localhost:19530).
"""

from __future__ import annotations

import socket
from types import SimpleNamespace

import pytest

import hermes.embedding_provider as embedding_provider
import hermes.milvus_client as mc

# Small, model-free dimension: this test is about row persistence, not embedding
# quality, so we patch the provider rather than load sentence-transformers.
_DIM = 8


def _milvus_up(host: str = "localhost", port: int = 19530) -> bool:
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


@pytest.mark.integration
@pytest.mark.skipif(not _milvus_up(), reason="dev Milvus not reachable on :19530")
async def test_persist_embedding_grows_row_count_by_n(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pymilvus import utility

    name = "test_111_persist_rowcount"
    monkeypatch.setenv("MILVUS_HOST", "localhost")
    monkeypatch.setenv("MILVUS_PORT", "19530")
    monkeypatch.setenv("MILVUS_COLLECTION_NAME", name)
    monkeypatch.delenv("LOGOS_EMBEDDING_DIM", raising=False)

    # Provider dimension is the source of truth for the collection schema; pin it
    # to a fake so the test neither loads a model nor depends on the live dim.
    monkeypatch.setattr(
        embedding_provider,
        "get_embedding_provider",
        lambda: SimpleNamespace(dimension=_DIM, model_name="fake-persist"),
    )

    # Reset module connection/cache globals so the patched env/provider take effect.
    mc._milvus_connected = False
    mc._milvus_collection = None
    mc._collection_name = None
    mc._milvus_host = None
    mc._milvus_port = None

    assert mc.connect_milvus(), "could not connect to dev Milvus"

    # Start from a clean collection so the row delta is unambiguous.
    if utility.has_collection(name):
        utility.drop_collection(name)
    mc._milvus_collection = None

    try:
        collection = mc.ensure_collection()
        assert collection is not None

        # Insert shape must match the schema dim (the num_rows-mismatch guard).
        schema_dim = mc._embedding_field_dim(collection)
        assert schema_dim == _DIM, f"collection dim {schema_dim} != provider {_DIM}"

        collection.flush()
        baseline = collection.num_entities
        assert baseline == 0, (
            f"fresh collection should start empty, got {baseline} "
            "(drop_collection may have failed or stale data exists)"
        )

        n = 5
        for i in range(n):
            ok = await mc.persist_embedding(
                embedding_id=f"emb-{i}",
                embedding=[0.1] * schema_dim,
                model="fake-persist",
                text=f"persist regression row {i}",
            )
            assert ok is True, f"persist_embedding returned False for row {i}"

        # num_entities only reflects sealed segments, so flush before counting.
        collection.flush()
        assert collection.num_entities == baseline + n

        # Read a row back to confirm the persisted vector carries the schema dim
        # (the "insert shape matches the collection schema dim" half of #111 —
        # guards against a silently truncated / reshaped insert, not just a
        # missing one).
        collection.load()
        rows = collection.query(
            expr='embedding_id == "emb-0"', output_fields=["embedding"]
        )
        assert len(rows) == 1
        assert len(rows[0]["embedding"]) == schema_dim
    finally:
        if utility.has_collection(name):
            utility.drop_collection(name)

"""ensure_collection must recreate a collection whose dim no longer matches the
provider, instead of silently reusing it (logos#535).

Integration test — needs a reachable Milvus (dev stack on localhost:19530).
"""

from __future__ import annotations

import socket
from types import SimpleNamespace

import pytest

import hermes.embedding_provider as embedding_provider
import hermes.milvus_client as mc


def _milvus_up(host: str = "localhost", port: int = 19530) -> bool:
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


@pytest.mark.skipif(not _milvus_up(), reason="dev Milvus not reachable on :19530")
def test_ensure_collection_recreates_on_dim_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pymilvus import (
        Collection,
        CollectionSchema,
        DataType,
        FieldSchema,
        utility,
    )

    name = "test_535_dim_recreate"
    monkeypatch.setenv("MILVUS_HOST", "localhost")
    monkeypatch.setenv("MILVUS_PORT", "19530")
    monkeypatch.setenv("MILVUS_COLLECTION_NAME", name)
    monkeypatch.delenv("LOGOS_EMBEDDING_DIM", raising=False)

    # reset module connection/cache globals
    mc._milvus_connected = False
    mc._milvus_collection = None
    mc._collection_name = None
    mc._milvus_host = None
    mc._milvus_port = None

    assert mc.connect_milvus(), "could not connect to dev Milvus"

    # Seed a STALE 384-dim collection (the pre-fix state).
    if utility.has_collection(name):
        utility.drop_collection(name)
    Collection(
        name=name,
        schema=CollectionSchema(
            [
                FieldSchema("embedding_id", DataType.VARCHAR, is_primary=True, max_length=64),
                FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=384),
                FieldSchema("model", DataType.VARCHAR, max_length=256),
                FieldSchema("text", DataType.VARCHAR, max_length=65535),
                FieldSchema("timestamp", DataType.INT64),
            ]
        ),
    )

    try:
        # Provider now declares 1536 — ensure_collection must recreate, not reuse.
        monkeypatch.setattr(
            embedding_provider,
            "get_embedding_provider",
            lambda: SimpleNamespace(dimension=1536, model_name="fake-1536"),
        )
        mc._milvus_collection = None

        collection = mc.ensure_collection()

        assert collection is not None
        assert mc._embedding_field_dim(collection) == 1536
    finally:
        if utility.has_collection(name):
            utility.drop_collection(name)

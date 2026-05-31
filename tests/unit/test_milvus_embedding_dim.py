"""The embedding collection dimension must come from the live provider.

Regression for logos#535: Hermes auto-selects the OpenAI provider (1536-dim)
when a key is present, but the collection was hardcoded to 384 -> every insert
failed with a num_rows mismatch and embeddings were silently dropped.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import hermes.embedding_provider as embedding_provider
from hermes import milvus_client


class TestEmbeddingDimensionFromProvider:
    def test_dimension_comes_from_provider_not_hardcoded_384(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A 1536-dim provider yields collection dim 1536, not the old 384."""
        monkeypatch.delenv("LOGOS_EMBEDDING_DIM", raising=False)
        monkeypatch.setattr(
            embedding_provider,
            "get_embedding_provider",
            lambda: SimpleNamespace(dimension=1536, model_name="fake-1536"),
        )
        assert milvus_client.get_embedding_dimension() == 1536

    def test_override_disagreeing_with_provider_fails_loud(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """LOGOS_EMBEDDING_DIM that the provider can't deliver raises, never silent."""
        from logos_config import EmbeddingDimMismatch

        monkeypatch.setenv("LOGOS_EMBEDDING_DIM", "512")
        monkeypatch.setattr(
            embedding_provider,
            "get_embedding_provider",
            lambda: SimpleNamespace(dimension=384, model_name="fake-384"),
        )
        with pytest.raises(EmbeddingDimMismatch):
            milvus_client.get_embedding_dimension()

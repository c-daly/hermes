"""Tests for embedding_provider module."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np


pytestmark = pytest.mark.unit


class TestSentenceTransformerProvider:
    """Tests for the local sentence-transformers provider."""

    def test_model_name(self):
        from hermes.embedding_provider import SentenceTransformerProvider

        provider = SentenceTransformerProvider(model="test-model")
        assert provider.model_name == "test-model"

    def test_default_model_name(self):
        from hermes.embedding_provider import SentenceTransformerProvider

        provider = SentenceTransformerProvider()
        assert provider.model_name == "all-MiniLM-L6-v2"

    def test_dimension_loads_model(self):
        from hermes.embedding_provider import SentenceTransformerProvider

        provider = SentenceTransformerProvider()
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        provider._model = mock_model

        assert provider.dimension == 384
        mock_model.get_sentence_embedding_dimension.assert_called_once()

    @pytest.mark.asyncio
    async def test_embed(self):
        from hermes.embedding_provider import SentenceTransformerProvider

        provider = SentenceTransformerProvider()
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
        provider._model = mock_model

        result = await provider.embed("hello")
        assert result == [0.1, 0.2, 0.3]


class TestOpenAIEmbeddingProvider:
    """Tests for the OpenAI API provider."""

    def test_properties(self):
        from hermes.embedding_provider import OpenAIEmbeddingProvider

        provider = OpenAIEmbeddingProvider(api_key="sk-test")
        assert provider.model_name == "text-embedding-3-small"
        assert provider.dimension == 1536

    def test_custom_dimensions(self):
        from hermes.embedding_provider import OpenAIEmbeddingProvider

        provider = OpenAIEmbeddingProvider(api_key="sk-test", dimensions=256)
        assert provider.dimension == 256

    def test_ada_002_dimension(self):
        from hermes.embedding_provider import OpenAIEmbeddingProvider

        provider = OpenAIEmbeddingProvider(
            api_key="sk-test", model="text-embedding-ada-002"
        )
        assert provider.dimension == 1536

    def test_large_model_dimension(self):
        from hermes.embedding_provider import OpenAIEmbeddingProvider

        provider = OpenAIEmbeddingProvider(
            api_key="sk-test", model="text-embedding-3-large"
        )
        assert provider.dimension == 3072

    @pytest.mark.asyncio
    async def test_embed_calls_api(self):
        from hermes.embedding_provider import OpenAIEmbeddingProvider

        provider = OpenAIEmbeddingProvider(
            api_key="sk-test", base_url="https://api.example.com/v1"
        )

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "data": [{"embedding": [0.1, 0.2, 0.3]}]
        }

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("hermes.embedding_provider.httpx.AsyncClient", return_value=mock_client):
            result = await provider.embed("hello")

        assert result == [0.1, 0.2, 0.3]
        mock_client.post.assert_called_once()
        call_kwargs = mock_client.post.call_args
        assert "dimensions" in call_kwargs.kwargs.get("json", call_kwargs[1].get("json", {}))

    @pytest.mark.asyncio
    async def test_embed_ada_no_dimensions_param(self):
        from hermes.embedding_provider import OpenAIEmbeddingProvider

        provider = OpenAIEmbeddingProvider(
            api_key="sk-test", model="text-embedding-ada-002"
        )

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "data": [{"embedding": [0.5]}]
        }

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("hermes.embedding_provider.httpx.AsyncClient", return_value=mock_client):
            result = await provider.embed("hello")

        # ada-002 should NOT pass dimensions param
        call_kwargs = mock_client.post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json", {})
        assert "dimensions" not in payload


class TestDetectBackend:
    """Tests for _detect_backend and get_embedding_provider factory."""

    def test_explicit_provider(self):
        import hermes.embedding_provider as mod

        with patch.object(mod, "get_env_value", side_effect=lambda k, **kw: "openai" if k == "EMBEDDING_PROVIDER" else None):
            assert mod._detect_backend() == "openai"

    def test_auto_detect_openai_with_key(self):
        import hermes.embedding_provider as mod

        def fake_env(k, **kw):
            if k == "EMBEDDING_PROVIDER":
                return None
            if k in ("HERMES_LLM_API_KEY", "OPENAI_API_KEY"):
                return "sk-test"
            return kw.get("default")

        with patch.object(mod, "get_env_value", side_effect=fake_env):
            assert mod._detect_backend() == "openai"

    def test_auto_detect_sentence_transformers_no_key(self):
        import hermes.embedding_provider as mod

        with patch.object(mod, "get_env_value", return_value=None):
            assert mod._detect_backend() == "sentence-transformers"

    def test_get_embedding_provider_openai(self):
        import hermes.embedding_provider as mod

        # Reset singleton
        mod._provider = None

        def fake_env(k, **kw):
            if k == "EMBEDDING_PROVIDER":
                return "openai"
            if k in ("HERMES_LLM_API_KEY", "OPENAI_API_KEY"):
                return "sk-test"
            return kw.get("default")

        with patch.object(mod, "get_env_value", side_effect=fake_env):
            provider = mod.get_embedding_provider()
            assert isinstance(provider, mod.OpenAIEmbeddingProvider)

        # Cleanup
        mod._provider = None

    def test_get_embedding_provider_sentence_transformers(self):
        import hermes.embedding_provider as mod

        mod._provider = None

        def fake_env(k, **kw):
            if k == "EMBEDDING_PROVIDER":
                return "sentence-transformers"
            return kw.get("default")

        with patch.object(mod, "get_env_value", side_effect=fake_env):
            provider = mod.get_embedding_provider()
            assert isinstance(provider, mod.SentenceTransformerProvider)

        mod._provider = None

    def test_get_embedding_provider_unknown_raises(self):
        import hermes.embedding_provider as mod

        mod._provider = None

        with patch.object(mod, "get_env_value", side_effect=lambda k, **kw: "unknown" if k == "EMBEDDING_PROVIDER" else None):
            with pytest.raises(ValueError, match="Unknown EMBEDDING_PROVIDER"):
                mod.get_embedding_provider()

        mod._provider = None

    def test_get_embedding_provider_openai_no_key_raises(self):
        import hermes.embedding_provider as mod

        mod._provider = None

        def fake_env(k, **kw):
            if k == "EMBEDDING_PROVIDER":
                return "openai"
            return None

        with patch.object(mod, "get_env_value", side_effect=fake_env):
            with pytest.raises(RuntimeError, match="no API key found"):
                mod.get_embedding_provider()

        mod._provider = None

    def test_singleton_returns_cached(self):
        import hermes.embedding_provider as mod

        sentinel = MagicMock()
        mod._provider = sentinel
        assert mod.get_embedding_provider() is sentinel
        mod._provider = None

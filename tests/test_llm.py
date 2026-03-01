"""Tests for hermes.llm -- LLM provider abstraction.

Covers:
- EchoProvider generation (deterministic, metadata scenario label)
- OpenAIProvider generation (mocked httpx)
- Provider selection (_get_provider, get_default_provider_name)
- generate_completion orchestration and error paths
- _normalize_choices edge cases
- _estimate_usage token estimation
- llm_service_health reporting
"""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from hermes.llm import (
    BaseLLMProvider,
    EchoProvider,
    LLMProviderError,
    LLMProviderNotConfiguredError,
    LLMProviderResponseError,
    OpenAIProvider,
    _estimate_usage,
    _get_provider,
    _normalize_choices,
    generate_completion,
    get_default_provider_name,
    llm_service_health,
    _PROVIDER_CACHE,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_provider_cache():
    """Clear the provider cache before each test for isolation."""
    _PROVIDER_CACHE.clear()
    yield
    _PROVIDER_CACHE.clear()


# ---------------------------------------------------------------------------
# EchoProvider
# ---------------------------------------------------------------------------


class TestEchoProvider:
    async def test_basic_echo(self):
        provider = EchoProvider()
        result = await provider.generate(
            messages=[{"role": "user", "content": "hello"}]
        )
        assert result["provider"] == "echo"
        assert result["model"] == "echo-stub"
        assert len(result["choices"]) == 1
        assert result["choices"][0]["message"]["role"] == "assistant"
        assert "[echo]" in result["choices"][0]["message"]["content"]
        assert "hello" in result["choices"][0]["message"]["content"]
        assert result["id"].startswith("echo-")
        assert "usage" in result

    async def test_echo_with_metadata_scenario(self):
        provider = EchoProvider()
        result = await provider.generate(
            messages=[{"role": "user", "content": "test"}],
            metadata={"scenario": "ner_extraction"},
        )
        content = result["choices"][0]["message"]["content"]
        assert "[ner_extraction]" in content

    async def test_echo_with_metadata_no_scenario(self):
        provider = EchoProvider()
        result = await provider.generate(
            messages=[{"role": "user", "content": "test"}],
            metadata={"other_key": "value"},
        )
        content = result["choices"][0]["message"]["content"]
        # No scenario label should be injected
        assert "[echo]" in content
        assert "]" in content  # just the echo prefix bracket

    async def test_echo_empty_messages(self):
        provider = EchoProvider()
        result = await provider.generate(messages=[])
        content = result["choices"][0]["message"]["content"]
        assert "(empty prompt)" in content

    async def test_echo_custom_model(self):
        provider = EchoProvider(default_model="custom-echo")
        result = await provider.generate(
            messages=[{"role": "user", "content": "hi"}],
            model="override-model",
        )
        assert result["model"] == "override-model"

    async def test_echo_default_model_fallback(self):
        provider = EchoProvider()
        result = await provider.generate(
            messages=[{"role": "user", "content": "hi"}],
        )
        assert result["model"] == "echo-stub"

    async def test_echo_multi_message_transcript(self):
        provider = EchoProvider()
        result = await provider.generate(
            messages=[
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "hello"},
            ]
        )
        content = result["choices"][0]["message"]["content"]
        assert "system: You are helpful" in content
        assert "user: hello" in content

    async def test_echo_raw_field(self):
        provider = EchoProvider()
        result = await provider.generate(
            messages=[{"role": "user", "content": "hi"}],
            metadata={"key": "val"},
        )
        assert result["raw"]["echo"] is True
        assert result["raw"]["metadata"] == {"key": "val"}

    async def test_echo_no_metadata(self):
        provider = EchoProvider()
        result = await provider.generate(
            messages=[{"role": "user", "content": "hi"}],
        )
        assert result["raw"]["metadata"] == {}


# ---------------------------------------------------------------------------
# OpenAIProvider
# ---------------------------------------------------------------------------


class TestOpenAIProvider:
    async def test_successful_generation(self):
        mock_response = httpx.Response(
            200,
            json={
                "id": "chatcmpl-123",
                "model": "gpt-4o-mini",
                "created": 1700000000,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello!"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            },
        )
        with patch("hermes.llm.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            provider = OpenAIProvider(
                api_key="test-key",
                base_url="https://api.openai.com/v1",
                default_model="gpt-4o-mini",
            )
            result = await provider.generate(
                messages=[{"role": "user", "content": "hi"}],
                temperature=0.5,
                max_tokens=100,
            )

            assert result["provider"] == "openai"
            assert result["model"] == "gpt-4o-mini"
            assert result["choices"][0]["message"]["content"] == "Hello!"
            assert result["usage"]["total_tokens"] == 15

    async def test_http_status_error(self):
        mock_request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        mock_response = httpx.Response(
            429,
            text="Rate limit exceeded",
            request=mock_request,
        )
        with patch("hermes.llm.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.post.side_effect = httpx.HTTPStatusError(
                "Rate limit", request=mock_request, response=mock_response
            )
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            provider = OpenAIProvider(
                api_key="test-key",
                base_url="https://api.openai.com/v1",
                default_model="gpt-4o-mini",
            )
            with pytest.raises(LLMProviderResponseError):
                await provider.generate(
                    messages=[{"role": "user", "content": "hi"}]
                )

    async def test_http_general_error(self):
        with patch("hermes.llm.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.ConnectError("Connection refused")
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            provider = OpenAIProvider(
                api_key="test-key",
                base_url="https://api.openai.com/v1",
                default_model="gpt-4o-mini",
            )
            with pytest.raises(LLMProviderError):
                await provider.generate(
                    messages=[{"role": "user", "content": "hi"}]
                )

    async def test_empty_choices_fallback(self):
        mock_response = httpx.Response(
            200,
            json={
                "id": "chatcmpl-123",
                "model": "gpt-4o-mini",
                "created": 1700000000,
                "choices": [],
            },
        )
        with patch("hermes.llm.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            provider = OpenAIProvider(
                api_key="test-key",
                base_url="https://api.openai.com/v1",
                default_model="gpt-4o-mini",
            )
            result = await provider.generate(
                messages=[{"role": "user", "content": "hi"}]
            )
            # Should get fallback empty choice
            assert len(result["choices"]) == 1
            assert result["choices"][0]["message"]["content"] == ""

    async def test_base_url_trailing_slash_stripped(self):
        provider = OpenAIProvider(
            api_key="test-key",
            base_url="https://api.openai.com/v1/",
            default_model="gpt-4o-mini",
        )
        assert provider.base_url == "https://api.openai.com/v1"

    async def test_default_model_override(self):
        mock_response = httpx.Response(
            200,
            json={
                "id": "chatcmpl-123",
                "model": "gpt-4",
                "created": 1700000000,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hi"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )
        with patch("hermes.llm.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            provider = OpenAIProvider(
                api_key="k", base_url="https://api.openai.com/v1", default_model="gpt-4"
            )
            result = await provider.generate(
                messages=[{"role": "user", "content": "hi"}],
                model=None,  # should use default
            )
            assert result["model"] == "gpt-4"


# ---------------------------------------------------------------------------
# BaseLLMProvider
# ---------------------------------------------------------------------------


class TestBaseLLMProvider:
    async def test_generate_not_implemented(self):
        provider = BaseLLMProvider()
        with pytest.raises(NotImplementedError):
            await provider.generate(messages=[])

    def test_default_model(self):
        provider = BaseLLMProvider()
        assert provider.default_model == "default"

    def test_custom_default_model(self):
        provider = BaseLLMProvider(default_model="my-model")
        assert provider.default_model == "my-model"


# ---------------------------------------------------------------------------
# Provider selection
# ---------------------------------------------------------------------------


class TestProviderSelection:
    def test_get_default_provider_echo_when_no_config(self):
        with patch("hermes.llm.get_env_value", return_value=None):
            assert get_default_provider_name() == "echo"

    def test_get_default_provider_from_env(self):
        def mock_env(key, **kwargs):
            if key == "HERMES_LLM_PROVIDER":
                return "openai"
            return None

        with patch("hermes.llm.get_env_value", side_effect=mock_env):
            assert get_default_provider_name() == "openai"

    def test_get_default_provider_openai_via_api_key(self):
        def mock_env(key, **kwargs):
            if key == "HERMES_LLM_PROVIDER":
                return None
            if key == "HERMES_LLM_API_KEY":
                return "sk-test"
            return None

        with patch("hermes.llm.get_env_value", side_effect=mock_env):
            assert get_default_provider_name() == "openai"

    def test_get_default_provider_empty_string_falls_to_echo(self):
        def mock_env(key, **kwargs):
            if key == "HERMES_LLM_PROVIDER":
                return "  "  # whitespace only
            return None

        with patch("hermes.llm.get_env_value", side_effect=mock_env):
            assert get_default_provider_name() == "echo"

    def test_get_provider_echo(self):
        with patch("hermes.llm.get_env_value", return_value=None):
            provider = _get_provider("echo")
            assert provider is not None
            assert isinstance(provider, EchoProvider)

    def test_get_provider_mock_alias(self):
        with patch("hermes.llm.get_env_value", return_value=None):
            provider = _get_provider("mock")
            assert isinstance(provider, EchoProvider)

    def test_get_provider_unknown_returns_none(self):
        provider = _get_provider("nonexistent_provider")
        assert provider is None

    def test_get_provider_openai_no_key_returns_none(self):
        with patch("hermes.llm.get_env_value", return_value=None):
            provider = _get_provider("openai")
            assert provider is None

    def test_get_provider_openai_with_key(self):
        def mock_env(key, **kwargs):
            if key == "HERMES_LLM_API_KEY":
                return "sk-test"
            if key == "OPENAI_API_KEY":
                return None
            if key == "HERMES_LLM_BASE_URL":
                return None
            if key == "HERMES_LLM_MODEL":
                return None
            return kwargs.get("default")

        with patch("hermes.llm.get_env_value", side_effect=mock_env):
            provider = _get_provider("openai")
            assert isinstance(provider, OpenAIProvider)

    def test_provider_caching(self):
        with patch("hermes.llm.get_env_value", return_value=None):
            p1 = _get_provider("echo")
            p2 = _get_provider("echo")
            assert p1 is p2  # Same instance from cache


# ---------------------------------------------------------------------------
# generate_completion
# ---------------------------------------------------------------------------


class TestGenerateCompletion:
    async def test_echo_completion(self):
        with patch("hermes.llm.get_env_value", return_value=None):
            result = await generate_completion(
                messages=[{"role": "user", "content": "hello"}],
                provider_override="echo",
            )
            assert result["provider"] == "echo"

    async def test_not_configured_provider_raises(self):
        with patch("hermes.llm._get_provider", return_value=None):
            with pytest.raises(LLMProviderNotConfiguredError):
                await generate_completion(
                    messages=[{"role": "user", "content": "hi"}],
                    provider_override="nonexistent",
                )

    async def test_empty_messages_raises(self):
        with patch("hermes.llm.get_env_value", return_value=None):
            with pytest.raises(LLMProviderError, match="At least one message"):
                await generate_completion(
                    messages=[],
                    provider_override="echo",
                )

    async def test_provider_override_takes_precedence(self):
        with patch("hermes.llm.get_env_value", return_value=None):
            result = await generate_completion(
                messages=[{"role": "user", "content": "hi"}],
                provider_override="ECHO",  # uppercase should be normalized
            )
            assert result["provider"] == "echo"


# ---------------------------------------------------------------------------
# _normalize_choices
# ---------------------------------------------------------------------------


class TestNormalizeChoices:
    def test_empty_choices(self):
        assert _normalize_choices(None) == []
        assert _normalize_choices([]) == []

    def test_standard_choice(self):
        choices = [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hi"},
                "finish_reason": "stop",
            }
        ]
        result = _normalize_choices(choices)
        assert len(result) == 1
        assert result[0]["message"]["content"] == "hi"

    def test_legacy_text_field(self):
        choices = [{"text": "legacy response"}]
        result = _normalize_choices(choices)
        assert result[0]["message"]["content"] == "legacy response"
        assert result[0]["message"]["role"] == "assistant"

    def test_missing_message_and_text(self):
        choices = [{"index": 0}]
        result = _normalize_choices(choices)
        assert result[0]["message"]["content"] == ""

    def test_index_defaults(self):
        choices = [
            {"message": {"role": "assistant", "content": "a"}},
            {"message": {"role": "assistant", "content": "b"}},
        ]
        result = _normalize_choices(choices)
        assert result[0]["index"] == 0
        assert result[1]["index"] == 1

    def test_finish_reason_default(self):
        choices = [{"message": {"role": "assistant", "content": "hi"}}]
        result = _normalize_choices(choices)
        assert result[0]["finish_reason"] == "stop"


# ---------------------------------------------------------------------------
# _estimate_usage
# ---------------------------------------------------------------------------


class TestEstimateUsage:
    def test_basic_estimation(self):
        usage = _estimate_usage("hello world", "hi there")
        assert usage["prompt_tokens"] == max(1, len("hello world") // 4)
        assert usage["completion_tokens"] == max(1, len("hi there") // 4)
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]

    def test_empty_prompt(self):
        usage = _estimate_usage("", "response")
        assert usage["prompt_tokens"] == 1  # min 1

    def test_short_text(self):
        usage = _estimate_usage("hi", "ok")
        assert usage["prompt_tokens"] >= 1
        assert usage["completion_tokens"] >= 1


# ---------------------------------------------------------------------------
# llm_service_health
# ---------------------------------------------------------------------------


class TestLLMServiceHealth:
    def test_echo_health(self):
        with patch("hermes.llm.get_env_value", return_value=None):
            health = llm_service_health()
            assert health["default_provider"] == "echo"
            assert health["configured"] is True
            assert health["providers"]["echo"] is True

    def test_openai_configured(self):
        def mock_env(key, **kwargs):
            if key == "HERMES_LLM_PROVIDER":
                return "openai"
            if key in ("HERMES_LLM_API_KEY", "OPENAI_API_KEY"):
                return "sk-test"
            return None

        with patch("hermes.llm.get_env_value", side_effect=mock_env):
            health = llm_service_health()
            assert health["default_provider"] == "openai"
            assert health["configured"] is True

    def test_unknown_provider_not_configured(self):
        def mock_env(key, **kwargs):
            if key == "HERMES_LLM_PROVIDER":
                return "unknown"
            return None

        with patch("hermes.llm.get_env_value", side_effect=mock_env):
            health = llm_service_health()
            assert health["configured"] is False

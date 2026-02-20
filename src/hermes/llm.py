"""LLM provider abstraction for Hermes' `/llm` endpoint."""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, List, Optional

import httpx
from logos_config import get_env_value

logger = logging.getLogger(__name__)


class LLMProviderError(Exception):
    """Base class for provider errors."""


class LLMProviderNotConfiguredError(LLMProviderError):
    """Raised when the requested provider is not configured."""


class LLMProviderResponseError(LLMProviderError):
    """Raised when a provider returns an error response."""


class BaseLLMProvider:
    """Abstract base class for LLM providers."""

    name = "base"

    def __init__(self, default_model: Optional[str] = None) -> None:
        self.default_model = default_model or "default"

    async def generate(
        self,
        *,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError


class EchoProvider(BaseLLMProvider):
    """Fallback provider that deterministically echoes the prompt."""

    name = "echo"

    async def generate(
        self,
        *,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        del temperature, max_tokens  # Unused but kept for parity with other providers
        model_name = model or self.default_model or "echo-stub"
        transcript = "\n".join(
            f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in messages
        ).strip()
        if not transcript:
            transcript = "(empty prompt)"

        context_label = ""
        if metadata:
            scenario = metadata.get("scenario")
            if scenario:
                context_label = f"[{scenario}] "

        response_text = f"[echo] {context_label}{transcript}"
        created_ts = int(time.time())
        usage = _estimate_usage(transcript, response_text)

        return {
            "id": f"echo-{uuid.uuid4().hex}",
            "provider": self.name,
            "model": model_name,
            "created": created_ts,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": response_text},
                    "finish_reason": "stop",
                }
            ],
            "usage": usage,
            "raw": {
                "echo": True,
                "metadata": metadata or {},
            },
        }


class OpenAIProvider(BaseLLMProvider):
    """OpenAI Chat Completions provider."""

    name = "openai"

    def __init__(self, api_key: str, base_url: str, default_model: str) -> None:
        super().__init__(default_model or "gpt-4o-mini")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = httpx.Timeout(60.0)

    async def generate(
        self,
        *,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        del metadata  # Not used directly, reserved for future routing hints
        payload: Dict[str, Any] = {
            "model": model or self.default_model,
            "messages": messages,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text
            logger.error("OpenAI provider error: %s", detail)
            raise LLMProviderResponseError(f"OpenAI provider error: {detail}") from exc
        except httpx.HTTPError as exc:
            logger.error("OpenAI request failed: %s", str(exc))
            raise LLMProviderError(f"OpenAI request failed: {str(exc)}") from exc

        data = response.json()
        normalized_choices = _normalize_choices(data.get("choices"))
        if not normalized_choices:
            normalized_choices = [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": ""},
                    "finish_reason": "stop",
                }
            ]

        return {
            "id": data.get("id", f"openai-{uuid.uuid4().hex}"),
            "provider": self.name,
            "model": data.get("model", payload["model"]),
            "created": data.get("created", int(time.time())),
            "choices": normalized_choices,
            "usage": data.get("usage"),
            "raw": data,
        }


def get_default_provider_name() -> str:
    configured = get_env_value("HERMES_LLM_PROVIDER")
    if configured:
        return configured.strip().lower() or "echo"
    # Fall back to OpenAI automatically if an API key is present
    has_openai_key = get_env_value("HERMES_LLM_API_KEY") or get_env_value(
        "OPENAI_API_KEY"
    )
    if has_openai_key:
        return "openai"
    return "echo"


async def generate_completion(
    *,
    messages: List[Dict[str, str]],
    provider_override: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generate a completion using the configured provider."""
    provider_name = (provider_override or get_default_provider_name()).strip().lower()
    provider = _get_provider(provider_name)
    if provider is None:
        raise LLMProviderNotConfiguredError(
            f"LLM provider '{provider_name}' is not configured. "
            "Set HERMES_LLM_PROVIDER and related credentials."
        )

    if not messages:
        raise LLMProviderError("At least one message is required to call the LLM.")

    return await provider.generate(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        metadata=metadata,
    )


def llm_service_health() -> Dict[str, Any]:
    """Return configuration metadata for health responses."""
    default_provider = get_default_provider_name()
    providers = {
        "echo": True,
        "openai": bool(
            get_env_value("HERMES_LLM_API_KEY") or get_env_value("OPENAI_API_KEY")
        ),
    }
    configured = providers.get(default_provider, False)
    return {
        "default_provider": default_provider,
        "configured": configured,
        "providers": providers,
    }


_PROVIDER_CACHE: Dict[str, BaseLLMProvider] = {}


def _get_provider(name: str) -> Optional[BaseLLMProvider]:
    normalized = name.lower()
    if normalized in _PROVIDER_CACHE:
        return _PROVIDER_CACHE[normalized]

    provider: Optional[BaseLLMProvider]
    if normalized in {"echo", "mock"}:
        provider = EchoProvider(
            default_model=get_env_value("HERMES_LLM_MODEL", default="echo-stub")
            or "echo-stub"
        )
    elif normalized == "openai":
        provider = _build_openai_provider()
    else:
        provider = None

    if provider is not None:
        _PROVIDER_CACHE[normalized] = provider
    return provider


def _build_openai_provider() -> Optional[OpenAIProvider]:
    api_key = get_env_value("HERMES_LLM_API_KEY") or get_env_value("OPENAI_API_KEY")
    if not api_key:
        logger.warning(
            "OpenAI provider requested but no API key found in HERMES_LLM_API_KEY or OPENAI_API_KEY."
        )
        return None
    base_url = (
        get_env_value("HERMES_LLM_BASE_URL", default="https://api.openai.com/v1")
        or "https://api.openai.com/v1"
    )
    default_model = (
        get_env_value("HERMES_LLM_MODEL", default="gpt-4o-mini") or "gpt-4o-mini"
    )
    return OpenAIProvider(
        api_key=api_key, base_url=base_url, default_model=default_model
    )


def _normalize_choices(choices: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    if not choices:
        return []
    normalized: List[Dict[str, Any]] = []
    for idx, choice in enumerate(choices):
        message = choice.get("message")
        if not message and "text" in choice:
            message = {"role": "assistant", "content": choice["text"]}
        normalized.append(
            {
                "index": choice.get("index", idx),
                "message": message or {"role": "assistant", "content": ""},
                "finish_reason": choice.get("finish_reason", "stop"),
            }
        )
    return normalized


def _estimate_usage(prompt_text: str, completion_text: str) -> Dict[str, int]:
    """Very rough token estimation for the echo provider."""
    prompt_tokens = max(1, len(prompt_text) // 4) if prompt_text else 1
    completion_tokens = max(1, len(completion_text) // 4)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }

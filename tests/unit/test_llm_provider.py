"""Unit tests for Hermes LLM provider helpers."""

import hermes.llm as llm_module
from hermes.llm import _build_openai_provider, get_default_provider_name  # type: ignore[attr-defined]


def _reset_cache():
    llm_module._PROVIDER_CACHE.clear()  # type: ignore[attr-defined]


def test_default_provider_uses_openai_when_key_present(monkeypatch):
    monkeypatch.delenv("HERMES_LLM_PROVIDER", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    _reset_cache()
    assert get_default_provider_name() == "openai"


def test_default_provider_prefers_explicit_setting(monkeypatch):
    monkeypatch.setenv("HERMES_LLM_PROVIDER", "echo")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    _reset_cache()
    assert get_default_provider_name() == "echo"


def test_build_openai_provider_uses_backup_key(monkeypatch):
    monkeypatch.delenv("HERMES_LLM_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    _reset_cache()
    provider = _build_openai_provider()
    assert provider is not None

"""Tests for ner_provider module."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

pytestmark = pytest.mark.unit


class TestSpacyNERProvider:
    """Tests for the local spaCy NER provider."""

    @pytest.mark.asyncio
    async def test_extract_entities(self):
        from hermes.ner_provider import SpacyNERProvider

        provider = SpacyNERProvider()

        # Create mock spaCy doc with entities
        mock_ent = MagicMock()
        mock_ent.text = "Paris"
        mock_ent.label_ = "GPE"
        mock_ent.start_char = 12
        mock_ent.end_char = 17

        mock_doc = MagicMock()
        mock_doc.ents = [mock_ent]

        mock_nlp = MagicMock(return_value=mock_doc)
        provider._nlp = mock_nlp

        result = await provider.extract_entities("I went to Paris")
        assert len(result) == 1
        assert result[0]["name"] == "Paris"
        assert result[0]["type"] == "location"
        assert result[0]["start"] == 12
        assert result[0]["end"] == 17

    @pytest.mark.asyncio
    async def test_extract_entities_unknown_label(self):
        from hermes.ner_provider import SpacyNERProvider

        provider = SpacyNERProvider()

        mock_ent = MagicMock()
        mock_ent.text = "Something"
        mock_ent.label_ = "UNKNOWN_LABEL"
        mock_ent.start_char = 0
        mock_ent.end_char = 9

        mock_doc = MagicMock()
        mock_doc.ents = [mock_ent]
        provider._nlp = MagicMock(return_value=mock_doc)

        result = await provider.extract_entities("Something happened")
        assert result[0]["type"] == "entity"  # fallback

    @pytest.mark.asyncio
    async def test_extract_entities_multiple(self):
        from hermes.ner_provider import SpacyNERProvider

        provider = SpacyNERProvider()

        ent1 = MagicMock(text="Google", label_="ORG", start_char=0, end_char=6)
        ent2 = MagicMock(text="Mountain View", label_="GPE", start_char=15, end_char=28)

        mock_doc = MagicMock()
        mock_doc.ents = [ent1, ent2]
        provider._nlp = MagicMock(return_value=mock_doc)

        result = await provider.extract_entities("Google is in Mountain View")
        assert len(result) == 2
        assert result[0]["type"] == "entity"  # ORG → entity
        assert result[1]["type"] == "location"  # GPE → location


class TestOpenAINERProvider:
    """Tests for the OpenAI-based NER provider."""

    @pytest.mark.asyncio
    async def test_extract_entities_success(self):
        from hermes.ner_provider import OpenAINERProvider

        provider = OpenAINERProvider()

        mock_result = {
            "choices": [
                {
                    "message": {
                        "content": '{"entities": [{"name": "Paris", "type": "location", "start": 12, "end": 17}]}'
                    }
                }
            ]
        }

        with patch(
            "hermes.llm.generate_completion", new_callable=AsyncMock
        ) as mock_llm:
            mock_llm.return_value = mock_result
            result = await provider.extract_entities("I went to Paris")

        assert len(result) == 1
        assert result[0]["name"] == "Paris"
        assert result[0]["type"] == "location"

    @pytest.mark.asyncio
    async def test_extract_entities_llm_failure(self):
        from hermes.ner_provider import OpenAINERProvider

        provider = OpenAINERProvider()

        with patch(
            "hermes.llm.generate_completion", new_callable=AsyncMock
        ) as mock_llm:
            mock_llm.side_effect = Exception("API error")
            result = await provider.extract_entities("test")

        assert result == []

    @pytest.mark.asyncio
    async def test_extract_entities_empty_response(self):
        from hermes.ner_provider import OpenAINERProvider

        provider = OpenAINERProvider()

        mock_result = {"choices": [{"message": {"content": '{"entities": []}'}}]}

        with patch(
            "hermes.llm.generate_completion", new_callable=AsyncMock
        ) as mock_llm:
            mock_llm.return_value = mock_result
            result = await provider.extract_entities("nothing here")

        assert result == []


class TestParseResponse:
    """Tests for OpenAINERProvider._parse_response."""

    def test_valid_json(self):
        from hermes.ner_provider import OpenAINERProvider

        result = OpenAINERProvider._parse_response(
            '{"entities": [{"name": "Paris", "type": "location", "start": 0, "end": 5}]}',
            "Paris is nice",
        )
        assert len(result) == 1
        assert result[0]["name"] == "Paris"

    def test_json_in_code_fence(self):
        from hermes.ner_provider import OpenAINERProvider

        content = '```json\n{"entities": [{"name": "Bob", "type": "entity", "start": 0, "end": 3}]}\n```'
        result = OpenAINERProvider._parse_response(content, "Bob walked")
        assert len(result) == 1
        assert result[0]["name"] == "Bob"

    def test_invalid_json_returns_empty(self):
        from hermes.ner_provider import OpenAINERProvider

        result = OpenAINERProvider._parse_response("not json at all", "test")
        assert result == []

    def test_invalid_json_in_code_fence(self):
        from hermes.ner_provider import OpenAINERProvider

        content = "```json\nnot valid json\n```"
        result = OpenAINERProvider._parse_response(content, "test")
        assert result == []

    def test_empty_entity_name_skipped(self):
        from hermes.ner_provider import OpenAINERProvider

        content = '{"entities": [{"name": "", "type": "entity"}]}'
        result = OpenAINERProvider._parse_response(content, "test")
        assert result == []

    def test_unknown_type_defaults_to_entity(self):
        from hermes.ner_provider import OpenAINERProvider

        content = '{"entities": [{"name": "Foo", "type": "invalid_type", "start": 0, "end": 3}]}'
        result = OpenAINERProvider._parse_response(content, "Foo bar")
        assert result[0]["type"] == "entity"

    def test_missing_offsets_found_in_text(self):
        from hermes.ner_provider import OpenAINERProvider

        content = '{"entities": [{"name": "Paris", "type": "location"}]}'
        result = OpenAINERProvider._parse_response(content, "Visit Paris today")
        assert result[0]["start"] == 6
        assert result[0]["end"] == 11

    def test_missing_offsets_not_in_text(self):
        from hermes.ner_provider import OpenAINERProvider

        content = '{"entities": [{"name": "Tokyo", "type": "location"}]}'
        result = OpenAINERProvider._parse_response(content, "Visit Paris")
        # Falls back to (0, len(name))
        assert result[0]["start"] == 0
        assert result[0]["end"] == 5


class TestNERDetectBackend:
    """Tests for _detect_backend and get_ner_provider factory."""

    def test_explicit_provider(self):
        import hermes.ner_provider as mod

        with patch.object(
            mod,
            "get_env_value",
            side_effect=lambda k, **kw: "spacy" if k == "NER_PROVIDER" else None,
        ):
            assert mod._detect_backend() == "spacy"

    def test_auto_detect_combined_with_key(self):
        import hermes.ner_provider as mod

        def fake_env(k, **kw):
            if k == "NER_PROVIDER":
                return None
            if k in ("HERMES_LLM_API_KEY", "OPENAI_API_KEY"):
                return "sk-test"
            return kw.get("default")

        with patch.object(mod, "get_env_value", side_effect=fake_env):
            assert mod._detect_backend() == "combined"

    def test_auto_detect_spacy_no_key(self):
        import hermes.ner_provider as mod

        with patch.object(mod, "get_env_value", return_value=None):
            assert mod._detect_backend() == "spacy"

    def test_get_ner_provider_openai(self):
        import hermes.ner_provider as mod

        mod._ner_provider = None
        with patch.object(mod, "_detect_backend", return_value="openai"):
            provider = mod.get_ner_provider()
            assert isinstance(provider, mod.OpenAINERProvider)
        mod._ner_provider = None

    def test_get_ner_provider_spacy(self):
        import hermes.ner_provider as mod

        mod._ner_provider = None
        with patch.object(mod, "_detect_backend", return_value="spacy"):
            provider = mod.get_ner_provider()
            assert isinstance(provider, mod.SpacyNERProvider)
        mod._ner_provider = None

    def test_get_ner_provider_unknown_raises(self):
        import hermes.ner_provider as mod

        mod._ner_provider = None
        with patch.object(mod, "_detect_backend", return_value="unknown"):
            with pytest.raises(ValueError, match="Unknown NER_PROVIDER"):
                mod.get_ner_provider()
        mod._ner_provider = None

    def test_singleton_returns_cached(self):
        import hermes.ner_provider as mod

        sentinel = MagicMock()
        mod._ner_provider = sentinel
        assert mod.get_ner_provider() is sentinel
        mod._ner_provider = None

"""Tests for OpenAICombinedExtractor — single-call NER + relation extraction."""

import json

import pytest
from unittest.mock import AsyncMock, patch


def _patch_ontology_client():
    """No-op kept for call-site compatibility: the extractor no longer
    consults the ontology client (hermes#148 — free typing)."""
    from contextlib import nullcontext

    return nullcontext()


@pytest.mark.asyncio
class TestCombinedExtraction:
    """Tests for extract_entities_and_relations."""

    async def test_returns_entities_and_relations(self):
        from hermes.combined_extractor import OpenAICombinedExtractor

        extractor = OpenAICombinedExtractor()

        llm_response = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "entities": [
                                    {
                                        "name": "Paris",
                                        "type": "location",
                                        "start": 0,
                                        "end": 5,
                                    },
                                    {
                                        "name": "France",
                                        "type": "location",
                                        "start": 20,
                                        "end": 26,
                                    },
                                ],
                                "relations": [
                                    {
                                        "source_name": "Paris",
                                        "target_name": "France",
                                        "relation": "CAPITAL_OF",
                                        "confidence": 0.95,
                                        "bidirectional": False,
                                    }
                                ],
                            }
                        )
                    }
                }
            ]
        }

        p1 = _patch_ontology_client()
        with (
            p1,
            patch(
                "hermes.llm.generate_completion",
                new_callable=AsyncMock,
                return_value=llm_response,
            ),
        ):
            entities, relations = await extractor.extract_entities_and_relations(
                "Paris is the capital of France"
            )

        assert len(entities) == 2
        assert entities[0]["name"] == "Paris"
        assert entities[0]["type"] == "location"
        assert entities[1]["name"] == "France"

        assert len(relations) == 1
        assert relations[0]["source_name"] == "Paris"
        assert relations[0]["target_name"] == "France"
        assert relations[0]["relation"] == "CAPITAL_OF"
        assert relations[0]["confidence"] == 0.95

    async def test_entity_offset_resolution(self):
        """Entities with missing offsets are resolved from the text."""
        from hermes.combined_extractor import OpenAICombinedExtractor

        extractor = OpenAICombinedExtractor()

        llm_response = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "entities": [
                                    {"name": "Alice", "type": "entity"},
                                    {"name": "Google", "type": "entity"},
                                ],
                                "relations": [],
                            }
                        )
                    }
                }
            ]
        }

        text = "Alice works at Google"
        p1 = _patch_ontology_client()
        with (
            p1,
            patch(
                "hermes.llm.generate_completion",
                new_callable=AsyncMock,
                return_value=llm_response,
            ),
        ):
            entities, _ = await extractor.extract_entities_and_relations(text)

        assert len(entities) == 2
        assert entities[0]["start"] == 0
        assert entities[0]["end"] == 5
        assert entities[1]["start"] == 15
        assert entities[1]["end"] == 21

    async def test_relation_entity_name_validation(self):
        """Relations referencing non-existent entities are filtered out."""
        from hermes.combined_extractor import OpenAICombinedExtractor

        extractor = OpenAICombinedExtractor()

        llm_response = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "entities": [
                                    {
                                        "name": "Paris",
                                        "type": "location",
                                        "start": 0,
                                        "end": 5,
                                    },
                                ],
                                "relations": [
                                    {
                                        "source_name": "Paris",
                                        "target_name": "NonExistent",
                                        "relation": "LOCATED_IN",
                                        "confidence": 0.9,
                                        "bidirectional": False,
                                    }
                                ],
                            }
                        )
                    }
                }
            ]
        }

        p1 = _patch_ontology_client()
        with (
            p1,
            patch(
                "hermes.llm.generate_completion",
                new_callable=AsyncMock,
                return_value=llm_response,
            ),
        ):
            entities, relations = await extractor.extract_entities_and_relations(
                "Paris"
            )

        assert len(entities) == 1
        assert len(relations) == 0  # filtered out

    async def test_graceful_degradation_on_bad_json(self):
        """Returns empty lists when LLM returns unparseable content."""
        from hermes.combined_extractor import OpenAICombinedExtractor

        extractor = OpenAICombinedExtractor()

        llm_response = {"choices": [{"message": {"content": "not valid json at all"}}]}

        p1 = _patch_ontology_client()
        with (
            p1,
            patch(
                "hermes.llm.generate_completion",
                new_callable=AsyncMock,
                return_value=llm_response,
            ),
        ):
            entities, relations = await extractor.extract_entities_and_relations(
                "some text"
            )

        assert entities == []
        assert relations == []

    async def test_graceful_degradation_on_exception(self):
        """Returns empty lists when LLM call raises."""
        from hermes.combined_extractor import OpenAICombinedExtractor

        extractor = OpenAICombinedExtractor()

        p1 = _patch_ontology_client()
        with (
            p1,
            patch(
                "hermes.llm.generate_completion",
                new_callable=AsyncMock,
                side_effect=RuntimeError("API down"),
            ),
        ):
            entities, relations = await extractor.extract_entities_and_relations(
                "some text"
            )

        assert entities == []
        assert relations == []

    async def test_handles_markdown_fenced_json(self):
        """Parses JSON wrapped in markdown code fences."""
        from hermes.combined_extractor import OpenAICombinedExtractor

        extractor = OpenAICombinedExtractor()

        raw = (
            "```json\n"
            + json.dumps(
                {
                    "entities": [
                        {"name": "Bob", "type": "entity", "start": 0, "end": 3}
                    ],
                    "relations": [],
                }
            )
            + "\n```"
        )

        llm_response = {"choices": [{"message": {"content": raw}}]}

        p1 = _patch_ontology_client()
        with (
            p1,
            patch(
                "hermes.llm.generate_completion",
                new_callable=AsyncMock,
                return_value=llm_response,
            ),
        ):
            entities, relations = await extractor.extract_entities_and_relations("Bob")

        assert len(entities) == 1
        assert entities[0]["name"] == "Bob"


@pytest.mark.asyncio
class TestFreeTypingPrompt:
    """The prompt must not constrain entity types (hermes#148).

    The 2026-06-11 ablation measured any type list in the prompt
    suppressing type_accuracy ~4x (0.457 free vs ~0.12 constrained) and
    taxing entity/link F1 — and sophia ignores the proposal type field
    (centroid classification), so the constraint had no consumer.
    """

    async def test_prompt_has_no_entity_types_section(self):
        from hermes.combined_extractor import OpenAICombinedExtractor
        from hermes.ner_provider import ONTOLOGY_TYPES

        prompt = OpenAICombinedExtractor()._build_system_prompt()
        assert "## Entity Types" not in prompt
        for type_name in ONTOLOGY_TYPES:
            assert f"- {type_name}:" not in prompt

    async def test_prompt_documents_free_type_field(self):
        from hermes.combined_extractor import OpenAICombinedExtractor

        prompt = OpenAICombinedExtractor()._build_system_prompt()
        assert "category you choose" in prompt
        assert "must be one of the types listed above" not in prompt

    async def test_extraction_runs_without_ontology_client(self):
        from hermes.combined_extractor import OpenAICombinedExtractor

        extractor = OpenAICombinedExtractor()
        llm_response = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "entities": [
                                    {
                                        "name": "Bob",
                                        "type": "person",
                                        "start": 0,
                                        "end": 3,
                                    }
                                ],
                                "relations": [],
                            }
                        )
                    }
                }
            ]
        }
        with patch(
            "hermes.llm.generate_completion",
            new_callable=AsyncMock,
            return_value=llm_response,
        ) as mock_llm:
            entities, _ = await extractor.extract_entities_and_relations("Bob")

        assert entities[0]["type"] == "person"
        system_msg = mock_llm.call_args[1]["messages"][0]["content"]
        assert "## Entity Types" not in system_msg


@pytest.mark.asyncio
class TestProtocolCompat:
    """Combined extractor satisfies both NERProvider and RelationExtractor."""

    async def test_extract_entities_protocol(self):
        from hermes.combined_extractor import OpenAICombinedExtractor

        extractor = OpenAICombinedExtractor()

        llm_response = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "entities": [
                                    {
                                        "name": "X",
                                        "type": "entity",
                                        "start": 0,
                                        "end": 1,
                                    }
                                ],
                                "relations": [],
                            }
                        )
                    }
                }
            ]
        }

        p1 = _patch_ontology_client()
        with (
            p1,
            patch(
                "hermes.llm.generate_completion",
                new_callable=AsyncMock,
                return_value=llm_response,
            ),
        ):
            entities = await extractor.extract_entities("X")

        assert len(entities) == 1
        assert entities[0]["name"] == "X"

    async def test_extract_protocol(self):
        from hermes.combined_extractor import OpenAICombinedExtractor

        extractor = OpenAICombinedExtractor()

        llm_response = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "entities": [
                                    {
                                        "name": "A",
                                        "type": "entity",
                                        "start": 0,
                                        "end": 1,
                                    },
                                    {
                                        "name": "B",
                                        "type": "entity",
                                        "start": 5,
                                        "end": 6,
                                    },
                                ],
                                "relations": [
                                    {
                                        "source_name": "A",
                                        "target_name": "B",
                                        "relation": "KNOWS",
                                        "confidence": 0.8,
                                        "bidirectional": True,
                                    }
                                ],
                            }
                        )
                    }
                }
            ]
        }

        p1 = _patch_ontology_client()
        with (
            p1,
            patch(
                "hermes.llm.generate_completion",
                new_callable=AsyncMock,
                return_value=llm_response,
            ),
        ):
            relations = await extractor.extract(
                "A knows B", [{"name": "A"}, {"name": "B"}]
            )

        assert len(relations) == 1
        assert relations[0]["relation"] == "KNOWS"


class TestSingleton:
    """Tests for get_combined_instance."""

    def test_returns_same_instance(self):
        import hermes.combined_extractor as mod

        mod._combined_instance = None
        first = mod.get_combined_instance()
        second = mod.get_combined_instance()
        assert first is second
        mod._combined_instance = None

    def test_isinstance_checks(self):
        from hermes.combined_extractor import (
            OpenAICombinedExtractor,
            get_combined_instance,
        )

        import hermes.combined_extractor as mod

        mod._combined_instance = None
        inst = get_combined_instance()
        assert isinstance(inst, OpenAICombinedExtractor)
        mod._combined_instance = None


class TestClosedVocabPrompt:
    """H5: closed-vocabulary clause injected into the combined RE prompt.

    The clause reuses the live match-before-mint vocabulary
    (``get_predicate_vocabulary().known()``) so the model prefers an existing
    predicate over minting a near-duplicate -- the source-side df=1 lever.
    """

    @staticmethod
    def _patch_vocab(monkeypatch, known):
        class _StubVocab:
            def known(self):
                return set(known)

        monkeypatch.setattr(
            "hermes.predicate_resolver.get_predicate_vocabulary",
            lambda: _StubVocab(),
        )

    def test_clause_injected_when_vocab_present(self, monkeypatch):
        from hermes.combined_extractor import OpenAICombinedExtractor

        self._patch_vocab(monkeypatch, {"LOCATED_IN", "PART_OF"})
        prompt = OpenAICombinedExtractor()._build_system_prompt()

        assert "## Known Relations" in prompt
        assert "LOCATED_IN" in prompt
        assert "PART_OF" in prompt
        assert "only mint a NEW" in prompt
        # the clause must sit before the JSON-only sign-off, not after it
        assert prompt.index("## Known Relations") < prompt.index(
            "Return ONLY valid JSON"
        )

    def test_clause_omitted_when_vocab_empty(self, monkeypatch):
        from hermes.combined_extractor import OpenAICombinedExtractor

        self._patch_vocab(monkeypatch, set())
        prompt = OpenAICombinedExtractor()._build_system_prompt()

        assert "## Known Relations" not in prompt

    def test_cap_respected(self, monkeypatch):
        from hermes.combined_extractor import OpenAICombinedExtractor

        self._patch_vocab(monkeypatch, {f"R{i:03d}" for i in range(200)})
        monkeypatch.setenv("RE_VOCAB_CAP", "150")
        prompt = OpenAICombinedExtractor()._build_system_prompt()

        # sorted, capped at 150 -> R000..R149 in, R150.. out
        assert "R149" in prompt
        assert "R150" not in prompt

    def test_non_int_cap_falls_back_without_raising(self, monkeypatch):
        from hermes.combined_extractor import OpenAICombinedExtractor

        self._patch_vocab(monkeypatch, {"LOCATED_IN", "PART_OF"})
        monkeypatch.setenv("RE_VOCAB_CAP", "not-a-number")
        # must not raise; falls back to the default cap and still injects
        prompt = OpenAICombinedExtractor()._build_system_prompt()

        assert "## Known Relations" in prompt
        assert "LOCATED_IN" in prompt

    def test_fail_soft_when_vocab_raises(self, monkeypatch):
        from hermes.combined_extractor import OpenAICombinedExtractor

        class _Boom:
            def known(self):
                raise RuntimeError("redis down")

        monkeypatch.setattr(
            "hermes.predicate_resolver.get_predicate_vocabulary",
            lambda: _Boom(),
        )
        # must not raise; falls back to the open prompt
        prompt = OpenAICombinedExtractor()._build_system_prompt()

        assert "## Known Relations" not in prompt

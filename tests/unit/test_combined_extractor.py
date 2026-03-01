"""Tests for OpenAICombinedExtractor — single-call NER + relation extraction."""

import json

import pytest
from unittest.mock import AsyncMock, patch


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

        with patch(
            "hermes.llm.generate_completion",
            new_callable=AsyncMock,
            return_value=llm_response,
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
        with patch(
            "hermes.llm.generate_completion",
            new_callable=AsyncMock,
            return_value=llm_response,
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

        with patch(
            "hermes.llm.generate_completion",
            new_callable=AsyncMock,
            return_value=llm_response,
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

        with patch(
            "hermes.llm.generate_completion",
            new_callable=AsyncMock,
            return_value=llm_response,
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

        with patch(
            "hermes.llm.generate_completion",
            new_callable=AsyncMock,
            side_effect=RuntimeError("API down"),
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

        with patch(
            "hermes.llm.generate_completion",
            new_callable=AsyncMock,
            return_value=llm_response,
        ):
            entities, relations = await extractor.extract_entities_and_relations("Bob")

        assert len(entities) == 1
        assert entities[0]["name"] == "Bob"


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

        with patch(
            "hermes.llm.generate_completion",
            new_callable=AsyncMock,
            return_value=llm_response,
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

        with patch(
            "hermes.llm.generate_completion",
            new_callable=AsyncMock,
            return_value=llm_response,
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

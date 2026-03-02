"""Tests for OpenAICombinedExtractor — single-call NER + relation extraction."""

import json

import pytest
from unittest.mock import AsyncMock, patch


def _patch_ontology_client():
    """Patch ontology client to return None (fallback to hardcoded types)."""
    return (
        patch("hermes.combined_extractor.fetch_type_list", new_callable=AsyncMock, return_value=None),
        patch("hermes.combined_extractor.fetch_edge_type_list", new_callable=AsyncMock, return_value=None),
    )


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

        p1, p2 = _patch_ontology_client()
        with p1, p2, patch(
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
        p1, p2 = _patch_ontology_client()
        with p1, p2, patch(
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

        p1, p2 = _patch_ontology_client()
        with p1, p2, patch(
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

        p1, p2 = _patch_ontology_client()
        with p1, p2, patch(
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

        p1, p2 = _patch_ontology_client()
        with p1, p2, patch(
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

        p1, p2 = _patch_ontology_client()
        with p1, p2, patch(
            "hermes.llm.generate_completion",
            new_callable=AsyncMock,
            return_value=llm_response,
        ):
            entities, relations = await extractor.extract_entities_and_relations("Bob")

        assert len(entities) == 1
        assert entities[0]["name"] == "Bob"


@pytest.mark.asyncio
class TestDynamicTypePrompt:
    """Tests for dynamic type list injection into the system prompt."""

    async def test_dynamic_types_included_when_available(self):
        from hermes.combined_extractor import OpenAICombinedExtractor

        extractor = OpenAICombinedExtractor()

        fetched_types = [
            {"name": "robot", "description": "an autonomous agent"},
            {"name": "sensor", "description": "a sensing device"},
        ]

        llm_response = {
            "choices": [{"message": {"content": json.dumps({"entities": [], "relations": []})}}]
        }

        with (
            patch("hermes.combined_extractor.fetch_type_list", new_callable=AsyncMock, return_value=fetched_types),
            patch("hermes.combined_extractor.fetch_edge_type_list", new_callable=AsyncMock, return_value=None),
            patch("hermes.llm.generate_completion", new_callable=AsyncMock, return_value=llm_response) as mock_llm,
        ):
            await extractor.extract_entities_and_relations("test")

        system_msg = mock_llm.call_args[1]["messages"][0]["content"]
        assert "- robot: an autonomous agent" in system_msg
        assert "- sensor: a sensing device" in system_msg
        assert "If none of these types fit, use 'object'." in system_msg

    async def test_falls_back_to_hardcoded_when_fetch_returns_none(self):
        from hermes.combined_extractor import OpenAICombinedExtractor
        from hermes.ner_provider import ONTOLOGY_TYPES

        extractor = OpenAICombinedExtractor()

        llm_response = {
            "choices": [{"message": {"content": json.dumps({"entities": [], "relations": []})}}]
        }

        p1, p2 = _patch_ontology_client()
        with p1, p2, patch(
            "hermes.llm.generate_completion", new_callable=AsyncMock, return_value=llm_response
        ) as mock_llm:
            await extractor.extract_entities_and_relations("test")

        system_msg = mock_llm.call_args[1]["messages"][0]["content"]
        for type_name, desc in ONTOLOGY_TYPES.items():
            assert f"- {type_name}: {desc}" in system_msg

    async def test_edge_types_included_when_available(self):
        from hermes.combined_extractor import OpenAICombinedExtractor

        extractor = OpenAICombinedExtractor()

        fetched_edge_types = [
            {"name": "LOCATED_IN", "description": "spatial containment"},
            {"name": "PART_OF", "description": "part-whole relation"},
        ]

        llm_response = {
            "choices": [{"message": {"content": json.dumps({"entities": [], "relations": []})}}]
        }

        with (
            patch("hermes.combined_extractor.fetch_type_list", new_callable=AsyncMock, return_value=None),
            patch("hermes.combined_extractor.fetch_edge_type_list", new_callable=AsyncMock, return_value=fetched_edge_types),
            patch("hermes.llm.generate_completion", new_callable=AsyncMock, return_value=llm_response) as mock_llm,
        ):
            await extractor.extract_entities_and_relations("test")

        system_msg = mock_llm.call_args[1]["messages"][0]["content"]
        assert "## Known Relation Types" in system_msg
        assert "- LOCATED_IN: spatial containment" in system_msg
        assert "- PART_OF: part-whole relation" in system_msg
        assert "You may also use other UPPER_SNAKE_CASE labels" in system_msg


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

        p1, p2 = _patch_ontology_client()
        with p1, p2, patch(
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

        p1, p2 = _patch_ontology_client()
        with p1, p2, patch(
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

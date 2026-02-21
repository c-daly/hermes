"""Tests for ProposalBuilder — builds structured proposals from text."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
class TestProposalBuilder:
    async def test_build_returns_required_fields(self):
        from hermes.proposal_builder import ProposalBuilder

        builder = ProposalBuilder()

        mock_provider = MagicMock()
        mock_provider.extract_entities = AsyncMock(return_value=[])

        mock_rel = MagicMock()
        mock_rel.extract = AsyncMock(return_value=[])

        emb_result = {
            "embedding": [0.1] * 384,
            "dimension": 384,
            "model": "all-MiniLM-L6-v2",
            "embedding_id": "doc-id",
        }

        with (
            patch(
                "hermes.proposal_builder.get_ner_provider",
                return_value=mock_provider,
            ),
            patch(
                "hermes.proposal_builder.get_relation_extractor",
                return_value=mock_rel,
            ),
            patch(
                "hermes.proposal_builder.generate_embeddings_batch",
                new_callable=AsyncMock,
            ) as mock_batch,
        ):
            # 0 entities -> batch embed gets just [text] -> 1 result (doc embedding)
            mock_batch.return_value = [emb_result]
            proposal = await builder.build(text="Hello world", metadata={})

        assert "proposal_id" in proposal
        assert "proposed_nodes" in proposal
        assert "document_embedding" in proposal
        assert proposal["source_service"] == "hermes"

    async def test_extracts_entities_as_proposed_nodes(self):
        from hermes.proposal_builder import ProposalBuilder

        builder = ProposalBuilder()

        mock_provider = MagicMock()
        mock_provider.extract_entities = AsyncMock(
            return_value=[
                {"name": "Eiffel Tower", "type": "object", "start": 4, "end": 16},
                {"name": "Paris", "type": "location", "start": 23, "end": 28},
            ]
        )

        mock_rel = MagicMock()
        mock_rel.extract = AsyncMock(return_value=[])

        emb_result = {
            "embedding": [0.1] * 384,
            "dimension": 384,
            "model": "all-MiniLM-L6-v2",
            "embedding_id": "test-id",
        }

        with (
            patch(
                "hermes.proposal_builder.get_ner_provider",
                return_value=mock_provider,
            ),
            patch(
                "hermes.proposal_builder.get_relation_extractor",
                return_value=mock_rel,
            ),
            patch(
                "hermes.proposal_builder.generate_embeddings_batch",
                new_callable=AsyncMock,
            ) as mock_batch,
        ):
            # 2 entity embeddings + 1 doc embedding = 3 results
            mock_batch.return_value = [emb_result, emb_result, emb_result]
            proposal = await builder.build(
                text="The Eiffel Tower is in Paris",
                metadata={},
            )

        assert len(proposal["proposed_nodes"]) == 2
        assert proposal["proposed_nodes"][0]["name"] == "Eiffel Tower"
        assert proposal["proposed_nodes"][0]["type"] == "object"
        assert proposal["proposed_nodes"][1]["type"] == "location"
        assert "embedding" in proposal["proposed_nodes"][0]

    async def test_degrades_gracefully_without_nlp(self):
        from hermes.proposal_builder import ProposalBuilder

        builder = ProposalBuilder()

        mock_provider = MagicMock()
        mock_provider.extract_entities = AsyncMock(
            side_effect=Exception("NER not available")
        )

        mock_rel = MagicMock()
        mock_rel.extract = AsyncMock(return_value=[])

        emb_result = {
            "embedding": [0.1] * 384,
            "dimension": 384,
            "model": "all-MiniLM-L6-v2",
            "embedding_id": "fallback-id",
        }

        with (
            patch(
                "hermes.proposal_builder.get_ner_provider",
                return_value=mock_provider,
            ),
            patch(
                "hermes.proposal_builder.get_relation_extractor",
                return_value=mock_rel,
            ),
            patch(
                "hermes.proposal_builder.generate_embeddings_batch",
                new_callable=AsyncMock,
            ) as mock_batch,
        ):
            # NER fails -> 0 entities -> batch embed gets just [text] -> 1 result (doc embedding)
            mock_batch.return_value = [emb_result]
            proposal = await builder.build(text="Hello", metadata={})

        assert proposal["proposed_nodes"] == []
        assert proposal["document_embedding"] is not None

    async def test_pipeline_metadata_populated(self):
        from hermes.proposal_builder import ProposalBuilder

        builder = ProposalBuilder()

        mock_ner = MagicMock()
        mock_ner.name = "spacy"
        mock_ner.extract_entities = AsyncMock(
            return_value=[
                {"name": "Paris", "type": "location", "start": 0, "end": 5},
            ]
        )

        mock_rel = MagicMock()
        mock_rel.extract = AsyncMock(return_value=[])

        mock_emb_provider = MagicMock()
        mock_emb_provider.model_name = "all-MiniLM-L6-v2"

        emb_result = {
            "embedding": [0.1] * 384,
            "dimension": 384,
            "model": "all-MiniLM-L6-v2",
            "embedding_id": "test-id",
        }

        with (
            patch(
                "hermes.proposal_builder.get_ner_provider",
                return_value=mock_ner,
            ),
            patch(
                "hermes.proposal_builder.get_relation_extractor",
                return_value=mock_rel,
            ),
            patch(
                "hermes.proposal_builder.get_embedding_provider",
                return_value=mock_emb_provider,
            ),
            patch(
                "hermes.proposal_builder.generate_embeddings_batch",
                new_callable=AsyncMock,
            ) as mock_batch,
        ):
            # 1 entity + 1 doc text = 2 batch results
            mock_batch.return_value = [emb_result, emb_result]
            proposal = await builder.build(text="Paris", metadata={"foo": "bar"})

        pipeline = proposal["metadata"]["pipeline"]
        assert pipeline["ner_provider"] == "spacy"
        assert pipeline["embedding_provider"] == "all-MiniLM-L6-v2"
        assert pipeline["entity_count"] == 1
        assert pipeline["edge_count"] == 0
        assert "ner_duration_ms" in pipeline
        assert "total_duration_ms" in pipeline
        # Original metadata preserved
        assert proposal["metadata"]["foo"] == "bar"

    async def test_experiment_tags_passed_through(self):
        from hermes.proposal_builder import ProposalBuilder

        builder = ProposalBuilder()

        mock_ner = MagicMock()
        mock_ner.name = "spacy"
        mock_ner.extract_entities = AsyncMock(return_value=[])

        mock_rel = MagicMock()
        mock_rel.extract = AsyncMock(return_value=[])

        mock_emb_provider = MagicMock()
        mock_emb_provider.model_name = "test-model"

        emb_result = {
            "embedding": [0.1] * 384,
            "dimension": 384,
            "model": "test-model",
            "embedding_id": "id",
        }

        with (
            patch(
                "hermes.proposal_builder.get_ner_provider",
                return_value=mock_ner,
            ),
            patch(
                "hermes.proposal_builder.get_relation_extractor",
                return_value=mock_rel,
            ),
            patch(
                "hermes.proposal_builder.get_embedding_provider",
                return_value=mock_emb_provider,
            ),
            patch(
                "hermes.proposal_builder.generate_embeddings_batch",
                new_callable=AsyncMock,
            ) as mock_batch,
        ):
            mock_batch.return_value = [emb_result]
            proposal = await builder.build(
                text="Hello",
                metadata={"experiment_tags": ["baseline", "v2"]},
            )

        assert proposal["metadata"]["experiment_tags"] == ["baseline", "v2"]
        assert "pipeline" in proposal["metadata"]

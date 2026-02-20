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

        with (
            patch(
                "hermes.proposal_builder.get_ner_provider",
                return_value=mock_provider,
            ),
            patch(
                "hermes.proposal_builder.generate_embedding", new_callable=AsyncMock
            ) as mock_emb,
        ):
            mock_emb.return_value = {
                "embedding": [0.1] * 384,
                "dimension": 384,
                "model": "all-MiniLM-L6-v2",
                "embedding_id": "doc-id",
            }
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

        with (
            patch(
                "hermes.proposal_builder.get_ner_provider",
                return_value=mock_provider,
            ),
            patch(
                "hermes.proposal_builder.generate_embedding", new_callable=AsyncMock
            ) as mock_emb,
        ):
            mock_emb.return_value = {
                "embedding": [0.1] * 384,
                "dimension": 384,
                "model": "all-MiniLM-L6-v2",
                "embedding_id": "test-id",
            }
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

        with (
            patch(
                "hermes.proposal_builder.get_ner_provider",
                return_value=mock_provider,
            ),
            patch(
                "hermes.proposal_builder.generate_embedding", new_callable=AsyncMock
            ) as mock_emb,
        ):
            mock_emb.return_value = {
                "embedding": [0.1] * 384,
                "dimension": 384,
                "model": "all-MiniLM-L6-v2",
                "embedding_id": "fallback-id",
            }
            proposal = await builder.build(text="Hello", metadata={})

        assert proposal["proposed_nodes"] == []
        assert proposal["document_embedding"] is not None

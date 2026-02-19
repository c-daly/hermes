"""Tests for ProposalBuilder — builds structured proposals from text."""

import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
class TestProposalBuilder:
    async def test_build_returns_required_fields(self):
        from hermes.proposal_builder import ProposalBuilder

        builder = ProposalBuilder()

        with (
            patch(
                "hermes.proposal_builder.process_nlp", new_callable=AsyncMock
            ) as mock_nlp,
            patch(
                "hermes.proposal_builder.generate_embedding", new_callable=AsyncMock
            ) as mock_emb,
        ):
            mock_nlp.return_value = {"entities": []}
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

        with (
            patch(
                "hermes.proposal_builder.process_nlp", new_callable=AsyncMock
            ) as mock_nlp,
            patch(
                "hermes.proposal_builder.generate_embedding", new_callable=AsyncMock
            ) as mock_emb,
        ):
            mock_nlp.return_value = {
                "entities": [
                    {"text": "Eiffel Tower", "label": "FAC", "start": 4, "end": 16},
                    {"text": "Paris", "label": "GPE", "start": 23, "end": 28},
                ]
            }
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
        assert proposal["proposed_nodes"][0]["type"] == "object"  # FAC -> object
        assert proposal["proposed_nodes"][1]["type"] == "location"  # GPE -> location
        assert "embedding" in proposal["proposed_nodes"][0]

    async def test_degrades_gracefully_without_nlp(self):
        from hermes.proposal_builder import ProposalBuilder

        builder = ProposalBuilder()

        with (
            patch(
                "hermes.proposal_builder.process_nlp", new_callable=AsyncMock
            ) as mock_nlp,
            patch(
                "hermes.proposal_builder.generate_embedding", new_callable=AsyncMock
            ) as mock_emb,
        ):
            mock_nlp.side_effect = Exception("spaCy not available")
            mock_emb.return_value = {
                "embedding": [0.1] * 384,
                "dimension": 384,
                "model": "all-MiniLM-L6-v2",
                "embedding_id": "fallback-id",
            }
            proposal = await builder.build(text="Hello", metadata={})

        assert proposal["proposed_nodes"] == []
        assert proposal["document_embedding"] is not None

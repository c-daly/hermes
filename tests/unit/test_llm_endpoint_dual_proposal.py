"""Tests for post-generation proposal in the /llm endpoint.

After generating an LLM response, a proposal is built from the
combined prompt + reply text before returning the response.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture
def mock_generate_llm_response():
    """Mock LLM response matching LLMResponse schema."""
    return {
        "id": "chatcmpl-test",
        "provider": "openai",
        "model": "gpt-4",
        "created": 1700000000,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Rottweilers are a loyal breed originating from Germany.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 12, "total_tokens": 22},
    }


@pytest.fixture
def mock_sophia_context():
    """Empty Sophia context."""
    return []


@pytest.fixture
def mock_proposal():
    """A minimal proposal dict."""
    return {
        "proposal_id": "test-proposal-id",
        "source_service": "hermes",
        "proposed_nodes": [],
        "proposed_edges": [],
        "document_embedding": None,
        "metadata": {},
    }


@pytest.mark.asyncio
class TestPostGenerationProposal:
    async def test_post_generation_proposal_built(
        self, mock_generate_llm_response, mock_sophia_context, mock_proposal
    ):
        """After /llm responds, a post-generation proposal is built."""
        build_calls: list[dict] = []

        async def tracking_build(text, metadata, **kwargs):
            build_calls.append({"text": text, "metadata": metadata})
            return mock_proposal

        mock_builder = MagicMock()
        mock_builder.build = AsyncMock(side_effect=tracking_build)

        with (
            patch("hermes.main._proposal_builder", mock_builder),
            patch(
                "hermes.main._get_sophia_context",
                new_callable=AsyncMock,
                return_value=mock_sophia_context,
            ),
            patch(
                "hermes.main.generate_llm_response",
                new_callable=AsyncMock,
                return_value=mock_generate_llm_response,
            ),
        ):
            from hermes.main import app
            from fastapi.testclient import TestClient

            client = TestClient(app)
            response = client.post(
                "/llm",
                json={
                    "messages": [{"role": "user", "content": "Tell me about rottweilers"}],
                },
            )

        assert response.status_code == 200
        assert len(build_calls) >= 1

    async def test_post_generation_proposal_contains_both_texts(
        self, mock_generate_llm_response, mock_sophia_context, mock_proposal
    ):
        """Post-generation proposal contains both user text and reply text."""
        build_calls: list[dict] = []

        async def tracking_build(text, metadata, **kwargs):
            build_calls.append({"text": text, "metadata": metadata})
            return mock_proposal

        mock_builder = MagicMock()
        mock_builder.build = AsyncMock(side_effect=tracking_build)

        with (
            patch("hermes.main._proposal_builder", mock_builder),
            patch(
                "hermes.main._get_sophia_context",
                new_callable=AsyncMock,
                return_value=mock_sophia_context,
            ),
            patch(
                "hermes.main.generate_llm_response",
                new_callable=AsyncMock,
                return_value=mock_generate_llm_response,
            ),
        ):
            from hermes.main import app
            from fastapi.testclient import TestClient

            client = TestClient(app)
            response = client.post(
                "/llm",
                json={
                    "messages": [{"role": "user", "content": "Tell me about rottweilers"}],
                },
            )

        assert response.status_code == 200
        post_gen_calls = [
            c for c in build_calls
            if "Tell me about rottweilers" in c["text"]
            and "Rottweilers are a loyal breed" in c["text"]
        ]
        assert len(post_gen_calls) == 1

    async def test_post_generation_proposal_has_extraction_source(
        self, mock_generate_llm_response, mock_sophia_context, mock_proposal
    ):
        """Post-generation proposal has extraction_source: prompt_and_reply."""
        build_calls: list[dict] = []

        async def tracking_build(text, metadata, **kwargs):
            build_calls.append({"text": text, "metadata": metadata})
            return mock_proposal

        mock_builder = MagicMock()
        mock_builder.build = AsyncMock(side_effect=tracking_build)

        with (
            patch("hermes.main._proposal_builder", mock_builder),
            patch(
                "hermes.main._get_sophia_context",
                new_callable=AsyncMock,
                return_value=mock_sophia_context,
            ),
            patch(
                "hermes.main.generate_llm_response",
                new_callable=AsyncMock,
                return_value=mock_generate_llm_response,
            ),
        ):
            from hermes.main import app
            from fastapi.testclient import TestClient

            client = TestClient(app)
            response = client.post(
                "/llm",
                json={
                    "messages": [{"role": "user", "content": "Tell me about rottweilers"}],
                },
            )

        assert response.status_code == 200
        post_gen_calls = [
            c for c in build_calls
            if c["metadata"].get("extraction_source") == "prompt_and_reply"
        ]
        assert len(post_gen_calls) == 1

    async def test_endpoint_succeeds_if_proposal_fails(
        self, mock_generate_llm_response, mock_sophia_context, mock_proposal
    ):
        """If the post-generation proposal build fails, endpoint still returns."""
        async def failing_build(text, metadata, **kwargs):
            if metadata.get("extraction_source") == "prompt_and_reply":
                raise RuntimeError("Proposal build exploded")
            return mock_proposal

        mock_builder = MagicMock()
        mock_builder.build = AsyncMock(side_effect=failing_build)

        with (
            patch("hermes.main._proposal_builder", mock_builder),
            patch(
                "hermes.main._get_sophia_context",
                new_callable=AsyncMock,
                return_value=mock_sophia_context,
            ),
            patch(
                "hermes.main.generate_llm_response",
                new_callable=AsyncMock,
                return_value=mock_generate_llm_response,
            ),
        ):
            from hermes.main import app
            from fastapi.testclient import TestClient

            client = TestClient(app)
            response = client.post(
                "/llm",
                json={
                    "messages": [{"role": "user", "content": "Tell me about rottweilers"}],
                },
            )

        assert response.status_code == 200

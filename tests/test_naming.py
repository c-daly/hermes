import pytest
from unittest.mock import AsyncMock, patch
from httpx import ASGITransport, AsyncClient
from hermes.main import app


@pytest.mark.asyncio
async def test_name_type_returns_name():
    """POST /name-type returns a type name for a cluster of node names."""
    with patch("hermes.main.generate_completion", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = {
            "choices": [{"message": {"content": '{"type_name": "temporal_reference"}'}}]
        }
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/name-type",
                json={
                    "node_names": ["13th century", "the 1200s", "medieval period"],
                    "parent_type": "state",
                },
            )
        assert resp.status_code == 200
        assert resp.json()["type_name"] == "temporal_reference"


@pytest.mark.asyncio
async def test_name_relationship_returns_label():
    """POST /name-relationship returns an edge label for a node pair."""
    with patch("hermes.main.generate_completion", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '{"relationship": "LOCATED_IN", "bidirectional": false}'
                    }
                }
            ]
        }
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/name-relationship",
                json={
                    "source_name": "Dublin",
                    "target_name": "Ireland",
                    "context": "Dublin is the capital of Ireland",
                },
            )
        assert resp.status_code == 200
        assert resp.json()["relationship"] == "LOCATED_IN"


@pytest.mark.asyncio
async def test_name_type_handles_code_fence():
    """POST /name-type handles LLM wrapping JSON in markdown code fences."""
    with patch("hermes.main.generate_completion", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '```json\n{"type_name": "geographic_region"}\n```'
                    }
                }
            ]
        }
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/name-type",
                json={"node_names": ["Europe", "Asia", "Africa"]},
            )
        assert resp.status_code == 200
        assert resp.json()["type_name"] == "geographic_region"


@pytest.mark.asyncio
async def test_name_type_empty_choices_returns_502():
    """POST /name-type returns 502 when LLM returns no choices."""
    with patch("hermes.main.generate_completion", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = {"choices": []}
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/name-type",
                json={"node_names": ["Dublin"]},
            )
        assert resp.status_code == 502


@pytest.mark.asyncio
async def test_name_type_empty_node_names_returns_422():
    """POST /name-type rejects empty node_names list."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/name-type",
            json={"node_names": []},
        )
    assert resp.status_code == 422

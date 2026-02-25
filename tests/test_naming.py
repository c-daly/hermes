import pytest
from unittest.mock import AsyncMock, patch
from httpx import AsyncClient, ASGITransport
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
            resp = await client.post("/name-type", json={
                "node_names": ["13th century", "the 1200s", "medieval period"],
                "parent_type": "state",
            })
        assert resp.status_code == 200
        assert resp.json()["type_name"] == "temporal_reference"

"""Tests for the Hermes API endpoints."""

import pytest
from fastapi.testclient import TestClient
from hermes.main import app

client = TestClient(app)

# Check if ML dependencies are available
try:
    from hermes import services

    ML_AVAILABLE = (
        services.WHISPER_AVAILABLE
        and services.TTS_AVAILABLE
        and services.SPACY_AVAILABLE
        and services.SENTENCE_TRANSFORMERS_AVAILABLE
    )
except ImportError:
    ML_AVAILABLE = False


def test_root_endpoint():
    """Test the root endpoint returns API information."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Hermes API"
    assert "version" in data
    assert "/stt" in data["endpoints"]
    assert "/tts" in data["endpoints"]
    assert "/simple_nlp" in data["endpoints"]
    assert "/embed_text" in data["endpoints"]
    assert "/llm" in data["endpoints"]


def test_health_endpoint():
    """Test the health endpoint returns service status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()

    # Check required fields (logos_config.health.HealthResponse schema)
    assert "status" in data
    assert "version" in data
    assert "service" in data
    assert data["service"] == "hermes"
    assert "timestamp" in data

    # Status should be "healthy", "degraded", or "unavailable"
    assert data["status"] in ["healthy", "degraded", "unavailable"]

    # Check dependencies structure
    assert "dependencies" in data
    deps = data["dependencies"]
    assert "milvus" in deps
    assert "llm" in deps

    # Each dependency should have status and connected fields
    for dep_name, dep_info in deps.items():
        assert "status" in dep_info
        assert dep_info["status"] in ["healthy", "degraded", "unavailable"]
        assert "connected" in dep_info

    # Check Milvus dependency details
    milvus = deps["milvus"]
    assert "details" in milvus
    milvus_details = milvus["details"]
    assert "host" in milvus_details
    assert "port" in milvus_details

    # Check capabilities structure
    assert "capabilities" in data
    caps = data["capabilities"]
    assert "stt" in caps
    assert "tts" in caps
    assert "nlp" in caps
    assert "embeddings" in caps

    # Each capability should be "available" or "unavailable"
    for cap_name, cap_status in caps.items():
        assert cap_status in ["available", "unavailable"]


def test_health_endpoint_head():
    """Test the health endpoint supports HEAD method."""
    response = client.head("/health")
    assert response.status_code == 200
    # HEAD should return no body
    assert response.content == b""
    # Should still have X-Request-ID header from middleware
    assert "x-request-id" in response.headers


def test_stt_endpoint_validation():
    """Test speech-to-text endpoint validates input."""
    # Test with non-audio file
    files = {"audio": ("test.txt", b"test text data", "text/plain")}
    response = client.post("/stt", files=files)
    assert response.status_code == 400
    assert "Invalid file type" in response.json()["detail"]


@pytest.mark.skipif(not ML_AVAILABLE, reason="ML dependencies not installed")
def test_stt_endpoint():
    """Test speech-to-text endpoint with ML dependencies."""
    # Create a minimal test WAV file
    files = {"audio": ("test.wav", b"RIFF" + b"\x00" * 40, "audio/wav")}
    response = client.post("/stt", files=files)
    # May fail with actual transcription but should handle gracefully
    assert response.status_code in [200, 500]  # 500 for invalid audio format


def test_tts_endpoint_validation():
    """Test text-to-speech endpoint validates input."""
    # Test with empty text
    request_data = {"text": "", "voice": "default", "language": "en-US"}
    response = client.post("/tts", json=request_data)
    assert response.status_code == 400
    assert "cannot be empty" in response.json()["detail"]


@pytest.mark.skipif(not ML_AVAILABLE, reason="ML dependencies not installed")
def test_tts_endpoint():
    """Test text-to-speech endpoint with ML dependencies."""
    request_data = {"text": "Hello, world!", "voice": "default", "language": "en-US"}
    response = client.post("/tts", json=request_data)
    # May succeed or fail depending on model availability
    assert response.status_code in [200, 500]
    if response.status_code == 200:
        assert response.headers["content-type"] == "audio/wav"
        assert len(response.content) > 0


def test_simple_nlp_validation():
    """Test simple NLP endpoint validates input."""
    # Test with empty text
    request_data = {"text": "", "operations": ["tokenize"]}
    response = client.post("/simple_nlp", json=request_data)
    assert response.status_code == 400
    assert "cannot be empty" in response.json()["detail"]

    # Test with invalid operation
    request_data = {"text": "Test text", "operations": ["invalid_op"]}
    response = client.post("/simple_nlp", json=request_data)
    assert response.status_code == 400
    assert "Invalid operations" in response.json()["detail"]


@pytest.mark.skipif(not ML_AVAILABLE, reason="ML dependencies not installed")
def test_simple_nlp_tokenize():
    """Test simple NLP with tokenization."""
    request_data = {"text": "This is a test sentence.", "operations": ["tokenize"]}
    response = client.post("/simple_nlp", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert "tokens" in data
    assert isinstance(data["tokens"], list)
    assert len(data["tokens"]) > 0


@pytest.mark.skipif(not ML_AVAILABLE, reason="ML dependencies not installed")
def test_simple_nlp_multiple_operations():
    """Test simple NLP with multiple operations."""
    request_data = {
        "text": "This is a test.",
        "operations": ["tokenize", "pos_tag", "lemmatize"],
    }
    response = client.post("/simple_nlp", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert "tokens" in data
    assert "pos_tags" in data
    assert "lemmas" in data


def test_embed_text_validation():
    """Test text embedding endpoint validates input."""
    # Test with empty text
    request_data = {"text": "", "model": "default"}
    response = client.post("/embed_text", json=request_data)
    assert response.status_code == 400
    assert "cannot be empty" in response.json()["detail"]


@pytest.mark.skipif(not ML_AVAILABLE, reason="ML dependencies not installed")
def test_embed_text_endpoint():
    """Test text embedding endpoint with ML dependencies."""
    request_data = {
        "text": "This is a test sentence for embedding.",
        "model": "default",
    }
    response = client.post("/embed_text", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert "embedding" in data
    assert "dimension" in data
    assert "model" in data
    assert isinstance(data["embedding"], list)
    assert len(data["embedding"]) == data["dimension"]
    assert data["dimension"] > 0  # Should be 384 for all-MiniLM-L6-v2


def test_llm_endpoint_requires_prompt_or_messages():
    """Hermes should validate LLM requests."""
    response = client.post("/llm", json={"prompt": "   "})
    assert response.status_code == 400
    assert "Either 'prompt' or 'messages'" in response.json()["detail"]


def test_llm_endpoint_with_prompt():
    """LLM endpoint should return an echo response when explicitly requested."""
    response = client.post("/llm", json={"prompt": "Hello Hermes", "provider": "echo"})
    assert response.status_code == 200
    data = response.json()
    assert data["provider"] == "echo"
    assert data["choices"]
    assert data["choices"][0]["message"]["content"].startswith("[echo]")


def test_llm_endpoint_with_messages():
    """LLM endpoint should honor explicit messages payloads."""
    payload = {
        "messages": [
            {"role": "system", "content": "You are a friendly assistant."},
            {"role": "user", "content": "Say hi"},
        ],
        "max_tokens": 32,
    }
    response = client.post("/llm", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert "hi" in data["choices"][0]["message"]["content"].lower()


def test_llm_forwards_to_sophia_with_provenance(monkeypatch):
    """LLM responses should be forwarded to Sophia with provenance metadata."""
    import httpx
    from unittest.mock import AsyncMock, MagicMock

    # Track the call to Sophia
    captured_request = {}

    class MockResponse:
        status_code = 201
        text = "Created"

    async def mock_post(url, json=None, headers=None):
        captured_request["url"] = url
        captured_request["json"] = json
        captured_request["headers"] = headers
        return MockResponse()

    # Mock httpx.AsyncClient
    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.post = mock_post

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: mock_client)

    # Set Sophia token to enable forwarding
    monkeypatch.setenv("SOPHIA_API_TOKEN", "test-token")
    monkeypatch.setenv("SOPHIA_HOST", "localhost")
    monkeypatch.setenv("SOPHIA_PORT", "8001")

    # Make LLM request
    response = client.post("/llm", json={"prompt": "Hello", "provider": "echo"})
    assert response.status_code == 200

    # Verify Sophia was called with correct provenance
    assert "hermes_proposal" in captured_request.get("url", "")
    payload = captured_request.get("json", {})
    assert payload.get("source_service") == "hermes"
    assert payload.get("llm_provider") == "echo"
    assert payload.get("confidence") == 0.7
    assert payload.get("metadata", {}).get("source") == "hermes_llm"
    assert payload.get("metadata", {}).get("derivation") == "observed"

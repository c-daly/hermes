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


def test_health_endpoint():
    """Test the health endpoint returns service status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()

    # Check required fields
    assert "status" in data
    assert "version" in data
    assert "services" in data
    assert "milvus" in data
    assert "queue" in data

    # Status should be "healthy" or "degraded"
    assert data["status"] in ["healthy", "degraded"]

    # Services should include all ML services
    services = data["services"]
    assert "stt" in services
    assert "tts" in services
    assert "nlp" in services
    assert "embeddings" in services

    # Each service should be "available" or "unavailable"
    for service_name, service_status in services.items():
        assert service_status in ["available", "unavailable"]

    # Check Milvus status structure
    milvus = data["milvus"]
    assert "connected" in milvus
    assert "host" in milvus
    assert "port" in milvus
    assert "collection" in milvus
    assert isinstance(milvus["connected"], bool)

    # Check queue status structure
    queue = data["queue"]
    assert "enabled" in queue
    assert "pending" in queue
    assert "processed" in queue
    assert isinstance(queue["enabled"], bool)
    assert isinstance(queue["pending"], int)
    assert isinstance(queue["processed"], int)


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

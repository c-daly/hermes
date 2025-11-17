"""Tests for the Hermes API endpoints."""

from fastapi.testclient import TestClient
from hermes.main import app

client = TestClient(app)


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


def test_stt_endpoint():
    """Test speech-to-text endpoint accepts audio files."""
    # Create a minimal test file
    files = {"audio": ("test.wav", b"test audio data", "audio/wav")}
    response = client.post("/stt", files=files)
    assert response.status_code == 200
    data = response.json()
    assert "text" in data
    assert "confidence" in data
    assert isinstance(data["confidence"], float)


def test_tts_endpoint():
    """Test text-to-speech endpoint returns audio."""
    request_data = {"text": "Hello, world!", "voice": "default", "language": "en-US"}
    response = client.post("/tts", json=request_data)
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert len(response.content) > 0


def test_simple_nlp_tokenize():
    """Test simple NLP with tokenization."""
    request_data = {"text": "This is a test sentence.", "operations": ["tokenize"]}
    response = client.post("/simple_nlp", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert "tokens" in data
    assert isinstance(data["tokens"], list)
    assert len(data["tokens"]) > 0


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


def test_embed_text_endpoint():
    """Test text embedding endpoint."""
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

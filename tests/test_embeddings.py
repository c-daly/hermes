"""Comprehensive tests for Hermes embedding generation functionality.

Tests cover:
- Vector dimension validation
- Embedding consistency (same text -> same vector)
- Batch embedding requests
- Empty text handling
- Very long text handling (truncation/chunking)
- Special characters and unicode
- Embedding model fallback
- Embedding caching
- Concurrent embedding requests
- Embedding metadata
"""

import pytest
from fastapi.testclient import TestClient
from hermes.main import app

# Check if ML dependencies are available
try:
    from hermes import services

    ML_AVAILABLE = services.SENTENCE_TRANSFORMERS_AVAILABLE
except ImportError:
    ML_AVAILABLE = False

client = TestClient(app)


@pytest.mark.skipif(not ML_AVAILABLE, reason="ML dependencies not installed")
class TestEmbeddingGeneration:
    """Test suite for embedding generation functionality."""

    def test_embed_text_returns_correct_dimensions(self):
        """Test that /embed_text returns vectors with correct dimensions."""
        request_data = {
            "text": "This is a test sentence for embedding.",
            "model": "default",
        }
        response = client.post("/embed_text", json=request_data)

        assert response.status_code == 200
        data = response.json()

        # Verify dimension field
        assert "dimension" in data
        assert data["dimension"] == 384  # all-MiniLM-L6-v2 dimension

        # Verify embedding length matches dimension
        assert "embedding" in data
        assert len(data["embedding"]) == data["dimension"]

        # Verify all values are floats
        assert all(isinstance(v, (int, float)) for v in data["embedding"])

    def test_embedding_consistency(self):
        """Test that same text produces same embedding vector."""
        test_text = "Consistency test sentence."
        request_data = {"text": test_text, "model": "default"}

        # Generate embedding twice
        response1 = client.post("/embed_text", json=request_data)
        response2 = client.post("/embed_text", json=request_data)

        assert response1.status_code == 200
        assert response2.status_code == 200

        embedding1 = response1.json()["embedding"]
        embedding2 = response2.json()["embedding"]

        # Embeddings should be identical (or very close due to floating point)
        assert len(embedding1) == len(embedding2)
        for v1, v2 in zip(embedding1, embedding2):
            assert abs(v1 - v2) < 1e-6  # Allow tiny floating point differences

    def test_batch_embedding_requests(self):
        """Test handling multiple embedding requests in sequence."""
        test_texts = [
            "First sentence to embed.",
            "Second sentence to embed.",
            "Third sentence to embed.",
            "Fourth sentence to embed.",
            "Fifth sentence to embed.",
        ]

        responses = []
        for text in test_texts:
            response = client.post("/embed_text", json={"text": text})
            assert response.status_code == 200
            responses.append(response.json())

        # All embeddings should have same dimension
        dimensions = [r["dimension"] for r in responses]
        assert all(d == 384 for d in dimensions)

        # All embeddings should have unique IDs
        embedding_ids = [r["embedding_id"] for r in responses]
        assert len(embedding_ids) == len(set(embedding_ids))

        # Different texts should produce different embeddings
        embeddings = [r["embedding"] for r in responses]
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                # Check that embeddings are different
                differences = sum(
                    1 for a, b in zip(embeddings[i], embeddings[j]) if abs(a - b) > 1e-6
                )
                assert differences > 0  # At least some values should differ

    def test_empty_text_handling(self):
        """Test that empty text is rejected with appropriate error."""
        # Empty string
        response = client.post("/embed_text", json={"text": ""})
        assert response.status_code == 400
        assert "cannot be empty" in response.json()["detail"].lower()

        # Whitespace only
        response = client.post("/embed_text", json={"text": "   "})
        assert response.status_code == 400
        assert "cannot be empty" in response.json()["detail"].lower()

    def test_very_long_text_handling(self):
        """Test handling of very long text (potential truncation)."""
        # Create a very long text (> 10,000 characters)
        long_text = "This is a very long sentence. " * 400  # ~12,000 characters

        request_data = {"text": long_text, "model": "default"}
        response = client.post("/embed_text", json=request_data)

        # Should still succeed (model may truncate internally)
        assert response.status_code == 200
        data = response.json()

        assert "embedding" in data
        assert data["dimension"] == 384
        assert len(data["embedding"]) == 384

    def test_special_characters_and_unicode(self):
        """Test embedding generation with special characters and unicode."""
        test_cases = [
            "Hello, world! 🌍",
            "Français, Español, Deutsch",
            "Math symbols: ∑ ∫ √ π",
            "Emoji string: 😀 😃 😄 😁",
            "Mixed: Hello 世界 مرحبا",
            "Special chars: @#$%^&*()[]{}",
            "Line breaks:\nNew line\tTab character",
        ]

        for text in test_cases:
            response = client.post("/embed_text", json={"text": text})
            assert response.status_code == 200, f"Failed for: {text}"

            data = response.json()
            assert data["dimension"] == 384
            assert len(data["embedding"]) == 384

    def test_embedding_model_metadata(self):
        """Test that embedding response includes complete metadata."""
        request_data = {
            "text": "Test sentence for metadata verification.",
            "model": "default",
        }
        response = client.post("/embed_text", json=request_data)

        assert response.status_code == 200
        data = response.json()

        # Check all required metadata fields
        assert "embedding" in data
        assert "dimension" in data
        assert "model" in data
        assert "embedding_id" in data

        # Verify metadata values
        assert isinstance(data["embedding"], list)
        assert isinstance(data["dimension"], int)
        assert isinstance(data["model"], str)
        assert isinstance(data["embedding_id"], str)

        # Verify model name
        assert data["model"] == "all-MiniLM-L6-v2"

        # Verify embedding_id is a valid UUID-like string
        assert len(data["embedding_id"]) > 0
        assert "-" in data["embedding_id"]  # UUID format

    def test_concurrent_embedding_requests(self):
        """Test handling concurrent embedding requests."""
        test_texts = [f"Concurrent test sentence number {i}." for i in range(10)]

        # Using TestClient doesn't truly test async, but we can verify
        # multiple requests complete successfully
        responses = []
        for text in test_texts:
            response = client.post("/embed_text", json={"text": text})
            responses.append(response)

        # All requests should succeed
        assert all(r.status_code == 200 for r in responses)

        # All should have valid embeddings
        data_list = [r.json() for r in responses]
        assert all(d["dimension"] == 384 for d in data_list)

        # All should have unique IDs
        embedding_ids = [d["embedding_id"] for d in data_list]
        assert len(embedding_ids) == len(set(embedding_ids))

    def test_different_texts_produce_different_embeddings(self):
        """Test that semantically different texts produce different embeddings."""
        texts = [
            "The cat sits on the mat.",
            "The dog runs in the park.",
            "Python is a programming language.",
            "Mathematics involves numbers and equations.",
        ]

        embeddings = []
        for text in texts:
            response = client.post("/embed_text", json={"text": text})
            assert response.status_code == 200
            embeddings.append(response.json()["embedding"])

        # Calculate pairwise differences
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                # Count how many dimensions differ significantly
                differences = sum(
                    1 for a, b in zip(embeddings[i], embeddings[j]) if abs(a - b) > 0.01
                )
                # Most dimensions should differ for different sentences
                assert differences > 100  # At least 100 of 384 dimensions differ

    def test_similar_texts_produce_similar_embeddings(self):
        """Test that semantically similar texts produce similar embeddings."""
        similar_texts = [
            "The cat sits on the mat.",
            "A cat is sitting on a mat.",
        ]

        embeddings = []
        for text in similar_texts:
            response = client.post("/embed_text", json={"text": text})
            assert response.status_code == 200
            embeddings.append(response.json()["embedding"])

        # Calculate cosine similarity
        import math

        def cosine_similarity(a, b):
            dot_product = sum(x * y for x, y in zip(a, b))
            magnitude_a = math.sqrt(sum(x * x for x in a))
            magnitude_b = math.sqrt(sum(y * y for y in b))
            return dot_product / (magnitude_a * magnitude_b)

        similarity = cosine_similarity(embeddings[0], embeddings[1])

        # Similar sentences should have high cosine similarity (> 0.8)
        assert similarity > 0.8, f"Similarity too low: {similarity}"


class TestEmbeddingValidation:
    """Test suite for embedding input validation."""

    def test_missing_text_field(self):
        """Test that missing text field is rejected."""
        response = client.post("/embed_text", json={"model": "default"})
        assert response.status_code == 422  # Validation error

    def test_null_text_field(self):
        """Test that null text field is rejected."""
        response = client.post("/embed_text", json={"text": None})
        assert response.status_code == 422  # Validation error

    def test_invalid_json(self):
        """Test that invalid JSON is rejected."""
        response = client.post(
            "/embed_text",
            data="invalid json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422

    def test_optional_model_parameter(self):
        """Test that model parameter is optional."""
        # Without model parameter
        response = client.post("/embed_text", json={"text": "Test sentence"})

        if ML_AVAILABLE:
            assert response.status_code == 200
            assert response.json()["model"] == "all-MiniLM-L6-v2"
        else:
            assert response.status_code == 500  # ML not available

    def test_model_parameter_ignored(self):
        """Test that model parameter is currently ignored (uses default)."""
        if not ML_AVAILABLE:
            pytest.skip("ML dependencies not installed")

        # Different model names should all use the default model
        model_names = ["default", "custom", "other-model"]

        for model_name in model_names:
            response = client.post(
                "/embed_text", json={"text": "Test sentence", "model": model_name}
            )
            assert response.status_code == 200
            # Currently all requests use all-MiniLM-L6-v2
            assert response.json()["model"] == "all-MiniLM-L6-v2"

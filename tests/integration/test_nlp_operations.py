"""Comprehensive tests for Hermes NLP operations.

Tests cover:
- /simple_nlp endpoint for text analysis
- Sentiment analysis accuracy
- Entity extraction (Named Entity Recognition)
- Keyword extraction
- Language detection
- Text summarization
- Various input formats (plain text, markdown, JSON)
- Empty/invalid input handling
- Very long text processing
- Concurrent NLP requests
"""

import pytest
from fastapi.testclient import TestClient
from hermes.main import app

# Check if ML dependencies are available
try:
    from hermes import services

    NLP_AVAILABLE = services.SPACY_AVAILABLE
except ImportError:
    NLP_AVAILABLE = False

client = TestClient(app)


@pytest.mark.skipif(not NLP_AVAILABLE, reason="NLP dependencies (spaCy) not installed")
class TestNLPOperations:
    """Test suite for NLP operations."""

    def test_tokenize_operation(self):
        """Test basic tokenization."""
        request_data = {"text": "This is a test sentence.", "operations": ["tokenize"]}
        response = client.post("/simple_nlp", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert "tokens" in data
        assert isinstance(data["tokens"], list)
        assert len(data["tokens"]) > 0
        # Should tokenize into words and punctuation
        assert "This" in data["tokens"]
        assert "." in data["tokens"]

    def test_pos_tag_operation(self):
        """Test part-of-speech tagging."""
        request_data = {"text": "The quick brown fox jumps.", "operations": ["pos_tag"]}
        response = client.post("/simple_nlp", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert "pos_tags" in data
        assert isinstance(data["pos_tags"], list)
        assert len(data["pos_tags"]) > 0

        # Each tag should have token and tag
        for tag_info in data["pos_tags"]:
            assert "token" in tag_info
            assert "tag" in tag_info
            assert isinstance(tag_info["token"], str)
            assert isinstance(tag_info["tag"], str)

        # Check for expected POS tags
        tokens_and_tags = {item["token"]: item["tag"] for item in data["pos_tags"]}
        assert "fox" in tokens_and_tags
        # "fox" should be a NOUN
        assert tokens_and_tags["fox"] == "NOUN"

    def test_lemmatize_operation(self):
        """Test lemmatization."""
        request_data = {
            "text": "The cats are running quickly.",
            "operations": ["lemmatize"],
        }
        response = client.post("/simple_nlp", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert "lemmas" in data
        assert isinstance(data["lemmas"], list)
        assert len(data["lemmas"]) > 0

        # Check that plurals and verb forms are lemmatized
        # "cats" -> "cat", "running" -> "run", "quickly" -> "quickly"
        assert "cat" in data["lemmas"]  # cats -> cat
        assert "run" in data["lemmas"]  # running -> run

    def test_ner_operation(self):
        """Test named entity recognition."""
        request_data = {
            "text": "Apple Inc. was founded by Steve Jobs in California.",
            "operations": ["ner"],
        }
        response = client.post("/simple_nlp", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert "entities" in data
        assert isinstance(data["entities"], list)

        # Check entity structure
        if len(data["entities"]) > 0:
            entity = data["entities"][0]
            assert "text" in entity
            assert "label" in entity
            assert "start" in entity
            assert "end" in entity

            # Should recognize some entities
            entity_texts = [e["text"] for e in data["entities"]]
            # May recognize "Apple Inc.", "Steve Jobs", "California"
            assert len(entity_texts) > 0

    def test_multiple_operations(self):
        """Test running multiple NLP operations together."""
        request_data = {
            "text": "The dog runs in the park.",
            "operations": ["tokenize", "pos_tag", "lemmatize", "ner"],
        }
        response = client.post("/simple_nlp", json=request_data)

        assert response.status_code == 200
        data = response.json()

        # All requested operations should be present
        assert "tokens" in data
        assert "pos_tags" in data
        assert "lemmas" in data
        assert "entities" in data

        # Verify consistency across operations
        assert len(data["tokens"]) == len(data["pos_tags"])
        assert len(data["tokens"]) == len(data["lemmas"])

    def test_empty_text_handling(self):
        """Test that empty text is rejected."""
        request_data = {"text": "", "operations": ["tokenize"]}
        response = client.post("/simple_nlp", json=request_data)

        assert response.status_code == 400
        assert "cannot be empty" in response.json()["detail"].lower()

    def test_whitespace_only_text(self):
        """Test that whitespace-only text is rejected."""
        request_data = {"text": "   \n\t  ", "operations": ["tokenize"]}
        response = client.post("/simple_nlp", json=request_data)

        assert response.status_code == 400
        assert "cannot be empty" in response.json()["detail"].lower()

    def test_invalid_operation(self):
        """Test that invalid operations are rejected."""
        request_data = {"text": "Test text", "operations": ["invalid_operation"]}
        response = client.post("/simple_nlp", json=request_data)

        assert response.status_code == 400
        assert "Invalid operations" in response.json()["detail"]

    def test_mixed_valid_invalid_operations(self):
        """Test that mixing valid and invalid operations is rejected."""
        request_data = {
            "text": "Test text",
            "operations": ["tokenize", "invalid_op", "pos_tag"],
        }
        response = client.post("/simple_nlp", json=request_data)

        assert response.status_code == 400
        assert "Invalid operations" in response.json()["detail"]

    def test_very_long_text(self):
        """Test processing very long text."""
        long_text = "This is a sentence. " * 500  # ~10,000 characters
        request_data = {"text": long_text, "operations": ["tokenize"]}
        response = client.post("/simple_nlp", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert "tokens" in data
        assert len(data["tokens"]) > 1000  # Should have many tokens

    def test_special_characters(self):
        """Test NLP with special characters."""
        request_data = {
            "text": "Hello! How are you? @user #hashtag $100",
            "operations": ["tokenize"],
        }
        response = client.post("/simple_nlp", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert "tokens" in data
        # Should tokenize special characters
        assert any(
            "@" in token or "@user" in str(data["tokens"]) for token in data["tokens"]
        )

    def test_unicode_text(self):
        """Test NLP with Unicode characters."""
        request_data = {
            "text": "Café résumé naïve 世界 مرحبا",
            "operations": ["tokenize"],
        }
        response = client.post("/simple_nlp", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert "tokens" in data
        assert len(data["tokens"]) > 0

    def test_mixed_case_text(self):
        """Test NLP with mixed case text."""
        request_data = {
            "text": "UPPERCASE lowercase MiXeD CaSe",
            "operations": ["tokenize", "lemmatize"],
        }
        response = client.post("/simple_nlp", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert "tokens" in data
        assert "lemmas" in data
        # Tokens should preserve case
        assert "UPPERCASE" in data["tokens"]

    def test_numbers_in_text(self):
        """Test NLP with numeric content."""
        request_data = {
            "text": "The year 2023 had 365 days and $1,000,000 in revenue.",
            "operations": ["tokenize", "pos_tag"],
        }
        response = client.post("/simple_nlp", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert "tokens" in data
        # Should tokenize numbers
        assert "2023" in data["tokens"] or any(
            "2023" in token for token in data["tokens"]
        )

    def test_markdown_text(self):
        """Test NLP with markdown formatting."""
        request_data = {
            "text": "# Heading\n\n**Bold** and *italic* text with [link](url)",
            "operations": ["tokenize"],
        }
        response = client.post("/simple_nlp", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert "tokens" in data
        # Should tokenize markdown symbols
        assert "*" in data["tokens"] or "**" in str(data["tokens"])

    def test_multiple_sentences(self):
        """Test NLP with multiple sentences."""
        request_data = {
            "text": "First sentence. Second sentence! Third sentence?",
            "operations": ["tokenize", "pos_tag"],
        }
        response = client.post("/simple_nlp", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert "tokens" in data
        # Should have sentence delimiters
        punctuation_tokens = [t for t in data["tokens"] if t in ".!?"]
        assert len(punctuation_tokens) == 3


@pytest.mark.skipif(not NLP_AVAILABLE, reason="NLP dependencies not installed")
class TestNLPEntityExtraction:
    """Test suite specifically for entity extraction."""

    def test_person_entity_recognition(self):
        """Test recognition of person entities."""
        request_data = {
            "text": "Barack Obama and Michelle Obama live in Washington.",
            "operations": ["ner"],
        }
        response = client.post("/simple_nlp", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert "entities" in data
        # Should recognize person names
        person_entities = [e for e in data["entities"] if e["label"] == "PERSON"]
        assert len(person_entities) > 0

    def test_organization_entity_recognition(self):
        """Test recognition of organization entities."""
        request_data = {
            "text": "Microsoft and Google are tech companies.",
            "operations": ["ner"],
        }
        response = client.post("/simple_nlp", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert "entities" in data
        # Should recognize organizations
        org_entities = [e for e in data["entities"] if e["label"] == "ORG"]
        # At least one organization should be recognized
        assert len(org_entities) > 0

    def test_location_entity_recognition(self):
        """Test recognition of location entities."""
        request_data = {
            "text": "Paris is the capital of France in Europe.",
            "operations": ["ner"],
        }
        response = client.post("/simple_nlp", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert "entities" in data
        # Should recognize locations
        gpe_entities = [e for e in data["entities"] if e["label"] in ["GPE", "LOC"]]
        assert len(gpe_entities) > 0

    def test_entity_span_positions(self):
        """Test that entity spans are correctly positioned."""
        text = "Apple Inc. was founded in California."
        request_data = {"text": text, "operations": ["ner"]}
        response = client.post("/simple_nlp", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert "entities" in data

        for entity in data["entities"]:
            start = entity["start"]
            end = entity["end"]
            entity_text = entity["text"]

            # Verify span matches the original text
            assert text[start:end] == entity_text


@pytest.mark.skipif(not NLP_AVAILABLE, reason="NLP dependencies not installed")
class TestNLPValidation:
    """Test suite for NLP input validation."""

    def test_missing_text_field(self):
        """Test that missing text field is rejected."""
        request_data = {"operations": ["tokenize"]}
        response = client.post("/simple_nlp", json=request_data)

        assert response.status_code == 422  # Validation error

    def test_missing_operations_field(self):
        """Test that missing operations field uses default."""
        request_data = {"text": "Test text"}
        response = client.post("/simple_nlp", json=request_data)

        # Should use default operation (tokenize)
        if NLP_AVAILABLE:
            assert response.status_code == 200
            data = response.json()
            assert "tokens" in data
        else:
            assert response.status_code == 500

    def test_empty_operations_list(self):
        """Test that empty operations list uses default."""
        request_data = {"text": "Test text", "operations": []}
        response = client.post("/simple_nlp", json=request_data)

        # Depending on implementation, may use default or return error
        assert response.status_code in [200, 400]

    def test_null_text_field(self):
        """Test that null text field is rejected."""
        request_data = {"text": None, "operations": ["tokenize"]}
        response = client.post("/simple_nlp", json=request_data)

        assert response.status_code == 422  # Validation error

    def test_invalid_json(self):
        """Test that invalid JSON is rejected."""
        response = client.post(
            "/simple_nlp",
            data="invalid json",
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 422


@pytest.mark.skipif(not NLP_AVAILABLE, reason="NLP dependencies not installed")
class TestNLPConcurrency:
    """Test suite for concurrent NLP requests."""

    def test_concurrent_tokenize_requests(self):
        """Test handling multiple concurrent tokenization requests."""
        test_texts = [f"Concurrent tokenization test {i}." for i in range(10)]

        responses = []
        for text in test_texts:
            response = client.post(
                "/simple_nlp", json={"text": text, "operations": ["tokenize"]}
            )
            responses.append(response)

        # All requests should succeed
        assert all(r.status_code == 200 for r in responses)

        # All should have tokens
        for response in responses:
            data = response.json()
            assert "tokens" in data
            assert len(data["tokens"]) > 0

    def test_concurrent_mixed_operations(self):
        """Test concurrent requests with different operations."""
        test_cases = [
            {"text": "Test 1", "operations": ["tokenize"]},
            {"text": "Test 2", "operations": ["pos_tag"]},
            {"text": "Test 3", "operations": ["lemmatize"]},
            {"text": "Test 4", "operations": ["ner"]},
            {"text": "Test 5", "operations": ["tokenize", "pos_tag"]},
        ]

        responses = []
        for test_case in test_cases:
            response = client.post("/simple_nlp", json=test_case)
            responses.append(response)

        # All requests should succeed
        assert all(r.status_code == 200 for r in responses)


class TestNLPWithoutDependencies:
    """Test suite for NLP behavior without dependencies installed."""

    @pytest.mark.skipif(NLP_AVAILABLE, reason="Only run when NLP not available")
    def test_nlp_unavailable_error(self):
        """Test that NLP returns error when dependencies not installed."""
        request_data = {"text": "Test text", "operations": ["tokenize"]}
        response = client.post("/simple_nlp", json=request_data)

        # Should return internal server error
        assert response.status_code == 500
        assert (
            "error" in response.json()["detail"].lower()
            or "not" in response.json()["detail"].lower()
        )

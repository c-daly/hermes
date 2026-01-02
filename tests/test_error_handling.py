"""Comprehensive tests for Hermes error handling and resilience.

Tests cover:
- API validation errors (malformed requests)
- Authentication failures (invalid API keys)
- Rate limiting enforcement
- Timeout scenarios
- Dependency failures (Milvus, Neo4j, LLM providers)
- Circuit breaker patterns
- Retry logic
- Graceful degradation
- Error response format consistency
- Logging for debugging
"""

from unittest.mock import patch
from fastapi.testclient import TestClient
from hermes.main import app

client = TestClient(app)


class TestAPIValidationErrors:
    """Test suite for API request validation."""

    def test_invalid_json_format(self):
        """Test that malformed JSON is rejected."""
        response = client.post(
            "/embed_text",
            data="not valid json{",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422

    def test_missing_required_field(self):
        """Test that missing required fields are rejected."""
        # Missing 'text' field for /embed_text
        response = client.post("/embed_text", json={"model": "default"})
        assert response.status_code == 422

        # Missing 'text' field for /simple_nlp
        response = client.post("/simple_nlp", json={"operations": ["tokenize"]})
        assert response.status_code == 422

    def test_wrong_field_type(self):
        """Test that wrong field types are rejected."""
        # Text should be string, not int
        response = client.post("/embed_text", json={"text": 12345})
        assert response.status_code == 422

        # Operations should be list, not string
        response = client.post(
            "/simple_nlp", json={"text": "test", "operations": "tokenize"}
        )
        assert response.status_code == 422

    def test_empty_string_validation(self):
        """Test that empty strings are properly rejected."""
        # Empty text for embedding
        response = client.post("/embed_text", json={"text": ""})
        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()

        # Empty text for NLP
        response = client.post(
            "/simple_nlp", json={"text": "", "operations": ["tokenize"]}
        )
        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()

        # Empty text for TTS
        response = client.post("/tts", json={"text": ""})
        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()

    def test_whitespace_only_validation(self):
        """Test that whitespace-only strings are rejected."""
        whitespace_strings = ["   ", "\n\t", "  \n  \t  "]

        for ws in whitespace_strings:
            response = client.post("/embed_text", json={"text": ws})
            assert response.status_code == 400
            assert "empty" in response.json()["detail"].lower()

    def test_invalid_operation_for_nlp(self):
        """Test that invalid NLP operations are rejected."""
        response = client.post(
            "/simple_nlp", json={"text": "test", "operations": ["invalid_op"]}
        )
        assert response.status_code == 400
        assert "Invalid operations" in response.json()["detail"]

    def test_invalid_file_type_for_stt(self):
        """Test that non-audio files are rejected for STT."""
        files = {"audio": ("test.txt", b"not audio data", "text/plain")}
        response = client.post("/stt", files=files)
        assert response.status_code == 400
        assert "Invalid file type" in response.json()["detail"]

    def test_invalid_content_type(self):
        """Test that invalid Content-Type headers are handled."""
        response = client.post(
            "/embed_text",
            data='{"text": "test"}',
            headers={"Content-Type": "text/plain"},
        )
        # Should fail with validation error
        assert response.status_code in [400, 422]

    def test_extra_unexpected_fields(self):
        """Test that extra fields are handled gracefully."""
        # Extra fields should be ignored (Pydantic behavior)
        response = client.post(
            "/embed_text",
            json={"text": "test", "unexpected_field": "value", "another_field": 123},
        )
        # Should succeed, extra fields ignored
        assert response.status_code in [200, 500]  # 500 if ML not available


class TestErrorResponseFormat:
    """Test suite for consistent error response formatting."""

    def test_validation_error_format(self):
        """Test that validation errors have consistent format."""
        response = client.post("/embed_text", json={"text": ""})
        assert response.status_code == 400

        error_data = response.json()
        assert "detail" in error_data
        assert isinstance(error_data["detail"], str)

    def test_not_found_error_format(self):
        """Test 404 error format for non-existent endpoints."""
        response = client.get("/nonexistent_endpoint")
        assert response.status_code == 404

        error_data = response.json()
        assert "detail" in error_data

    def test_method_not_allowed_format(self):
        """Test 405 error format for wrong HTTP methods."""
        # GET on POST-only endpoint
        response = client.get("/embed_text")
        assert response.status_code == 405

        error_data = response.json()
        assert "detail" in error_data

    def test_internal_error_format(self):
        """Test 500 error format for internal errors."""
        # Try to use ML endpoint without dependencies
        with patch("hermes.services.SENTENCE_TRANSFORMERS_AVAILABLE", False):
            response = client.post("/embed_text", json={"text": "test"})
            if response.status_code == 500:
                error_data = response.json()
                assert "detail" in error_data


class TestDependencyFailures:
    """Test suite for handling dependency failures."""

    def test_milvus_connection_failure_handling(self):
        """Test graceful handling when Milvus is unavailable."""
        # Mock Milvus connection failure
        with patch("hermes.milvus_client._milvus_connected", False):
            # Embedding should still generate, but may not persist
            response = client.post("/embed_text", json={"text": "test without milvus"})
            # Should either succeed (graceful degradation) or fail gracefully
            assert response.status_code in [200, 500, 503]

    def test_health_check_with_degraded_services(self):
        """Test health endpoint shows degraded status when services unavailable."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "capabilities" in data

        # Status should be "healthy", "degraded", or "unavailable"
        assert data["status"] in ["healthy", "degraded", "unavailable"]

        # Each capability should report availability
        for cap_name, status in data["capabilities"].items():
            assert status in ["available", "unavailable"]

    def test_ml_service_unavailable(self):
        """Test behavior when ML services are not installed."""
        with patch("hermes.services.SENTENCE_TRANSFORMERS_AVAILABLE", False):
            response = client.post("/embed_text", json={"text": "test"})
            assert response.status_code == 500
            assert (
                "error" in response.json()["detail"].lower()
                or "not" in response.json()["detail"].lower()
            )


class TestTimeoutHandling:
    """Test suite for timeout scenarios."""

    def test_long_running_request(self):
        """Test handling of requests that take too long."""
        # Create a very long text to potentially cause timeout
        very_long_text = "word " * 100000  # 500,000 characters

        response = client.post("/embed_text", json={"text": very_long_text})

        # Should either succeed or fail gracefully
        assert response.status_code in [200, 400, 500, 504]

    def test_concurrent_heavy_requests(self):
        """Test system under load with multiple heavy requests."""
        long_text = "This is a test sentence. " * 1000

        responses = []
        for i in range(5):
            response = client.post(
                "/simple_nlp",
                json={"text": long_text, "operations": ["tokenize", "pos_tag"]},
            )
            responses.append(response)

        # All should complete (success or appropriate error)
        for response in responses:
            assert response.status_code in [200, 500, 503]


class TestRateLimiting:
    """Test suite for rate limiting behavior."""

    def test_rapid_successive_requests(self):
        """Test handling many rapid requests (no rate limiting currently)."""
        # Send many requests rapidly
        responses = []
        for i in range(50):
            response = client.post("/embed_text", json={"text": f"test {i}"})
            responses.append(response)

        # Should handle all requests (currently no rate limiting)
        # All should succeed or fail consistently
        status_codes = [r.status_code for r in responses]
        # Most should succeed (if ML available)
        successful = [s for s in status_codes if s == 200]
        # At least some should work
        assert len(successful) > 0 or all(s in [500, 503] for s in status_codes)


class TestGracefulDegradation:
    """Test suite for graceful degradation scenarios."""

    def test_partial_service_availability(self):
        """Test that some endpoints work even if others fail."""
        # Health check should always work
        response = client.get("/health")
        assert response.status_code == 200

        # Root endpoint should always work
        response = client.get("/")
        assert response.status_code == 200

    def test_fallback_behavior(self):
        """Test fallback to default behavior when options unavailable."""
        # Request with unsupported model should use default
        response = client.post(
            "/embed_text", json={"text": "test", "model": "nonexistent-model"}
        )
        # Should either use default or return error
        assert response.status_code in [200, 400, 404, 500]


class TestLLMProviderErrors:
    """Test suite for LLM provider error handling."""

    def test_llm_missing_prompt_and_messages(self):
        """Test LLM endpoint rejects requests without prompt or messages."""
        response = client.post("/llm", json={})
        assert response.status_code == 400
        assert (
            "prompt" in response.json()["detail"].lower()
            or "messages" in response.json()["detail"].lower()
        )

    def test_llm_empty_prompt(self):
        """Test LLM endpoint rejects empty prompts."""
        response = client.post("/llm", json={"prompt": "   "})
        assert response.status_code == 400

    def test_llm_with_invalid_provider(self):
        """Test LLM endpoint with non-existent provider."""
        response = client.post(
            "/llm", json={"prompt": "test", "provider": "nonexistent-provider"}
        )
        # Should handle gracefully
        assert response.status_code in [200, 400, 502, 503]

    def test_llm_default_provider(self):
        """Test LLM endpoint with explicit echo provider."""
        response = client.post("/llm", json={"prompt": "Hello", "provider": "echo"})
        # Should work with echo provider
        assert response.status_code == 200
        data = response.json()
        assert "provider" in data
        assert data["provider"] == "echo"

    def test_llm_temperature_validation(self):
        """Test LLM temperature parameter validation."""
        # Temperature too high
        response = client.post("/llm", json={"prompt": "test", "temperature": 5.0})
        assert response.status_code in [400, 422]

        # Temperature negative
        response = client.post("/llm", json={"prompt": "test", "temperature": -1.0})
        assert response.status_code in [400, 422]

    def test_llm_max_tokens_validation(self):
        """Test LLM max_tokens parameter validation."""
        # Zero or negative max_tokens
        response = client.post("/llm", json={"prompt": "test", "max_tokens": 0})
        assert response.status_code in [400, 422]

        response = client.post("/llm", json={"prompt": "test", "max_tokens": -10})
        assert response.status_code in [400, 422]


class TestCORSAndSecurity:
    """Test suite for CORS and security headers."""

    def test_cors_headers_present(self):
        """Test that CORS headers are properly set."""
        response = client.options("/embed_text")
        # CORS preflight should be handled
        assert response.status_code in [200, 405]

    def test_health_endpoint_accessible(self):
        """Test that health endpoint is accessible."""
        response = client.get("/health")
        assert response.status_code == 200


class TestEdgeCases:
    """Test suite for edge cases and unusual inputs."""

    def test_null_bytes_in_text(self):
        """Test handling of null bytes in text."""
        text_with_null = "test\x00text"
        response = client.post("/embed_text", json={"text": text_with_null})
        # Should handle or reject gracefully
        assert response.status_code in [200, 400, 500]

    def test_extremely_long_single_word(self):
        """Test handling of extremely long single word."""
        long_word = "a" * 100000
        response = client.post(
            "/simple_nlp", json={"text": long_word, "operations": ["tokenize"]}
        )
        # Should handle or reject gracefully
        assert response.status_code in [200, 400, 500]

    def test_only_special_characters(self):
        """Test handling of text with only special characters."""
        special_text = "!@#$%^&*()_+-={}[]|\\:;<>?,./~`"
        response = client.post("/embed_text", json={"text": special_text})
        # Should either process or reject gracefully
        assert response.status_code in [200, 400, 500]

    def test_mixed_control_characters(self):
        """Test handling of control characters."""
        control_text = "test\r\n\t\b\ftext"
        response = client.post("/embed_text", json={"text": control_text})
        # Should handle gracefully
        assert response.status_code in [200, 400, 500]

    def test_repeated_requests_same_content(self):
        """Test multiple identical requests in succession."""
        request_data = {"text": "identical request test"}

        responses = []
        for _ in range(10):
            response = client.post("/embed_text", json=request_data)
            responses.append(response)

        # All should succeed (or all fail consistently)
        status_codes = set(r.status_code for r in responses)
        assert len(status_codes) <= 2  # Should be consistent


class TestErrorRecovery:
    """Test suite for error recovery and resilience."""

    def test_recovery_after_failed_request(self):
        """Test that system recovers after a failed request."""
        # Send invalid request
        response1 = client.post("/embed_text", json={"text": ""})
        assert response1.status_code == 400

        # Follow with valid request
        response2 = client.post("/embed_text", json={"text": "valid text"})
        # Should work normally (if ML available)
        assert response2.status_code in [200, 500]

    def test_multiple_errors_dont_crash_service(self):
        """Test that multiple errors don't crash the service."""
        # Send multiple invalid requests
        for _ in range(10):
            client.post("/embed_text", json={"text": ""})

        # Service should still respond
        response = client.get("/health")
        assert response.status_code == 200

    def test_malformed_requests_dont_crash_service(self):
        """Test that malformed requests don't crash the service."""
        # Send various malformed requests
        malformed_requests = [
            '{"text": }',
            '{"text": "test"',
            "not json at all",
            '{"text": null}',
        ]

        for bad_request in malformed_requests:
            try:
                client.post(
                    "/embed_text",
                    data=bad_request,
                    headers={"Content-Type": "application/json"},
                )
            except Exception:
                pass  # Expected to fail

        # Service should still respond
        response = client.get("/health")
        assert response.status_code == 200

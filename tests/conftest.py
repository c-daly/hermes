"""Shared test fixtures and utilities for Hermes tests.

This module provides reusable fixtures for:
- Test data (sample texts, embeddings)
- Database connections (Milvus, Neo4j)
- Mock services
- Test client configuration

Standardization Note:
    logos_test_utils.fixtures provides shared fixtures for all LOGOS repos:
    - stack_env: Parsed .env.test environment
    - neo4j_config: Neo4jConfig dataclass
    - neo4j_driver: Session-scoped driver with wait_for_neo4j

    Hermes-specific fixtures use dict-based configs from hermes.env for
    compatibility with existing tests. New tests should prefer the standardized
    fixtures when possible.
"""

import pytest
from typing import Any
from fastapi.testclient import TestClient
from hermes.main import app
from hermes.env import get_milvus_config, get_neo4j_config, load_env_file

# Re-export shared fixtures from logos_test_utils
# These can be used directly in tests that need the standardized interface
from logos_test_utils.fixtures import stack_env  # noqa: F401

# Load environment configuration at module level (dict-based for compatibility)
_env = load_env_file()
_milvus_config = get_milvus_config(_env)
_neo4j_config = get_neo4j_config(_env)

# Sample test data
SAMPLE_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is transforming technology.",
    "Natural language processing enables human-computer interaction.",
    "Python is a versatile programming language.",
    "Data science combines statistics and programming.",
]

SAMPLE_LONG_TEXT = """
This is a longer sample text for testing purposes. It contains multiple sentences
and paragraphs to simulate real-world document processing scenarios.

The text includes various linguistic features like punctuation, capitalization,
and different sentence structures. This helps ensure that NLP operations work
correctly with realistic input data.

Performance testing also benefits from having representative sample data that
matches the characteristics of production workloads.
"""

SAMPLE_UNICODE_TEXT = [
    "Hello world! 🌍",
    "Café résumé naïve",
    "世界 مرحبا Привет",
    "Special chars: @#$%^&*()",
]


@pytest.fixture
def test_client() -> TestClient:
    """Provide a FastAPI test client.

    Note: This fixture does NOT trigger lifespan events.
    Use `lifespan_client` for tests requiring Milvus initialization.
    """
    return TestClient(app)


@pytest.fixture
def lifespan_client():
    """Provide a FastAPI test client with lifespan management.

    This fixture properly triggers startup/shutdown events, which
    initializes Milvus connection. Use this for integration tests
    that require Milvus persistence.
    """
    with TestClient(app) as client:
        yield client


@pytest.fixture
def sample_short_text() -> str:
    """Provide a short sample text."""
    return "Quick test sentence."


@pytest.fixture
def sample_medium_text() -> str:
    """Provide a medium-length sample text."""
    return " ".join(SAMPLE_TEXTS)


@pytest.fixture
def sample_long_text() -> str:
    """Provide a long sample text."""
    return SAMPLE_LONG_TEXT


@pytest.fixture
def sample_texts() -> list[str]:
    """Provide a list of sample texts."""
    return SAMPLE_TEXTS.copy()


@pytest.fixture
def sample_unicode_texts() -> list[str]:
    """Provide sample texts with Unicode characters."""
    return SAMPLE_UNICODE_TEXT.copy()


# Milvus fixtures
@pytest.fixture
def milvus_connection():
    """Provide a Milvus connection for testing."""
    try:
        from pymilvus import connections

        connections.connect(
            alias="test",
            host=_milvus_config["host"],
            port=_milvus_config["port"],
            timeout=5,
        )
        yield connections
        connections.disconnect("test")
    except Exception as e:
        pytest.skip(f"Milvus not available: {e}")


@pytest.fixture
def milvus_collection_name() -> str:
    """Provide the test collection name."""
    return _milvus_config["collection_name"]


# Neo4j fixtures
@pytest.fixture
def neo4j_driver():
    """Provide a Neo4j driver for testing."""
    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(
            _neo4j_config["uri"],
            auth=(_neo4j_config["user"], _neo4j_config["password"]),
        )
        driver.verify_connectivity()
        yield driver
        driver.close()
    except Exception as e:
        pytest.skip(f"Neo4j not available: {e}")


@pytest.fixture
def neo4j_cleanup(neo4j_driver):
    """Clean up Neo4j test data before and after tests."""
    with neo4j_driver.session() as session:
        session.run("MATCH (n:TestNode) DETACH DELETE n")

    yield

    with neo4j_driver.session() as session:
        session.run("MATCH (n:TestNode) DETACH DELETE n")


# ML model availability fixtures
@pytest.fixture
def ml_available() -> bool:
    """Check if ML dependencies are available."""
    try:
        from hermes import services

        return services.SENTENCE_TRANSFORMERS_AVAILABLE
    except ImportError:
        return False


@pytest.fixture
def nlp_available() -> bool:
    """Check if NLP dependencies are available."""
    try:
        from hermes import services

        return services.SPACY_AVAILABLE
    except ImportError:
        return False


# Mock data fixtures
@pytest.fixture
def mock_embedding_response() -> dict[str, Any]:
    """Provide a mock embedding response."""
    return {
        "embedding": [0.1] * 384,
        "dimension": 384,
        "model": "all-MiniLM-L6-v2",
        "embedding_id": "mock-embedding-001",
    }


@pytest.fixture
def mock_nlp_response() -> dict[str, Any]:
    """Provide a mock NLP response."""
    return {
        "tokens": ["This", "is", "a", "test", "."],
        "pos_tags": [
            {"token": "This", "tag": "DET"},
            {"token": "is", "tag": "VERB"},
            {"token": "a", "tag": "DET"},
            {"token": "test", "tag": "NOUN"},
            {"token": ".", "tag": "PUNCT"},
        ],
        "lemmas": ["this", "be", "a", "test", "."],
        "entities": [],
    }


# Test data generators
@pytest.fixture
def generate_test_texts():
    """Factory fixture to generate test texts."""

    def _generate(count: int, prefix: str = "Test") -> list[str]:
        return [f"{prefix} sentence number {i}." for i in range(count)]

    return _generate


@pytest.fixture
def generate_embedding_data():
    """Factory fixture to generate embedding test data."""

    def _generate(count: int) -> list[dict[str, Any]]:
        return [
            {"text": f"Test embedding {i}", "model": "default"} for i in range(count)
        ]

    return _generate


# Performance test helpers
@pytest.fixture
def measure_latency():
    """Helper to measure operation latency."""
    import time

    def _measure(func, *args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        latency = (time.time() - start) * 1000  # Convert to ms
        return result, latency

    return _measure


@pytest.fixture
def performance_threshold():
    """Provide performance thresholds for tests."""
    return {
        "health_p50": 50,  # ms
        "health_p95": 100,  # ms
        "embedding_p50": 1000,  # ms
        "embedding_p95": 2000,  # ms
        "nlp_p50": 500,  # ms
        "nlp_p95": 1000,  # ms
    }


# Configuration fixtures
@pytest.fixture
def test_config() -> dict[str, Any]:
    """Provide test configuration."""
    return {
        "milvus": {
            "host": _milvus_config["host"],
            "port": _milvus_config["port"],
            "collection": _milvus_config["collection_name"],
        },
        "neo4j": {
            "uri": _neo4j_config["uri"],
            "user": _neo4j_config["user"],
            "password": _neo4j_config["password"],
        },
        "api": {"base_url": "http://localhost:8080", "timeout": 30},
    }


# Cleanup utilities
@pytest.fixture
def cleanup_milvus():
    """Clean up Milvus test data."""
    # Placeholder for future cleanup logic
    yield
    # Cleanup after test if needed


@pytest.fixture
def cleanup_neo4j_test_nodes(neo4j_driver):
    """Clean up specific test node labels."""

    def _cleanup(*labels):
        with neo4j_driver.session() as session:
            for label in labels:
                session.run(f"MATCH (n:{label}) DETACH DELETE n")

    yield _cleanup

    # Final cleanup
    with neo4j_driver.session() as session:
        session.run("MATCH (n:TestNode) DETACH DELETE n")

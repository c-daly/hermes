"""Shared test fixtures for Hermes tests.

Standardized fixtures from logos_test_utils are re-exported here so that
downstream test files can consume them without importing logos_test_utils
directly.  Hermes-specific overrides (e.g. ``neo4j_config`` with
``repo="hermes"``) ensure correct port defaults.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from hermes.main import app

# Re-export shared fixtures from logos_test_utils (optional dependency)
try:
    from logos_test_utils.fixtures import stack_env  # noqa: F401
    from logos_test_utils.neo4j import (
        Neo4jConfig,
        get_neo4j_config,
        get_neo4j_driver,
        wait_for_neo4j,
    )

    _HAS_TEST_UTILS = True
except ImportError:
    _HAS_TEST_UTILS = False


# ---------------------------------------------------------------------------
# Neo4j fixtures (hermes-specific: repo="hermes" for correct port defaults)
# ---------------------------------------------------------------------------

if _HAS_TEST_UTILS:

    @pytest.fixture(scope="session")
    def neo4j_config(stack_env: dict[str, str]) -> Neo4jConfig:  # noqa: F811
        """Return hermes-specific Neo4j configuration."""
        return get_neo4j_config(stack_env, repo="hermes")

    @pytest.fixture(scope="session")
    def neo4j_driver(neo4j_config: Neo4jConfig):
        """Provide a shared Neo4j driver for hermes tests."""
        try:
            wait_for_neo4j(neo4j_config)
        except Exception:
            pytest.skip("Neo4j is not available")
        driver = get_neo4j_driver(neo4j_config)
        yield driver
        driver.close()


# ---------------------------------------------------------------------------
# FastAPI test client fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def test_client() -> TestClient:
    """Provide a FastAPI test client.

    Note: This fixture does NOT trigger lifespan events.
    Use ``lifespan_client`` for tests requiring Milvus initialization.
    """
    return TestClient(app)


@pytest.fixture
def lifespan_client():
    """Provide a FastAPI test client with lifespan management.

    This fixture properly triggers startup/shutdown events, which
    initializes Milvus connection.  Use this for integration tests
    that require Milvus persistence.
    """
    with TestClient(app) as client:
        yield client

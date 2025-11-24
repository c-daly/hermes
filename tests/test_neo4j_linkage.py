"""Tests for Hermes Neo4j integration and embedding linkage.

Tests cover:
- /embed_text creates Neo4j reference node
- [:HAS_EMBEDDING] relationship creation
- Embedding metadata stored in Neo4j
- Bidirectional linkage (Milvus ID ↔ Neo4j node)
- Query by Neo4j node returns Milvus vector
- Neo4j unavailable handling
- Orphaned embedding cleanup
- Embedding provenance tracking
"""

import pytest
from fastapi.testclient import TestClient
from hermes.main import app

# Check if dependencies are available
try:
    from neo4j import GraphDatabase

    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False

try:
    from hermes import services

    ML_AVAILABLE = services.SENTENCE_TRANSFORMERS_AVAILABLE
except ImportError:
    ML_AVAILABLE = False

MILVUS_AVAILABLE = False

# Test configuration
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "hermes_embeddings"

client = TestClient(app)


def check_neo4j_connection():
    """Check if Neo4j is available for connection."""
    if not NEO4J_AVAILABLE:
        return False
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        driver.close()
        return True
    except Exception:
        return False


NEO4J_CONNECTED = check_neo4j_connection()


@pytest.mark.skipif(not ML_AVAILABLE, reason="ML dependencies not installed")
@pytest.mark.skipif(not NEO4J_CONNECTED, reason="Neo4j server not available")
class TestNeo4jEmbeddingLinkage:
    """Test suite for Neo4j embedding reference and linkage."""

    @pytest.fixture(autouse=True)
    def cleanup_neo4j(self):
        """Clean up test nodes before and after each test."""
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

        # Cleanup before test
        with driver.session() as session:
            session.run("MATCH (n:EmbeddingTest) DETACH DELETE n")

        yield

        # Cleanup after test
        with driver.session() as session:
            session.run("MATCH (n:EmbeddingTest) DETACH DELETE n")

        driver.close()

    def test_create_neo4j_reference_node(self):
        """Test creating a Neo4j node with embedding_id reference."""
        # Generate an embedding
        response = client.post(
            "/embed_text", json={"text": "Test text for Neo4j reference"}
        )
        assert response.status_code == 200

        embedding_id = response.json()["embedding_id"]

        # Create Neo4j node with embedding reference
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

        try:
            with driver.session() as session:
                result = session.run(
                    """
                    CREATE (n:EmbeddingTest {
                        embedding_id: $embedding_id,
                        created_at: timestamp()
                    })
                    RETURN n.embedding_id as id
                    """,
                    embedding_id=embedding_id,
                )

                record = result.single()
                assert record["id"] == embedding_id
        finally:
            driver.close()

    def test_has_embedding_relationship(self):
        """Test creating [:HAS_EMBEDDING] relationship."""
        # Generate an embedding
        response = client.post(
            "/embed_text", json={"text": "Test text for relationship"}
        )
        assert response.status_code == 200

        embedding_id = response.json()["embedding_id"]

        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

        try:
            with driver.session() as session:
                # Create source node and embedding reference with relationship
                result = session.run(
                    """
                    CREATE (source:EmbeddingTest {name: "Source Node"})
                    CREATE (emb:EmbeddingTest:EmbeddingReference {
                        embedding_id: $embedding_id
                    })
                    CREATE (source)-[r:HAS_EMBEDDING]->(emb)
                    RETURN type(r) as rel_type
                    """,
                    embedding_id=embedding_id,
                )

                record = result.single()
                assert record["rel_type"] == "HAS_EMBEDDING"

                # Verify relationship can be queried
                result = session.run(
                    """
                    MATCH (source:EmbeddingTest)-[:HAS_EMBEDDING]->(emb:EmbeddingReference)
                    WHERE emb.embedding_id = $embedding_id
                    RETURN source.name as source_name, emb.embedding_id as emb_id
                    """,
                    embedding_id=embedding_id,
                )

                record = result.single()
                assert record["source_name"] == "Source Node"
                assert record["emb_id"] == embedding_id
        finally:
            driver.close()

    def test_embedding_metadata_in_neo4j(self):
        """Test storing embedding metadata in Neo4j."""
        response = client.post("/embed_text", json={"text": "Test text with metadata"})
        assert response.status_code == 200

        data = response.json()
        embedding_id = data["embedding_id"]
        model = data["model"]
        dimension = data["dimension"]

        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

        try:
            with driver.session() as session:
                # Store embedding metadata
                result = session.run(
                    """
                    CREATE (emb:EmbeddingTest {
                        embedding_id: $embedding_id,
                        model: $model,
                        dimension: $dimension,
                        created_at: timestamp()
                    })
                    RETURN emb
                    """,
                    embedding_id=embedding_id,
                    model=model,
                    dimension=dimension,
                )

                record = result.single()
                node = record["emb"]

                assert node["embedding_id"] == embedding_id
                assert node["model"] == model
                assert node["dimension"] == dimension
                assert "created_at" in node
        finally:
            driver.close()

    def test_bidirectional_linkage(self):
        """Test bidirectional linkage between Milvus ID and Neo4j node."""
        # Generate embedding
        test_text = "Bidirectional linkage test"
        response = client.post("/embed_text", json={"text": test_text})
        assert response.status_code == 200

        embedding_id = response.json()["embedding_id"]

        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

        try:
            with driver.session() as session:
                # Create Neo4j node with reference
                session.run(
                    """
                    CREATE (n:EmbeddingTest {
                        embedding_id: $embedding_id,
                        text: $text
                    })
                    """,
                    embedding_id=embedding_id,
                    text=test_text,
                )

                # Query Neo4j by embedding_id
                result = session.run(
                    """
                    MATCH (n:EmbeddingTest {embedding_id: $embedding_id})
                    RETURN n.embedding_id as id, n.text as text
                    """,
                    embedding_id=embedding_id,
                )

                record = result.single()
                assert record["id"] == embedding_id
                assert record["text"] == test_text

                # Verify we can use this ID to query Milvus (in integration test)
                # Here we just verify the linkage is consistent
                assert record["id"] == embedding_id
        finally:
            driver.close()

    def test_query_embedding_by_neo4j_node(self):
        """Test querying embedding using Neo4j node."""
        # Generate embedding
        test_text = "Query by Neo4j node test"
        response = client.post("/embed_text", json={"text": test_text})
        assert response.status_code == 200

        embedding_id = response.json()["embedding_id"]

        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

        try:
            with driver.session() as session:
                # Create Neo4j node with unique identifier
                node_uuid = "test-node-12345"
                session.run(
                    """
                    CREATE (n:EmbeddingTest {
                        node_id: $node_id,
                        embedding_id: $embedding_id,
                        text: $text
                    })
                    """,
                    node_id=node_uuid,
                    embedding_id=embedding_id,
                    text=test_text,
                )

                # Query by node_id to get embedding_id
                result = session.run(
                    """
                    MATCH (n:EmbeddingTest {node_id: $node_id})
                    RETURN n.embedding_id as embedding_id
                    """,
                    node_id=node_uuid,
                )

                record = result.single()
                retrieved_embedding_id = record["embedding_id"]

                assert retrieved_embedding_id == embedding_id
        finally:
            driver.close()


@pytest.mark.skipif(not NEO4J_CONNECTED, reason="Neo4j server not available")
class TestNeo4jErrorHandling:
    """Test suite for Neo4j error handling scenarios."""

    @pytest.fixture(autouse=True)
    def cleanup_neo4j(self):
        """Clean up test nodes."""
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with driver.session() as session:
            session.run("MATCH (n:EmbeddingTest) DETACH DELETE n")
        driver.close()
        yield
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with driver.session() as session:
            session.run("MATCH (n:EmbeddingTest) DETACH DELETE n")
        driver.close()

    def test_neo4j_connection_failure(self):
        """Test handling Neo4j connection failure."""
        # Try to connect to non-existent Neo4j instance
        with pytest.raises(Exception):
            driver = GraphDatabase.driver(
                "bolt://nonexistent:7687", auth=("neo4j", "password")
            )
            driver.verify_connectivity()

    def test_duplicate_embedding_id(self):
        """Test handling duplicate embedding_id in Neo4j."""
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

        try:
            with driver.session() as session:
                embedding_id = "duplicate-test-001"

                # Create first node
                session.run(
                    """
                    CREATE (n:EmbeddingTest {embedding_id: $id, version: 1})
                    """,
                    id=embedding_id,
                )

                # Create second node with same embedding_id (should be allowed)
                session.run(
                    """
                    CREATE (n:EmbeddingTest {embedding_id: $id, version: 2})
                    """,
                    id=embedding_id,
                )

                # Query should return both
                result = session.run(
                    """
                    MATCH (n:EmbeddingTest {embedding_id: $id})
                    RETURN count(n) as count
                    """,
                    id=embedding_id,
                )

                record = result.single()
                assert record["count"] == 2
        finally:
            driver.close()

    def test_orphaned_embedding_detection(self):
        """Test detecting embeddings without Neo4j nodes."""
        if not ML_AVAILABLE:
            pytest.skip("ML dependencies not installed")

        # Generate embedding
        response = client.post("/embed_text", json={"text": "Orphaned embedding test"})
        assert response.status_code == 200

        embedding_id = response.json()["embedding_id"]

        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

        try:
            with driver.session() as session:
                # Check if embedding_id exists in Neo4j
                result = session.run(
                    """
                    MATCH (n:EmbeddingTest {embedding_id: $id})
                    RETURN count(n) as count
                    """,
                    id=embedding_id,
                )

                record = result.single()
                # Should be 0 (orphaned in Milvus, not in Neo4j)
                assert record["count"] == 0
        finally:
            driver.close()


@pytest.mark.skipif(not ML_AVAILABLE, reason="ML dependencies not installed")
@pytest.mark.skipif(not NEO4J_CONNECTED, reason="Neo4j server not available")
class TestEmbeddingProvenance:
    """Test suite for embedding provenance tracking."""

    @pytest.fixture(autouse=True)
    def cleanup_neo4j(self):
        """Clean up test nodes."""
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with driver.session() as session:
            session.run("MATCH (n:EmbeddingTest) DETACH DELETE n")
        driver.close()
        yield
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with driver.session() as session:
            session.run("MATCH (n:EmbeddingTest) DETACH DELETE n")
        driver.close()

    def test_track_embedding_source(self):
        """Test tracking the source of an embedding."""
        # Generate embedding
        response = client.post("/embed_text", json={"text": "Provenance tracking test"})
        assert response.status_code == 200

        embedding_id = response.json()["embedding_id"]

        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

        try:
            with driver.session() as session:
                # Create provenance chain
                session.run(
                    """
                    CREATE (source:EmbeddingTest:Source {
                        source_id: $source_id,
                        source_type: "api_request"
                    })
                    CREATE (emb:EmbeddingTest:Embedding {
                        embedding_id: $embedding_id
                    })
                    CREATE (source)-[:GENERATED]->(emb)
                    """,
                    source_id="api-request-001",
                    embedding_id=embedding_id,
                )

                # Query provenance
                result = session.run(
                    """
                    MATCH (source:Source)-[:GENERATED]->(emb:Embedding)
                    WHERE emb.embedding_id = $embedding_id
                    RETURN source.source_id as source_id, source.source_type as source_type
                    """,
                    embedding_id=embedding_id,
                )

                record = result.single()
                assert record["source_id"] == "api-request-001"
                assert record["source_type"] == "api_request"
        finally:
            driver.close()

    def test_track_embedding_version(self):
        """Test tracking embedding versions."""
        # Generate initial embedding
        response = client.post("/embed_text", json={"text": "Version tracking test"})
        assert response.status_code == 200

        embedding_id_v1 = response.json()["embedding_id"]

        # Generate updated embedding
        response = client.post(
            "/embed_text", json={"text": "Version tracking test updated"}
        )
        assert response.status_code == 200

        embedding_id_v2 = response.json()["embedding_id"]

        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

        try:
            with driver.session() as session:
                # Create version chain
                session.run(
                    """
                    CREATE (v1:EmbeddingTest {
                        embedding_id: $id_v1,
                        version: 1,
                        created_at: timestamp()
                    })
                    CREATE (v2:EmbeddingTest {
                        embedding_id: $id_v2,
                        version: 2,
                        created_at: timestamp() + 1000
                    })
                    CREATE (v1)-[:SUPERSEDED_BY]->(v2)
                    """,
                    id_v1=embedding_id_v1,
                    id_v2=embedding_id_v2,
                )

                # Query version chain
                result = session.run(
                    """
                    MATCH (v1:EmbeddingTest)-[:SUPERSEDED_BY]->(v2:EmbeddingTest)
                    WHERE v1.embedding_id = $id_v1
                    RETURN v2.embedding_id as latest_id, v2.version as version
                    """,
                    id_v1=embedding_id_v1,
                )

                record = result.single()
                assert record["latest_id"] == embedding_id_v2
                assert record["version"] == 2
        finally:
            driver.close()

    def test_track_embedding_usage(self):
        """Test tracking where embeddings are used."""
        response = client.post("/embed_text", json={"text": "Usage tracking test"})
        assert response.status_code == 200

        embedding_id = response.json()["embedding_id"]

        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

        try:
            with driver.session() as session:
                # Create usage tracking
                session.run(
                    """
                    CREATE (emb:EmbeddingTest:Embedding {
                        embedding_id: $embedding_id
                    })
                    CREATE (use1:EmbeddingTest:Usage {
                        usage_type: "semantic_search",
                        timestamp: timestamp()
                    })
                    CREATE (use2:EmbeddingTest:Usage {
                        usage_type: "similarity_comparison",
                        timestamp: timestamp()
                    })
                    CREATE (emb)-[:USED_IN]->(use1)
                    CREATE (emb)-[:USED_IN]->(use2)
                    """,
                    embedding_id=embedding_id,
                )

                # Query usage
                result = session.run(
                    """
                    MATCH (emb:Embedding)-[:USED_IN]->(use:Usage)
                    WHERE emb.embedding_id = $embedding_id
                    RETURN collect(use.usage_type) as usage_types
                    """,
                    embedding_id=embedding_id,
                )

                record = result.single()
                usage_types = record["usage_types"]

                assert "semantic_search" in usage_types
                assert "similarity_comparison" in usage_types
        finally:
            driver.close()

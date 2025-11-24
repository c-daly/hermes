"""Comprehensive integration tests for Hermes workflows.

Tests cover:
- Complete embedding workflow: text → embed → Milvus → Neo4j
- Semantic search: query → embed → Milvus search → results
- Cross-service integration: Sophia → Hermes → embeddings
- Proposal ingestion (text proposals, not media)
- Embedding versioning (model updates)
- Data consistency across Milvus and Neo4j
"""

import pytest
import time
from fastapi.testclient import TestClient
from hermes.main import app

# Check if dependencies are available
try:
    from pymilvus import connections, Collection

    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False

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

# Test configuration
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "hermes_embeddings"
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"

client = TestClient(app)


def check_milvus_connection():
    """Check if Milvus is available."""
    if not MILVUS_AVAILABLE:
        return False
    try:
        connections.connect(
            alias="default", host=MILVUS_HOST, port=MILVUS_PORT, timeout=5
        )
        connections.disconnect("default")
        return True
    except Exception:
        return False


def check_neo4j_connection():
    """Check if Neo4j is available."""
    if not NEO4J_AVAILABLE:
        return False
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        driver.close()
        return True
    except Exception:
        return False


MILVUS_CONNECTED = check_milvus_connection()
NEO4J_CONNECTED = check_neo4j_connection()
ALL_SERVICES = ML_AVAILABLE and MILVUS_CONNECTED and NEO4J_CONNECTED


@pytest.mark.skipif(not ALL_SERVICES, reason="Requires ML, Milvus, and Neo4j")
class TestCompleteEmbeddingWorkflow:
    """Test complete embedding workflow end-to-end."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        """Clean up test data before and after tests."""
        # Cleanup Neo4j
        if NEO4J_CONNECTED:
            driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
            with driver.session() as session:
                session.run("MATCH (n:IntegrationTest) DETACH DELETE n")
            driver.close()

        yield

        # Cleanup after test
        if NEO4J_CONNECTED:
            driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
            with driver.session() as session:
                session.run("MATCH (n:IntegrationTest) DETACH DELETE n")
            driver.close()

    def test_text_to_embedding_to_storage(self):
        """Test complete workflow: text → embedding → Milvus → Neo4j."""
        test_text = "Integration test: complete workflow"

        # Step 1: Generate embedding via API
        response = client.post("/embed_text", json={"text": test_text})
        assert response.status_code == 200

        data = response.json()
        embedding_id = data["embedding_id"]
        model = data["model"]

        # Step 2: Verify embedding in Milvus
        connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)

        try:
            collection = Collection(COLLECTION_NAME)
            collection.load()
            time.sleep(0.5)

            results = collection.query(
                expr=f'embedding_id == "{embedding_id}"',
                output_fields=["embedding_id", "text", "model"],
            )

            assert len(results) == 1
            assert results[0]["embedding_id"] == embedding_id
            assert results[0]["text"] == test_text
            assert results[0]["model"] == model

        finally:
            connections.disconnect("default")

        # Step 3: Create Neo4j reference
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

        try:
            with driver.session() as session:
                session.run(
                    """
                    CREATE (n:IntegrationTest:TextNode {
                        embedding_id: $embedding_id,
                        original_text: $text
                    })
                    """,
                    embedding_id=embedding_id,
                    text=test_text,
                )

                # Verify Neo4j storage
                result = session.run(
                    """
                    MATCH (n:IntegrationTest {embedding_id: $embedding_id})
                    RETURN n.embedding_id as id, n.original_text as text
                    """,
                    embedding_id=embedding_id,
                )

                record = result.single()
                assert record["id"] == embedding_id
                assert record["text"] == test_text
        finally:
            driver.close()

    def test_semantic_search_workflow(self):
        """Test semantic search: query → embed → Milvus search → results."""
        # Step 1: Insert test documents
        test_docs = [
            "The cat sits on the mat.",
            "A dog runs in the park.",
            "Machine learning is fascinating.",
        ]

        embedding_ids = []
        for doc in test_docs:
            response = client.post("/embed_text", json={"text": doc})
            assert response.status_code == 200
            embedding_ids.append(response.json()["embedding_id"])

        time.sleep(1)  # Allow time for indexing

        # Step 2: Generate query embedding
        query_text = "Cats and dogs"
        response = client.post("/embed_text", json={"text": query_text})
        assert response.status_code == 200

        query_embedding = response.json()["embedding"]

        # Step 3: Search Milvus for similar embeddings
        connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)

        try:
            collection = Collection(COLLECTION_NAME)
            collection.load()

            search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
            results = collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=3,
                output_fields=["embedding_id", "text"],
            )

            # Should find the cat and dog documents
            assert len(results) == 1
            assert len(results[0]) >= 2

            top_texts = [hit.entity.get("text") for hit in results[0]]
            # Cat and dog sentences should be in top results
            assert any("cat" in text.lower() for text in top_texts)

        finally:
            connections.disconnect("default")

    def test_multiple_embeddings_linkage(self):
        """Test managing multiple embeddings linked to same entity."""
        # Create entity in Neo4j
        entity_id = "entity-001"

        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

        try:
            with driver.session() as session:
                session.run(
                    """
                    CREATE (e:IntegrationTest:Entity {
                        entity_id: $entity_id,
                        name: "Test Entity"
                    })
                    """,
                    entity_id=entity_id,
                )

            # Generate multiple embeddings for different aspects
            aspects = [
                "This entity is about technology.",
                "This entity relates to innovation.",
                "This entity involves data processing.",
            ]

            for aspect_text in aspects:
                # Generate embedding
                response = client.post("/embed_text", json={"text": aspect_text})
                assert response.status_code == 200

                embedding_id = response.json()["embedding_id"]

                # Link embedding to entity in Neo4j
                with driver.session() as session:
                    session.run(
                        """
                        MATCH (e:Entity {entity_id: $entity_id})
                        CREATE (emb:IntegrationTest:Embedding {
                            embedding_id: $embedding_id,
                            text: $text
                        })
                        CREATE (e)-[:HAS_EMBEDDING]->(emb)
                        """,
                        entity_id=entity_id,
                        embedding_id=embedding_id,
                        text=aspect_text,
                    )

            # Verify all embeddings linked
            with driver.session() as session:
                result = session.run(
                    """
                    MATCH (e:Entity {entity_id: $entity_id})-[:HAS_EMBEDDING]->(emb:Embedding)
                    RETURN count(emb) as count
                    """,
                    entity_id=entity_id,
                )

                record = result.single()
                assert record["count"] == len(aspects)

        finally:
            driver.close()


@pytest.mark.skipif(not ALL_SERVICES, reason="Requires ML, Milvus, and Neo4j")
class TestDataConsistency:
    """Test data consistency across Milvus and Neo4j."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        """Clean up test data."""
        if NEO4J_CONNECTED:
            driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
            with driver.session() as session:
                session.run("MATCH (n:IntegrationTest) DETACH DELETE n")
            driver.close()
        yield
        if NEO4J_CONNECTED:
            driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
            with driver.session() as session:
                session.run("MATCH (n:IntegrationTest) DETACH DELETE n")
            driver.close()

    def test_embedding_id_consistency(self):
        """Test that embedding_id is consistent across services."""
        test_text = "Consistency check text"

        # Generate embedding
        response = client.post("/embed_text", json={"text": test_text})
        assert response.status_code == 200

        embedding_id = response.json()["embedding_id"]

        # Check Milvus
        connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)

        try:
            collection = Collection(COLLECTION_NAME)
            collection.load()
            time.sleep(0.5)

            milvus_results = collection.query(
                expr=f'embedding_id == "{embedding_id}"',
                output_fields=["embedding_id", "text"],
            )

            assert len(milvus_results) == 1
            milvus_id = milvus_results[0]["embedding_id"]
        finally:
            connections.disconnect("default")

        # Store in Neo4j
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

        try:
            with driver.session() as session:
                session.run(
                    """
                    CREATE (n:IntegrationTest {embedding_id: $id})
                    """,
                    id=embedding_id,
                )

                result = session.run(
                    """
                    MATCH (n:IntegrationTest {embedding_id: $id})
                    RETURN n.embedding_id as neo4j_id
                    """,
                    id=embedding_id,
                )

                neo4j_id = result.single()["neo4j_id"]
        finally:
            driver.close()

        # All IDs should match
        assert embedding_id == milvus_id == neo4j_id

    def test_metadata_consistency(self):
        """Test that metadata is consistent across services."""
        test_text = "Metadata consistency test"

        response = client.post("/embed_text", json={"text": test_text})
        assert response.status_code == 200

        data = response.json()
        embedding_id = data["embedding_id"]
        model = data["model"]

        # Check Milvus metadata
        connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)

        try:
            collection = Collection(COLLECTION_NAME)
            collection.load()
            time.sleep(0.5)

            results = collection.query(
                expr=f'embedding_id == "{embedding_id}"',
                output_fields=["model", "text"],
            )

            milvus_model = results[0]["model"]
            milvus_text = results[0]["text"]
        finally:
            connections.disconnect("default")

        # Store in Neo4j
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

        try:
            with driver.session() as session:
                session.run(
                    """
                    CREATE (n:IntegrationTest {
                        embedding_id: $id,
                        model: $model,
                        text: $text
                    })
                    """,
                    id=embedding_id,
                    model=model,
                    text=test_text,
                )

                result = session.run(
                    """
                    MATCH (n:IntegrationTest {embedding_id: $id})
                    RETURN n.model as neo4j_model, n.text as neo4j_text
                    """,
                    id=embedding_id,
                )

                record = result.single()
                neo4j_model = record["neo4j_model"]
                neo4j_text = record["neo4j_text"]
        finally:
            driver.close()

        # Metadata should match
        assert model == milvus_model == neo4j_model
        assert test_text == milvus_text == neo4j_text


@pytest.mark.skipif(not ML_AVAILABLE, reason="Requires ML dependencies")
class TestProposalIngestion:
    """Test text proposal ingestion workflow."""

    def test_ingest_text_proposal(self):
        """Test ingesting a text proposal and generating embeddings."""
        proposal = {
            "title": "Improve Documentation",
            "description": "We should improve the documentation for better clarity.",
            "author": "test_user",
        }

        # Generate embeddings for title and description
        title_response = client.post("/embed_text", json={"text": proposal["title"]})
        desc_response = client.post(
            "/embed_text", json={"text": proposal["description"]}
        )

        assert title_response.status_code == 200
        assert desc_response.status_code == 200

        title_embedding_id = title_response.json()["embedding_id"]
        desc_embedding_id = desc_response.json()["embedding_id"]

        # Embeddings should be different
        assert title_embedding_id != desc_embedding_id

    def test_multi_paragraph_proposal(self):
        """Test ingesting a multi-paragraph proposal."""
        long_proposal = """
        Paragraph 1: This is the introduction to our proposal.
        
        Paragraph 2: This section describes the problem we're trying to solve.
        
        Paragraph 3: Here we present our proposed solution.
        
        Paragraph 4: Finally, we discuss the expected benefits.
        """

        response = client.post("/embed_text", json={"text": long_proposal})
        assert response.status_code == 200

        data = response.json()
        assert data["dimension"] == 384
        assert len(data["embedding"]) == 384


@pytest.mark.skipif(not ML_AVAILABLE, reason="Requires ML dependencies")
class TestCrossServiceIntegration:
    """Test integration with other LOGOS services."""

    def test_nlp_then_embedding(self):
        """Test NLP processing followed by embedding generation."""
        test_text = "This is a test for cross-service integration."

        # Step 1: Process with NLP
        nlp_response = client.post(
            "/simple_nlp", json={"text": test_text, "operations": ["tokenize", "ner"]}
        )
        assert nlp_response.status_code == 200

        nlp_data = nlp_response.json()
        assert "tokens" in nlp_data

        # Step 2: Generate embedding
        embed_response = client.post("/embed_text", json={"text": test_text})
        assert embed_response.status_code == 200

        embed_data = embed_response.json()
        assert "embedding_id" in embed_data

    def test_batch_processing_workflow(self):
        """Test batch processing of multiple texts."""
        texts = [
            "First document for batch processing.",
            "Second document for batch processing.",
            "Third document for batch processing.",
        ]

        # Process all texts
        results = []
        for text in texts:
            response = client.post("/embed_text", json={"text": text})
            assert response.status_code == 200
            results.append(response.json())

        # All should have unique IDs
        embedding_ids = [r["embedding_id"] for r in results]
        assert len(embedding_ids) == len(set(embedding_ids))

        # All should have same dimension
        dimensions = [r["dimension"] for r in results]
        assert all(d == 384 for d in dimensions)


@pytest.mark.skipif(not ALL_SERVICES, reason="Requires ML, Milvus, and Neo4j")
class TestEmbeddingVersioning:
    """Test embedding versioning when models are updated."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        """Clean up test data."""
        if NEO4J_CONNECTED:
            driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
            with driver.session() as session:
                session.run("MATCH (n:IntegrationTest) DETACH DELETE n")
            driver.close()
        yield
        if NEO4J_CONNECTED:
            driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
            with driver.session() as session:
                session.run("MATCH (n:IntegrationTest) DETACH DELETE n")
            driver.close()

    def test_track_embedding_versions(self):
        """Test tracking different versions of embeddings."""
        test_text = "Version tracking test"

        # Generate version 1
        response_v1 = client.post("/embed_text", json={"text": test_text})
        assert response_v1.status_code == 200

        embedding_id_v1 = response_v1.json()["embedding_id"]
        model_v1 = response_v1.json()["model"]

        # Store in Neo4j with version
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

        try:
            with driver.session() as session:
                session.run(
                    """
                    CREATE (v1:IntegrationTest:EmbeddingVersion {
                        embedding_id: $id,
                        model: $model,
                        version: 1,
                        text: $text,
                        created_at: timestamp()
                    })
                    """,
                    id=embedding_id_v1,
                    model=model_v1,
                    text=test_text,
                )

                # Verify version tracking
                result = session.run(
                    """
                    MATCH (v:EmbeddingVersion {text: $text})
                    RETURN count(v) as version_count
                    """,
                    text=test_text,
                )

                count = result.single()["version_count"]
                assert count == 1
        finally:
            driver.close()

    def test_model_metadata_tracking(self):
        """Test tracking which model generated each embedding."""
        response = client.post("/embed_text", json={"text": "Model tracking test"})
        assert response.status_code == 200

        data = response.json()
        model = data["model"]

        # Model should be tracked
        assert model == "all-MiniLM-L6-v2"

        # Verify in Milvus
        connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)

        try:
            collection = Collection(COLLECTION_NAME)
            collection.load()
            time.sleep(0.5)

            results = collection.query(
                expr=f'embedding_id == "{data["embedding_id"]}"',
                output_fields=["model"],
            )

            assert len(results) == 1
            assert results[0]["model"] == model
        finally:
            connections.disconnect("default")

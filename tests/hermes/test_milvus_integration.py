"""Functional test for Hermes embeddings persisted to Milvus.

This test verifies:
1. Hermes generates embeddings with a small model (all-MiniLM-L6-v2)
2. Embeddings are stored in Milvus with the correct schema
3. Response contains embedding_id and model metadata
4. Embeddings can be read back from Milvus
5. (Optional) embedding_id can be written to and read from Neo4j

Test runs with skip flag if Milvus is unavailable.
"""

import pytest
import time
from fastapi.testclient import TestClient
from hermes.main import app

# Check if integration dependencies are available
try:
    from pymilvus import (
        connections,
        Collection,
        CollectionSchema,
        FieldSchema,
        DataType,
        utility,
    )

    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False

try:
    from neo4j import GraphDatabase

    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False

# Check if ML dependencies are available
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
    """Check if Milvus is available for connection."""
    if not MILVUS_AVAILABLE:
        return False
    try:
        connections.connect(
            alias="default", host=MILVUS_HOST, port=MILVUS_PORT, timeout=5
        )
        return True
    except Exception:
        return False
    finally:
        try:
            connections.disconnect("default")
        except Exception:
            pass


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


# Determine if tests should run
MILVUS_CONNECTED = check_milvus_connection()
NEO4J_CONNECTED = check_neo4j_connection()


def create_milvus_collection():
    """Get or create Milvus collection with the schema from c-daly/logos#155.

    Schema for hermes_embeddings collection:
    - embedding_id: VARCHAR (primary key)
    - embedding: FLOAT_VECTOR (dimension 384 for all-MiniLM-L6-v2)
    - model: VARCHAR
    - text: VARCHAR (original text)
    - timestamp: INT64 (creation timestamp)

    Note: This function no longer drops the collection if it exists, to avoid
    invalidating collection references held by the FastAPI app.
    """
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)

    # Check if collection already exists (created by app on startup)
    if utility.has_collection(COLLECTION_NAME):
        # Use the existing collection instead of dropping/recreating
        collection = Collection(name=COLLECTION_NAME)
        return collection

    # Define schema (only used if collection doesn't exist yet)
    fields = [
        FieldSchema(
            name="embedding_id",
            dtype=DataType.VARCHAR,
            is_primary=True,
            max_length=64,
        ),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
        FieldSchema(name="model", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="timestamp", dtype=DataType.INT64),
    ]

    schema = CollectionSchema(
        fields=fields, description="Hermes text embeddings collection"
    )

    # Create collection
    collection = Collection(name=COLLECTION_NAME, schema=schema)

    # Create index for vector field
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 128},
    }
    collection.create_index(field_name="embedding", index_params=index_params)

    return collection


@pytest.mark.skipif(
    not ML_AVAILABLE, reason="ML dependencies (sentence-transformers) not installed"
)
@pytest.mark.skipif(not MILVUS_CONNECTED, reason="Milvus server not available")
def test_embedding_persisted_to_milvus():
    """Test that embeddings are generated and persisted to Milvus correctly.

    This test verifies:
    1. POST /embed_text generates an embedding with the small model
    2. Response contains embedding_id, embedding, dimension, and model
    3. Embedding is stored in Milvus with correct schema
    4. Embedding can be read back from Milvus
    """
    # Create Milvus collection
    collection = create_milvus_collection()

    try:
        # Step 1: Call /embed_text endpoint
        test_text = "This is a test sentence for embedding."
        request_data = {"text": test_text, "model": "default"}

        response = client.post("/embed_text", json=request_data)
        assert response.status_code == 200

        # Step 2: Verify response contains required fields
        data = response.json()
        assert "embedding" in data
        assert "dimension" in data
        assert "model" in data
        assert "embedding_id" in data

        embedding_id = data["embedding_id"]
        embedding = data["embedding"]
        model = data["model"]
        dimension = data["dimension"]

        # Verify embedding properties
        assert isinstance(embedding_id, str)
        assert len(embedding_id) > 0
        assert isinstance(embedding, list)
        assert dimension == 384  # all-MiniLM-L6-v2 dimension
        assert len(embedding) == dimension
        assert model == "all-MiniLM-L6-v2"

        # Step 3: Verify embedding was automatically persisted to Milvus
        # Get a fresh collection reference to see the data persisted by the endpoint
        collection = Collection(name=COLLECTION_NAME)

        # Load collection to enable search (compatible with multiple pymilvus versions)
        try:
            collection.load()
        except Exception:
            # Some pymilvus builds may raise on load or behave differently; ignore
            # and rely on the query retry loop below to detect persistence.
            pass

        # Wait for collection to be fully loaded if the API exposes is_loaded.
        load_timeout = 30  # seconds
        load_start = time.time()
        if hasattr(collection, "is_loaded"):
            while (
                not collection.is_loaded
                and (time.time() - load_start) < load_timeout
            ):
                time.sleep(0.5)
        else:
            # Give Milvus a short grace period before starting queries (the query
            # retry loop below will handle longer waits).
            time.sleep(1)

        # Step 4: Read back embedding from Milvus with retry logic
        # Query by embedding_id (primary key)
        # Use retry mechanism since Milvus indexing can be slow in CI
        max_retries = 20  # Increased for slow CI environments
        retry_delay = 2  # Start with 2 seconds
        results = []

        for attempt in range(max_retries):
            results = collection.query(
                expr=f'embedding_id == "{embedding_id}"',
                output_fields=["embedding_id", "model", "text", "timestamp"],
            )
            if len(results) > 0:
                break
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                # Exponential backoff with max of 5 seconds
                retry_delay = min(retry_delay * 1.5, 5)

        # Verify the embedding was automatically persisted
        assert (
            len(results) == 1
        ), f"Expected 1 result, got {len(results)} after {max_retries} retries"
        result = results[0]
        assert result["embedding_id"] == embedding_id
        assert result["model"] == model
        assert result["text"] == test_text
        assert "timestamp" in result

        # Step 5: Verify vector search works
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        search_results = collection.search(
            data=[embedding],
            anns_field="embedding",
            param=search_params,
            limit=1,
            output_fields=["embedding_id", "text"],
        )

        assert len(search_results) == 1
        assert len(search_results[0]) == 1
        top_result = search_results[0][0]
        assert top_result.entity.get("embedding_id") == embedding_id
        assert top_result.entity.get("text") == test_text

    finally:
        # Cleanup
        connections.disconnect("default")


@pytest.mark.skipif(
    not ML_AVAILABLE, reason="ML dependencies (sentence-transformers) not installed"
)
@pytest.mark.skipif(not MILVUS_CONNECTED, reason="Milvus server not available")
@pytest.mark.skipif(not NEO4J_CONNECTED, reason="Neo4j server not available")
def test_embedding_id_in_neo4j():
    """Optional test: Write embedding_id to Neo4j node and read it back.

    This test verifies:
    1. embedding_id can be written to a Neo4j node
    2. embedding_id can be read back from Neo4j
    """
    # Generate an embedding
    test_text = "Test sentence for Neo4j integration."
    request_data = {"text": test_text, "model": "default"}

    response = client.post("/embed_text", json=request_data)
    assert response.status_code == 200

    data = response.json()
    embedding_id = data["embedding_id"]

    # Connect to Neo4j
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    try:
        # Write embedding_id to Neo4j node
        with driver.session() as session:
            # Create a node with the embedding_id
            result = session.run(
                """
                CREATE (n:EmbeddingTest {
                    embedding_id: $embedding_id,
                    text: $text,
                    created_at: timestamp()
                })
                RETURN n.embedding_id as embedding_id
                """,
                embedding_id=embedding_id,
                text=test_text,
            )
            record = result.single()
            assert record["embedding_id"] == embedding_id

            # Read back the embedding_id
            result = session.run(
                """
                MATCH (n:EmbeddingTest {embedding_id: $embedding_id})
                RETURN n.embedding_id as embedding_id, n.text as text
                """,
                embedding_id=embedding_id,
            )
            record = result.single()
            assert record is not None
            assert record["embedding_id"] == embedding_id
            assert record["text"] == test_text

            # Cleanup: delete test node
            session.run(
                """
                MATCH (n:EmbeddingTest {embedding_id: $embedding_id})
                DELETE n
                """,
                embedding_id=embedding_id,
            )

    finally:
        driver.close()


@pytest.mark.skipif(
    not ML_AVAILABLE, reason="ML dependencies (sentence-transformers) not installed"
)
def test_embedding_response_includes_metadata():
    """Test that embedding response includes embedding_id and model metadata.

    This test can run without Milvus/Neo4j and just verifies the API response.
    """
    test_text = "Test sentence for metadata verification."
    request_data = {"text": test_text, "model": "default"}

    response = client.post("/embed_text", json=request_data)
    assert response.status_code == 200

    data = response.json()

    # Verify all required fields are present
    assert "embedding" in data
    assert "dimension" in data
    assert "model" in data
    assert "embedding_id" in data

    # Verify field types and values
    assert isinstance(data["embedding_id"], str)
    assert len(data["embedding_id"]) > 0
    assert isinstance(data["embedding"], list)
    assert isinstance(data["dimension"], int)
    assert isinstance(data["model"], str)

    # Verify specific values
    assert data["dimension"] == 384
    assert data["model"] == "all-MiniLM-L6-v2"
    assert len(data["embedding"]) == data["dimension"]

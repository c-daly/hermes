"""Comprehensive tests for Hermes Milvus integration.

This test suite covers:
1. Vector insertion to Milvus collection
2. Vector search (similarity query)
3. Batch insertion
4. Collection creation/initialization
5. Schema validation
6. Duplicate handling
7. Vector retrieval by ID
8. Filtering with metadata
9. Connection error handling
10. Milvus unavailable scenario
11. Index creation and optimization

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
NEO4J_PASSWORD = "neo4jtest"

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
    """Create Milvus collection with the schema from c-daly/logos#155.

    Schema for hermes_embeddings collection:
    - embedding_id: VARCHAR (primary key)
    - embedding: FLOAT_VECTOR (dimension 384 for all-MiniLM-L6-v2)
    - model: VARCHAR
    - text: VARCHAR (original text)
    - timestamp: INT64 (creation timestamp)
    """
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)

    # Drop collection if it exists
    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)

    # Define schema
    fields = [
        FieldSchema(
            name="embedding_id", dtype=DataType.VARCHAR, is_primary=True, max_length=64
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
        # Load collection to enable search
        collection.load()

        # Give Milvus a moment to index
        time.sleep(0.5)

        # Step 4: Read back embedding from Milvus
        # Query by embedding_id (primary key)
        results = collection.query(
            expr=f'embedding_id == "{embedding_id}"',
            output_fields=["embedding_id", "model", "text", "timestamp"],
        )

        # Verify the embedding was automatically persisted
        assert len(results) == 1
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


# Additional comprehensive Milvus tests


@pytest.mark.skipif(not MILVUS_CONNECTED, reason="Milvus server not available")
class TestMilvusVectorOperations:
    """Test suite for Milvus vector insertion and retrieval."""

    def test_vector_insertion(self):
        """Test inserting a single vector into Milvus."""
        collection = create_milvus_collection()

        try:
            # Insert test vector
            test_id = "test-vector-001"
            test_vector = [0.1] * 384
            test_model = "test-model"
            test_text = "Test insertion text"
            test_timestamp = int(time.time() * 1000)

            data = [
                [test_id],
                [test_vector],
                [test_model],
                [test_text],
                [test_timestamp],
            ]
            collection.insert(data)
            collection.flush()

            # Load and query
            collection.load()
            time.sleep(0.5)

            results = collection.query(
                expr=f'embedding_id == "{test_id}"',
                output_fields=["embedding_id", "model", "text", "timestamp"],
            )

            assert len(results) == 1
            assert results[0]["embedding_id"] == test_id
            assert results[0]["model"] == test_model
            assert results[0]["text"] == test_text

        finally:
            connections.disconnect("default")

    def test_batch_vector_insertion(self):
        """Test inserting multiple vectors in a batch."""
        collection = create_milvus_collection()

        try:
            # Create batch of test vectors
            num_vectors = 100
            ids = [f"batch-vector-{i:03d}" for i in range(num_vectors)]
            vectors = [[0.1 * (i % 10)] * 384 for i in range(num_vectors)]
            models = ["batch-model"] * num_vectors
            texts = [f"Batch text {i}" for i in range(num_vectors)]
            timestamps = [int(time.time() * 1000)] * num_vectors

            data = [ids, vectors, models, texts, timestamps]
            collection.insert(data)
            collection.flush()

            # Load and count
            collection.load()
            time.sleep(0.5)

            results = collection.query(
                expr='model == "batch-model"', output_fields=["embedding_id"], limit=200
            )

            assert len(results) == num_vectors

        finally:
            connections.disconnect("default")

    def test_vector_search_similarity(self):
        """Test similarity search for nearest neighbors."""
        collection = create_milvus_collection()

        try:
            # Insert test vectors with known similarities
            base_vector = [1.0] * 384
            similar_vector = [0.99] * 384
            different_vector = [-1.0] * 384

            ids = ["similar-base", "similar-close", "similar-far"]
            vectors = [base_vector, similar_vector, different_vector]
            models = ["test"] * 3
            texts = ["base", "close", "far"]
            timestamps = [int(time.time() * 1000)] * 3

            data = [ids, vectors, models, texts, timestamps]
            collection.insert(data)
            collection.flush()

            # Load and search
            collection.load()
            time.sleep(0.5)

            search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
            results = collection.search(
                data=[base_vector],
                anns_field="embedding",
                param=search_params,
                limit=3,
                output_fields=["embedding_id", "text"],
            )

            assert len(results) == 1
            assert len(results[0]) >= 1

            # The closest result should be base itself or similar
            top_result = results[0][0]
            assert top_result.entity.get("embedding_id") in [
                "similar-base",
                "similar-close",
            ]

        finally:
            connections.disconnect("default")

    def test_vector_retrieval_by_id(self):
        """Test retrieving specific vector by embedding_id."""
        collection = create_milvus_collection()

        try:
            test_id = "retrieve-by-id-001"
            test_vector = [0.5] * 384
            test_text = "Retrieval test text"

            data = [
                [test_id],
                [test_vector],
                ["retrieval-model"],
                [test_text],
                [int(time.time() * 1000)],
            ]
            collection.insert(data)
            collection.flush()

            collection.load()
            time.sleep(0.5)

            # Query by primary key
            results = collection.query(
                expr=f'embedding_id == "{test_id}"',
                output_fields=["embedding_id", "text", "model"],
            )

            assert len(results) == 1
            assert results[0]["embedding_id"] == test_id
            assert results[0]["text"] == test_text

        finally:
            connections.disconnect("default")


@pytest.mark.skipif(not MILVUS_CONNECTED, reason="Milvus server not available")
class TestMilvusCollectionManagement:
    """Test suite for Milvus collection management."""

    def test_collection_creation(self):
        """Test creating a new collection with schema."""
        connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)

        try:
            test_collection_name = "test_collection_creation"

            # Drop if exists
            if utility.has_collection(test_collection_name):
                utility.drop_collection(test_collection_name)

            # Create collection
            fields = [
                FieldSchema(
                    name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64
                ),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=128),
            ]
            schema = CollectionSchema(fields=fields)
            Collection(name=test_collection_name, schema=schema)

            assert utility.has_collection(test_collection_name)

            # Cleanup
            utility.drop_collection(test_collection_name)

        finally:
            connections.disconnect("default")

    def test_collection_schema_validation(self):
        """Test that collection has correct schema."""
        collection = create_milvus_collection()

        try:
            schema = collection.schema

            # Check field names
            field_names = [field.name for field in schema.fields]
            assert "embedding_id" in field_names
            assert "embedding" in field_names
            assert "model" in field_names
            assert "text" in field_names
            assert "timestamp" in field_names

            # Check primary key
            primary_field = next(f for f in schema.fields if f.is_primary)
            assert primary_field.name == "embedding_id"

            # Check vector dimension
            vector_field = next(f for f in schema.fields if f.name == "embedding")
            assert vector_field.dtype == DataType.FLOAT_VECTOR
            assert vector_field.params["dim"] == 384

        finally:
            connections.disconnect("default")

    def test_duplicate_id_handling(self):
        """Test that duplicate embedding_id is handled appropriately."""
        collection = create_milvus_collection()

        try:
            test_id = "duplicate-test-001"
            vector1 = [0.1] * 384
            vector2 = [0.2] * 384

            # Insert first vector
            data1 = [
                [test_id],
                [vector1],
                ["model1"],
                ["text1"],
                [int(time.time() * 1000)],
            ]
            collection.insert(data1)
            collection.flush()

            # Attempt to insert duplicate ID
            # Milvus allows this but we should be aware
            data2 = [
                [test_id],
                [vector2],
                ["model2"],
                ["text2"],
                [int(time.time() * 1000)],
            ]

            # This should work (Milvus allows duplicates in primary key before 2.4)
            # or fail (Milvus 2.4+ enforces unique primary keys)
            try:
                collection.insert(data2)
                collection.flush()
            except Exception:
                # Expected in newer Milvus versions
                pass

        finally:
            connections.disconnect("default")

    def test_index_creation(self):
        """Test creating index on vector field."""
        connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)

        try:
            test_collection_name = "test_index_collection"

            if utility.has_collection(test_collection_name):
                utility.drop_collection(test_collection_name)

            # Create collection
            fields = [
                FieldSchema(
                    name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64
                ),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=128),
            ]
            schema = CollectionSchema(fields=fields)
            collection = Collection(name=test_collection_name, schema=schema)

            # Create index
            index_params = {
                "index_type": "IVF_FLAT",
                "metric_type": "L2",
                "params": {"nlist": 128},
            }
            collection.create_index(field_name="vector", index_params=index_params)

            # Verify index exists
            index_info = collection.index()
            assert index_info is not None

            # Cleanup
            utility.drop_collection(test_collection_name)

        finally:
            connections.disconnect("default")


@pytest.mark.skipif(not MILVUS_CONNECTED, reason="Milvus server not available")
class TestMilvusErrorHandling:
    """Test suite for Milvus error handling."""

    def test_query_nonexistent_id(self):
        """Test querying for non-existent embedding_id."""
        collection = create_milvus_collection()

        try:
            collection.load()

            results = collection.query(
                expr='embedding_id == "nonexistent-id-999"',
                output_fields=["embedding_id"],
            )

            # Should return empty results
            assert len(results) == 0

        finally:
            connections.disconnect("default")

    def test_invalid_vector_dimension(self):
        """Test inserting vector with wrong dimension."""
        collection = create_milvus_collection()

        try:
            # Try to insert 256-dim vector into 384-dim field
            data = [
                ["wrong-dim-001"],
                [[0.1] * 256],  # Wrong dimension
                ["test"],
                ["test text"],
                [int(time.time() * 1000)],
            ]

            with pytest.raises(Exception):
                collection.insert(data)
                collection.flush()

        finally:
            connections.disconnect("default")

    def test_connection_timeout(self):
        """Test handling connection timeout."""
        # Try to connect to non-existent host
        with pytest.raises(Exception):
            connections.connect(
                alias="timeout_test", host="nonexistent-host", port="19530", timeout=1
            )


@pytest.mark.skipif(not MILVUS_CONNECTED, reason="Milvus server not available")
class TestMilvusMetadataFiltering:
    """Test suite for metadata filtering in Milvus queries."""

    def test_filter_by_model(self):
        """Test filtering vectors by model metadata."""
        collection = create_milvus_collection()

        try:
            # Insert vectors with different models
            ids = [f"model-filter-{i}" for i in range(10)]
            vectors = [[float(i)] * 384 for i in range(10)]
            models = ["model-A"] * 5 + ["model-B"] * 5
            texts = [f"text {i}" for i in range(10)]
            timestamps = [int(time.time() * 1000)] * 10

            data = [ids, vectors, models, texts, timestamps]
            collection.insert(data)
            collection.flush()

            collection.load()
            time.sleep(0.5)

            # Query for specific model
            results = collection.query(
                expr='model == "model-A"',
                output_fields=["embedding_id", "model"],
                limit=10,
            )

            assert len(results) == 5
            assert all(r["model"] == "model-A" for r in results)

        finally:
            connections.disconnect("default")

    def test_filter_by_timestamp_range(self):
        """Test filtering vectors by timestamp range."""
        collection = create_milvus_collection()

        try:
            current_time = int(time.time() * 1000)

            # Insert vectors with different timestamps
            ids = [f"time-filter-{i}" for i in range(10)]
            vectors = [[float(i)] * 384 for i in range(10)]
            models = ["time-test"] * 10
            texts = [f"text {i}" for i in range(10)]
            timestamps = [current_time + i * 1000 for i in range(10)]

            data = [ids, vectors, models, texts, timestamps]
            collection.insert(data)
            collection.flush()

            collection.load()
            time.sleep(0.5)

            # Query for timestamp range
            mid_time = current_time + 5000
            results = collection.query(
                expr=f"timestamp > {mid_time}",
                output_fields=["embedding_id", "timestamp"],
                limit=10,
            )

            assert len(results) > 0
            assert all(r["timestamp"] > mid_time for r in results)

        finally:
            connections.disconnect("default")

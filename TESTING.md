# Testing the Milvus Integration Implementation

This document provides a guide for testing the new Milvus integration functionality.

## Quick Test (No External Services)

Test that the API now returns `embedding_id`:

```bash
# Start the Hermes server
poetry run hermes

# In another terminal, test the API
curl -X POST http://localhost:8080/embed_text \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, world!", "model": "default"}'
```

Expected response should include:
```json
{
  "embedding": [...],
  "dimension": 384,
  "model": "all-MiniLM-L6-v2",
  "embedding_id": "uuid-here"
}
```

## Full Integration Test with Milvus

### 1. Start Services

Run the shared test stack helper (recommended):

```bash
./scripts/run_integration_stack.sh
```

The script will:

- Warn you about any conflicting ports (7474/7687/19530/9091)
- Start `etcd`, `minio`, `milvus`, and `neo4j` via `docker-compose.test.yml`
- Wait for each container to report healthy, tailing logs automatically on failure
- Export the expected `NEO4J_*` and `MILVUS_*` variables
- Run `poetry run pytest tests/test_milvus_integration.py -v` (pass additional pytest args to override)

Environment overrides (`NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `MILVUS_HOST`, `MILVUS_PORT`) are honored, making it easy to target an already running shared stack instead of launching new containers.

If you need to keep the stack running for manual testing, stop the script with `CTRL+C` *after* the services report healthy and before pytest launches, or start services manually with:

```bash
docker compose -f docker-compose.test.yml up -d
```

This starts:
- Milvus (with etcd and MinIO dependencies)
- Neo4j
- Hermes

### 2. Wait for Services to be Ready

Skip this step if you used `./scripts/run_integration_stack.sh`; it already polls container health and dumps logs on failure.

```bash
# Wait for Milvus
timeout 120 bash -c 'until curl -f http://localhost:19530/healthz 2>/dev/null; do echo "Waiting for Milvus..."; sleep 2; done'

# Wait for Neo4j
timeout 60 bash -c 'until nc -z localhost 7687; do echo "Waiting for Neo4j..."; sleep 2; done'
```

### 3. Install ML Dependencies

```bash
poetry run pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
poetry run pip install sentence-transformers
```

### 4. Run Integration Tests

```bash
# Run all integration tests
poetry run pytest tests/test_milvus_integration.py -v

# Or run specific tests
poetry run pytest tests/test_milvus_integration.py::test_embedding_persisted_to_milvus -v
poetry run pytest tests/test_milvus_integration.py::test_embedding_id_in_neo4j -v
```

### 5. Cleanup

```bash
docker-compose -f docker-compose.test.yml down -v
```

## Manual Testing Steps

### Test 1: Verify embedding_id in Response

```python
from fastapi.testclient import TestClient
from hermes.main import app

client = TestClient(app)
response = client.post("/embed_text", json={"text": "Test text", "model": "default"})
data = response.json()

assert "embedding_id" in data
assert isinstance(data["embedding_id"], str)
assert len(data["embedding_id"]) > 0
print(f"✓ Embedding ID: {data['embedding_id']}")
```

### Test 2: Verify Milvus Persistence

```python
from pymilvus import connections, Collection
import time

# Connect to Milvus
connections.connect(alias="default", host="localhost", port="19530")

# Get the collection (assumes test created it)
collection = Collection("hermes_embeddings")
collection.load()

# Query for a specific embedding_id
embedding_id = "your-embedding-id-here"
results = collection.query(
    expr=f'embedding_id == "{embedding_id}"',
    output_fields=["embedding_id", "model", "text"]
)

print(f"✓ Found {len(results)} embeddings in Milvus")
for result in results:
    print(f"  - ID: {result['embedding_id']}")
    print(f"  - Model: {result['model']}")
    print(f"  - Text: {result['text']}")
```

### Test 3: Verify Neo4j Integration

```python
from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

with driver.session() as session:
    # Query for test nodes
    result = session.run(
        "MATCH (n:EmbeddingTest) RETURN n.embedding_id as id, n.text as text LIMIT 10"
    )
    
    for record in result:
        print(f"✓ Neo4j Node: {record['id']} - {record['text']}")

driver.close()
```

## Troubleshooting

### Issue: Tests Skip Due to Missing ML Dependencies

**Solution:** Install sentence-transformers:
```bash
poetry run pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
poetry run pip install sentence-transformers
```

### Issue: Cannot Connect to Milvus

**Check if Milvus is running:**
```bash
docker ps | grep milvus
curl http://localhost:19530/healthz
```

**View Milvus logs:**
```bash
docker logs milvus-standalone
```

### Issue: Cannot Connect to Neo4j

**Check if Neo4j is running:**
```bash
docker ps | grep neo4j
```

**Test connection:**
```bash
cypher-shell -u neo4j -p password "RETURN 1"
```

### Issue: Port Already in Use

If you get "port already in use" errors:

```bash
# Check what's using the port
lsof -i :19530  # Milvus
lsof -i :7687   # Neo4j
lsof -i :8080   # Hermes

# Stop existing services
docker-compose -f docker-compose.test.yml down
```

## CI/CD Testing

The integration tests run automatically in GitHub Actions CI when:
1. Code is pushed to `main` or `develop` branches
2. A PR has the `integration-test` label

To trigger integration tests on a PR:
```bash
gh pr edit <PR_NUMBER> --add-label integration-test
```

## Expected Test Results

All tests should pass with the following outcomes:

1. **test_embedding_response_includes_metadata**: 
   - Verifies API response contains embedding_id
   - Should pass even without ML dependencies installed
   - Skip: Only if sentence-transformers not installed

2. **test_embedding_persisted_to_milvus**:
   - Creates Milvus collection with proper schema
   - Stores embedding with metadata
   - Reads back and verifies data
   - Skip: If Milvus not available or ML deps not installed

3. **test_embedding_id_in_neo4j**:
   - Writes embedding_id to Neo4j node
   - Reads back and verifies
   - Skip: If Neo4j not available or ML deps not installed

## Performance Notes

- First embedding request may take 10-30 seconds (model loading)
- Subsequent requests are much faster (~100-500ms)
- Milvus startup takes 60-90 seconds
- Neo4j startup takes 30-40 seconds

## References

- [Milvus Documentation](https://milvus.io/docs)
- [Neo4j Python Driver](https://neo4j.com/docs/python-manual/current/)
- [Sentence Transformers](https://www.sbert.net/)
- [tests/README.md](tests/README.md) - Integration test documentation

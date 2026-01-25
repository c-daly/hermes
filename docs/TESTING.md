# Hermes Testing Guide

This document provides a comprehensive guide for testing Hermes, including CI behavior, local testing, and ML dependency management.

## CI Testing Behavior

### Coverage Requirement

All CI jobs enforce a **60% minimum coverage** threshold. Tests will fail if coverage drops below 60%.

Current coverage: **~63%** (without ML deps)

### CI Jobs Overview

| Job | Triggers | ML Deps | Services | Purpose |
|-----|----------|---------|----------|---------|
| **standard** | All PRs/pushes | ❌ No | ✅ Milvus, Neo4j | Lint + type check + full test suite (via reusable workflow) |
| **ml-full-test** | Weekly (Sun 3am UTC), manual dispatch, or `ml-test` label | ✅ Yes | ✅ Milvus, Neo4j | Full test suite with all ML dependencies |

### Standard CI

The **standard** job runs on every PR and push. It uses the reusable CI workflow from logos and:
- Starts Milvus and Neo4j via docker-compose
- Runs ALL tests in `tests/` directory
- Does NOT install ML dependencies (saves 5-10 minutes)
- Enforces 75% minimum coverage
- ML tests skip gracefully with clear messages

**Expected output:**
```
=========== 97 passed, 93 skipped ===========
```

The 93 skipped tests are ML/NLP tests - this is expected behavior.

### Full ML Test Job (Opt-in)

The **ml-full-test** job runs the complete test suite with ML dependencies. Trigger it by:

1. **Manual dispatch**: Go to Actions → CI/CD → Run workflow → Check "Install ML dependencies"
2. **PR label**: Add the `ml-test` label to your PR

This job:
- Installs `poetry install -E dev -E ml`
- Installs CPU-only PyTorch for speed
- Runs against live Milvus + Neo4j
- Should have minimal skips (~1-2)

**Use this before merging ML/NLP changes!**

### Before Merging ML/NLP Changes

**You MUST run the full test suite locally with ML dependencies before merging changes to ML or NLP code:**

```bash
cd hermes
poetry install -E dev -E ml  # Install ML extras
poetry run pytest tests/ -v  # Run full suite
```

All ML tests should PASS locally before merging.

---

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

- Warn you about any conflicting ports (17530/17091 for Milvus)
- Start `milvus-etcd`, `milvus-minio`, and `milvus` via `tests/e2e/stack/hermes/docker-compose.test.yml`
- Wait for each container to report healthy, tailing logs automatically on failure
- Export the expected `MILVUS_*` variables (port 17530)
- Run `poetry run pytest tests/test_milvus_integration.py -v` (pass additional pytest args to override)

Environment overrides (`NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `MILVUS_HOST`, `MILVUS_PORT`) are honored, making it easy to target an already running shared stack instead of launching new containers.

If you need to keep the stack running for manual testing, stop the script with `CTRL+C` *after* the services report healthy and before pytest launches, or start services manually with:

```bash
docker compose -f tests/e2e/stack/hermes/docker-compose.test.yml up -d
```

This starts:
- Milvus (with etcd and MinIO dependencies) on port 17530

### 2. Wait for Services to be Ready

Skip this step if you used `./scripts/run_integration_stack.sh`; it already polls container health and dumps logs on failure.

```bash
# Wait for Milvus (port 17091 for health, 17530 for gRPC)
timeout 120 bash -c 'until curl -f http://localhost:17091/healthz 2>/dev/null; do echo "Waiting for Milvus..."; sleep 2; done'
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
docker compose -f tests/e2e/stack/hermes/docker-compose.test.yml down -v
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
connections.connect(alias="default", host="localhost", port="17530")

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

driver = GraphDatabase.driver("bolt://localhost:17687", auth=("neo4j", "password"))

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
curl http://localhost:17530/healthz
```

**View Milvus logs:**
```bash
docker logs hermes-test-milvus
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
lsof -i :17530  # Milvus
lsof -i :17687   # Neo4j
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

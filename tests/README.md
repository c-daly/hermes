# Hermes Test Suite - Phase 2 Testing

This directory contains comprehensive tests for the Hermes API, covering Phase 2 requirements as specified in [issue #24](https://github.com/c-daly/hermes/issues/24).

## Test Files Overview

### Unit Tests
- **`test_embeddings.py`** - Embedding generation tests (vector dimensions, consistency, batching, edge cases)
- **`test_nlp_operations.py`** - NLP functionality tests (tokenization, POS tagging, NER, etc.)
- **`test_error_handling.py`** - Error handling and resilience tests
- **`test_api.py`** - API endpoint and contract tests (existing)

### Integration Tests
- **`test_milvus_integration.py`** - Comprehensive Milvus vector database integration tests
- **`test_neo4j_linkage.py`** - Neo4j graph database integration and relationship tests
- **`test_hermes_integration.py`** - End-to-end workflow tests

### Performance Tests
- **`test_performance.py`** - Latency, throughput, and load testing

### Test Infrastructure
- **`conftest.py`** - Shared fixtures and test utilities

---

# Original Milvus Integration Testing Documentation

## Overview

The `test_milvus_integration.py` file contains tests that verify:

1. **Embedding Generation**: Hermes generates embeddings with a small model (all-MiniLM-L6-v2)
2. **Milvus Persistence**: Embeddings are stored in Milvus with the correct schema
3. **Response Metadata**: Response contains `embedding_id` and `model` metadata
4. **Data Retrieval**: Embeddings can be read back from Milvus
5. **Neo4j Integration (Optional)**: `embedding_id` can be written to and read from Neo4j

## Running Integration Tests Locally

### Recommended: Shared Test Stack Helper

Launch the dependencies and run the Milvus/Neo4j integration suite in one step:

```bash
./scripts/run_integration_stack.sh
```

The helper checks for port conflicts, waits for each container to become healthy, streams logs on failure, and finally executes `poetry run pytest tests/test_milvus_integration.py -v`. Pass any additional pytest arguments to the script to override the default command.

Environment overrides such as `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `MILVUS_HOST`, and `MILVUS_PORT` are respected, so you can point the tests at an already running stack without restarting containers.

### Prerequisites

1. Install dependencies:
```bash
poetry install --extras dev
poetry run pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
poetry run pip install sentence-transformers
```

2. Start Milvus services:
```bash
docker compose -f tests/e2e/stack/hermes/docker-compose.test.yml up -d
```

3. Wait for services to be ready:
```bash
# Wait for Milvus (port 17091 for health)
timeout 120 bash -c 'until curl -f http://localhost:17091/healthz 2>/dev/null; do sleep 2; done'
```

### Running Tests

Run all integration tests:
```bash
poetry run pytest tests/test_milvus_integration.py -v
```

Run specific tests:
```bash
# Test basic metadata response (no external services needed)
poetry run pytest tests/test_milvus_integration.py::test_embedding_response_includes_metadata -v

# Test Milvus integration (requires Milvus)
poetry run pytest tests/test_milvus_integration.py::test_embedding_persisted_to_milvus -v

# Test Neo4j integration (requires both Milvus and Neo4j)
poetry run pytest tests/test_milvus_integration.py::test_embedding_id_in_neo4j -v
```

### Cleanup

Stop and remove services:
```bash
docker compose -f tests/e2e/stack/hermes/docker-compose.test.yml down -v
```

## Test Behavior

### Skip Flags

Tests automatically skip if required dependencies are not available:

- **ML Dependencies**: Tests requiring sentence-transformers will skip if not installed
- **Milvus**: Tests requiring Milvus will skip if the service is not available
- **Neo4j**: Tests requiring Neo4j will skip if the service is not available

### Test Configuration

Default connection settings (can be overridden via environment variables):

- **Milvus**: `localhost:17530`
- **Neo4j**: `bolt://localhost:17687` (user: `neo4j`, password: `neo4jtest`)

## Milvus Schema

The tests use the following schema for the `hermes_embeddings` collection:

| Field Name     | Data Type     | Description                    |
|---------------|---------------|--------------------------------|
| embedding_id  | VARCHAR(64)   | Primary key, unique identifier |
| embedding     | FLOAT_VECTOR  | 384-dimensional vector         |
| model         | VARCHAR(256)  | Model name (all-MiniLM-L6-v2) |
| text          | VARCHAR(65535)| Original text                  |
| timestamp     | INT64         | Creation timestamp (ms)        |

The collection uses IVF_FLAT indexing with L2 distance metric.

## CI/CD Integration

Integration tests run automatically in CI/CD:

- On pushes to `main` or `develop` branches
- On PRs with the `integration-test` label

The CI workflow:
1. Sets up Milvus, Neo4j, etcd, and MinIO services
2. Installs ML dependencies (CPU-only PyTorch)
3. Waits for services to be ready
4. Runs integration tests with coverage reporting

## Troubleshooting

### Milvus Connection Failed

If Milvus connection fails:
1. Check if services are running: `docker compose -f tests/e2e/stack/hermes/docker-compose.test.yml ps`
2. Check Milvus logs: `docker compose -f tests/e2e/stack/hermes/docker-compose.test.yml logs milvus`
3. Verify port is accessible: `curl http://localhost:17091/healthz`

### Neo4j Connection Failed

If Neo4j connection fails:
1. Check Neo4j logs: `docker compose -f tests/e2e/stack/hermes/docker-compose.test.yml logs neo4j`
2. Verify credentials: `cypher-shell -u neo4j -p neo4jtest "RETURN 1"`
3. Ensure port 7687 is not already in use

### ML Dependencies Not Found

If sentence-transformers is not found:
1. Install explicitly: `poetry run pip install sentence-transformers`
2. Verify installation: `poetry run python -c "import sentence_transformers"`

## References

- [Milvus Documentation](https://milvus.io/docs)
- [Neo4j Python Driver](https://neo4j.com/docs/python-manual/current/)
- [Sentence Transformers](https://www.sbert.net/)
- [Project LOGOS Schema (c-daly/logos#155)](https://github.com/c-daly/logos/issues/155)

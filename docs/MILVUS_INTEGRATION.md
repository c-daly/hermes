# Milvus Integration for Text Embeddings

This document describes the Milvus integration for persisting text embeddings in Hermes.

## Overview

Hermes now automatically persists text embeddings to Milvus when the `/embed_text` endpoint is called. This provides:

- **Automatic persistence**: Embeddings are stored immediately when generated
- **Schema compatibility**: Uses the schema defined in c-daly/logos#155
- **Graceful degradation**: If Milvus is unavailable, embeddings are still returned but not persisted

## Configuration

Configure Milvus connection via environment variables:

```bash
# .env file or tests/e2e/stack/hermes/.env.test
MILVUS_HOST=localhost
MILVUS_PORT=17530
MILVUS_COLLECTION_NAME=hermes_embeddings
```

## Schema

The `hermes_embeddings` collection has the following schema:

| Field | Type | Description |
|-------|------|-------------|
| `embedding_id` | VARCHAR(64) | Primary key, unique identifier for the embedding |
| `embedding` | FLOAT_VECTOR(384) | Vector embedding from all-MiniLM-L6-v2 model |
| `model` | VARCHAR(256) | Model name used for embedding |
| `text` | VARCHAR(65535) | Original text that was embedded |
| `timestamp` | INT64 | Creation timestamp in milliseconds |

## Usage

### Starting Milvus

For testing and development, use Docker Compose:

```bash
# Start Milvus and dependencies
docker compose -f tests/e2e/stack/hermes/docker-compose.test.yml up -d

# Check status
docker ps --filter "name=hermes-test-milvus"

# Stop services
docker compose -f tests/e2e/stack/hermes/docker-compose.test.yml down -v
```

### API Usage

The embedding endpoint automatically persists to Milvus:

```bash
# Generate and persist an embedding
curl -X POST http://localhost:8080/embed_text \
  -H "Content-Type: application/json" \
  -d '{"text": "This is a test sentence.", "model": "default"}'
```

Response includes the embedding_id for later retrieval:

```json
{
  "embedding": [0.1, 0.2, ...],
  "dimension": 384,
  "model": "all-MiniLM-L6-v2",
  "embedding_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

### Querying Milvus

You can query the persisted embeddings using the pymilvus client:

```python
from pymilvus import connections, Collection

# Connect to Milvus
connections.connect(host="localhost", port="17530")

# Get collection
collection = Collection("hermes_embeddings")
collection.load()

# Query by embedding_id
results = collection.query(
    expr='embedding_id == "550e8400-e29b-41d4-a716-446655440000"',
    output_fields=["embedding_id", "model", "text", "timestamp"]
)

# Vector similarity search
search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
search_results = collection.search(
    data=[embedding_vector],
    anns_field="embedding",
    param=search_params,
    limit=5,
    output_fields=["embedding_id", "text"]
)
```

## Deployment

### Docker

The production Docker image includes pymilvus. Set Milvus connection in the environment:

```bash
docker run -d \
  -p 8080:8080 \
  -e MILVUS_HOST=milvus-host \
  -e MILVUS_PORT=17530 \
  hermes:latest
```

### Docker Compose

The `tests/e2e/stack/hermes/docker-compose.test.yml` includes a complete Milvus stack:

```bash
docker compose -f tests/e2e/stack/hermes/docker-compose.test.yml up -d
```

### Kubernetes

Add Milvus connection to your deployment:

```yaml
env:
  - name: MILVUS_HOST
    value: "milvus-service"
  - name: MILVUS_PORT
    value: "17530"
```

## Testing

### Unit Tests

Unit tests verify the Milvus client logic without requiring a running Milvus instance:

```bash
pytest tests/test_milvus_client.py -v
```

### Integration Tests

Integration tests require a running Milvus instance and ML dependencies:

```bash
# Start Milvus
docker compose -f tests/e2e/stack/hermes/docker-compose.test.yml up -d

# Install ML dependencies
pip install sentence-transformers

# Run integration tests
pytest tests/test_milvus_integration.py -v
```

## Graceful Degradation

If Milvus is not available:

1. The `/embed_text` endpoint still works and returns embeddings
2. A warning is logged: "Milvus not available, skipping persistence"
3. The application continues normally without persistence

This ensures Hermes can operate even without Milvus for development or testing.

## Monitoring

Check Milvus status via the health endpoint:

```bash
curl http://localhost:8080/health
```

The health endpoint returns Milvus connectivity status:

```json
{
  "status": "healthy",
  "version": "0.1.0",
  "services": {
    "stt": "available",
    "tts": "available",
    "nlp": "available",
    "embeddings": "available"
  },
  "milvus": {
    "connected": true,
    "host": "localhost",
    "port": "17530",
    "collection": "hermes_embeddings"
  },
  "queue": {
    "enabled": false,
    "pending": 0,
    "processed": 0
  }
}
```

Logs will also show initialization status:

```
INFO:hermes.milvus_client:Initializing Milvus integration...
INFO:hermes.milvus_client:Connected to Milvus at localhost:17530
INFO:hermes.milvus_client:Milvus integration initialized successfully
```

## Troubleshooting

### Connection Issues

If Hermes can't connect to Milvus:

1. Check Milvus is running: `docker ps --filter "name=milvus"`
2. Check network connectivity: `curl http://localhost:9091/healthz`
3. Verify environment variables: `echo $MILVUS_HOST`
4. Check logs: `docker logs milvus-standalone`

### Collection Issues

If the collection can't be created:

1. Check Milvus health: `curl http://localhost:9091/healthz`
2. Verify collection doesn't exist with wrong schema
3. Drop and recreate: Use `utility.drop_collection()` then restart Hermes

### Model Download Issues

If sentence-transformers can't download the model:

1. Check internet connectivity
2. Pre-download model: `python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"`
3. Use offline mode if needed

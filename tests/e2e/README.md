# Hermes E2E Test Infrastructure

This directory contains end-to-end test infrastructure for Hermes.

## Directory Structure

```
tests/e2e/
├── README.md           # This file
└── stack/
    └── hermes/
        ├── .env.test              # Environment variables for test stack
        ├── docker-compose.test.yml # Milvus stack (etcd, minio, milvus)
        └── STACK_VERSION          # Hash of generated stack files
```

## Generated Assets

The files in `stack/hermes/` are **generated** by the LOGOS render script:

```bash
cd /path/to/logos
python infra/scripts/render_test_stacks.py --repo hermes
```

**Do not hand-edit these files.** Make changes in the LOGOS template and regenerate instead.

## Running Integration Tests

Use the helper script from the repo root:

```bash
./scripts/run_integration_stack.sh
```

This will:
1. Start the Milvus stack (etcd, minio, milvus) on Hermes-specific ports (17530, 17091)
2. Wait for services to become healthy
3. Run the integration tests
4. Clean up containers on exit

## Port Assignments

Hermes uses unique ports to avoid conflicts with other LOGOS repos:

| Service | Host Port | Container Port | Description |
|---------|-----------|----------------|-------------|
| Milvus gRPC | 17530 | 19530 | Vector database API |
| Milvus health | 17091 | 9091 | Health check endpoint |
| MinIO API | 17900 | 9000 | Object storage |
| MinIO Console | 17901 | 9001 | MinIO web UI |

## Manual Operations

```bash
# Start stack manually
docker compose -f tests/e2e/stack/hermes/docker-compose.test.yml up -d

# Check status
docker compose -f tests/e2e/stack/hermes/docker-compose.test.yml ps

# View logs
docker compose -f tests/e2e/stack/hermes/docker-compose.test.yml logs milvus

# Stop and clean up
docker compose -f tests/e2e/stack/hermes/docker-compose.test.yml down -v
```

## Environment Variables

The `.env.test` file exports these variables:

- `MILVUS_HOST=milvus` (container name for internal access)
- `MILVUS_PORT=17530` (Hermes-specific port)
- `MILVUS_HEALTHCHECK=http://milvus:17091/healthz`
- `NEO4J_URI=bolt://neo4j:17687` (Hermes-specific port)
- `NEO4J_USER=neo4j`
- `NEO4J_PASSWORD=neo4jtest`

When running tests from the host, use `localhost` instead of container names.

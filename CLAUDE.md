# CLAUDE.md — hermes

## What This Is

Stateless language processing and embedding service for LOGOS. Consumed by Sophia and Apollo.

Provides: Speech-to-Text (STT), Text-to-Speech (TTS), simple NLP, text embeddings,
LLM gateway, media ingestion, NER/relation extraction, and naming services.

All endpoints live in `src/hermes/main.py` (no `api/` subdirectory). The canonical
contract is `logos/contracts/hermes.openapi.yaml` — update it first for API changes.

## Dependencies

- **Milvus** (19530) — vector storage for embeddings (only persistence hermes writes)
- **Redis** (6379) — context cache, type registry pub/sub
- **Python** >=3.12, **Poetry** for dependency management
- **logos-foundry** (git dep) — provides `logos_config`, `logos_test_utils`, `logos_observability`
- **Optional ML extras**: `poetry install -E ml` for torch, whisper, spacy, sentence-transformers, TTS
- **Optional GPU extras**: `poetry install -E ml-gpu` for CUDA-enabled ML
- **Optional OTEL**: `poetry install -E otel` for OpenTelemetry instrumentation

Hermes must remain stateless: no direct Neo4j/HCG access, no session state between requests.

## Key Commands

```bash
# Install
poetry install --with dev                     # core + dev tools
poetry install --with dev -E ml               # + ML models (CPU)
poetry install --with dev -E ml -E otel       # + ML + observability

# Lint & format
poetry run ruff check --fix .
poetry run ruff format .
poetry run black .
poetry run mypy src/

# Test
poetry run pytest tests/unit/ -x -q           # unit only (no services)
poetry run pytest tests/ -x -q                # all tests
poetry run pytest tests/integration/ -x -q    # needs Milvus

# Integration stack
./scripts/run_integration_stack.sh
# Or manually:
docker compose -f tests/e2e/stack/hermes/docker-compose.test.yml up -d
poetry run pytest tests/test_milvus_integration.py -v
docker compose -f tests/e2e/stack/hermes/docker-compose.test.yml down -v

# Full CI locally
./.github/workflows/run_ci.sh

# Dev server
poetry run hermes                             # uses entry point hermes.main:main
```

---

## Architecture

```
src/hermes/
├── main.py               # FastAPI app — ALL endpoints defined here
├── services.py           # Service layer: STT, TTS, NLP, embedding implementations
├── llm.py                # LLM provider abstraction (OpenAI, local, echo)
├── embedding_provider.py # Pluggable embedding providers (sentence-transformers, OpenAI)
├── ner_provider.py       # Pluggable NER providers (spaCy, OpenAI)
├── relation_extractor.py # Pluggable relation extraction (spaCy, OpenAI)
├── combined_extractor.py # Merged NER + relation extraction in single LLM call
├── proposal_builder.py   # Converts text → structured graph-ready proposals for Sophia
├── milvus_client.py      # Milvus connection and embedding persistence
├── context_cache.py      # Redis-backed context/proposal queue cache
├── type_registry.py      # Live ontology type cache via Redis pub/sub from Sophia
├── ontology_client.py    # Fetches ontology types from Sophia (with fallback)
├── name_normalizer.py    # Entity name normalization (lowercase, lemma, dedup)
├── env.py                # Path resolution and environment config utilities
├── static/               # Test UI assets (served at /ui)
└── py.typed              # PEP 561 marker
```

### Test Layout

```
tests/
├── unit/                 # Fast, no external services
├── integration/          # Needs Milvus
├── e2e/                  # Full stack tests
│   └── stack/hermes/     # Docker Compose for test infrastructure
├── conftest.py           # Shared fixtures
├── test_naming.py        # Naming endpoint tests
└── test_type_registry.py # Type registry tests
```

---

## Endpoints

All endpoints are in `src/hermes/main.py`. API port: **17000**.

### Core Language Services

| Method | Path | Description |
|--------|------|-------------|
| POST | `/stt` | Audio file → text transcription |
| POST | `/tts` | Text → audio/wav speech synthesis |
| POST | `/simple_nlp` | Tokenization, POS tagging, lemmatization, NER |
| POST | `/embed_text` | Text → vector embedding (persisted to Milvus) |

### LLM & Cognitive

| Method | Path | Description |
|--------|------|-------------|
| POST | `/llm` | Chat completion gateway (OpenAI, local, echo providers) |
| POST | `/ingest/media` | Media ingestion — processes audio/image/video, forwards to Sophia |
| POST | `/feedback` | Receives execution feedback from Sophia |

### Naming & Ontology

| Method | Path | Description |
|--------|------|-------------|
| POST | `/name-type` | Suggest type name for a cluster of node names |
| POST | `/name-relationship` | Suggest relationship name between node types |

### Infrastructure

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | API info and available endpoints |
| GET | `/ui` | Test UI (serves static/index.html) |
| GET/HEAD | `/health` | Health check — Milvus, LLM status, capabilities |

---

## Conventions & Gotchas

- **Ruff** for linting, **black** for formatting — both enforced in CI
- `main.py` has `# noqa: E402` exemption — `load_dotenv()` must run before config imports
- ML dependencies are optional — endpoints degrade gracefully when models are unavailable
- The `/health` endpoint reports capability status based on which ML libs are installed
- Embedding providers are selected via `EMBEDDING_PROVIDER` env var (local or openai)
- NER providers are selected via `NER_PROVIDER` env var (spacy or openai)
- `context_cache.py` and `type_registry.py` require Redis for full functionality
- Integration tests need Milvus running — use `./scripts/run_integration_stack.sh`
- Commit both `pyproject.toml` and `poetry.lock` together when changing dependencies
- Contract changes require upstream update in `logos/contracts/hermes.openapi.yaml` first
- Cross-repo consumers: Sophia and Apollo depend on hermes endpoints

## Docs

Documentation in `docs/`:
- `MILVUS_INTEGRATION.md` — vector storage integration details
- `TESTING.md` — test infrastructure and patterns
- `DOCKER.md` — container deployment
- `DEPLOYMENT_SUMMARY.md` — deployment overview
- `DEPENDENCY_TROUBLESHOOTING.md` — common dependency issues
- `CROSS_REPO_DEVELOPMENT.md` — working across LOGOS repos

Helper scripts in `scripts/`:
- `dev.sh` — start dev server
- `lint.sh` — run linting
- `test.sh` — run tests
- `run_integration_stack.sh` — start Milvus + run integration tests
- `setup-local-dev.sh` — local development setup

## Issue Templates

| Template | Use For |
|----------|---------|
| `task.yml` | Hermes-specific tasks |
| `config.yml` | Issue template chooser config |

For cross-repo templates (infrastructure, research, documentation), use the
[logos issue templates](https://github.com/c-daly/logos/tree/main/.github/ISSUE_TEMPLATE).

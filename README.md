# Hermes

[![CI/CD](https://github.com/c-daly/hermes/actions/workflows/ci.yml/badge.svg)](https://github.com/c-daly/hermes/actions/workflows/ci.yml)

Hermes: Stateless language & embedding tools for Project LOGOS

## Overview

Hermes is a component of [Project LOGOS](https://github.com/c-daly/logos) that provides stateless language processing and embedding services. It implements the canonical OpenAPI specification defined in the LOGOS meta repository.

### Features

- **Speech-to-Text (STT)**: Convert audio input to text transcription
- **Text-to-Speech (TTS)**: Synthesize speech from text
- **Simple NLP**: Basic NLP preprocessing (tokenization, POS tagging, lemmatization, NER)
- **Text Embeddings**: Generate vector embeddings for text with automatic Milvus persistence
- **LLM Gateway**: `/llm` endpoint proxies chat completions through configurable providers (OpenAI today, local/echo fallbacks included)

All endpoints are stateless and designed to be used by other LOGOS components (Sophia, Talos, Apollo).

### Milvus Integration

Text embeddings are automatically persisted to Milvus when generated, providing vector storage for semantic search and retrieval. See [Milvus Integration Guide](docs/MILVUS_INTEGRATION.md) for details.

## Installation

### Prerequisites

- Python 3.11 or higher
- [Poetry](https://python-poetry.org/) 1.0 or higher

### Setup

1. Clone the repository:
```bash
git clone https://github.com/c-daly/hermes.git
cd hermes
```

2. Install Poetry (if not already installed):
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Or via pip:
```bash
pip install poetry
```

3. Install dependencies:

**For development only (without ML models):**
```bash
poetry install --extras dev
```

**For production with ML capabilities (CPU-only, default):**
```bash
# Install PyTorch CPU version first to avoid CUDA packages
poetry run pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
# Then install ML extras
poetry install --extras "dev ml"
```

**For production with ML and GPU support (requires CUDA):**
```bash
poetry install --extras "dev ml-gpu"
# Then install PyTorch with CUDA support:
poetry run pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

This will create a virtual environment and install all required dependencies including development tools.

### Alternative: Using pip

If you prefer not to use Poetry, you can still install with pip:

**For development only (without ML models):**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

**For production with ML capabilities (CPU-only, default):**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
# Install PyTorch CPU version first to avoid CUDA packages
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
# Then install ML extras
pip install -e ".[dev,ml]"
```

**For production with ML and GPU support (requires CUDA):**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
# Install PyTorch with CUDA first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# Then install the rest
pip install -e ".[dev,ml-gpu]"
```

### CPU vs GPU Installation

**CPU-only (Default, Recommended for most users):**
- Faster installation without large CUDA packages (~500MB vs ~2GB)
- Works on all systems (Linux, macOS, Windows, ARM)
- Suitable for development and moderate workloads
- Use `ml` extra with CPU PyTorch (see commands above)
- **Important**: Install PyTorch CPU version first using `--index-url https://download.pytorch.org/whl/cpu` to avoid downloading CUDA libraries

**GPU with CUDA (For high-performance inference):**
- Requires NVIDIA GPU with CUDA support
- Larger installation (includes CUDA libraries ~2GB+)
- Significantly faster inference for large batches
- Use `ml-gpu` extra and install PyTorch with CUDA separately
- Install CUDA PyTorch using `--index-url https://download.pytorch.org/whl/cu118`

**Note:** The ML dependencies (Whisper, TTS, spaCy, sentence-transformers) are large and will download pre-trained models on first use. Without these dependencies, the API endpoints will return helpful error messages indicating that ML dependencies need to be installed.

## Running the Server

Start the Hermes API server:

```bash
poetry run hermes
```

Or using uvicorn directly:

```bash
poetry run uvicorn hermes.main:app --host 0.0.0.0 --port 8080 --reload
```

If not using Poetry:

```bash
hermes
# or
uvicorn hermes.main:app --host 0.0.0.0 --port 8080 --reload
```

The API will be available at `http://localhost:8080`

## Local Testing (CI Parity)

Hermes uses the shared LOGOS workflow template. Run these commands locally to match the GitHub Actions gate:

```bash
poetry install --with dev
poetry run ruff check src tests
poetry run black --check src tests
poetry run mypy src
poetry run pytest --cov=hermes --cov-report=term --cov-report=xml
```

If you need the Milvus/Neo4j integration tests, run `poetry run pytest tests/test_milvus_integration.py -v` with the docker-compose stack described in `.github/workflows/ci.yml`.

### LLM Provider Configuration

The `/llm` endpoint proxies completions through a configurable provider. By default,
Hermes uses a deterministic `echo` provider so tests and local demos run without
external credentials. To call a real LLM, set the following environment variables
before starting the server (or add them to your process manager/Docker config):

- `HERMES_LLM_PROVIDER` — Provider identifier (`echo`, `openai`, `mock`). Defaults to `echo`, or `openai` automatically when `OPENAI_API_KEY`/`HERMES_LLM_API_KEY` is set.
- `HERMES_LLM_API_KEY` — Required when `HERMES_LLM_PROVIDER=openai` (falls back to `OPENAI_API_KEY` if defined).
- `HERMES_LLM_MODEL` — Optional model override (for example `gpt-4o-mini`).
- `HERMES_LLM_BASE_URL` — Override the OpenAI base URL (useful for compatible gateways).
- `HERMES_CORS_ORIGINS` — Comma-separated origins for browser access (defaults to `*`).

Partial configuration falls back to the echo provider, and `/health` reports the
current default provider plus whether credentials are loaded.

### Docker Deployment

Hermes provides production-ready Docker images with ML capabilities and healthchecks.

#### Production Deployment

Build and run the production image with full ML capabilities:

```bash
# Build the production image (includes ML models)
docker build -t hermes:latest .

# Or inject your local Hermes LLM env vars during build
docker build \
  --build-arg HERMES_LLM_PROVIDER=${HERMES_LLM_PROVIDER:-openai} \
  --build-arg HERMES_LLM_API_KEY=${HERMES_LLM_API_KEY} \
  --build-arg HERMES_LLM_MODEL=${HERMES_LLM_MODEL:-gpt-4o-mini} \
  --build-arg HERMES_LLM_BASE_URL=${HERMES_LLM_BASE_URL:-https://api.openai.com/v1} \
  -t hermes:latest .
```

> ℹ️ Build args simply seed the image defaults. You can (and should) still override secrets at runtime via `docker run -e ...` or Compose environment entries.

```bash
# Run the container
docker run -d -p 8080:8080 --name hermes hermes:latest

# Check health status
docker inspect --format='{{.State.Health.Status}}' hermes
```

Or use Docker Compose for a complete setup:

```bash
# Start the service
docker-compose up -d

# View logs
docker-compose logs -f hermes

# Stop the service
docker-compose down
```

The production Docker image includes:
- ✅ Multi-layer security (non-root user, read-only filesystem support)
- ✅ Built-in healthcheck endpoint monitoring
- ✅ CPU-optimized PyTorch (~500MB instead of ~2GB with CUDA)
- ✅ All ML models (Whisper, TTS, spaCy, sentence-transformers)
- ✅ Production-ready resource limits and logging
- ✅ OCI-compliant image labels

#### Development Deployment

For faster iteration without ML dependencies:

```bash
# Build lightweight development image
docker build -f Dockerfile.dev -t hermes:dev .

# Run with source code mounting for live reload
docker-compose -f docker-compose.dev.yml up
```

#### Docker Configuration

**Environment Variables:**
- `PYTHONUNBUFFERED=1` - Enable real-time logging output

**Resource Limits** (docker-compose.yml):
- CPU: 2.0 cores max, 0.5 cores reserved
- Memory: 4GB max, 1GB reserved

**Healthcheck:**
- Checks `GET /` endpoint every 30 seconds
- 40-second startup grace period for model loading
- 3 retries before marking unhealthy

**Security Features:**
- Runs as non-root user (`hermes:1000`)
- Read-only root filesystem with tmpfs exceptions
- No privilege escalation allowed
- Minimal attack surface

For more deployment options (Kubernetes, Docker Swarm) and advanced configurations, see:
- **[Deployment Guide](deployments/)** - Kubernetes, Docker Swarm, and validation tools
- **[Docker Guide](DOCKER.md)** - Detailed Docker deployment instructions
- **[Integration Guide](examples/INTEGRATION.md)** - Integrating with Sophia and Apollo
- **[Environment Configuration](.env.example)** - All available configuration options

### API Documentation

Once the server is running, access the interactive API documentation:

- Swagger UI: http://localhost:8080/docs
- ReDoc: http://localhost:8080/redoc

## API Endpoints

### GET / - API Information

Returns basic API information and available endpoints.

**Response:**
```json
{
  "name": "Hermes API",
  "version": "0.1.0",
  "description": "Stateless language & embedding tools for Project LOGOS",
  "endpoints": ["/stt", "/tts", "/simple_nlp", "/embed_text", "/llm"]
}
```

### GET /health - Health Check

Returns detailed health status including ML service availability, Milvus connectivity, and internal queue status. Useful for monitoring and integration testing.

**Response:**
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "services": {
    "stt": "available",
    "tts": "available",
    "nlp": "available",
    "embeddings": "available",
    "llm": "available"
  },
  "milvus": {
    "connected": true,
    "host": "localhost",
    "port": "19530",
    "collection": "hermes_embeddings"
  },
  "queue": {
    "enabled": false,
    "pending": 0,
    "processed": 0
  },
  "llm": {
    "default_provider": "openai",
    "configured": true,
    "providers": {
      "echo": true,
      "openai": true
    }
  }
}
```

Status values:
- `healthy`: All ML services are available
- `degraded`: Some ML services are unavailable (dependencies not installed)

### POST /stt - Speech to Text

Convert audio to text transcription.

**Request:**
- Multipart form data with audio file
- Optional language parameter

**Response:**
```json
{
  "text": "transcribed text",
  "confidence": 0.95
}
```

### POST /tts - Text to Speech

Convert text to synthesized speech.

**Request:**
```json
{
  "text": "Hello, world!",
  "voice": "default",
  "language": "en-US"
}
```

**Response:**
- Audio file (WAV format)

### POST /simple_nlp - NLP Processing

Perform basic NLP operations.

**Request:**
```json
{
  "text": "This is a sentence.",
  "operations": ["tokenize", "pos_tag", "lemmatize", "ner"]
}
```

**Response:**
```json
{
  "tokens": ["This", "is", "a", "sentence", "."],
  "pos_tags": [...],
  "lemmas": [...],
  "entities": [...]
}
```

### POST /embed_text - Text Embedding

Generate vector embeddings for text.

**Request:**
```json
{
  "text": "Text to embed",
  "model": "default"
}
```

**Response:**
```json
{
  "embedding": [0.1, 0.2, ...],
  "dimension": 384,
  "model": "default"
}
```

### POST /llm - LLM Gateway

Proxy chat completions through the configured LLM provider. Apollo and other
clients should call this endpoint instead of talking to OpenAI (or equivalents)
directly so telemetry and persona context stay centralized in Hermes.

**Request:**
```json
{
  "messages": [
    {"role": "system", "content": "You are a concise assistant."},
    {"role": "user", "content": "Say hi"}
  ],
  "provider": "openai",
  "model": "gpt-4o-mini",
  "temperature": 0.6,
  "metadata": {"scenario": "browser_llm"}
}
```

**Response:**
```json
{
  "id": "openai-abc123",
  "provider": "openai",
  "model": "gpt-4o-mini",
  "created": 1731111111,
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! 👋"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 20,
    "completion_tokens": 7,
    "total_tokens": 27
  }
}
```

When `HERMES_LLM_PROVIDER` is unset, the endpoint falls back to the echo provider
so development and automated tests remain deterministic.

## Development

### Running Tests

```bash
poetry run pytest
```

With coverage:

```bash
poetry run pytest --cov=hermes --cov-report=html
```

### Code Quality

Lint and format code:

```bash
# Format code
poetry run black src tests

# Lint
poetry run ruff check src tests

# Type checking
poetry run mypy src
```

### Managing Dependencies

Add a new dependency:
```bash
poetry add <package-name>
```

Add a development dependency:
```bash
poetry add --group dev <package-name>
```

Update dependencies:
```bash
poetry update
```

View installed packages:
```bash
poetry show
```

## Architecture

Hermes is a stateless REST API built with FastAPI. It follows the canonical OpenAPI specification from the [LOGOS meta repository](https://github.com/c-daly/logos/blob/main/contracts/hermes.openapi.yaml).

**Key Design Principles:**
- Stateless: No session management or state storage
- Focused: Single responsibility - language processing only
- Independent: Does not interact directly with the HCG (Hybrid Causal Graph)
- Composable: Designed to be used by other LOGOS components

### Implementation Details

**Machine Learning Models:**
- **Speech-to-Text**: OpenAI Whisper (base model) for audio transcription
- **Text-to-Speech**: Coqui TTS with Tacotron2-DDC model for speech synthesis
- **NLP Processing**: spaCy (en_core_web_sm) for tokenization, POS tagging, lemmatization, and NER
- **Text Embeddings**: sentence-transformers (all-MiniLM-L6-v2) for 384-dimensional embeddings

**Model Loading:**
- Models are lazy-loaded on first use to minimize startup time
- Models are cached in memory for subsequent requests
- First requests to each endpoint will be slower due to model initialization

**Optional Dependencies:**
- ML dependencies are optional and can be installed with `pip install -e ".[ml]"`
- Without ML dependencies, endpoints return helpful error messages
- Validation tests run without ML dependencies
- Integration tests require ML dependencies and are automatically skipped if not available

## Project LOGOS Ecosystem

- [`c-daly/logos`](https://github.com/c-daly/logos) - Meta repository with specifications and infrastructure
- [`c-daly/sophia`](https://github.com/c-daly/sophia) - Non-linguistic cognitive core
- [`c-daly/hermes`](https://github.com/c-daly/hermes) - Language & embedding tools (this repo)
- [`c-daly/talos`](https://github.com/c-daly/talos) - Sensor/actuator abstraction layer
- [`c-daly/apollo`](https://github.com/c-daly/apollo) - UI and command layer

## License

MIT License - See LICENSE file for details

## Contributing

See CONTRIBUTING.md for development guidelines and contribution process.

# Hermes

[![CI/CD](https://github.com/c-daly/hermes/actions/workflows/ci.yml/badge.svg)](https://github.com/c-daly/hermes/actions/workflows/ci.yml)

Hermes: Stateless language & embedding tools for Project LOGOS

## Overview

Hermes is a component of [Project LOGOS](https://github.com/c-daly/logos) that provides stateless language processing and embedding services. It implements the canonical OpenAPI specification defined in the LOGOS meta repository.

### Features

- **Speech-to-Text (STT)**: Convert audio input to text transcription
- **Text-to-Speech (TTS)**: Synthesize speech from text
- **Simple NLP**: Basic NLP preprocessing (tokenization, POS tagging, lemmatization, NER)
- **Text Embeddings**: Generate vector embeddings for text

All endpoints are stateless and designed to be used by other LOGOS components (Sophia, Talos, Apollo).

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
```bash
poetry install
```

This will create a virtual environment and install all required dependencies.

### Alternative: Using pip

If you prefer not to use Poetry, you can still install with pip:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

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

### Docker Deployment

Build and run with Docker:

```bash
# Build the image
docker build -t hermes:latest .

# Run the container
docker run -d -p 8080:8080 --name hermes hermes:latest
```

Or use Docker Compose:

```bash
docker-compose up -d
```

### API Documentation

Once the server is running, access the interactive API documentation:

- Swagger UI: http://localhost:8080/docs
- ReDoc: http://localhost:8080/redoc

## API Endpoints

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

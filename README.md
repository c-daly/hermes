# Hermes

[![CI/CD](https://github.com/c-daly/hermes/actions/workflows/ci.yml/badge.svg)](https://github.com/c-daly/hermes/actions/workflows/ci.yml)

**Stateless language & embedding tools for [Project LOGOS](https://github.com/c-daly/logos)**

Hermes provides language processing services: speech-to-text, text-to-speech, embeddings, NLP, and an LLM gateway. All endpoints are stateless.

## Quick Start

```bash
# Install
poetry install --extras "dev ml"

# Run
poetry run hermes
# or: poetry run uvicorn hermes.main:app --host 0.0.0.0 --port 8080 --reload

# Test
poetry run pytest tests/unit/ -v
```

### Docker

```bash
docker pull ghcr.io/c-daly/hermes:latest
docker run -p 8080:8080 -e MILVUS_HOST=localhost -e MILVUS_PORT=17530 ghcr.io/c-daly/hermes:latest
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/stt` | POST | Speech-to-text (audio → text) |
| `/tts` | POST | Text-to-speech (text → audio) |
| `/embed` | POST | Generate text embeddings (auto-persisted to Milvus) |
| `/nlp` | POST | NLP preprocessing (tokenize, POS, NER) |
| `/llm` | POST | LLM chat completions proxy |
| `/health` | GET | Health check |

📖 API docs: `http://localhost:8080/docs` (when running)

## Integration Tests

```bash
./scripts/run_integration_stack.sh
```

Uses port 17xxx range (Neo4j 17474/17687, Milvus 17530).

## ML Dependencies

- **CPU (default)**: `poetry install --extras "dev ml"` + PyTorch CPU
- **GPU (CUDA)**: `poetry install --extras "dev ml-gpu"` + PyTorch CUDA

Models download on first use. Without ML extras, endpoints return informative errors.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MILVUS_HOST` | localhost | Milvus server host |
| `MILVUS_PORT` | 17530 | Milvus gRPC port |
| `OPENAI_API_KEY` | - | For LLM gateway |

## Documentation

- [LOGOS Getting Started](https://github.com/c-daly/logos/blob/main/docs/GETTING_STARTED.md)
- [Architecture Overview](https://github.com/c-daly/logos/blob/main/docs/ARCHITECTURE.md)
- [Testing Guide](https://github.com/c-daly/logos/blob/main/docs/TESTING.md)
- [Milvus Integration](docs/MILVUS_INTEGRATION.md)

## License

MIT - see [LICENSE](LICENSE)

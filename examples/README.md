# Hermes API Examples

This directory contains example scripts demonstrating how to use the Hermes API.

## Prerequisites

### Option 1: Local Installation

1. Install Hermes:
```bash
pip install -e ".[dev]"
```

2. Start the Hermes server:
```bash
uvicorn hermes.main:app --host 0.0.0.0 --port 8080
```

Or simply:
```bash
hermes
```

### Option 2: Using Docker

1. Start the server with Docker Compose:
```bash
docker-compose up -d
```

2. Check server health:
```bash
docker-compose ps
curl http://localhost:8080/
```

3. View logs:
```bash
docker-compose logs -f hermes
```

## Examples

### simple_usage.py

A complete example showing how to interact with all Hermes endpoints:
- Root endpoint (API information)
- Speech-to-Text (STT)
- Text-to-Speech (TTS)
- Simple NLP preprocessing
- Text embedding generation
- LLM gateway (`/llm`) for provider-backed completions

Run the example:
```bash
python examples/simple_usage.py
```

Expected output:
```
============================================================
Hermes API Usage Examples
============================================================

Testing root endpoint...
Status: 200
Response: {'name': 'Hermes API', 'version': '0.1.0', ...}

Testing STT endpoint...
Status: 200
Response: {'text': '...', 'confidence': 0.0}

Testing TTS endpoint...
Status: 200
Content-Type: audio/wav
Audio data length: 44 bytes

Testing Simple NLP endpoint...
Status: 200
Response: {'tokens': [...], 'pos_tags': [...], 'lemmas': [...]}

Testing Embed Text endpoint...
Status: 200
Embedding dimension: 384
Model: default
First 10 values: [0.0, 0.0, 0.0, ...]

============================================================
All examples completed successfully!
============================================================
```

## Using with Python Requests

```python
import requests

# Get API info
response = requests.get("http://localhost:8080/")
print(response.json())

# Generate text embeddings
response = requests.post(
    "http://localhost:8080/embed_text",
    json={"text": "Hello, world!", "model": "default"}
)
embeddings = response.json()["embedding"]
```

## Using with cURL

```bash
# Get API info
curl http://localhost:8080/

# Simple NLP
curl -X POST http://localhost:8080/simple_nlp \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "operations": ["tokenize"]}'

# Text embedding
curl -X POST http://localhost:8080/embed_text \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "model": "default"}'
```

## Integration with Other LOGOS Components

Hermes is designed to be used by other Project LOGOS components:

- **Sophia**: Uses Hermes for language understanding and embedding generation
- **Apollo**: Uses Hermes for speech-to-text and text-to-speech in the UI
- **Talos**: Can use Hermes for processing sensor data that includes text or speech

See the [Project LOGOS documentation](https://github.com/c-daly/logos) for more details on component integration.

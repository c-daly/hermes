# Integration Guide: Hermes with Sophia and Apollo

This guide explains how to integrate Hermes API with other Project LOGOS components (Sophia and Apollo).

## Overview

Hermes provides stateless language processing and embedding services to other LOGOS components:

- **Sophia** (Cognitive Core): Uses Hermes for text embeddings and NLP processing
- **Apollo** (UI Layer): Uses Hermes for speech-to-text and text-to-speech capabilities
- **Talos** (Sensor Layer): May use Hermes for processing text or speech sensor data

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Apollo (UI)                       │
│                                                     │
│  - Speech input/output via Hermes STT/TTS          │
│  - User interface and command processing           │
└──────────────────┬──────────────────────────────────┘
                   │
                   │ HTTP/REST
                   │
┌──────────────────▼──────────────────────────────────┐
│                Sophia (Cognitive)                   │
│                                                     │
│  - Text embeddings via Hermes                      │
│  - NLP processing via Hermes                       │
│  - Decision making and reasoning                   │
└──────────────────┬──────────────────────────────────┘
                   │
                   │ HTTP/REST
                   │
┌──────────────────▼──────────────────────────────────┐
│               Hermes (Language)                     │
│                                                     │
│  - Speech-to-Text (STT)                            │
│  - Text-to-Speech (TTS)                            │
│  - Text Embeddings                                 │
│  - Simple NLP (tokenization, POS, NER, etc.)       │
└─────────────────────────────────────────────────────┘
```

## Integration Methods

### Method 1: Docker Compose (Development)

Create a `docker-compose.logos.yml` file to run all components together:

```yaml
version: '3.8'

services:
  hermes:
    image: hermes:latest
    container_name: hermes
    ports:
      - "8080:8080"
    environment:
      - PYTHONUNBUFFERED=1
      - LOG_LEVEL=INFO
    networks:
      - logos
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  sophia:
    image: sophia:latest
    container_name: sophia
    ports:
      - "8000:8000"
    environment:
      - HERMES_URL=http://hermes:8080
    depends_on:
      hermes:
        condition: service_healthy
    networks:
      - logos

  apollo:
    image: apollo:latest
    container_name: apollo
    ports:
      - "3000:3000"
    environment:
      - HERMES_URL=http://hermes:8080
      - SOPHIA_URL=http://sophia:8000
    depends_on:
      - hermes
      - sophia
    networks:
      - logos

networks:
  logos:
    driver: bridge
```

Start the stack:
```bash
docker-compose -f docker-compose.logos.yml up -d
```

### Method 2: Kubernetes (Production)

Deploy all components to the same namespace with service discovery:

```yaml
# logos-namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: logos

---
# hermes-deployment.yaml (use from deployments/kubernetes/)
# sophia-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sophia
  namespace: logos
spec:
  replicas: 2
  selector:
    matchLabels:
      app: sophia
  template:
    metadata:
      labels:
        app: sophia
    spec:
      containers:
      - name: sophia
        image: sophia:latest
        env:
        - name: HERMES_URL
          value: "http://hermes.logos.svc.cluster.local:8080"
        ports:
        - containerPort: 8000

---
# sophia-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: sophia
  namespace: logos
spec:
  selector:
    app: sophia
  ports:
  - port: 8000
    targetPort: 8000

---
# apollo-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: apollo
  namespace: logos
spec:
  replicas: 2
  selector:
    matchLabels:
      app: apollo
  template:
    metadata:
      labels:
        app: apollo
    spec:
      containers:
      - name: apollo
        image: apollo:latest
        env:
        - name: HERMES_URL
          value: "http://hermes.logos.svc.cluster.local:8080"
        - name: SOPHIA_URL
          value: "http://sophia.logos.svc.cluster.local:8000"
        ports:
        - containerPort: 3000

---
# apollo-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: apollo
  namespace: logos
spec:
  type: LoadBalancer
  selector:
    app: apollo
  ports:
  - port: 3000
    targetPort: 3000
```

Deploy:
```bash
kubectl apply -f logos-namespace.yaml
kubectl apply -f deployments/kubernetes/deployment.yaml
kubectl apply -f sophia-deployment.yaml
kubectl apply -f apollo-deployment.yaml
```

### Method 3: Docker Swarm

Use the stack file from `deployments/swarm/` and extend it:

```yaml
version: '3.8'

services:
  hermes:
    # ... (from deployments/swarm/stack.yml)
    
  sophia:
    image: sophia:latest
    environment:
      - HERMES_URL=http://hermes:8080
    networks:
      - logos-network
    deploy:
      replicas: 2
    
  apollo:
    image: apollo:latest
    environment:
      - HERMES_URL=http://hermes:8080
      - SOPHIA_URL=http://sophia:8000
    ports:
      - "3000:3000"
    networks:
      - logos-network
    deploy:
      replicas: 2

networks:
  logos-network:
    driver: overlay
```

## API Integration Examples

### Sophia Integration

**Use Case**: Generating text embeddings for semantic search

```python
# sophia/services/embedding.py
import httpx
import os

HERMES_URL = os.getenv("HERMES_URL", "http://localhost:8080")

async def get_text_embedding(text: str) -> list[float]:
    """Get text embedding from Hermes."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{HERMES_URL}/embed_text",
            json={"text": text, "model": "default"}
        )
        response.raise_for_status()
        result = response.json()
        return result["embedding"]

# Usage in Sophia
embedding = await get_text_embedding("What is the weather today?")
# Use embedding for semantic search in HCG
```

**Use Case**: NLP processing for entity extraction

```python
# sophia/services/nlp.py
import httpx
import os

HERMES_URL = os.getenv("HERMES_URL", "http://localhost:8080")

async def extract_entities(text: str) -> list[dict]:
    """Extract named entities from text."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{HERMES_URL}/simple_nlp",
            json={
                "text": text,
                "operations": ["tokenize", "ner"]
            }
        )
        response.raise_for_status()
        result = response.json()
        return result.get("entities", [])

# Usage
entities = await extract_entities("John works at Google in New York")
# [{"text": "John", "label": "PERSON"}, {"text": "Google", "label": "ORG"}, ...]
```

### Apollo Integration

**Use Case**: Speech-to-text for voice commands

```python
# apollo/services/speech.py
import httpx
import os

HERMES_URL = os.getenv("HERMES_URL", "http://localhost:8080")

async def transcribe_audio(audio_data: bytes, language: str = "en-US") -> str:
    """Transcribe audio to text."""
    async with httpx.AsyncClient() as client:
        files = {"audio": ("audio.wav", audio_data, "audio/wav")}
        data = {"language": language}
        
        response = await client.post(
            f"{HERMES_URL}/stt",
            files=files,
            data=data
        )
        response.raise_for_status()
        result = response.json()
        return result["text"]

# Usage in Apollo voice interface
audio_bytes = await record_audio()
command_text = await transcribe_audio(audio_bytes)
# Process command_text
```

**Use Case**: Text-to-speech for responses

```python
# apollo/services/speech.py
import httpx
import os

HERMES_URL = os.getenv("HERMES_URL", "http://localhost:8080")

async def synthesize_speech(text: str, language: str = "en-US") -> bytes:
    """Synthesize speech from text."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{HERMES_URL}/tts",
            json={
                "text": text,
                "voice": "default",
                "language": language
            }
        )
        response.raise_for_status()
        return response.content

# Usage
audio_data = await synthesize_speech("Hello, how can I help you?")
# Play audio_data to user
```

## Configuration

### Environment Variables

**Sophia Configuration:**
```bash
# .env or docker-compose environment
HERMES_URL=http://hermes:8080
HERMES_TIMEOUT=30  # seconds
HERMES_RETRY_ATTEMPTS=3
```

**Apollo Configuration:**
```bash
# .env or docker-compose environment
HERMES_URL=http://hermes:8080
SOPHIA_URL=http://sophia:8000
HERMES_TIMEOUT=30
SOPHIA_TIMEOUT=10
```

### Service Discovery

**Docker Compose / Swarm:**
- Use service names: `http://hermes:8080`
- Automatic DNS resolution within the network

**Kubernetes:**
- Use full service DNS: `http://hermes.logos.svc.cluster.local:8080`
- Or short name within namespace: `http://hermes:8080`

**Local Development:**
- Use localhost: `http://localhost:8080`
- Ensure port forwarding is configured

## Error Handling

### Recommended Pattern

```python
import httpx
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10)
)
async def call_hermes_with_retry(endpoint: str, **kwargs):
    """Call Hermes API with automatic retry."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{HERMES_URL}/{endpoint}",
                **kwargs
            )
            response.raise_for_status()
            return response
    except httpx.HTTPError as e:
        logger.error(f"Hermes API error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error calling Hermes: {e}")
        raise

# Usage
try:
    response = await call_hermes_with_retry(
        "embed_text",
        json={"text": "sample text"}
    )
    result = response.json()
except Exception as e:
    # Handle error appropriately
    logger.error(f"Failed to get embedding: {e}")
    # Use fallback or return error to user
```

## Health Checks and Monitoring

### Check Hermes Availability

```python
async def check_hermes_health() -> bool:
    """Check if Hermes is available."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{HERMES_URL}/")
            return response.status_code == 200
    except Exception:
        return False

# Usage in Sophia/Apollo startup
if not await check_hermes_health():
    logger.warning("Hermes is not available. Some features may be limited.")
```

### Integration Testing

```python
import pytest
import httpx

@pytest.mark.asyncio
async def test_hermes_integration():
    """Test integration with Hermes API."""
    async with httpx.AsyncClient() as client:
        # Test root endpoint
        response = await client.get("http://hermes:8080/")
        assert response.status_code == 200
        
        # Test embedding endpoint
        response = await client.post(
            "http://hermes:8080/embed_text",
            json={"text": "test", "model": "default"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "embedding" in data
        assert len(data["embedding"]) > 0
```

## Performance Considerations

### Caching

Cache embeddings to reduce redundant API calls:

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_cached_embedding(text: str) -> list[float]:
    """Get cached embedding or fetch from Hermes."""
    # Implementation
    pass
```

### Batch Processing

Process multiple items efficiently:

```python
import asyncio

async def batch_embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed multiple texts concurrently."""
    tasks = [get_text_embedding(text) for text in texts]
    return await asyncio.gather(*tasks)
```

### Timeouts

Set appropriate timeouts for different operations:

```python
# Short timeout for quick operations
HERMES_TIMEOUT_SHORT = 10  # seconds

# Long timeout for ML operations (STT, TTS)
HERMES_TIMEOUT_LONG = 60  # seconds

# Use in requests
async with httpx.AsyncClient(timeout=HERMES_TIMEOUT_LONG) as client:
    response = await client.post(...)
```

## Security

### Authentication (Future)

When authentication is added to Hermes:

```python
HERMES_API_KEY = os.getenv("HERMES_API_KEY")

headers = {
    "Authorization": f"Bearer {HERMES_API_KEY}"
}

async with httpx.AsyncClient() as client:
    response = await client.post(
        f"{HERMES_URL}/embed_text",
        headers=headers,
        json={"text": "..."}
    )
```

### Network Policies (Kubernetes)

Restrict network access to Hermes:

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: hermes-access
  namespace: logos
spec:
  podSelector:
    matchLabels:
      app: hermes
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: sophia
    - podSelector:
        matchLabels:
          app: apollo
    ports:
    - protocol: TCP
      port: 8080
```

## Troubleshooting

### Connection Refused

**Symptoms:** Cannot connect to Hermes from Sophia/Apollo

**Solutions:**
1. Check if Hermes is running: `docker ps | grep hermes`
2. Verify network connectivity: `docker exec sophia ping hermes`
3. Check firewall rules
4. Verify correct URL in environment variables

### Timeout Errors

**Symptoms:** Requests to Hermes time out

**Solutions:**
1. Increase timeout values
2. Check Hermes resource usage (CPU/memory)
3. Verify models are loaded (first request is slower)
4. Scale Hermes replicas if under heavy load

### 503 Service Unavailable

**Symptoms:** Hermes returns 503 errors

**Solutions:**
1. Check Hermes health: `curl http://hermes:8080/`
2. View Hermes logs: `docker logs hermes`
3. Verify ML dependencies are installed
4. Check if models failed to load

## Additional Resources

- [Hermes API Documentation](https://github.com/c-daly/hermes)
- [Sophia Repository](https://github.com/c-daly/sophia)
- [Apollo Repository](https://github.com/c-daly/apollo)
- [Project LOGOS Meta Repository](https://github.com/c-daly/logos)
- [OpenAPI Specification](https://github.com/c-daly/logos/blob/main/contracts/hermes.openapi.yaml)

## Support

For integration issues:
1. Check this guide and the Hermes documentation
2. Review logs from all components
3. Test each component individually
4. Open an issue in the respective GitHub repository

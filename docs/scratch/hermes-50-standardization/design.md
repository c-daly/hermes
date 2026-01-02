# Hermes #50 + #26 Standardization Design

## Overview

Adopt LOGOS standardization patterns in Hermes: structured logging via `setup_logging()`, unified health response schema from `logos_config.health`, request ID middleware for tracing, and HEAD method support for health checks. This mirrors the sophia standardization completed in Phase 1.

## Approach

**Selected:** Direct pattern adoption from sophia (no alternatives considered - consistency is the goal).

Copy the exact patterns used in `sophia/src/sophia/api/app.py`:
1. `setup_logging("hermes")` replaces `logging.basicConfig()`
2. `RequestIDMiddleware` class copied verbatim
3. `HealthResponse` from `logos_config.health` replaces custom `HealthResponse`
4. Milvus config becomes lazy-loaded (function call at use-time, not import-time)

## Components

### 1. Logging Setup
- **Location:** `src/hermes/main.py` (top of file)
- **Responsibility:** Configure structured JSON logging for hermes
- **Interface:**
  ```python
  from logos_test_utils import setup_logging
  logger = setup_logging("hermes")
  ```
- **Dependencies:** `logos_test_utils` (new dependency)

### 2. Request ID Middleware
- **Location:** `src/hermes/main.py` (class definition before app creation)
- **Responsibility:** Add X-Request-ID header to all requests for distributed tracing
- **Interface:**
  ```python
  class RequestIDMiddleware(BaseHTTPMiddleware):
      async def dispatch(
          self, request: Request, call_next: RequestResponseEndpoint
      ) -> Response:
          request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
          request.state.request_id = request_id
          response = await call_next(request)
          response.headers["X-Request-ID"] = request_id
          return response
  ```
- **Dependencies:** `starlette.middleware.base`, `uuid`

### 3. Health Endpoint
- **Location:** `src/hermes/main.py` (replace existing `/health` endpoint)
- **Responsibility:** Return standardized health response with dependency status
- **Interface:**
  ```python
  from logos_config.health import HealthResponse, DependencyStatus

  @app.api_route("/health", methods=["GET", "HEAD"], response_model=HealthResponse)
  async def health() -> HealthResponse:
      # Returns HealthResponse with milvus and llm dependencies
  ```
- **Dependencies:** `logos_config.health`

### 4. Lazy Milvus Config
- **Location:** `src/hermes/milvus_client.py`
- **Responsibility:** Defer env var reads until first use (not import time)
- **Interface:**
  ```python
  # Module-level: just defaults
  _milvus_host: str | None = None
  _milvus_port: str | None = None
  _collection_name: str | None = None

  def get_milvus_host() -> str:
      global _milvus_host
      if _milvus_host is None:
          _milvus_host = get_env_value("MILVUS_HOST", default="localhost") or "localhost"
      return _milvus_host
  ```

## Behavior Specification

### B1: Structured Logging

**Preconditions:** Application starts

**Input:** None (automatic on import)

**Processing:**
1. `setup_logging("hermes")` called at module load
2. Creates logger named "hermes"
3. Configures JSON formatter by default
4. If `LOG_FORMAT=human` env var set, uses human-readable format

**Output:** Logger instance with structured output

**Postconditions:** All `logger.info()` calls output JSON

**Example:**
```json
{"timestamp": "2026-01-02T10:00:00Z", "level": "INFO", "logger": "hermes", "message": "Starting Hermes API..."}
```

### B2: Request ID Propagation

**Preconditions:** HTTP request received by FastAPI

**Input:** HTTP request, optionally with `X-Request-ID` header

**Processing:**
1. Check for `X-Request-ID` in request headers
2. If present, use that value
3. If absent, generate UUID4
4. Store in `request.state.request_id`
5. Add to response headers

**Output:** Response with `X-Request-ID` header

**Postconditions:** Every response has `X-Request-ID` header

**Example:**
```
Request: GET /health (no X-Request-ID header)
Response headers: X-Request-ID: 550e8400-e29b-41d4-a716-446655440000

Request: GET /health with X-Request-ID: my-trace-123
Response headers: X-Request-ID: my-trace-123
```

### B3: Health Endpoint (GET)

**Preconditions:** None

**Input:** GET /health

**Processing:**
1. Check Milvus connectivity via `_milvus_connected` flag
2. Check LLM provider status via `get_llm_health()`
3. Compute overall status:
   - "healthy" if Milvus connected (Milvus is critical)
   - "degraded" if Milvus disconnected but app running
4. Build HealthResponse with dependencies dict

**Output:**
```json
{
  "status": "healthy",
  "service": "hermes",
  "version": "0.1.0",
  "timestamp": "2026-01-02T10:00:00Z",
  "dependencies": {
    "milvus": {
      "status": "healthy",
      "connected": true,
      "details": {"host": "localhost", "port": "17530", "collection": "hermes_embeddings"}
    },
    "llm": {
      "status": "healthy",
      "connected": true,
      "details": {"provider": "openai", "model": "gpt-4"}
    }
  },
  "capabilities": {
    "stt": "available",
    "tts": "available",
    "nlp": "available",
    "embeddings": "available"
  }
}
```

**Postconditions:** Response is valid `HealthResponse` JSON

### B4: Health Endpoint (HEAD)

**Preconditions:** None

**Input:** HEAD /health

**Processing:** Same as GET but FastAPI returns headers only (no body)

**Output:** HTTP 200 with headers, empty body

**Example:**
```
$ curl -I http://localhost:8080/health
HTTP/1.1 200 OK
content-type: application/json
x-request-id: 550e8400-e29b-41d4-a716-446655440000
```

### B5: Lazy Milvus Config

**Preconditions:** `import hermes.milvus_client` executed

**Input:** None at import time

**Processing:**
1. At import: module-level variables are `None`
2. On first call to `get_milvus_host()`: read env var, cache result
3. Subsequent calls: return cached value

**Output:** No env var reads at import time

**Postconditions:** Can `import hermes.main` without MILVUS_HOST set

**Example:**
```python
# Before: fails if MILVUS_HOST not set
import hermes.milvus_client  # reads env immediately

# After: succeeds even without env vars
import hermes.milvus_client  # no env reads
hermes.milvus_client.connect_milvus()  # reads env here
```

## Edge Cases & Error Handling

### E1: Milvus Not Available
- **Condition:** pymilvus not installed or connection fails
- **Behavior:** Health returns `status: "degraded"`, milvus dependency shows `status: "unavailable"`, `connected: false`
- **Example:** `{"status": "degraded", "dependencies": {"milvus": {"status": "unavailable", "connected": false}}}`

### E2: LLM Not Configured
- **Condition:** No LLM provider env vars set
- **Behavior:** Health returns llm dependency as `status: "unavailable"`
- **Example:** `{"dependencies": {"llm": {"status": "unavailable", "connected": false, "details": {"configured": false}}}}`

### E3: Missing Request ID Header
- **Condition:** Client sends request without X-Request-ID
- **Behavior:** Generate UUID4, use as request ID
- **Example:** Response includes `X-Request-ID: <generated-uuid>`

## Testing Strategy

### Unit Tests

**T1: Logging format test**
- Import `hermes.main`, check logger has StructuredFormatter
- With `LOG_FORMAT=human`, check HumanFormatter used

**T2: Request ID middleware test**
- Mock request without X-Request-ID → verify UUID generated
- Mock request with X-Request-ID → verify same ID in response

**T3: Health endpoint test**
- Mock Milvus connected → verify status "healthy"
- Mock Milvus disconnected → verify status "degraded"
- Verify response matches HealthResponse schema

### Integration Tests

**T4: HEAD method test**
```python
response = client.head("/health")
assert response.status_code == 200
assert response.content == b""  # No body
assert "x-request-id" in response.headers
```

**T5: Import without env vars**
```python
# In subprocess with clean env
import hermes.main  # Should not raise
```

## Files Affected

| File | Action | Description |
|------|--------|-------------|
| `pyproject.toml` | Modify | Add `logos_test_utils` to dependencies |
| `src/hermes/main.py` | Modify | Replace logging, add middleware, update /health |
| `src/hermes/milvus_client.py` | Modify | Make config lazy-loaded |

## Out of Scope

- Changes to other endpoints (STT, TTS, NLP, etc.)
- Changes to LLM provider logic
- Adding new health checks beyond Milvus and LLM
- Request ID logging integration (future enhancement)
- Removing `hermes/env.py` (keep for backward compatibility)

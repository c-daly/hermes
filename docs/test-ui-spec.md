# Hermes Test UI Specification

## Overview

A bundled test UI for interacting with Hermes API endpoints, served from FastAPI at `/ui`.

## Requirements

### Tech Stack
- **HTML**: Static, no templating engine
- **JavaScript**: Vanilla JS (no framework, no build step)
- **CSS**: Tailwind CSS via CDN
- **Serving**: FastAPI StaticFiles mount at `/static`, HTML endpoint at `/ui`

### Endpoints to Support (Phase 1)

| Endpoint | Method | Form Fields | Response Display |
|----------|--------|-------------|------------------|
| `/health` | GET | None (button only) | Status badge + JSON |
| `/llm` | POST | prompt, provider, model, temperature, max_tokens | Formatted message + JSON toggle |
| `/tts` | POST | text, voice, language | Audio player + download |
| `/simple_nlp` | POST | text, operations (checkboxes) | Tokens/POS/Lemmas/Entities cards |
| `/embed_text` | POST | text, model | Dimension + truncated vector + JSON |

### Deferred (Phase 2)
- `/stt` - requires audio file upload
- `/ingest/media` - requires file upload
- `/feedback` - complex nested JSON payload

## File Structure

```
src/hermes/
├── static/
│   ├── css/
│   │   └── custom.css       # Minimal overrides (if needed)
│   ├── js/
│   │   └── app.js           # Form handlers, API calls, response rendering
│   └── index.html           # Main UI page
└── main.py                  # Add /ui route + static mount
```

## UI Layout

```
┌─────────────────────────────────────────────────────────────┐
│  Hermes API Test UI                          [Health: ●]    │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐                                            │
│  │ Endpoints   │  ┌─────────────────────────────────────┐   │
│  │ ○ Health    │  │                                     │   │
│  │ ● LLM       │  │  [Form for selected endpoint]       │   │
│  │ ○ TTS       │  │                                     │   │
│  │ ○ NLP       │  │  Prompt: [________________]         │   │
│  │ ○ Embed     │  │  Provider: [echo ▼]                 │   │
│  │             │  │  Model: [________________]          │   │
│  │             │  │  Temperature: [0.7____]             │   │
│  │             │  │                                     │   │
│  │             │  │  [Submit]                           │   │
│  └─────────────┘  └─────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ Response                              [JSON] [Formatted]││
│  │ ─────────────────────────────────────────────────────── ││
│  │ Assistant: Hello! I received your message...           ││
│  │                                                         ││
│  │ Provider: echo | Model: echo-stub | Tokens: 42          ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

## Implementation Details

### FastAPI Changes (main.py)

```python
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

# Mount static files
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/ui")
async def serve_ui() -> FileResponse:
    """Serve the test UI."""
    return FileResponse(static_dir / "index.html")
```

### JavaScript API (app.js)

```javascript
// Core functions needed:
async function callApi(endpoint, method, body) { ... }
function renderResponse(endpoint, data) { ... }
function toggleJsonView() { ... }

// Per-endpoint form handlers:
function submitLlm() { ... }
function submitTts() { ... }
function submitNlp() { ... }
function submitEmbed() { ... }
function checkHealth() { ... }
```

### Form Specifications

#### /llm Form
- `prompt` (textarea, required)
- `provider` (select: echo, openai)
- `model` (text input, optional)
- `temperature` (number input, 0.0-2.0, default 0.7)
- `max_tokens` (number input, optional)

#### /tts Form
- `text` (textarea, required)
- `voice` (select: default, or text input)
- `language` (select: en-US, es-ES, fr-FR, etc.)

#### /simple_nlp Form
- `text` (textarea, required)
- `operations` (checkboxes: tokenize, pos_tag, lemmatize, ner)

#### /embed_text Form
- `text` (textarea, required)
- `model` (text input, default: "default")

## Test Cases

### Unit Tests (test_ui.py)

1. `test_ui_endpoint_returns_html` - GET /ui returns HTML with correct content-type
2. `test_static_files_accessible` - Static assets load correctly
3. `test_ui_contains_endpoint_forms` - HTML contains forms for each endpoint

### Integration Tests

4. `test_llm_form_submission` - Submit LLM form, verify API called correctly
5. `test_health_indicator_updates` - Health check updates UI status

## Acceptance Criteria

1. [ ] `/ui` serves HTML page with Tailwind styling
2. [ ] Sidebar shows all 5 endpoints
3. [ ] LLM form submits and displays response
4. [ ] Response toggle switches between JSON and formatted view
5. [ ] TTS form plays audio response
6. [ ] NLP form shows parsed results in cards
7. [ ] Embed form shows vector dimension and preview
8. [ ] Health button updates status indicator

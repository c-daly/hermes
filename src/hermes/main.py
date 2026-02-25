"""Main FastAPI application for Hermes API.

Implements the canonical Hermes OpenAPI contract from Project LOGOS.
See: https://github.com/c-daly/logos/blob/main/contracts/hermes.openapi.yaml
"""

import importlib.util
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile

# Load .env early so all module-level config reads see the values.
load_dotenv()
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

try:
    from logos_config import get_env_value
    from logos_config.health import DependencyStatus, HealthResponse
    from logos_config.ports import get_repo_ports
except ImportError:

    def get_env_value(key: str, default: str | None = None) -> str | None:  # type: ignore[misc]
        return os.environ.get(key, default)

    # Minimal Pydantic stubs so the health endpoint still works
    class DependencyStatus(BaseModel):  # type: ignore[no-redef]
        status: str = "unavailable"
        connected: bool = False
        details: Dict[str, Any] = Field(default_factory=dict)

    class HealthResponse(BaseModel):  # type: ignore[no-redef]
        status: str = "degraded"
        service: str = ""
        version: str = ""
        dependencies: Dict[str, Any] = Field(default_factory=dict)
        capabilities: Dict[str, str] = Field(default_factory=dict)

    from collections import namedtuple

    _FallbackPorts = namedtuple(
        "_FallbackPorts",
        ["neo4j_http", "neo4j_bolt", "milvus_grpc", "milvus_metrics", "api"],
    )

    def get_repo_ports(repo: str) -> Any:  # type: ignore[misc]
        _defaults = {
            "hermes": _FallbackPorts(17474, 17687, 17530, 17091, 17000),
            "sophia": _FallbackPorts(47474, 47687, 47530, 47091, 47000),
        }
        return _defaults.get(repo, _FallbackPorts(7474, 7687, 19530, 9091, 8000))


try:
    from logos_observability import get_tracer, setup_telemetry  # noqa: E402
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor  # noqa: E402
    from opentelemetry.trace import StatusCode  # noqa: E402

    _OTEL_AVAILABLE = True
except ImportError:
    from types import SimpleNamespace  # noqa: E402

    class _NoopSpan:
        """No-op span stub when OTel is not installed."""

        def set_attribute(self, *a: Any) -> None:
            pass

        def set_status(self, *a: Any) -> None:
            pass

        def record_exception(self, *a: Any) -> None:
            pass

        def __enter__(self) -> "_NoopSpan":
            return self

        def __exit__(self, *a: Any) -> None:
            pass

    class _NoopTracer:
        """No-op tracer stub when OTel is not installed."""

        def start_as_current_span(self, name: str, **kw: Any) -> "_NoopSpan":
            return _NoopSpan()

    def get_tracer(name: str) -> Any:  # type: ignore[misc]
        """Return no-op tracer when OTel is not installed."""
        return _NoopTracer()

    setup_telemetry = None  # type: ignore[assignment]
    FastAPIInstrumentor = None  # type: ignore[assignment,misc]
    StatusCode = SimpleNamespace(ERROR=None, OK=None)  # type: ignore[assignment,misc]
    _OTEL_AVAILABLE = False

try:
    from logos_test_utils import setup_logging  # type: ignore[import-not-found]
except ImportError:
    setup_logging = None  # type: ignore[assignment]
from starlette.middleware.base import (
    BaseHTTPMiddleware,
    RequestResponseEndpoint,
)
from starlette.responses import Response as StarletteResponse

from hermes import __version__, milvus_client
from hermes.context_cache import ContextCache
from hermes.llm import (
    LLMProviderError,
    LLMProviderNotConfiguredError,
    generate_completion,
)
from hermes.proposal_builder import ProposalBuilder
from hermes.services import (
    generate_embedding,
    generate_llm_response,
    get_llm_health,
    process_nlp,
    synthesize_speech,
    transcribe_audio,
)

# Centralized port defaults from logos_config
_HERMES_PORTS = get_repo_ports("hermes")
_SOPHIA_PORTS = get_repo_ports("sophia")

# Configure structured logging for hermes
logger = (
    setup_logging("hermes")
    if setup_logging is not None
    else logging.getLogger("hermes")
)
tracer = get_tracer("hermes.api")

# Proposal builder for cognitive-loop context injection
_proposal_builder = ProposalBuilder()

# Redis context cache (lazily initialised on first use)
_context_cache: ContextCache | None = None


# -------------------------------------------------------------------
# Cognitive-loop helpers: context retrieval from Sophia
# -------------------------------------------------------------------


def _get_context_cache() -> ContextCache | None:
    """Return (and lazily create) the module-level ContextCache."""
    global _context_cache
    if _context_cache is None:
        redis_url = (
            get_env_value("REDIS_URL", default="redis://localhost:6379/0")
            or "redis://localhost:6379/0"
        )
        _context_cache = ContextCache(redis_url)
    return _context_cache


async def _get_sophia_context(text: str, request_id: str, metadata: dict) -> list[dict]:
    """Retrieve relevant context for the LLM prompt.

    Strategy:
    1. Check Redis for cached context from a prior Sophia processing turn.
       If found, return it immediately (no synchronous Sophia call).
    2. Build a proposal and enqueue it to Redis for background processing.
    3. If Redis is unavailable, fall back to the original synchronous
       Sophia call so the cognitive loop still works.

    Never raises -- if everything fails, returns empty list.
    """
    # --- Fast path: try Redis cache first ---
    cache = _get_context_cache()
    conversation_id = metadata.get("conversation_id") or request_id
    reusable_proposal: dict | None = None

    if cache is not None and cache.available:
        cached = cache.get_context(conversation_id)
        if cached:
            logger.debug(
                "Using cached Sophia context for %s (%d items)",
                conversation_id,
                len(cached),
            )
            # Fire-and-forget: enqueue proposal for background processing
            try:
                proposal = await _proposal_builder.build(
                    text=text,
                    metadata=metadata or {},
                    correlation_id=request_id,
                )
                cache.enqueue_proposal(proposal, conversation_id=conversation_id)
            except Exception as e:
                logger.warning(
                    f"Background proposal enqueue failed: {e}", exc_info=True
                )
            return cached

        # No cached context yet — still enqueue and fall through
        try:
            reusable_proposal = await _proposal_builder.build(
                text=text,
                metadata=metadata or {},
                correlation_id=request_id,
            )
            cache.enqueue_proposal(reusable_proposal, conversation_id=conversation_id)
            logger.debug(
                "No cached context for %s; proposal enqueued for background processing",
                conversation_id,
            )
        except Exception as e:
            logger.warning(f"Proposal build/enqueue failed: {e}", exc_info=True)

        # For a first-turn conversation with no cache, fall through to
        # synchronous Sophia call so the user still gets context.

    # --- Fallback: synchronous Sophia call ---
    sophia_host = get_env_value("SOPHIA_HOST", default="localhost") or "localhost"
    sophia_port = get_env_value("SOPHIA_PORT", default=str(_SOPHIA_PORTS.api)) or str(
        _SOPHIA_PORTS.api
    )
    sophia_token = get_env_value("SOPHIA_API_KEY") or get_env_value("SOPHIA_API_TOKEN")

    if not sophia_token:
        logger.debug(
            "SOPHIA_API_KEY/SOPHIA_API_TOKEN not configured -- context disabled"
        )
        return []

    # Reuse proposal from the enqueue path if available
    proposal: dict | None = reusable_proposal  # type: ignore[no-redef]
    if proposal is None:
        try:
            proposal = await _proposal_builder.build(
                text=text,
                metadata=metadata or {},
                correlation_id=request_id,
            )
        except Exception as e:
            logger.warning(f"Proposal building failed: {e}", exc_info=True)
            return []

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
            response = await client.post(
                f"http://{sophia_host}:{sophia_port}/ingest/hermes_proposal",
                json=proposal,
                headers={"Authorization": f"Bearer {sophia_token}"},
            )
            if response.status_code == 201:
                data: dict[str, list[dict[str, Any]]] = response.json()
                return list(data.get("relevant_context", []))
            logger.warning(
                f"Sophia returned {response.status_code}: {response.text[:200]}"
            )
            return []
    except httpx.ConnectError as e:
        logger.warning(f"Cannot reach Sophia at {sophia_host}:{sophia_port}: {e}")
        return []
    except httpx.TimeoutException:
        logger.warning(f"Sophia request timed out ({sophia_host}:{sophia_port})")
        return []
    except Exception as e:
        logger.error(
            f"Unexpected error during Sophia context retrieval: {e}", exc_info=True
        )
        return []


def _build_context_message(context: list[dict]) -> dict | None:
    """Translate Sophia's graph context into a system message for the LLM."""
    if not context:
        return None

    lines = ["Relevant knowledge from memory:"]
    for item in context:
        name = item.get("name", "unknown")
        node_type = item.get("type", "")
        props = item.get("properties") or {}
        desc = f"- {name}"
        if node_type:
            desc += f" ({node_type})"
        prop_str = ", ".join(
            f"{k}={v}"
            for k, v in props.items()
            if k
            not in (
                "source",
                "derivation",
                "confidence",
                "raw_text",
                "created_at",
                "updated_at",
            )
        )
        if prop_str:
            desc += f": {prop_str}"
        lines.append(desc)

    return {"role": "system", "content": "\n".join(lines)}


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware to add request ID to all requests for tracing."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> StarletteResponse:
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        # Store request_id in request state for access in handlers
        request.state.request_id = request_id

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore
    """Lifespan event handler for application startup and shutdown."""
    # Startup
    # Initialize OpenTelemetry
    if _OTEL_AVAILABLE:
        otlp_endpoint = get_env_value("OTEL_EXPORTER_OTLP_ENDPOINT")
        setup_telemetry(
            service_name=get_env_value("OTEL_SERVICE_NAME", default="hermes")
            or "hermes",
            export_to_console=(
                get_env_value("OTEL_CONSOLE_EXPORT", default="false") or "false"
            ).lower()
            == "true",
            otlp_endpoint=otlp_endpoint,
        )
        logger.info(
            "OpenTelemetry initialized",
            extra={"otlp_endpoint": otlp_endpoint or "none"},
        )
    else:
        logger.info("OpenTelemetry not available, skipping initialization")
    logger.info("Starting Hermes API...")
    # Initialize Milvus connection and collection
    milvus_client.initialize_milvus()
    logger.info("Hermes API startup complete")
    yield
    # Shutdown
    logger.info("Shutting down Hermes API...")
    milvus_client.disconnect_milvus()


# Create FastAPI app
app = FastAPI(
    title="Hermes API",
    version=__version__,
    description="Stateless language & embedding tools for Project LOGOS",
    lifespan=lifespan,
)
if _OTEL_AVAILABLE:
    FastAPIInstrumentor.instrument_app(app)

raw_origins = get_env_value("HERMES_CORS_ORIGINS", default="*") or "*"
if raw_origins.strip() == "*":
    cors_origins = ["*"]
else:
    cors_origins = [
        origin.strip() for origin in raw_origins.split(",") if origin.strip()
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestIDMiddleware)

# Mount static files for test UI
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


# Request/Response Models
class TTSRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize")
    voice: str = Field(default="default", description="Optional voice identifier")
    language: str = Field(default="en-US", description="Language code")


class SimpleNLPRequest(BaseModel):
    text: str = Field(..., description="Text to process")
    operations: List[str] = Field(
        default=["tokenize"], description="List of NLP operations to perform"
    )


class EmbedTextRequest(BaseModel):
    text: str = Field(..., description="Text to embed")
    model: str = Field(
        default="default", description="Optional embedding model identifier"
    )


class STTResponse(BaseModel):
    text: str = Field(..., description="Transcribed text")
    confidence: float = Field(..., description="Confidence score (0.0 to 1.0)")


class POSTag(BaseModel):
    token: str
    tag: str


class Entity(BaseModel):
    text: str
    label: str
    start: int
    end: int


class SimpleNLPResponse(BaseModel):
    tokens: Optional[List[str]] = None
    pos_tags: Optional[List[POSTag]] = None
    lemmas: Optional[List[str]] = None
    entities: Optional[List[Entity]] = None


class EmbedTextResponse(BaseModel):
    embedding: List[float] = Field(..., description="Vector embedding")
    dimension: int = Field(..., description="Embedding dimension")
    model: str = Field(..., description="Model used for embedding")
    embedding_id: str = Field(..., description="Unique identifier for this embedding")


class LLMMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"] = Field(
        ..., description="Role of the message."
    )
    content: str = Field(..., description="Text content of the message.")
    name: Optional[str] = Field(
        default=None, description="Optional identifier for tool/function calls."
    )


class LLMChoice(BaseModel):
    index: int = Field(..., description="Choice index.")
    message: LLMMessage = Field(..., description="Assistant message payload.")
    finish_reason: str = Field(
        default="stop", description="Reason generation completed."
    )


class LLMUsage(BaseModel):
    prompt_tokens: int = Field(..., description="Prompt token count.")
    completion_tokens: int = Field(..., description="Completion token count.")
    total_tokens: int = Field(..., description="Total token count.")


class LLMResponse(BaseModel):
    id: str = Field(..., description="Provider response identifier.")
    provider: str = Field(..., description="Provider used for the completion.")
    model: str = Field(..., description="Provider model identifier.")
    created: int = Field(..., description="Epoch timestamp when created.")
    choices: List[LLMChoice] = Field(..., description="Choices returned by provider.")
    usage: Optional[LLMUsage] = Field(
        default=None, description="Token usage metadata if returned by provider."
    )
    raw: Optional[Dict[str, Any]] = Field(
        default=None, description="Raw provider response for diagnostics."
    )


class LLMRequest(BaseModel):
    prompt: Optional[str] = Field(
        default=None,
        description="Convenience field converted to a single user message.",
    )
    messages: Optional[List[LLMMessage]] = Field(
        default=None,
        description="Conversation history to send to the provider.",
    )
    provider: Optional[str] = Field(
        default=None,
        description="Override configured provider (e.g., `openai`, `echo`).",
    )
    model: Optional[str] = Field(
        default=None, description="Override provider model for this request."
    )
    temperature: Optional[float] = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature to forward to the provider.",
    )
    max_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        description="Optional maximum tokens for the completion.",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata stored alongside the request.",
    )
    experiment_tags: Optional[List[str]] = Field(
        default=None,
        description="Tags for experiment tracking (e.g. ['baseline', 'v2-ner']).",
    )


# Note: HealthResponse is now imported from logos_config.health


# API Endpoints
@app.get("/")
async def root() -> Dict[str, Any]:
    """Root endpoint with API information."""
    return {
        "name": "Hermes API",
        "version": __version__,
        "description": "Stateless language & embedding tools for Project LOGOS",
        "endpoints": ["/stt", "/tts", "/simple_nlp", "/embed_text", "/llm"],
    }


@app.get("/ui")
async def serve_ui() -> FileResponse:
    """Serve the test UI."""
    return FileResponse(static_dir / "index.html")


@app.api_route("/health", methods=["GET", "HEAD"], response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint with detailed service status.

    Returns the overall health status and availability of ML services,
    Milvus connectivity, and LLM provider status.
    Supports both GET and HEAD methods for health probes.
    """
    # Build dependency status for Milvus
    milvus_connected = milvus_client._milvus_connected
    milvus_dep = DependencyStatus(
        status="healthy" if milvus_connected else "unavailable",
        connected=milvus_connected,
        details={
            "host": milvus_client.get_milvus_host(),
            "port": milvus_client.get_milvus_port(),
            "collection": (
                milvus_client.get_collection_name() if milvus_connected else None
            ),
        },
    )

    # Build dependency status for LLM
    llm_health = get_llm_health()
    llm_configured = llm_health.get("configured", False)
    llm_dep = DependencyStatus(
        status="healthy" if llm_configured else "unavailable",
        connected=llm_configured,
        details=llm_health,
    )

    # Build capabilities from ML library availability
    capabilities: Dict[str, str] = {}
    capabilities["stt"] = (
        "available" if importlib.util.find_spec("whisper") else "unavailable"
    )
    capabilities["tts"] = (
        "available" if importlib.util.find_spec("TTS") else "unavailable"
    )
    capabilities["nlp"] = (
        "available" if importlib.util.find_spec("spacy") else "unavailable"
    )
    capabilities["embeddings"] = (
        "available"
        if importlib.util.find_spec("sentence_transformers")
        else "unavailable"
    )

    # Determine overall status (Milvus is critical)
    if not milvus_connected:
        overall_status: Literal["healthy", "degraded", "unavailable"] = "degraded"
    elif not llm_configured:
        overall_status = "degraded"
    else:
        overall_status = "healthy"

    return HealthResponse(
        status=overall_status,
        service="hermes",
        version=__version__,
        dependencies={
            "milvus": milvus_dep,
            "llm": llm_dep,
        },
        capabilities=capabilities,
    )


@app.post("/stt", response_model=STTResponse)
async def speech_to_text(
    audio: UploadFile = File(...), language: str = "en-US"
) -> STTResponse:
    """Convert audio input to text transcription.

    Args:
        audio: Audio file to transcribe
        language: Optional language hint (e.g., "en-US")

    Returns:
        STTResponse with transcribed text and confidence score
    """
    with tracer.start_as_current_span("hermes.stt") as span:
        span.set_attribute("stt.format", audio.content_type or "unknown")
        try:
            # Validate audio file
            if not audio.content_type or not audio.content_type.startswith("audio/"):
                raise HTTPException(
                    status_code=400, detail="Invalid file type. Expected audio file."
                )

            # Read audio bytes
            audio_bytes = await audio.read()

            # Extract language code (e.g., "en-US" -> "en")
            lang_code = language.split("-")[0] if language else "en"

            # Transcribe
            result = await transcribe_audio(audio_bytes, lang_code)

            logger.info(f"STT request completed for language: {language}")
            return STTResponse(text=result["text"], confidence=result["confidence"])

        except HTTPException:
            raise
        except Exception as e:
            span.record_exception(e)
            span.set_status(StatusCode.ERROR, str(e))
            logger.error(f"STT error: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/tts")
async def text_to_speech(request: TTSRequest) -> Response:
    """Convert text to synthesized speech audio.

    Args:
        request: TTSRequest with text, voice, and language

    Returns:
        Audio file in WAV format
    """
    with tracer.start_as_current_span("hermes.tts") as span:
        span.set_attribute("tts.text_length", len(request.text))
        try:
            # Validate request
            if not request.text or len(request.text.strip()) == 0:
                raise HTTPException(status_code=400, detail="Text cannot be empty")

            logger.info(f"TTS request received for text: {request.text[:50]}...")

            # Synthesize speech
            audio_bytes = await synthesize_speech(
                request.text, request.voice, request.language
            )

            return Response(content=audio_bytes, media_type="audio/wav")

        except HTTPException:
            raise
        except Exception as e:
            span.record_exception(e)
            span.set_status(StatusCode.ERROR, str(e))
            logger.error(f"TTS error: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/simple_nlp", response_model=SimpleNLPResponse)
async def simple_nlp(request: SimpleNLPRequest) -> SimpleNLPResponse:
    """Perform basic NLP preprocessing.

    Args:
        request: SimpleNLPRequest with text and operations

    Returns:
        SimpleNLPResponse with requested NLP results
    """
    with tracer.start_as_current_span("hermes.nlp") as span:
        span.set_attribute("nlp.operations", str(request.operations))
        span.set_attribute("nlp.text_length", len(request.text))
        try:
            # Validate request
            if not request.text or len(request.text.strip()) == 0:
                raise HTTPException(status_code=400, detail="Text cannot be empty")

            # Validate operations
            valid_operations = {"tokenize", "pos_tag", "lemmatize", "ner"}
            invalid_ops = set(request.operations) - valid_operations
            if invalid_ops:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid operations: {invalid_ops}. Valid operations are: {valid_operations}",
                )

            logger.info(f"NLP request received with operations: {request.operations}")

            # Process NLP
            result = await process_nlp(request.text, request.operations)

            return SimpleNLPResponse(**result)

        except HTTPException:
            raise
        except Exception as e:
            span.record_exception(e)
            span.set_status(StatusCode.ERROR, str(e))
            logger.error(f"NLP error: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/embed_text", response_model=EmbedTextResponse)
async def embed_text(request: EmbedTextRequest) -> EmbedTextResponse:
    """Generate vector embeddings for input text.

    Args:
        request: EmbedTextRequest with text and model

    Returns:
        EmbedTextResponse with embedding vector
    """
    with tracer.start_as_current_span("hermes.embed_text") as span:
        span.set_attribute("embedding.model", request.model)
        span.set_attribute("embedding.text_length", len(request.text))
        try:
            # Validate request
            if not request.text or len(request.text.strip()) == 0:
                raise HTTPException(status_code=400, detail="Text cannot be empty")

            logger.info(f"Embedding request received for text: {request.text[:50]}...")

            # Generate embedding
            result = await generate_embedding(request.text, request.model)
            span.set_attribute("embedding.dimension", result.get("dimension", 0))

            return EmbedTextResponse(
                embedding=result["embedding"],
                dimension=result["dimension"],
                model=result["model"],
                embedding_id=result["embedding_id"],
            )

        except HTTPException:
            raise
        except Exception as e:
            span.record_exception(e)
            span.set_status(StatusCode.ERROR, str(e))
            logger.error(f"Embedding error: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/llm", response_model=LLMResponse)
async def llm_generate(request: LLMRequest, http_request: Request) -> LLMResponse:
    """Proxy language model completions through Hermes.

    Flow (cognitive loop):
    1. Extract user text from the request
    2. Send structured proposal to Sophia, get relevant context back
    3. Inject context as a system message into the LLM prompt
    4. Generate LLM response with enriched context
    """
    with tracer.start_as_current_span("hermes.llm") as span:
        span.set_attribute("llm.provider", request.provider or "default")
        span.set_attribute("llm.model", request.model or "default")
        span.set_attribute("llm.max_tokens", request.max_tokens or 0)
        try:
            normalized_messages: List[LLMMessage] = list(request.messages or [])
            if not normalized_messages:
                prompt = (request.prompt or "").strip()
                if not prompt:
                    raise HTTPException(
                        status_code=400,
                        detail="Either 'prompt' or 'messages' must be provided.",
                    )
                normalized_messages = [LLMMessage(role="user", content=prompt)]

            # --- Cognitive loop: retrieve context from Sophia BEFORE generation ---
            request_id = getattr(http_request.state, "request_id", str(uuid.uuid4()))

            # Extract user text from the last user message for proposal building
            user_text = ""
            for msg in reversed(normalized_messages):
                if msg.role == "user":
                    user_text = msg.content
                    break

            if user_text:
                # Merge experiment_tags into metadata for pipeline tracking
                ctx_metadata = dict(request.metadata or {})
                if request.experiment_tags:
                    ctx_metadata["experiment_tags"] = request.experiment_tags
                sophia_context = await _get_sophia_context(
                    user_text,
                    request_id,
                    ctx_metadata,
                )
                context_msg = _build_context_message(sophia_context)
                if context_msg:
                    span.set_attribute("llm.sophia_context_items", len(sophia_context))
                    # Inject the context system message right before the last user message
                    inject_idx = 0
                    for i in range(len(normalized_messages) - 1, -1, -1):
                        if normalized_messages[i].role == "user":
                            inject_idx = i
                            break
                    normalized_messages.insert(
                        inject_idx,
                        LLMMessage(role="system", content=context_msg["content"]),
                    )

            result = await generate_llm_response(
                messages=[
                    msg.model_dump(exclude_none=True) for msg in normalized_messages
                ],
                provider=request.provider,
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                metadata=request.metadata,
            )

            return LLMResponse(**result)
        except HTTPException:
            raise
        except LLMProviderNotConfiguredError as exc:
            span.record_exception(exc)
            span.set_status(StatusCode.ERROR, str(exc))
            logger.error("LLM provider not configured: %s", exc)
            raise HTTPException(
                status_code=503, detail="LLM provider not configured"
            ) from exc
        except LLMProviderError as exc:
            span.record_exception(exc)
            span.set_status(StatusCode.ERROR, str(exc))
            logger.error("LLM provider error: %s", exc)
            raise HTTPException(status_code=502, detail="LLM provider error") from exc
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(StatusCode.ERROR, str(exc))
            logger.error("LLM endpoint failure: %s", str(exc))
            raise HTTPException(status_code=500, detail="LLM provider failure") from exc


# ---------------------------------------------------------------------
# Media Ingestion Endpoint
# ---------------------------------------------------------------------


class MediaType(str):
    """Media type enumeration."""

    IMAGE = "IMAGE"
    VIDEO = "VIDEO"
    AUDIO = "AUDIO"


class MediaIngestResponse(BaseModel):
    """Response from media ingestion."""

    sample_id: str = Field(..., description="Unique identifier for the media sample")
    file_path: str = Field(..., description="Path where media is stored")
    media_type: str = Field(..., description="Type of media (IMAGE, VIDEO, AUDIO)")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Extracted metadata"
    )
    neo4j_node_id: Optional[str] = Field(None, description="Neo4j node ID if persisted")
    embedding_id: Optional[str] = Field(
        None, description="Milvus embedding ID if generated"
    )
    transcription: Optional[str] = Field(
        None, description="Transcription for audio/video"
    )
    message: str = Field(..., description="Status message")


@app.post("/ingest/media", response_model=MediaIngestResponse)
async def ingest_media(
    file: UploadFile = File(...),
    media_type: str = "image",
    question: Optional[str] = None,
) -> MediaIngestResponse:
    """Ingest media, process it through Hermes, and forward to Sophia.

    This endpoint:
    1. Receives media from Apollo
    2. Processes it (STT for audio, embedding generation, etc.)
    3. Forwards to Sophia for storage and perception workflows

    Args:
        file: Media file to ingest (image/video/audio)
        media_type: Type of media (image, video, audio)
        question: Optional question context for perception

    Returns:
        MediaIngestResponse with sample_id and processing results
    """
    with tracer.start_as_current_span("hermes.ingest.media") as span:
        span.set_attribute("ingest.content_type", file.content_type or "unknown")

        # Normalize media_type to lowercase for Sophia compatibility
        media_type = media_type.lower()

        # Get Sophia configuration from environment
        sophia_host = get_env_value("SOPHIA_HOST", default="localhost") or "localhost"
        sophia_port = get_env_value(
            "SOPHIA_PORT", default=str(_SOPHIA_PORTS.api)
        ) or str(_SOPHIA_PORTS.api)
        sophia_url = f"http://{sophia_host}:{sophia_port}"
        sophia_token = get_env_value("SOPHIA_API_KEY") or get_env_value(
            "SOPHIA_API_TOKEN"
        )

        if not sophia_token:
            raise HTTPException(
                status_code=503,
                detail="Sophia API token not configured. Set SOPHIA_API_KEY or SOPHIA_API_TOKEN.",
            )

        transcription: Optional[str] = None
        embedding_id: Optional[str] = None

        try:
            # Read file content
            file_content = await file.read()
            await file.seek(0)  # Reset for forwarding

            # Process based on media type
            if media_type == "audio":
                # Transcribe audio using Hermes STT
                try:
                    stt_result = await transcribe_audio(file_content)
                    transcription = stt_result.get("text")
                    logger.info(
                        f"Transcribed audio: {transcription[:100] if transcription else 'empty'}..."
                    )

                    # Generate embedding for transcription
                    if transcription:
                        embed_result = await generate_embedding(
                            transcription, "default"
                        )
                        embedding_id = embed_result.get("embedding_id")
                except Exception as e:
                    logger.warning(
                        f"Audio processing failed, continuing with forward: {e}"
                    )

            elif media_type == "video":
                # For video, we could extract audio and transcribe
                # For now, just log and forward
                logger.info(f"Video ingestion: {file.filename}")

            elif media_type == "image":
                # For images, we could generate description/caption
                # For now, just log and forward
                logger.info(f"Image ingestion: {file.filename}")

            # Forward to Sophia for storage and perception
            await file.seek(0)  # Reset file pointer
            files = {"file": (file.filename, file.file, file.content_type)}
            data = {
                "media_type": media_type,
                # Provenance metadata for HCG node attribution
                "source": "ingestion",
                "derivation": "observed",
            }
            if question:
                data["question"] = question

            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{sophia_url}/ingest/media",
                    files=files,
                    data=data,
                    headers={"Authorization": f"Bearer {sophia_token}"},
                )
                response.raise_for_status()
                sophia_result = response.json()

            # Merge Hermes processing with Sophia result
            result = MediaIngestResponse(
                sample_id=sophia_result.get("sample_id", "unknown"),
                file_path=sophia_result.get("file_path", ""),
                media_type=media_type,
                metadata=sophia_result.get("metadata", {}),
                neo4j_node_id=sophia_result.get("neo4j_node_id"),
                embedding_id=embedding_id,
                transcription=transcription,
                message=f"Media ingested via Hermes. {sophia_result.get('message', '')}",
            )

            logger.info(f"Media ingested: {result.sample_id} ({media_type})")
            return result

        except HTTPException:
            raise
        except httpx.HTTPStatusError as exc:
            span.record_exception(exc)
            span.set_status(StatusCode.ERROR, str(exc))
            logger.error(
                f"Sophia rejected media: {exc.response.status_code} - {exc.response.text}"
            )
            raise HTTPException(
                status_code=exc.response.status_code,
                detail="Sophia ingestion failed",
            ) from exc
        except httpx.RequestError as exc:
            span.record_exception(exc)
            span.set_status(StatusCode.ERROR, str(exc))
            logger.error(f"Cannot connect to Sophia: {exc}")
            raise HTTPException(
                status_code=503,
                detail="Cannot connect to Sophia service",
            ) from exc
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(StatusCode.ERROR, str(exc))
            logger.error(f"Media ingestion failed: {exc}")
            raise HTTPException(
                status_code=500,
                detail="Media ingestion failed",
            ) from exc


# ---------------------------------------------------------------------
# Feedback Endpoint (Sophia → Hermes)
# ---------------------------------------------------------------------


class StepResult(BaseModel):
    """Result of a single plan step execution."""

    step_index: int
    action: str
    outcome: Literal["success", "failure", "skipped"]
    error: Optional[str] = None
    duration_ms: Optional[int] = None


class StateDiff(BaseModel):
    """Changes to CWM state."""

    added_nodes: List[str] = Field(default_factory=list)
    removed_nodes: List[str] = Field(default_factory=list)
    modified_nodes: List[str] = Field(default_factory=list)


class FeedbackPayload(BaseModel):
    """Feedback sent from Sophia to Hermes."""

    # Correlation (at least one required)
    correlation_id: Optional[str] = None
    plan_id: Optional[str] = None
    execution_id: Optional[str] = None

    # Outcome
    feedback_type: Literal["observation", "plan", "execution", "validation"]
    outcome: Literal["accepted", "rejected", "created", "success", "failure", "partial"]
    reason: str

    # Details (optional, type-dependent)
    state_diff: Optional[StateDiff] = None
    step_results: Optional[List[StepResult]] = None
    node_ids_created: Optional[List[str]] = None

    # Metadata
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source_service: str = "sophia"

    def model_post_init(self, __context: Any) -> None:
        """Validate at least one correlation key is present."""
        if not any([self.correlation_id, self.plan_id, self.execution_id]):
            raise ValueError(
                "At least one of correlation_id, plan_id, or execution_id required"
            )


class FeedbackResponse(BaseModel):
    """Response to feedback submission."""

    status: str = "accepted"
    message: str = "Feedback received"


@app.post("/feedback", response_model=FeedbackResponse, status_code=201)
async def receive_feedback(
    payload: FeedbackPayload, request: Request
) -> FeedbackResponse:
    """Receive feedback from Sophia about proposal/execution outcomes.

    This endpoint accepts structured feedback tied to correlation IDs,
    plan IDs, or execution IDs. The feedback is logged for observability.

    Args:
        payload: Feedback payload from Sophia
        request: HTTP request for correlation ID extraction

    Returns:
        FeedbackResponse acknowledging receipt
    """
    with tracer.start_as_current_span("hermes.feedback") as span:
        span.set_attribute("feedback.type", payload.feedback_type)
        span.set_attribute("feedback.correlation_id", payload.correlation_id or "")
        request_id = getattr(request.state, "request_id", "unknown")

        # Log structured feedback for observability
        logger.info(
            "Received feedback",
            extra={
                "request_id": request_id,
                "feedback_type": payload.feedback_type,
                "outcome": payload.outcome,
                "correlation_id": payload.correlation_id,
                "plan_id": payload.plan_id,
                "execution_id": payload.execution_id,
                "source_service": payload.source_service,
                "reason": payload.reason,
            },
        )

        return FeedbackResponse(
            status="accepted",
            message=f"Feedback received for {payload.feedback_type}: {payload.outcome}",
        )


# ---------------------------------------------------------------------
# Naming Endpoints (Sophia type-classification support)
# ---------------------------------------------------------------------


class NameTypeRequest(BaseModel):
    node_names: list[str] = Field(..., description="Cluster of node names to classify")
    parent_type: str | None = Field(
        default=None, description="Optional parent type hint"
    )


class NameTypeResponse(BaseModel):
    type_name: str = Field(..., description="Suggested type name for the cluster")


@app.post("/name-type", response_model=NameTypeResponse)
async def name_type(request: NameTypeRequest) -> NameTypeResponse:
    """Suggest a type name for a cluster of node names."""
    names_list = ", ".join(request.node_names)
    prompt = f"Given these node names that are clustered together: [{names_list}]"
    if request.parent_type:
        prompt += f"\nTheir current parent type is: {request.parent_type}"
    prompt += (
        "\n\nSuggest a concise, snake_case type name that describes this cluster. "
        'Return ONLY a JSON object: {"type_name": "<name>"}.'
    )
    result = await generate_completion(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=128,
    )
    content = result["choices"][0]["message"]["content"]
    data = json.loads(content)
    return NameTypeResponse(type_name=data["type_name"])


class NameRelationshipRequest(BaseModel):
    source_name: str = Field(..., description="Source node name")
    target_name: str = Field(..., description="Target node name")
    context: str | None = Field(default=None, description="Optional context sentence")


class NameRelationshipResponse(BaseModel):
    relationship: str = Field(
        ..., description="Suggested edge label (UPPER_SNAKE_CASE)"
    )
    bidirectional: bool = Field(
        default=False, description="Whether the relationship is bidirectional"
    )


@app.post("/name-relationship", response_model=NameRelationshipResponse)
async def name_relationship(
    request: NameRelationshipRequest,
) -> NameRelationshipResponse:
    """Suggest a relationship label for a pair of nodes."""
    prompt = f'Given source node "{request.source_name}" and target node "{request.target_name}"'
    if request.context:
        prompt += f'\nContext: "{request.context}"'
    prompt += (
        "\n\nSuggest an UPPER_SNAKE_CASE relationship label for the directed edge "
        "from source to target, and whether it is bidirectional. "
        'Return ONLY a JSON object: {"relationship": "<LABEL>", "bidirectional": <true|false>}.'
    )
    result = await generate_completion(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=128,
    )
    content = result["choices"][0]["message"]["content"]
    data = json.loads(content)
    return NameRelationshipResponse(
        relationship=data["relationship"],
        bidirectional=data.get("bidirectional", False),
    )


def main() -> None:
    """Entry point for running the Hermes server."""
    import uvicorn

    port = int(
        get_env_value("HERMES_PORT", default=str(_HERMES_PORTS.api))
        or str(_HERMES_PORTS.api)
    )
    host = get_env_value("HERMES_HOST", default="0.0.0.0") or "0.0.0.0"
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()

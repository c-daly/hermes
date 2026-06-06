"""Main FastAPI application for Hermes API.

Implements the canonical Hermes OpenAPI contract from Project LOGOS.
See: https://github.com/c-daly/logos/blob/main/contracts/hermes.openapi.yaml
"""

import asyncio
import importlib.util
import json
import logging
import os
import threading
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
from pydantic import BaseModel, Field, field_validator

try:
    from logos_config import get_env_value
    from logos_config import RedisConfig  # type: ignore[attr-defined]
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

    class RedisConfig:  # type: ignore[no-redef]
        """Minimal stub when logos_config is not installed."""

        def __init__(self) -> None:
            self.url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

    from collections import namedtuple

    _FallbackPorts = namedtuple(
        "_FallbackPorts",
        ["neo4j_http", "neo4j_bolt", "milvus_grpc", "milvus_metrics", "api"],
    )

    def get_repo_ports(repo: str) -> Any:  # type: ignore[misc]
        _defaults = {
            "hermes": _FallbackPorts(7474, 7687, 19530, 9091, 17000),
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
from hermes.canonical import canonicalize
from hermes.context_cache import ContextCache
from hermes.llm import (
    LLMProviderError,
    LLMProviderNotConfiguredError,
    generate_completion,
)
from hermes.embedding_provider import get_visual_embedding_providers
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

_MAX_UPLOAD_BYTES = 16 * 1024 * 1024  # 16 MB

# Proposal builder for cognitive-loop context injection
_proposal_builder = ProposalBuilder()

# Redis context cache (lazily initialised on first use)
_context_cache: ContextCache | None = None

# Strong references to fire-and-forget background tasks. asyncio holds only weak
# references to tasks, so a task kept solely in a local variable can be garbage
# collected mid-execution. Tasks are registered here and discard themselves on
# completion (see _spawn_background_task).
_background_tasks: set[asyncio.Task] = set()  # type: ignore[type-arg]


# -------------------------------------------------------------------
# Cognitive-loop helpers: context retrieval from Sophia
# -------------------------------------------------------------------


def _get_context_cache() -> ContextCache | None:
    """Return (and lazily create) the module-level ContextCache."""
    global _context_cache
    if _context_cache is None:
        _context_cache = ContextCache(RedisConfig())
    return _context_cache


async def _build_and_enqueue_proposal(
    text: str, request_id: str, metadata: dict, conversation_id: str
) -> None:
    """Build a proposal from *text* and enqueue it for background ingestion.

    Best-effort, fire-and-forget: runs off the response path. If Redis (the
    proposal queue) is unavailable the turn is simply not ingested -- context is
    eventually-consistent, so an occasional miss is acceptable.
    """
    cache = _get_context_cache()
    if cache is None or not cache.available:
        return
    try:
        proposal = await _proposal_builder.build(
            text=text,
            metadata=metadata or {},
            correlation_id=request_id,
        )
        cache.enqueue_proposal(proposal, conversation_id=conversation_id)
    except Exception as e:
        logger.warning(f"Background proposal build/enqueue failed: {e}", exc_info=True)


async def _get_sophia_context(text: str, request_id: str, metadata: dict) -> list[dict]:
    """Opportunistically return cached context; enqueue ingestion in the background.

    Non-blocking by design -- the response never waits on Sophia:
    - Context is opportunistic: if the conversation already has cached context
      (written by the background worker as it ingests prior turns), use it; on a
      miss, return [] and generate without graph context. We never call Sophia
      synchronously and never wait/compute to obtain context.
    - The proposal is built and enqueued fire-and-forget for background ingestion.

    Context is eventually-consistent: a turn or two of lag is acceptable and the
    cache warms over the conversation. Never raises -- returns [] on any failure.
    """
    cache = _get_context_cache()
    conversation_id = metadata.get("conversation_id") or request_id

    context: list[dict] = []
    if cache is not None and cache.available:
        context = cache.get_context(conversation_id)
        # Fire-and-forget background ingestion -- never blocks, never calls Sophia
        # synchronously.
        _spawn_background_task(
            _build_and_enqueue_proposal(text, request_id, metadata, conversation_id)
        )

    return context


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


_type_registry = None
_type_registry_event_bus = None
_type_registry_listener = None
_type_registry_redis_client = None


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
    # Initialize TypeRegistry for ontology type sync
    global _type_registry, _type_registry_event_bus, _type_registry_listener, _type_registry_redis_client
    try:
        from hermes.type_registry import TypeRegistry
        import redis

        _redis_config = RedisConfig()
        _type_registry_redis_client = redis.from_url(_redis_config.url)
        _type_registry = TypeRegistry(_type_registry_redis_client)
        logger.info(
            "TypeRegistry initialized with %d types",
            len(_type_registry.get_type_names()),
        )

        # Subscribe to ontology changes for live updates
        from logos_events import EventBus  # type: ignore[import-not-found]

        _type_registry_event_bus = EventBus(_redis_config)
        _type_registry_event_bus.subscribe(
            "logos:sophia:proposal_processed",
            _type_registry.on_proposal_processed,
        )
        _type_registry_listener = threading.Thread(
            target=_type_registry_event_bus.listen,
            daemon=True,
            name="type-registry-listener",
        )
        _type_registry_listener.start()
        logger.info("TypeRegistry subscribed to ontology changes")
    except Exception:
        logger.warning(
            "TypeRegistry unavailable — ontology type sync disabled",
            exc_info=True,
        )
        if _type_registry_redis_client is not None:
            try:
                _type_registry_redis_client.close()
            except Exception:
                pass
            _type_registry_redis_client = None
        _type_registry = None
        _type_registry_event_bus = None
        _type_registry_listener = None
    logger.info("Hermes API startup complete")
    yield
    # Shutdown
    logger.info("Shutting down Hermes API...")
    # Stop TypeRegistry event listener
    if _type_registry_event_bus is not None:
        _type_registry_event_bus.stop()
    if _type_registry_listener is not None:
        _type_registry_listener.join(timeout=5)
        if _type_registry_listener.is_alive():
            logger.warning("TypeRegistry listener thread did not stop within 5s")
        else:
            logger.info("TypeRegistry event listener stopped")
    if _type_registry_redis_client is not None:
        _type_registry_redis_client.close()
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


@app.get("/health", response_model=HealthResponse, operation_id="getHealth")
@app.head("/health", include_in_schema=False)
async def health() -> HealthResponse:
    """Health check endpoint with detailed service status.

    Returns the overall health status and availability of ML services,
    Milvus connectivity, and LLM provider status.
    Supports both GET and HEAD methods for health probes.

    The HEAD variant is registered for lightweight liveness probes but is
    excluded from the OpenAPI schema so the served document does not emit a
    duplicate ``operationId`` for the same path.
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
    # Visual embedding providers
    visual_providers = get_visual_embedding_providers()
    if visual_providers:
        capabilities["visual_embeddings"] = ",".join(sorted(visual_providers.keys()))
    else:
        capabilities["visual_embeddings"] = "unavailable"

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


def _log_background_task_error(task: asyncio.Task) -> None:  # type: ignore[type-arg]
    """Log exceptions from fire-and-forget background tasks."""
    if not task.cancelled() and task.exception() is not None:
        logger.warning("Background proposal task failed: %s", task.exception())


def _spawn_background_task(coro) -> asyncio.Task:  # type: ignore[type-arg, no-untyped-def]
    """Schedule *coro* fire-and-forget while holding a strong reference.

    asyncio keeps only a weak reference to a task, so a task held solely by a
    local variable can be garbage-collected before it completes. Registering
    it in a module-level set (and discarding on completion) keeps it alive.
    """
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    task.add_done_callback(_log_background_task_error)
    return task


@app.post("/embed_visual")
async def embed_visual(file: UploadFile = File(...)) -> dict[str, Any]:  # type: ignore[assignment]
    """Generate visual embeddings for an uploaded image."""
    if file.size and file.size > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="File too large (max 16MB)")
    with tracer.start_as_current_span("hermes.embed_visual") as span:
        media = await file.read()
        if len(media) > _MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=400, detail="File too large (max 16MB)")
        media_type = file.content_type or "application/octet-stream"
        span.set_attribute("embed.media_type", media_type)

        providers = get_visual_embedding_providers()
        if not providers:
            raise HTTPException(
                503,
                "No visual embedding providers configured. Set EMBEDDING_PROVIDER_VISUAL env var.",
            )

        embeddings: dict[str, Any] = {}
        errors: dict[str, str] = {}

        async def _run_provider(
            name: str, provider: Any
        ) -> tuple[str, dict[str, Any] | None, str | None]:
            try:
                embedding = await provider.embed(media, media_type)
                return (
                    name,
                    {
                        "embedding": embedding,
                        "dim": provider.dimension,
                        "model": provider.model_name,
                    },
                    None,
                )
            except Exception as e:
                logger.error(
                    "Embedding failed for provider %s: %s", name, e, exc_info=True
                )
                span.record_exception(e)
                return name, None, str(e)

        results = await asyncio.gather(
            *(_run_provider(n, p) for n, p in providers.items())
        )
        for pname, pdata, perror in results:
            if pdata is not None:
                embeddings[pname] = pdata
            if perror is not None:
                errors[pname] = perror

        if not embeddings:
            span.set_status(StatusCode.ERROR, "All visual providers failed")
            raise HTTPException(
                500, detail=f"All visual embedding providers failed: {errors}"
            )

        result: dict[str, Any] = {"embeddings": embeddings, "media_type": media_type}
        if errors:
            result["errors"] = errors
        return result


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

            # --- Post-generation: extract NER from prompt + reply ---
            if user_text:
                try:
                    reply_text = (
                        result.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                    )
                    if reply_text:
                        combined_text = f"{user_text}\n\n{reply_text}"
                        post_meta = dict(request.metadata or {})
                        post_meta["extraction_source"] = "prompt_and_reply"
                        if request.experiment_tags:
                            post_meta["experiment_tags"] = request.experiment_tags

                        _spawn_background_task(
                            _proposal_builder.build(
                                text=combined_text,
                                metadata=post_meta,
                                correlation_id=request_id,
                            )
                        )
                except Exception as e:
                    logger.warning("Post-generation proposal failed: %s", e)

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
    node_names: list[str] = Field(
        ..., min_length=1, description="Cluster of node names to classify"
    )
    parent_type: str | None = Field(
        default=None, description="Optional parent type hint"
    )


class NameTypeResponse(BaseModel):
    type_name: str = Field(..., description="Suggested type name for the cluster")


def _extract_json(text: str) -> dict[str, Any]:
    """Parse JSON from LLM output, handling markdown code fences."""
    try:
        result: dict[str, Any] = json.loads(text)
        return result
    except json.JSONDecodeError:
        # LLMs sometimes wrap JSON in ```json ... ``` fences.
        # Extract by finding the first { and last } instead of regex
        # to avoid ReDoS on adversarial input.
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            result = json.loads(text[start : end + 1])
            return result
        raise


@app.post("/name-type", response_model=NameTypeResponse)
async def name_type(request: NameTypeRequest) -> NameTypeResponse:
    """Suggest a type name for a cluster of node names."""
    names_list = ", ".join(request.node_names)
    system_msg = (
        "You are a concise naming assistant. "
        'Return ONLY a JSON object: {"type_name": "<name>"}. '
        "The type_name must be snake_case."
    )
    user_msg = f"Given these node names that are clustered together: [{names_list}]"
    if request.parent_type:
        user_msg += f"\nTheir current parent type is: {request.parent_type}"
    user_msg += (
        "\n\nSuggest a concise, snake_case type name that describes this cluster."
    )

    result = await generate_completion(
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,
        max_tokens=128,
    )
    choices = result.get("choices", [])
    if not choices:
        raise HTTPException(status_code=502, detail="LLM returned no choices")
    content = choices[0]["message"]["content"]
    data = _extract_json(content)
    return NameTypeResponse(type_name=data["type_name"])


class NameClusterMember(BaseModel):
    name: str
    id: Optional[str] = None
    type: Optional[str] = None
    hermes_type_hint: Optional[str] = None
    neighbors: List[Dict[str, Any]] = Field(default_factory=list)


class NameClusterRequest(BaseModel):
    members: List[NameClusterMember] = Field(
        ..., min_length=1, description="Cluster members to name (must be non-empty)"
    )
    candidates: List[str] = Field(default_factory=list)


class NameClusterResponse(BaseModel):
    label: str
    description: str = ""
    confidence: float = 0.5
    removed: List[str] = Field(
        default_factory=list,
        description="Member ids that do not fit the named category (outliers).",
    )
    parent: Optional[str] = Field(
        default=None,
        description=(
            "When `label` is a newly coined type (not one of the supplied "
            "candidates), the existing candidate to graft it under; None when "
            "reusing a candidate or no listed category is a sensible parent."
        ),
    )


@app.post("/name-cluster", response_model=NameClusterResponse)
async def name_cluster(request: NameClusterRequest) -> NameClusterResponse:
    """Name the single category that binds a cluster of nodes (Sophia emergence #505).

    Sophia knows *that* the members belong together; Hermes says *what* they are.
    """
    member_lines = []
    for mem in request.members:
        line = f"- {mem.name}"
        if mem.hermes_type_hint:
            line += f" (hint: {mem.hermes_type_hint})"
        if mem.neighbors:
            rels = ", ".join(
                f"{n.get('relation', '?')}->{n.get('neighbor_name', '?')}"
                for n in mem.neighbors
            )
            line += f"; relations: {rels}"
        member_lines.append(line)
    members_block = "\n".join(member_lines)
    candidates = ", ".join(request.candidates) if request.candidates else "(none)"

    system_msg = (
        "You name the single category that binds a cluster of entities together. "
        "Find the common thread; never refuse. Reuse one of the existing categories "
        "ONLY if this cluster is the same kind of thing as that category. If it "
        "differs from every existing category, coin a NEW, specific lowercase noun "
        "that distinguishes it from them -- do NOT reuse a broad existing label for "
        "a distinct group (e.g. do not name three different groups all 'ecosystem'; "
        "use 'forest ecosystem', 'coral reef', etc.). "
        "When you coin a NEW category, also set 'parent' to the single existing "
        "category above that best generalizes it (its closest broader supertype) "
        "so it can be grafted into the hierarchy; use null for 'parent' when you "
        "reuse an existing category or none is a sensible parent. "
        "Most members share one obvious category, but a few may NOT belong -- a "
        "PART of another member, or a different kind of thing -- and would force a "
        "looser, more general name. List any such members by their EXACT name in "
        "'removed', then name the category of the REMAINING coherent majority. "
        "Leave 'removed' empty when every member fits one specific category "
        "(same-kind nesting, e.g. a component within a component, is NOT a removal). "
        "Return ONLY a JSON object: "
        '{"label": "<noun>", "description": "<short>", "confidence": <0.0-1.0>, '
        '"removed": ["<member name>", ...], "parent": "<existing category or null>"}.'
    )
    user_msg = (
        f"Existing categories: {candidates}\n\n"
        "These entities were grouped together (semantically and structurally "
        f"similar):\n{members_block}\n\nWhat single category binds them?"
    )

    result = await generate_completion(
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,
        # Label + description are short, but the response now also carries a
        # `removed` array of member names; 128 tokens could truncate it mid-array
        # into unparseable JSON (a 502). Give the outlier list room to complete.
        max_tokens=512,
    )
    choices = result.get("choices", [])
    if not choices:
        raise HTTPException(status_code=502, detail="LLM returned no choices")
    # The LLM response shape is untrusted: a missing key, non-dict JSON, or
    # unparseable content must surface as a 502 (bad upstream response), not an
    # unhandled 500. Confidence is clamped to the documented [0.0, 1.0] range.
    try:
        data = _extract_json(choices[0]["message"]["content"])
        if not isinstance(data, dict):
            raise ValueError("LLM response is not a JSON object")
        label = data.get("label")
        if not label:
            raise KeyError("label")
        try:
            confidence = float(data.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to parse LLM cluster-name response: {e}",
        )
    # Hermes flags outliers by name; map them back to the caller's member ids so
    # Sophia can exclude them from the minted type without re-matching names.
    # `removed` is optional and secondary: a malformed value (non-list, null, or
    # a truncated array) must not 500 or discard an otherwise-valid name -- degrade
    # to "no outliers", mirroring how confidence is defaulted above.
    removed_raw = data.get("removed", [])
    if not isinstance(removed_raw, list):
        removed_raw = []
    removed_names = {
        str(n).strip().lower() for n in removed_raw if isinstance(n, (str, int, float))
    }
    removed_ids = [
        mem.id
        for mem in request.members
        if mem.id and mem.name.strip().lower() in removed_names
    ]
    # Graft parent: when Hermes coins a NEW label it may name an existing
    # candidate to attach the new type under. Validate against the supplied
    # candidates (case-insensitive); a hallucinated or self-referential parent
    # degrades to None so Sophia falls back to its default parent rather than
    # grafting onto a dangling target (closed-world, mirrors `removed`).
    label_norm = str(label).strip().lower()
    parent_raw = data.get("parent")
    graft_parent: Optional[str] = None
    if isinstance(parent_raw, str):
        cand_by_lower = {c.strip().lower(): c.strip() for c in request.candidates}
        # `parent` applies only to a NEWLY coined label. If the label is itself an
        # existing candidate (reuse), the contract requires `parent=None` --
        # otherwise Sophia would try to graft an existing type under a new parent
        # (review: "null on reuse").
        if label_norm not in cand_by_lower:
            matched = cand_by_lower.get(parent_raw.strip().lower())
            if matched and matched.lower() != label_norm:
                graft_parent = matched
    return NameClusterResponse(
        label=label_norm,
        description=str(data.get("description", "")),
        confidence=min(1.0, max(0.0, confidence)),
        removed=removed_ids,
        parent=graft_parent,
    )


# ---------------------------------------------------------------------
# v2 typing pass: POST /type-cluster (naming-driven typing experiment T4)
#
# Parallel to /name-cluster but catalog-aware: one LLM pass per cluster
# emits a hypernym + IS_A chain to a realm root + reuse/mint decision per
# subgroup. This module owns the contract + fail-closed server-side
# validation only; the placement cascade lives in the offline harness.
# The catalog is read from the module-level Redis-synced ``_type_registry``
# (duck-typed: get_type_names() / get_type(name)); tests monkeypatch an
# in-process stub (experiment path A).
# ---------------------------------------------------------------------

# The three realm roots every IS_A chain must terminate in.
_TYPE_ROOTS: set[str] = {"entity", "concept", "process"}

# Hard ceiling on chain length (the root link is always kept).
_MAX_CHAIN: int = 8

# Over-specification ceiling (computed on the RAW name, pre-canonicalize).
MAX_WORDS: int = 3
_OVER_SPEC_TOKENS: set[str] = {
    "and",
    "or",
    "&",
    "/",
    "related",
    "feature-of",
    "part-of",
    "with",
}


def _is_over_specified(raw_name: str) -> bool:
    """Ceiling check on the RAW name: >MAX_WORDS words OR a conjunction."""
    lowered = (raw_name or "").strip().lower()
    if not lowered:
        return False
    words = lowered.split()
    if len(words) > MAX_WORDS:
        return True
    word_set = set(words)
    for tok in _OVER_SPEC_TOKENS:
        # whole-word match for word tokens; substring for symbols (& and /).
        if tok.isalpha() or "-" in tok:
            if tok in word_set:
                return True
        elif tok in lowered:
            return True
    return False


def _build_catalog_context() -> tuple[str, dict[str, str], set[str], set[str]]:
    """Build the catalog system-prompt block + closed-world resolution maps.

    Returns ``(catalog_block, alias_to_uuid, published_uuids, root_uuids)``.
    Entries are sorted by (root, name) so the block is stable across requests
    (prompt-cache friendly). Non-root types get bracketed short aliases
    ``[t_xxxx]`` (the ``assign_to`` token, mapped back to a uuid server-side);
    realm roots are listed GRAFT-ONLY and never receive an assignable alias.
    With no registry installed everything is empty (catalog-agnostic mode).
    """
    registry = _type_registry
    if registry is None:
        return "", {}, set(), set()
    try:
        names = list(registry.get_type_names())
    except Exception:
        logger.warning("type_cluster: type registry unavailable; no catalog")
        return "", {}, set(), set()
    entries: list[dict[str, Any]] = []
    for name in names:
        info = registry.get_type(name)
        if not isinstance(info, dict):
            continue
        type_uuid = info.get("uuid")
        if not isinstance(type_uuid, str) or not type_uuid:
            continue
        entries.append(
            {
                "name": name,
                "uuid": type_uuid,
                "root": str(info.get("root") or ""),
                "chain": info.get("chain"),
                "is_root": bool(info.get("is_root")) or name in _TYPE_ROOTS,
            }
        )
    entries.sort(key=lambda e: (e["root"], e["name"]))
    published_uuids = {e["uuid"] for e in entries}
    root_uuids = {e["uuid"] for e in entries if e["is_root"]}
    alias_to_uuid: dict[str, str] = {}
    lines = [
        "PUBLISHED TYPE CATALOG (you may ONLY assign_to or graft onto an id below):"
    ]
    assignable = [e for e in entries if not e["is_root"]]
    for idx, entry in enumerate(assignable):
        alias = f"t_{idx:04d}"
        alias_to_uuid[alias] = entry["uuid"]
        entry_name = entry["name"]
        entry_root = entry["root"] or "?"
        line = f"  [{alias}] {entry_name} (root: {entry_root})"
        chain = entry["chain"]
        if isinstance(chain, list) and chain:
            line += "  chain: " + " > ".join(str(c) for c in chain)
        lines.append(line)
    roots_csv = ", ".join(sorted(_TYPE_ROOTS))
    lines.append(
        "GRAFT-ONLY ROOTS (valid as a parent in a chain, NEVER as "
        f"assign_to): {roots_csv}"
    )
    return "\n".join(lines), alias_to_uuid, published_uuids, root_uuids


def _resolve_assign_to(
    raw_assign: Any,
    alias_to_uuid: dict[str, str],
    published_uuids: set[str],
    root_uuids: set[str],
    has_catalog: bool,
) -> str:
    """Closed-world ``assign_to`` resolution (fail-closed; roots graft-only).

    Bracketed ``[t_xxxx]`` aliases resolve through the injected catalog;
    bare tokens must be published uuids when a catalog is present. Anything
    unresolvable -- and anything resolving to a protected realm root -- is
    coerced to the literal ``NEW``. Without a catalog the contract rule
    applies: bracketed aliases are unresolvable (coerce), bare tokens are
    accepted as already-resolved uuids.
    """
    if not isinstance(raw_assign, str):
        return "NEW"
    token = raw_assign.strip()
    if not token or token == "NEW":
        return "NEW"
    if token.startswith("[") or token.endswith("]"):
        alias = token.strip("[]").strip()
        resolved = alias_to_uuid.get(alias)
        if resolved is None:
            logger.info(
                "type_cluster: closed_world_coerce on unresolved alias %s",
                token,
            )
            return "NEW"
        if resolved in root_uuids:
            logger.info("type_cluster: protected_root_coerce on assign_to %s", token)
            return "NEW"
        return resolved
    if has_catalog:
        if token not in published_uuids:
            logger.info("type_cluster: closed_world_coerce on unknown uuid %s", token)
            return "NEW"
        if token in root_uuids:
            logger.info("type_cluster: protected_root_coerce on assign_to %s", token)
            return "NEW"
        return token
    return token


class TypeClusterMember(BaseModel):
    id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    hermes_type_hint: Optional[str] = None
    neighbors: List[Dict[str, Any]] = Field(default_factory=list)


class TypeClusterRequest(BaseModel):
    members: List[TypeClusterMember] = Field(
        ..., min_length=1, description="Cluster members to type (non-empty)"
    )
    request_id: Optional[str] = None
    # catalog_k is OUT OF SCOPE for T4; null => full catalog (forward-compat).
    catalog_k: Optional[int] = None

    @field_validator("members")
    @classmethod
    def _member_ids_unique(
        cls, value: List[TypeClusterMember]
    ) -> List[TypeClusterMember]:
        ids = [member.id for member in value]
        if len(ids) != len(set(ids)):
            raise ValueError("member ids must be unique (total partition over ids)")
        return value


class TypeClusterResponse(BaseModel):
    """As-designed contract (#127): Hermes NAMES the cluster, picks an existing
    PARENT when minting, and flags the OUTLIERS. The cascade asserts placement
    (reuse / graft / root + the full IS_A chain) from the name + parent.

    parent semantics: null  => `name` is an existing type to REUSE.
                       set   => mint NEW `name` under this existing parent
                                (which may be a domain root).
    The model works in NAMES only -- never a chain, never an id."""

    request_id: Optional[str] = None
    name: str = ""  # canonical type name for the whole cluster
    parent: Optional[str] = None  # existing parent to graft under; null => reuse
    over_specified: bool = False
    residual_ids: List[str] = Field(default_factory=list)  # members that don't fit
    raw_partition_ok: bool = True  # every returned outlier name resolved to a member


@app.post("/type-cluster", response_model=TypeClusterResponse)
async def type_cluster(request: TypeClusterRequest) -> TypeClusterResponse:
    """Name the category that binds a cluster; pick an existing parent or reuse.

    Naming-driven typing v2, as designed (#127): Sophia points (the coarse
    cluster); Hermes NAMES it and -- choosing FROM WHAT EXISTS -- either reuses
    the most specific existing type that fits (parent=null) or mints a new
    `name` under the most specific existing `parent` (which may be a domain
    root; minting under a root is always legal). It also flags OUTLIERS:
    members that don't belong. Hermes does NOT emit an IS_A chain, sub-partition
    the cluster, or echo ids -- the placement cascade asserts the full chain
    from the name + parent. The model reasons over member NAMES and existing
    type NAMES, and returns NAMES; ids never reach it (verbatim id echo was the
    source of the truncation / no-usable-groups failures).
    """
    member_lines = [
        f"- {mem.name}" + (f" (hint: {mem.hermes_type_hint})" if mem.hermes_type_hint else "")
        for mem in request.members
    ]
    members_block = "\n".join(member_lines)

    catalog_block, _alias_unused, _pub_unused, _root_unused = _build_catalog_context()

    system_msg = (
        "You type a cluster of entities. You are given the cluster's members "
        "and the existing type catalog. Choose FROM WHAT EXISTS: return the "
        "most specific existing type that fits the WHOLE cluster as `name` "
        "with `parent`: null (reuse it). If no existing type fits, mint a new "
        "type: return the new `name` (a lowercase singular noun) and `parent` "
        "= the most specific existing type to place it under (a domain root -- "
        "entity, concept, or process -- is a valid parent). Also return "
        "`outliers`: the EXACT names of any listed members that do not belong "
        "under `name`. Do not invent ids or chains. Return ONLY a JSON object: "
        '{"name": "<noun>", "parent": "<existing type>" or null, '
        '"outliers": ["<member name>", ...]}.'
    )
    if catalog_block:
        system_msg += "\n\n" + catalog_block
    user_msg = (
        "These entities were grouped together (embedding-coarse cluster):\n"
        f"{members_block}\n\nType them: name the category that binds them "
        "(reuse an existing type if one fits, else mint under the most "
        "specific existing parent), and list any members that do not fit."
    )

    # Bounded response: one name + one parent + a short outlier list. The
    # per-member token scaling the partition contract needed (#126) is moot.
    result = await generate_completion(
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,
        max_tokens=512,
        metadata={"request_id": request.request_id},
    )
    choices = result.get("choices", [])
    if not choices:
        raise HTTPException(status_code=502, detail="LLM returned no choices")
    content = choices[0]["message"]["content"]

    try:
        data = _extract_json(content)
    except Exception:
        logger.warning("type_cluster: unparseable JSON")
        raise HTTPException(
            status_code=502, detail="LLM typing response was unparseable"
        )
    if not isinstance(data, dict):
        raise HTTPException(
            status_code=502, detail="LLM typing response is not a JSON object"
        )

    raw_name = data.get("name")
    if not isinstance(raw_name, str) or not raw_name.strip():
        raise HTTPException(
            status_code=502, detail="LLM typing response had no usable name"
        )
    over_specified = _is_over_specified(raw_name)
    name = canonicalize(raw_name)

    # parent: null => reuse `name`; else the existing type to graft under.
    # Canonicalize for the cascade's by_norm lookup; placement validity (does
    # the parent actually exist?) is the cascade's call, not Hermes'.
    raw_parent = data.get("parent")
    parent: Optional[str] = None
    if isinstance(raw_parent, str) and raw_parent.strip():
        parent = canonicalize(raw_parent)

    # Map outlier NAMES back to input ids over the known member set: exact,
    # then case/space-normalized. Names the model invented (no member match)
    # are dropped. Names, not ids -- the model reproduces the medium it
    # reasoned over far more reliably than a 36-char uuid (#127).
    raw_outliers = data.get("outliers")
    if not isinstance(raw_outliers, list):
        raw_outliers = []

    def _norm(value: str) -> str:
        return " ".join(str(value).strip().lower().split())

    by_norm: dict[str, list[str]] = {}
    for mem in request.members:
        by_norm.setdefault(_norm(mem.name), []).append(mem.id)

    claimed: set[str] = set()
    residual_ids: list[str] = []
    named_outliers = [o for o in raw_outliers if isinstance(o, str)]
    for outlier in named_outliers:
        for mid in by_norm.get(_norm(outlier), []):
            if mid not in claimed:
                claimed.add(mid)
                residual_ids.append(mid)
                break

    raw_partition_ok = len(claimed) == len(named_outliers)

    return TypeClusterResponse(
        request_id=request.request_id,
        name=name,
        parent=parent,
        over_specified=over_specified,
        residual_ids=sorted(residual_ids),
        raw_partition_ok=raw_partition_ok,
    )


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
    system_msg = (
        "You are a concise naming assistant. "
        'Return ONLY a JSON object: {"relationship": "<LABEL>", "bidirectional": <true|false>}. '
        "The relationship must be UPPER_SNAKE_CASE."
    )
    user_msg = (
        f'Given source node "{request.source_name}" '
        f'and target node "{request.target_name}"'
    )
    if request.context:
        user_msg += f'\nContext: "{request.context}"'
    user_msg += (
        "\n\nSuggest an UPPER_SNAKE_CASE relationship label for the directed edge "
        "from source to target, and whether it is bidirectional."
    )

    result = await generate_completion(
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,
        max_tokens=128,
    )
    choices = result.get("choices", [])
    if not choices:
        raise HTTPException(status_code=502, detail="LLM returned no choices")
    content = choices[0]["message"]["content"]
    data = _extract_json(content)
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

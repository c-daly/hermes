"""Main FastAPI application for Hermes API.

Implements the canonical Hermes OpenAPI contract from Project LOGOS.
See: https://github.com/c-daly/logos/blob/main/contracts/hermes.openapi.yaml
"""

from dotenv import load_dotenv

# Load .env file before any pydantic-settings models are instantiated
load_dotenv()

import importlib.util  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import uuid  # noqa: E402
from contextlib import asynccontextmanager  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, List, Literal, Optional  # noqa: E402

from fastapi import FastAPI, File, HTTPException, Request, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse, Response  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from logos_config import get_env_value  # noqa: E402
from logos_config.health import DependencyStatus, HealthResponse  # noqa: E402
from logos_observability import setup_telemetry  # noqa: E402
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor  # noqa: E402

# TODO: Remove type ignore once logos-foundry publishes py.typed marker (logos #472)
try:
    from logos_test_utils import setup_logging  # type: ignore[import-untyped,import-not-found]  # noqa: E402
except ImportError:
    setup_logging = None  # type: ignore[assignment]
from pydantic import BaseModel, Field  # noqa: E402
from starlette.middleware.base import (  # noqa: E402
    BaseHTTPMiddleware,
    RequestResponseEndpoint,
)
from starlette.responses import Response as StarletteResponse  # noqa: E402

from hermes import __version__, milvus_client  # noqa: E402
from hermes.llm import LLMProviderError, LLMProviderNotConfiguredError  # noqa: E402
from hermes.services import (  # noqa: E402
    generate_embedding,
    generate_llm_response,
    get_llm_health,
    process_nlp,
    synthesize_speech,
    transcribe_audio,
)

# Configure structured logging for hermes
logger = (
    setup_logging("hermes")
    if setup_logging is not None
    else logging.getLogger("hermes")
)


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
    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    setup_telemetry(
        service_name="hermes",
        export_to_console=os.getenv("OTEL_CONSOLE_EXPORT", "false").lower() == "true",
        otlp_endpoint=otlp_endpoint,
    )
    logger.info("OpenTelemetry initialized", extra={"otlp_endpoint": otlp_endpoint or "none"})
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
        logger.error(f"STT error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/tts")
async def text_to_speech(request: TTSRequest) -> Response:
    """Convert text to synthesized speech audio.

    Args:
        request: TTSRequest with text, voice, and language

    Returns:
        Audio file in WAV format
    """
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
        logger.error(f"TTS error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/simple_nlp", response_model=SimpleNLPResponse)
async def simple_nlp(request: SimpleNLPRequest) -> SimpleNLPResponse:
    """Perform basic NLP preprocessing.

    Args:
        request: SimpleNLPRequest with text and operations

    Returns:
        SimpleNLPResponse with requested NLP results
    """
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
        logger.error(f"NLP error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/embed_text", response_model=EmbedTextResponse)
async def embed_text(request: EmbedTextRequest) -> EmbedTextResponse:
    """Generate vector embeddings for input text.

    Args:
        request: EmbedTextRequest with text and model

    Returns:
        EmbedTextResponse with embedding vector
    """
    try:
        # Validate request
        if not request.text or len(request.text.strip()) == 0:
            raise HTTPException(status_code=400, detail="Text cannot be empty")

        logger.info(f"Embedding request received for text: {request.text[:50]}...")

        # Generate embedding
        result = await generate_embedding(request.text, request.model)

        return EmbedTextResponse(
            embedding=result["embedding"],
            dimension=result["dimension"],
            model=result["model"],
            embedding_id=result["embedding_id"],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Embedding error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


async def _forward_llm_to_sophia(
    result: Dict[str, Any],
    request_id: str,
) -> None:
    """Forward LLM response to Sophia for cognitive processing.

    Sends the LLM response to Sophia's /ingest/hermes_proposal endpoint
    with appropriate provenance metadata.

    Args:
        result: The LLM response dict from generate_llm_response
        request_id: Correlation ID for request tracing
    """
    import httpx

    sophia_host = get_env_value("SOPHIA_HOST", default="localhost") or "localhost"
    sophia_port = get_env_value("SOPHIA_PORT", default="8001") or "8001"
    sophia_url = f"http://{sophia_host}:{sophia_port}"
    sophia_token = get_env_value("SOPHIA_API_KEY") or get_env_value("SOPHIA_API_TOKEN")

    if not sophia_token:
        logger.debug("Sophia API token not configured, skipping LLM forwarding")
        return

    # Extract the assistant message content
    choices = result.get("choices", [])
    if not choices:
        logger.debug("No choices in LLM response, skipping forwarding")
        return

    raw_text = choices[0].get("message", {}).get("content", "")
    if not raw_text:
        logger.debug("Empty LLM response content, skipping forwarding")
        return

    # Build HermesProposalRequest payload
    proposal = {
        "proposal_id": str(uuid.uuid4()),
        "correlation_id": request_id,
        "source_service": "hermes",
        "llm_provider": result.get("provider", "unknown"),
        "model": result.get("model", "unknown"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        # Provenance: LLM-mediated content uses lower confidence
        "confidence": 0.7,
        "raw_text": raw_text,
        "metadata": {
            "source": "hermes_llm",
            "derivation": "observed",
        },
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{sophia_url}/ingest/hermes_proposal",
                json=proposal,
                headers={"Authorization": f"Bearer {sophia_token}"},
            )
            if response.status_code == 201:
                logger.info(
                    f"Forwarded LLM response to Sophia: {proposal['proposal_id']}"
                )
            else:
                logger.warning(
                    f"Sophia rejected LLM proposal: {response.status_code} - {response.text}"
                )
    except httpx.RequestError as exc:
        logger.warning(f"Failed to forward LLM response to Sophia: {exc}")
    except Exception as exc:
        logger.warning(f"Unexpected error forwarding to Sophia: {exc}")


@app.post("/llm", response_model=LLMResponse)
async def llm_generate(request: LLMRequest, http_request: Request) -> LLMResponse:
    """Proxy language model completions through Hermes."""
    normalized_messages: List[LLMMessage] = list(request.messages or [])
    if not normalized_messages:
        prompt = (request.prompt or "").strip()
        if not prompt:
            raise HTTPException(
                status_code=400,
                detail="Either 'prompt' or 'messages' must be provided.",
            )
        normalized_messages = [LLMMessage(role="user", content=prompt)]

    try:
        result = await generate_llm_response(
            messages=[msg.model_dump(exclude_none=True) for msg in normalized_messages],
            provider=request.provider,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            metadata=request.metadata,
        )

        # Forward LLM response to Sophia for cognitive processing
        request_id = getattr(http_request.state, "request_id", str(uuid.uuid4()))
        await _forward_llm_to_sophia(result, request_id)

        return LLMResponse(**result)
    except LLMProviderNotConfiguredError as exc:
        logger.error("LLM provider not configured: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except LLMProviderError as exc:
        logger.error("LLM provider error: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
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
    import httpx

    # Normalize media_type to lowercase for Sophia compatibility
    media_type = media_type.lower()

    # Get Sophia configuration from environment
    sophia_host = get_env_value("SOPHIA_HOST", default="localhost") or "localhost"
    sophia_port = get_env_value("SOPHIA_PORT", default="8001") or "8001"
    sophia_url = f"http://{sophia_host}:{sophia_port}"
    sophia_token = get_env_value("SOPHIA_API_KEY") or get_env_value("SOPHIA_API_TOKEN")

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
                    embed_result = await generate_embedding(transcription, "default")
                    embedding_id = embed_result.get("embedding_id")
            except Exception as e:
                logger.warning(f"Audio processing failed, continuing with forward: {e}")

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

    except httpx.HTTPStatusError as exc:
        logger.error(
            f"Sophia rejected media: {exc.response.status_code} - {exc.response.text}"
        )
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=f"Sophia ingestion failed: {exc.response.text}",
        ) from exc
    except httpx.RequestError as exc:
        logger.error(f"Cannot connect to Sophia: {exc}")
        raise HTTPException(
            status_code=503,
            detail=f"Cannot connect to Sophia service: {str(exc)}",
        ) from exc
    except Exception as exc:
        logger.error(f"Media ingestion failed: {exc}")
        raise HTTPException(
            status_code=500,
            detail=f"Media ingestion failed: {str(exc)}",
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


def main() -> None:
    """Entry point for running the Hermes server."""
    import uvicorn

    port = int(get_env_value("HERMES_PORT", default="8080") or "8080")
    host = get_env_value("HERMES_HOST", default="0.0.0.0") or "0.0.0.0"
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()

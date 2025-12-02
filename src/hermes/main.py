"""Main FastAPI application for Hermes API.

Implements the canonical Hermes OpenAPI contract from Project LOGOS.
See: https://github.com/c-daly/logos/blob/main/contracts/hermes.openapi.yaml
"""

import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Literal, Optional
from contextlib import asynccontextmanager
import logging
import importlib.util

from fastapi.middleware.cors import CORSMiddleware

from hermes import __version__
from hermes.llm import LLMProviderError, LLMProviderNotConfiguredError
from hermes.services import (
    transcribe_audio,
    synthesize_speech,
    process_nlp,
    generate_embedding,
    generate_llm_response,
    get_llm_health,
)
from hermes import milvus_client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore
    """Lifespan event handler for application startup and shutdown."""
    # Startup
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

raw_origins = os.getenv("HERMES_CORS_ORIGINS", "*")
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


class HealthResponse(BaseModel):
    status: str = Field(..., description="Overall health status")
    version: str = Field(..., description="API version")
    services: Dict[str, str] = Field(..., description="Status of individual services")
    milvus: Dict[str, Any] = Field(..., description="Milvus connectivity status")
    queue: Dict[str, Any] = Field(..., description="Internal queue status")
    llm: Dict[str, Any] = Field(..., description="LLM provider configuration status")


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


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint with detailed service status.

    Returns the overall health status and availability of ML services,
    Milvus connectivity, and internal queue status.
    This is useful for monitoring and integration with other LOGOS components.
    """
    services_status = {}

    # Check STT (Whisper) availability
    services_status["stt"] = (
        "available" if importlib.util.find_spec("whisper") else "unavailable"
    )

    # Check TTS availability
    services_status["tts"] = (
        "available" if importlib.util.find_spec("TTS") else "unavailable"
    )

    # Check NLP (spaCy) availability
    services_status["nlp"] = (
        "available" if importlib.util.find_spec("spacy") else "unavailable"
    )

    # Check embeddings (sentence-transformers) availability
    services_status["embeddings"] = (
        "available"
        if importlib.util.find_spec("sentence_transformers")
        else "unavailable"
    )

    # Check Milvus connectivity
    milvus_status = {
        "connected": milvus_client._milvus_connected,
        "host": milvus_client.MILVUS_HOST,
        "port": milvus_client.MILVUS_PORT,
        "collection": (
            milvus_client.COLLECTION_NAME if milvus_client._milvus_connected else None
        ),
    }

    # Internal queue status (placeholder for Phase 2)
    # Note: Hermes is currently stateless with no internal queues
    # This is a placeholder for future async processing capabilities
    queue_status = {
        "enabled": False,
        "pending": 0,
        "processed": 0,
    }

    # LLM provider status
    llm_status = get_llm_health()
    services_status["llm"] = "available" if llm_status["configured"] else "unavailable"

    # Determine overall status
    all_available = all(status == "available" for status in services_status.values())
    overall_status = "healthy" if all_available else "degraded"

    return HealthResponse(
        status=overall_status,
        version=__version__,
        services=services_status,
        milvus=milvus_status,
        queue=queue_status,
        llm=llm_status,
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


@app.post("/llm", response_model=LLMResponse)
async def llm_generate(request: LLMRequest) -> LLMResponse:
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
    media_type: str = "IMAGE",
    question: Optional[str] = None,
) -> MediaIngestResponse:
    """Ingest media, process it through Hermes, and forward to Sophia.

    This endpoint:
    1. Receives media from Apollo
    2. Processes it (STT for audio, embedding generation, etc.)
    3. Forwards to Sophia for storage and perception workflows

    Args:
        file: Media file to ingest (image/video/audio)
        media_type: Type of media (IMAGE, VIDEO, AUDIO)
        question: Optional question context for perception

    Returns:
        MediaIngestResponse with sample_id and processing results
    """
    import httpx

    # Get Sophia configuration from environment
    sophia_host = os.getenv("SOPHIA_HOST", "localhost")
    sophia_port = os.getenv("SOPHIA_PORT", "8001")
    sophia_url = f"http://{sophia_host}:{sophia_port}"
    sophia_token = os.getenv("SOPHIA_API_KEY") or os.getenv("SOPHIA_API_TOKEN")

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
        if media_type == "AUDIO":
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

        elif media_type == "VIDEO":
            # For video, we could extract audio and transcribe
            # For now, just log and forward
            logger.info(f"Video ingestion: {file.filename}")

        elif media_type == "IMAGE":
            # For images, we could generate description/caption
            # For now, just log and forward
            logger.info(f"Image ingestion: {file.filename}")

        # Forward to Sophia for storage and perception
        await file.seek(0)  # Reset file pointer
        files = {"file": (file.filename, file.file, file.content_type)}
        data = {"media_type": media_type}
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


def main() -> None:
    """Entry point for running the Hermes server."""
    import uvicorn

    port = int(os.getenv("HERMES_PORT", "8080"))
    host = os.getenv("HERMES_HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()

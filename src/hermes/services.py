"""Service layer for Hermes API endpoints.

This module contains the actual implementations for STT, TTS, NLP, and embedding services.
"""

import logging
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes import milvus_client
from hermes.embedding_provider import get_embedding_provider
from hermes.llm import generate_completion, llm_service_health

logger = logging.getLogger(__name__)

# Try importing ML libraries
try:
    import whisper

    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    logger.warning("Whisper not available. Install with: pip install openai-whisper")

try:
    from TTS.api import TTS

    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    logger.warning("TTS not available. Install with: pip install TTS")

try:
    import spacy

    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("spaCy not available. Install with: pip install spacy")

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning(
        "sentence-transformers not available. Install with: pip install sentence-transformers"
    )


# Global model instances (lazy loaded)
_whisper_model: Optional[Any] = None
_tts_model: Optional[Any] = None
_spacy_model: Optional[Any] = None
_embedding_model: Optional[Any] = None


def get_whisper_model() -> Any:
    """Get or initialize the Whisper model."""
    if not WHISPER_AVAILABLE:
        raise RuntimeError(
            "Whisper not installed. Install ML dependencies with: pip install -e '.[ml]'"
        )

    global _whisper_model
    if _whisper_model is None:
        logger.info("Loading Whisper model (base)...")
        _whisper_model = whisper.load_model("base")
        logger.info("Whisper model loaded successfully")
    return _whisper_model


def get_tts_model() -> Any:
    """Get or initialize the TTS model."""
    if not TTS_AVAILABLE:
        raise RuntimeError(
            "TTS not installed. Install ML dependencies with: pip install -e '.[ml]'"
        )

    global _tts_model
    if _tts_model is None:
        logger.info("Loading TTS model...")
        # Use a lightweight English TTS model
        _tts_model = TTS(
            model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False
        )
        logger.info("TTS model loaded successfully")
    return _tts_model


def get_spacy_model() -> Any:
    """Get or initialize the spaCy model."""
    if not SPACY_AVAILABLE:
        raise RuntimeError(
            "spaCy not installed. Install ML dependencies with: pip install -e '.[ml]'"
        )

    global _spacy_model
    if _spacy_model is None:
        logger.info("Loading spaCy model...")
        try:
            _spacy_model = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found, downloading en_core_web_sm...")
            import subprocess

            subprocess.run(
                ["python", "-m", "spacy", "download", "en_core_web_sm"], check=True
            )
            _spacy_model = spacy.load("en_core_web_sm")
        logger.info("spaCy model loaded successfully")
    return _spacy_model


def get_embedding_model() -> Any:
    """Get or initialize the sentence embedding model.

    .. deprecated:: Prefer ``get_embedding_provider()`` from
       ``hermes.embedding_provider`` for new code.
    """
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        raise RuntimeError(
            "sentence-transformers not installed. Install ML dependencies with: pip install -e '.[ml]'"
        )

    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading sentence-transformers model...")
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Embedding model loaded successfully")
    return _embedding_model


async def transcribe_audio(audio_bytes: bytes, language: str = "en") -> Dict[str, Any]:
    """Transcribe audio to text using Whisper.

    Args:
        audio_bytes: Audio file bytes
        language: Language code (e.g., "en", "es")

    Returns:
        Dictionary with 'text' and 'confidence' keys
    """
    try:
        model = get_whisper_model()

        # Save audio to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_audio.write(audio_bytes)
            temp_audio_path = temp_audio.name

        try:
            # Transcribe
            result = model.transcribe(temp_audio_path, language=language)

            # Extract confidence (Whisper doesn't provide direct confidence,
            # so we'll use a proxy based on language probability)
            confidence = result.get("language_probability", 0.9)
            if "segments" in result and len(result["segments"]) > 0:
                # Average confidence from segments if available
                avg_no_speech = sum(
                    seg.get("no_speech_prob", 0.0) for seg in result["segments"]
                ) / len(result["segments"])
                confidence = max(0.0, min(1.0, 1.0 - avg_no_speech))

            return {"text": result["text"].strip(), "confidence": float(confidence)}
        finally:
            # Clean up temporary file
            Path(temp_audio_path).unlink(missing_ok=True)

    except Exception as e:
        logger.error(f"Error transcribing audio: {str(e)}")
        raise


async def synthesize_speech(
    text: str, voice: str = "default", language: str = "en-US"
) -> bytes:
    """Synthesize speech from text using TTS.

    Args:
        text: Text to synthesize
        voice: Voice identifier (currently ignored, uses default)
        language: Language code

    Returns:
        WAV audio bytes
    """
    try:
        model = get_tts_model()

        # Synthesize to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_audio_path = temp_audio.name

        try:
            # Generate speech
            model.tts_to_file(text=text, file_path=temp_audio_path)

            # Read the generated audio
            with open(temp_audio_path, "rb") as f:
                audio_bytes = f.read()

            return audio_bytes
        finally:
            # Clean up temporary file
            Path(temp_audio_path).unlink(missing_ok=True)

    except Exception as e:
        logger.error(f"Error synthesizing speech: {str(e)}")
        raise


async def process_nlp(text: str, operations: List[str]) -> Dict[str, Any]:
    """Process text with spaCy NLP pipeline.

    Args:
        text: Text to process
        operations: List of operations to perform (tokenize, pos_tag, lemmatize, ner)

    Returns:
        Dictionary with requested NLP results
    """
    try:
        nlp = get_spacy_model()
        doc = nlp(text)

        result: Dict[str, Any] = {}

        if "tokenize" in operations:
            result["tokens"] = [token.text for token in doc]

        if "pos_tag" in operations:
            result["pos_tags"] = [
                {"token": token.text, "tag": token.pos_} for token in doc
            ]

        if "lemmatize" in operations:
            result["lemmas"] = [token.lemma_ for token in doc]

        if "ner" in operations:
            result["entities"] = [
                {
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                }
                for ent in doc.ents
            ]

        return result

    except Exception as e:
        logger.error(f"Error processing NLP: {str(e)}")
        raise


async def generate_embedding(text: str, model_name: str = "default") -> Dict[str, Any]:
    """Generate text embedding using the configured embedding provider.

    Args:
        text: Text to embed
        model_name: Model identifier (currently ignored, uses configured provider)

    Returns:
        Dictionary with 'embedding', 'dimension', 'model', and 'embedding_id' keys
    """
    try:
        provider = get_embedding_provider()
        embedding_list = await provider.embed(text)

        embedding_id = str(uuid.uuid4())
        model_name_used = provider.model_name

        # Persist to Milvus if available
        await milvus_client.persist_embedding(
            embedding_id=embedding_id,
            embedding=embedding_list,
            model=model_name_used,
            text=text,
        )

        return {
            "embedding": embedding_list,
            "dimension": len(embedding_list),
            "model": model_name_used,
            "embedding_id": embedding_id,
        }

    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        raise


async def generate_llm_response(
    *,
    messages: List[Dict[str, str]],
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Proxy LLM requests to the configured provider."""
    return await generate_completion(
        messages=messages,
        provider_override=provider,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        metadata=metadata,
    )


def get_llm_health() -> Dict[str, Any]:
    """Return configuration metadata for the LLM subsystem."""
    return llm_service_health()

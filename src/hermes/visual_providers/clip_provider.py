"""CLIP visual embedding provider for Hermes.

Implements ``VisualEmbeddingProvider`` using OpenAI's CLIP ViT-L/14 model
via the ``open_clip`` library.  Requires ``open_clip_torch``, ``torch``, and
``Pillow``.  The module is importable without these deps; ``CLIPVisualProvider``
raises ``ImportError`` on *instantiation* (not import) when they are absent.
"""

from __future__ import annotations

import asyncio
import io
import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OTel — try-import so the module works without opentelemetry installed
# ---------------------------------------------------------------------------
try:
    from opentelemetry import metrics as _otel_metrics

    _meter = _otel_metrics.get_meter("hermes.clip")
    _inference_count = _meter.create_counter(
        "hermes.clip.inference_count",
        description="Number of CLIP inference calls",
    )
    _inference_latency = _meter.create_histogram(
        "hermes.clip.inference_latency_ms",
        description="CLIP inference latency in milliseconds",
        unit="ms",
    )
    _model_load_time = _meter.create_histogram(
        "hermes.clip.model_load_time_ms",
        description="CLIP model load time in milliseconds",
        unit="ms",
    )
    _OTEL_AVAILABLE = True
except Exception:  # noqa: BLE001
    _OTEL_AVAILABLE = False
    _inference_count = None  # type: ignore[assignment]
    _inference_latency = None  # type: ignore[assignment]
    _model_load_time = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Heavy optional deps — wrapped so the module remains importable without them
# ---------------------------------------------------------------------------
try:
    import open_clip  # type: ignore[import]
    import torch  # type: ignore[import]
    from PIL import Image  # type: ignore[import]

    _DEPS_AVAILABLE = True
except ImportError:
    _DEPS_AVAILABLE = False


class CLIPVisualProvider:
    """Visual embedding provider using CLIP ViT-L/14 (open_clip).

    Model is lazy-loaded on the first ``embed`` / ``embed_batch`` call.
    Device is controlled by the ``CLIP_DEVICE`` environment variable
    (default: ``"cpu"``).

    Raises:
        ImportError: On instantiation when ``open_clip_torch``, ``torch``,
            or ``Pillow`` are not installed.
    """

    _MODEL_ARCH: str = "ViT-L-14"
    _MODEL_PRETRAINED: str = "openai"
    _DIM: int = 768

    def __init__(self) -> None:
        if not _DEPS_AVAILABLE:
            raise ImportError(
                "CLIPVisualProvider requires 'open_clip_torch', 'torch', and 'Pillow'. "
                "Install with: pip install open_clip_torch torch Pillow"
            )
        try:
            from logos_config import get_env_value
        except ImportError:
            import os
            from collections.abc import Mapping

            def get_env_value(
                key: str,
                env: Mapping[str, str] | None = None,
                default: str | None = None,
            ) -> str | None:
                return (env or os.environ).get(key, default)

        self._device: str = get_env_value("CLIP_DEVICE") or "cpu"
        self._model: Any = None
        self._preprocess: Any = None
        self._load_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Protocol properties
    # ------------------------------------------------------------------

    @property
    def dimension(self) -> int:
        """Output embedding dimensionality (768 for ViT-L/14)."""
        return self._DIM

    @property
    def model_name(self) -> str:
        """Canonical provider identifier."""
        return "clip-vit-l14"

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _load(self) -> tuple[Any, Any]:
        """Lazy-load CLIP model and preprocessor (once per instance)."""
        if self._model is not None:
            return self._model, self._preprocess
        with self._load_lock:
            if self._model is not None:
                return self._model, self._preprocess
            t0 = time.monotonic()
            logger.info(
                "Loading CLIP model %s/%s on device=%s",
                self._MODEL_ARCH,
                self._MODEL_PRETRAINED,
                self._device,
            )
            model, _, preprocess = open_clip.create_model_and_transforms(
                self._MODEL_ARCH, pretrained=self._MODEL_PRETRAINED
            )
            self._model = model.to(self._device).eval()
            self._preprocess = preprocess
            elapsed_ms = (time.monotonic() - t0) * 1000
            logger.info("CLIP model loaded in %.1f ms", elapsed_ms)
            if _OTEL_AVAILABLE and _model_load_time is not None:
                _model_load_time.record(elapsed_ms)
        return self._model, self._preprocess

    # ------------------------------------------------------------------
    # Synchronous inference helpers (called via asyncio.to_thread)
    # ------------------------------------------------------------------

    def _decode_image(self, media: bytes, media_type: str) -> Any:
        """Decode raw bytes to a PIL Image.

        Raises:
            ValueError: If ``media`` cannot be decoded as an image.
        """
        try:
            return Image.open(io.BytesIO(media)).convert("RGB")
        except Exception as exc:
            raise ValueError(
                f"Cannot decode image bytes (media_type={media_type!r}): {exc}"
            ) from exc

    def _infer_single(self, media: bytes, media_type: str) -> list[float]:
        """Synchronous single-image inference."""
        image = self._decode_image(media, media_type)
        model, preprocess = self._load()

        image_tensor = preprocess(image).unsqueeze(0).to(self._device)
        t0 = time.monotonic()
        with torch.no_grad():
            embedding = model.encode_image(image_tensor)
        elapsed_ms = (time.monotonic() - t0) * 1000

        if _OTEL_AVAILABLE:
            if _inference_count is not None:
                _inference_count.add(1)
            if _inference_latency is not None:
                _inference_latency.record(elapsed_ms)

        embedding = embedding.squeeze(0)
        norm = embedding.norm()
        if norm > 0:
            embedding = embedding / norm
        result: list[float] = embedding.cpu().numpy().tolist()
        return result

    def _infer_batch(
        self, media_list: list[bytes], media_type: str
    ) -> list[list[float]]:
        """Synchronous batched inference — single forward pass."""
        model, preprocess = self._load()

        tensors = []
        for media in media_list:
            image = self._decode_image(media, media_type)
            tensors.append(preprocess(image))

        batch = torch.stack(tensors).to(self._device)
        t0 = time.monotonic()
        with torch.no_grad():
            embeddings = model.encode_image(batch)
        elapsed_ms = (time.monotonic() - t0) * 1000

        if _OTEL_AVAILABLE:
            if _inference_count is not None:
                _inference_count.add(len(media_list))
            if _inference_latency is not None:
                _inference_latency.record(elapsed_ms)

        norms = embeddings.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        embeddings = embeddings / norms
        results: list[list[float]] = embeddings.cpu().numpy().tolist()
        return results

    # ------------------------------------------------------------------
    # VisualEmbeddingProvider protocol
    # ------------------------------------------------------------------

    async def embed(self, media: bytes, media_type: str) -> list[float]:
        """Embed a single image, returning a 768-dim float vector.

        Args:
            media: Raw image bytes (JPEG, PNG, etc.).
            media_type: MIME type hint (e.g. ``"image/jpeg"``).

        Returns:
            A list of 768 floats.

        Raises:
            ValueError: If ``media`` cannot be decoded as an image.
        """
        return await asyncio.to_thread(self._infer_single, media, media_type)

    async def embed_batch(
        self, media_list: list[bytes], media_type: str
    ) -> list[list[float]]:
        """Embed a batch of images in a single forward pass.

        Args:
            media_list: List of raw image bytes.
            media_type: MIME type hint shared across the batch.

        Returns:
            A list of 768-dim float vectors, one per input image.

        Raises:
            ValueError: If any image in ``media_list`` cannot be decoded.
        """
        if not media_list:
            return []
        return await asyncio.to_thread(self._infer_batch, media_list, media_type)

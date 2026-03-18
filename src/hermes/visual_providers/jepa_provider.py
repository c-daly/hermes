"""V-JEPA visual embedding provider for Hermes.

Implements the VisualEmbeddingProvider protocol using Facebook's
V-JEPA ViT-H/14 model (1024-dim patch tokens, mean-pooled).

Optional dependencies: torch, torchvision, Pillow.  The module is importable
without them; instantiation raises ImportError if they are absent.

Env vars:
    JEPA_DEVICE       torch device string (default: "cpu")
    JEPA_WEIGHTS_PATH path to local weights file used as fallback when the
                      torch-hub download is unavailable
"""

from __future__ import annotations

import asyncio
import io
import logging
import math
import os
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy dependencies -- try-import so the module stays importable
# ---------------------------------------------------------------------------
try:
    import torch
    from torchvision import transforms
    from PIL import Image

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# ---------------------------------------------------------------------------
# Optional OTel metrics -- try-import pattern
# ---------------------------------------------------------------------------
try:
    from opentelemetry import metrics as _otel_metrics

    _meter = _otel_metrics.get_meter("hermes.jepa")
    _inference_count = _meter.create_counter(
        "hermes.jepa.inference_count",
        description="Total number of JEPA inference calls",
    )
    _inference_latency = _meter.create_histogram(
        "hermes.jepa.inference_latency_ms",
        description="JEPA per-item inference latency",
        unit="ms",
    )
    _model_load_time = _meter.create_histogram(
        "hermes.jepa.model_load_time_ms",
        description="Time taken to load the V-JEPA model",
        unit="ms",
    )
    _OTEL_AVAILABLE = True
except Exception:  # noqa: BLE001
    _OTEL_AVAILABLE = False
    _inference_count = None  # type: ignore[assignment]
    _inference_latency = None  # type: ignore[assignment]
    _model_load_time = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Standard ImageNet normalisation (shared with CLIP / most ViT models)
# ---------------------------------------------------------------------------
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


class JEPAVisualProvider:
    """Visual embedding provider backed by V-JEPA ViT-H/14.

    Satisfies the VisualEmbeddingProvider protocol defined in
    hermes.embedding_provider.  The model is loaded lazily on the first
    embed / embed_batch call so startup time is not penalised.

    Raises:
        ImportError: on instantiation when torch / torchvision / Pillow are
            absent from the environment.
    """

    def __init__(self) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "JEPAVisualProvider requires torch, torchvision, and Pillow. "
                "Install them with: pip install torch torchvision Pillow"
            )

        try:
            from logos_config import get_env_value
        except ImportError:
            from collections.abc import Mapping

            def get_env_value(
                key: str,
                env: Mapping[str, str] | None = None,
                default: str | None = None,
            ) -> str | None:
                return (env or os.environ).get(key, default)

        device_str = get_env_value("JEPA_DEVICE") or "cpu"
        self._device = torch.device(device_str)

        dtype_str = (get_env_value("JEPA_DTYPE") or "fp32").lower()
        _dtype_map = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        self._dtype = _dtype_map.get(dtype_str, torch.float32)

        self._weights_path: str | None = get_env_value("JEPA_WEIGHTS_PATH")
        self._load_lock = threading.Lock()
        self._model: Any = None

        self._transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ]
        )

    # ------------------------------------------------------------------
    # Protocol properties
    # ------------------------------------------------------------------

    @property
    def dimension(self) -> int:
        """Embedding dimensionality: 1024 (ViT-H/14 patch tokens)."""
        return 1024

    @property
    def model_name(self) -> str:
        return "vjepa-vit-h14"

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> Any:
        """Load V-JEPA ViT-H/14 (idempotent -- only runs once)."""
        if self._model is not None:
            return self._model

        t0 = time.monotonic()
        logger.info("Loading V-JEPA ViT-H/14 (device=%s)...", self._device)

        model: Any = None

        # Primary path: torch hub
        try:
            model = torch.hub.load(
                "facebookresearch/jepa",
                "vjepa_vith14",
                pretrained=True,
            )
            logger.info("V-JEPA loaded from torch hub")
        except Exception as hub_exc:
            logger.warning("torch.hub load failed (%s); trying local weights", hub_exc)

            # Fallback: local weights file from JEPA_WEIGHTS_PATH
            if self._weights_path and os.path.exists(self._weights_path):
                try:
                    model = torch.load(
                        self._weights_path,
                        map_location=self._device,
                        weights_only=True,
                    )
                    logger.info("V-JEPA loaded from %s", self._weights_path)
                except Exception as load_exc:
                    raise RuntimeError(
                        f"V-JEPA load failed via hub ({hub_exc}) "
                        f"and via weights file ({load_exc})"
                    ) from load_exc
            else:
                # CPU fallback: retry hub without pretrained weights so
                # architecture is at least available for testing
                try:
                    model = torch.hub.load(
                        "facebookresearch/jepa",
                        "vjepa_vith14",
                        pretrained=False,
                    )
                    logger.warning(
                        "V-JEPA loaded without pretrained weights (hub pretrained "
                        "download failed and JEPA_WEIGHTS_PATH not set)"
                    )
                except Exception as fallback_exc:
                    raise RuntimeError(
                        f"V-JEPA load failed: hub={hub_exc}, fallback={fallback_exc}"
                    ) from fallback_exc

        model = model.to(self._device)
        model = model.to(dtype=self._dtype)
        model.eval()
        self._model = model

        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.info("V-JEPA model ready in %.1f ms", elapsed_ms)
        if _OTEL_AVAILABLE:
            _model_load_time.record(elapsed_ms)

        return self._model

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self, media: bytes, media_type: str) -> "torch.Tensor":
        """Decode media bytes to a single-frame video tensor.

        Returns shape [1, 3, 1, 224, 224] (batch=1, C=3, T=1, H=224, W=224).

        Raises:
            ValueError: if media cannot be decoded as an image.
        """
        try:
            image = Image.open(io.BytesIO(media)).convert("RGB")
        except Exception as exc:
            raise ValueError(
                f"Cannot decode media bytes as an image ({media_type!r}): {exc}"
            ) from exc

        # frame: (C, H, W)
        frame: torch.Tensor = self._transform(image)
        # video: (B=1, C=3, T=1, H=224, W=224)
        video = (
            frame.unsqueeze(0).unsqueeze(2).to(device=self._device, dtype=self._dtype)
        )
        return video

    # ------------------------------------------------------------------
    # Synchronous inference (runs in a thread via asyncio.to_thread)
    # ------------------------------------------------------------------

    def _run_inference(self, media: bytes, media_type: str) -> list[float]:
        """Run a single forward pass; return a 1024-dim embedding.

        Raises:
            ValueError: on invalid/undecodable media.
            RuntimeError: if model output shape is unexpected.
        """
        model = self._load_model()
        video = self._preprocess(media, media_type)

        t0 = time.monotonic()
        with torch.no_grad():
            output = model(video)
        elapsed_ms = (time.monotonic() - t0) * 1000

        if _OTEL_AVAILABLE:
            _inference_count.add(1)
            _inference_latency.record(elapsed_ms)

        # Unwrap tuple/list outputs (some hub models return (features, masks, ...))
        if isinstance(output, (list, tuple)):
            output = output[0]

        # Normalise to 2-D (tokens, dim) then mean-pool -> 1-D (dim,)
        if output.dim() == 3:
            # (batch, tokens, dim) -- take first batch item, mean over tokens
            embedding: torch.Tensor = output[0].mean(dim=0)
        elif output.dim() == 2:
            # (tokens, dim) -- mean over tokens
            embedding = output.mean(dim=0)
        elif output.dim() == 1:
            # Already a flat vector
            embedding = output
        else:
            raise RuntimeError(f"Unexpected model output shape: {tuple(output.shape)}")

        result: list[float] = embedding.cpu().float().tolist()

        if len(result) != self.dimension:
            raise RuntimeError(
                f"Model returned {len(result)}-dim embedding; expected {self.dimension}"
            )
        if not all(math.isfinite(v) for v in result):
            raise RuntimeError("Model output contains non-finite values")

        return result

    # ------------------------------------------------------------------
    # Async public API (VisualEmbeddingProvider protocol)
    # ------------------------------------------------------------------

    async def embed(self, media: bytes, media_type: str) -> list[float]:
        """Return a 1024-dim embedding for media.

        Args:
            media:      Raw bytes of the image/video.
            media_type: MIME type hint (e.g. image/jpeg).

        Raises:
            ValueError: if the bytes cannot be decoded as an image.
        """
        return await asyncio.to_thread(self._run_inference, media, media_type)

    async def embed_batch(
        self, media_list: list[bytes], media_type: str
    ) -> list[list[float]]:
        """Return embeddings for every item in media_list.

        Items are processed concurrently via asyncio.gather; each item
        runs _run_inference in its own thread.
        """
        if not media_list:
            return []
        results = await asyncio.gather(
            *(self.embed(media, media_type) for media in media_list)
        )
        return list(results)

"""Tests that exercise CLIPVisualProvider through real code paths.

Mock open_clip to return a fake model, but let the actual preprocessing,
normalization, and inference pipeline run.
"""

import asyncio
import math
import struct
import sys
import zlib
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from torchvision import transforms


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tiny_png() -> bytes:
    """Minimal valid 1x1 red PNG."""

    def chunk(ctype: bytes, data: bytes) -> bytes:
        c = ctype + data
        return (
            struct.pack(">I", len(data))
            + c
            + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
        )

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    raw = zlib.compress(b"\x00\xff\x00\x00")
    idat = chunk(b"IDAT", raw)
    iend = chunk(b"IEND", b"")
    return sig + ihdr + idat + iend


def _make_fake_clip(dim: int = 768) -> tuple:
    """Return (model, None, preprocess) mimicking open_clip."""
    model = MagicMock()
    model.eval = MagicMock(return_value=model)
    model.to = MagicMock(return_value=model)

    def encode_image(x: Any) -> torch.Tensor:
        batch = x.shape[0]
        # Non-uniform magnitudes to verify normalization works
        return torch.randn(batch, dim) * 5.0

    model.encode_image = encode_image

    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    return model, None, preprocess


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _ensure_logos_config():
    import os

    if "logos_config" not in sys.modules or not hasattr(
        sys.modules["logos_config"], "get_env_value"
    ):
        mock_lc = MagicMock()
        mock_lc.get_env_value = lambda key, env=None, default=None: os.environ.get(
            key, default
        )
        sys.modules["logos_config"] = mock_lc


@pytest.fixture()
def clip_provider():
    """Real CLIPVisualProvider with mocked open_clip."""
    fake_model, _, fake_preprocess = _make_fake_clip()

    mock_oc = MagicMock()
    mock_oc.create_model_and_transforms = MagicMock(
        return_value=(fake_model, None, fake_preprocess)
    )
    sys.modules["open_clip"] = mock_oc

    import importlib
    import hermes.visual_providers.clip_provider as mod

    importlib.reload(mod)
    provider = mod.CLIPVisualProvider()
    return provider


# ---------------------------------------------------------------------------
# Tests: normalization
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCLIPNormalization:

    def test_single_embed_is_unit_vector(self, clip_provider):
        result = asyncio.get_event_loop().run_until_complete(
            clip_provider.embed(_tiny_png(), "image/png")
        )
        assert len(result) == 768
        magnitude = math.sqrt(sum(v * v for v in result))
        assert abs(magnitude - 1.0) < 1e-4, f"Not unit-norm: magnitude={magnitude}"

    def test_batch_embeds_are_unit_vectors(self, clip_provider):
        png = _tiny_png()
        results = asyncio.get_event_loop().run_until_complete(
            clip_provider.embed_batch([png, png, png], "image/png")
        )
        for i, emb in enumerate(results):
            magnitude = math.sqrt(sum(v * v for v in emb))
            assert (
                abs(magnitude - 1.0) < 1e-4
            ), f"Embedding {i} not unit-norm: {magnitude}"


# ---------------------------------------------------------------------------
# Tests: full pipeline
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCLIPInferencePipeline:

    def test_embed_produces_768_dim(self, clip_provider):
        result = asyncio.get_event_loop().run_until_complete(
            clip_provider.embed(_tiny_png(), "image/png")
        )
        assert len(result) == 768
        assert all(isinstance(v, float) for v in result)

    def test_embed_all_finite(self, clip_provider):
        result = asyncio.get_event_loop().run_until_complete(
            clip_provider.embed(_tiny_png(), "image/png")
        )
        assert all(math.isfinite(v) for v in result)

    def test_embed_invalid_bytes_raises(self, clip_provider):
        with pytest.raises((ValueError, RuntimeError)):
            asyncio.get_event_loop().run_until_complete(
                clip_provider.embed(b"garbage", "image/png")
            )

    def test_load_idempotent(self, clip_provider):
        m1, _ = clip_provider._load()
        m2, _ = clip_provider._load()
        assert m1 is m2

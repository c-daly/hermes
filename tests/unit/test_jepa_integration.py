"""Tests that exercise JEPAVisualProvider through real code paths.

Mock torch.hub.load to return a fake nn.Module, but let the actual
preprocessing, inference pipeline, and validation run.
"""

import asyncio
import math
import struct
import sys
import zlib
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch")


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


def _make_fake_jepa_model(dim: int = 1024, n_tokens: int = 32) -> MagicMock:
    """Fake nn.Module producing (batch, n_tokens, dim) output."""
    model = MagicMock()
    model.eval = MagicMock(return_value=model)
    model.to = MagicMock(return_value=model)

    def forward(x: Any) -> torch.Tensor:
        batch = x.shape[0]
        torch.manual_seed(42)
        return torch.randn(batch, n_tokens, dim)

    model.side_effect = forward
    return model


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _ensure_logos_config():
    """Mock logos_config if not installed."""
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
def jepa_provider():
    """Real JEPAVisualProvider with mocked hub load."""
    import importlib
    import hermes.visual_providers.jepa_provider as mod

    fake_model = _make_fake_jepa_model()
    with patch("torch.hub.load", return_value=fake_model):
        importlib.reload(mod)
        provider = mod.JEPAVisualProvider()
        # Trigger model load while hub mock is active
        provider._load_model()
    return provider


# ---------------------------------------------------------------------------
# Tests: _load_model
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestJEPALoadModel:

    def test_returns_not_none(self, jepa_provider):
        assert jepa_provider._model is not None

    def test_idempotent(self):
        import importlib
        import hermes.visual_providers.jepa_provider as mod

        fake_model = _make_fake_jepa_model()
        with patch("torch.hub.load", return_value=fake_model) as mock_hub:
            importlib.reload(mod)
            provider = mod.JEPAVisualProvider()
            m1 = provider._load_model()
            m2 = provider._load_model()
        assert m1 is m2
        assert mock_hub.call_count == 1

    def test_hub_failure_no_weights_raises(self):
        import importlib
        import hermes.visual_providers.jepa_provider as mod

        with patch("torch.hub.load", side_effect=RuntimeError("no network")):
            importlib.reload(mod)
            provider = mod.JEPAVisualProvider()
            with pytest.raises(RuntimeError, match="V-JEPA model load failed"):
                provider._load_model()


# ---------------------------------------------------------------------------
# Tests: full inference pipeline
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestJEPAInferencePipeline:

    def test_embed_produces_1024_dim(self, jepa_provider):
        result = asyncio.get_event_loop().run_until_complete(
            jepa_provider.embed(_tiny_png(), "image/png")
        )
        assert isinstance(result, list)
        assert len(result) == 1024

    def test_embed_all_finite(self, jepa_provider):
        result = asyncio.get_event_loop().run_until_complete(
            jepa_provider.embed(_tiny_png(), "image/png")
        )
        assert all(math.isfinite(v) for v in result)

    def test_embed_batch_shapes(self, jepa_provider):
        png = _tiny_png()
        batch = asyncio.get_event_loop().run_until_complete(
            jepa_provider.embed_batch([png, png], "image/png")
        )
        assert len(batch) == 2
        assert all(len(emb) == 1024 for emb in batch)

    def test_embed_invalid_bytes_raises(self, jepa_provider):
        with pytest.raises((ValueError, RuntimeError)):
            asyncio.get_event_loop().run_until_complete(
                jepa_provider.embed(b"not an image", "image/png")
            )

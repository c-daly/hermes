"""Unit tests for CLIPVisualProvider."""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Helpers: build mock modules for open_clip / torch / PIL / logos_config
# ---------------------------------------------------------------------------


def _make_mock_torch() -> MagicMock:
    """Create a mock ``torch`` module."""
    mock_torch = MagicMock(name="torch")
    mock_torch.no_grad.return_value.__enter__ = MagicMock(return_value=None)
    mock_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)
    mock_torch.stack = MagicMock()
    return mock_torch


def _make_mock_open_clip() -> MagicMock:
    """Create a mock ``open_clip`` module."""
    mock_oc = MagicMock(name="open_clip")
    return mock_oc


def _make_mock_pil() -> tuple[MagicMock, MagicMock]:
    """Create a mock ``PIL`` module."""
    mock_pil = MagicMock(name="PIL")
    mock_pil_image = MagicMock(name="PIL.Image")
    mock_pil.Image = mock_pil_image
    return mock_pil, mock_pil_image


def _make_mock_logos_config() -> MagicMock:
    """Create a mock ``logos_config`` module."""
    mock_lc = MagicMock(name="logos_config")
    mock_lc.get_env_value = MagicMock(return_value=None)
    return mock_lc


def _inject_mocks() -> tuple[dict[str, ModuleType | MagicMock], dict[str, MagicMock]]:
    """Build all mock modules.

    Returns (modules_dict_for_sys_modules, mocks_by_name).
    """
    mock_torch = _make_mock_torch()
    mock_oc = _make_mock_open_clip()
    mock_pil, mock_pil_image = _make_mock_pil()
    mock_lc = _make_mock_logos_config()

    modules: dict[str, ModuleType | MagicMock] = {
        "torch": mock_torch,
        "open_clip": mock_oc,
        "PIL": mock_pil,
        "PIL.Image": mock_pil_image,
        "logos_config": mock_lc,
    }
    mocks = {
        "torch": mock_torch,
        "open_clip": mock_oc,
        "pil": mock_pil,
        "pil_image": mock_pil_image,
        "logos_config": mock_lc,
    }
    return modules, mocks


@pytest.fixture()
def provider_env():
    """Import CLIPVisualProvider with all heavy deps mocked.

    Yields (provider_instance, mocks_dict, module).
    """
    modules, mocks = _inject_mocks()

    mod_name = "hermes.visual_providers.clip_provider"
    saved = sys.modules.pop(mod_name, None)

    with patch.dict(sys.modules, modules):
        import importlib

        import hermes.visual_providers.clip_provider as cp_mod

        importlib.reload(cp_mod)
        provider = cp_mod.CLIPVisualProvider()
        yield provider, mocks, cp_mod

    if saved is not None:
        sys.modules[mod_name] = saved
    else:
        sys.modules.pop(mod_name, None)


# ------------------------------------------------------------------
# Test classes
# ------------------------------------------------------------------


class TestProtocolCompliance:
    def test_protocol_compliance(self, provider_env):
        """CLIPVisualProvider has the required VisualEmbeddingProvider interface."""
        provider, _mocks, _mod = provider_env

        assert hasattr(provider, "dimension")
        assert hasattr(provider, "model_name")
        assert hasattr(provider, "embed")
        assert hasattr(provider, "embed_batch")
        assert callable(provider.embed)
        assert callable(provider.embed_batch)


class TestProperties:
    def test_dimension_is_768(self, provider_env):
        provider, _mocks, _mod = provider_env
        assert provider.dimension == 768

    def test_model_name(self, provider_env):
        provider, _mocks, _mod = provider_env
        assert isinstance(provider.model_name, str)
        assert len(provider.model_name) > 0


class TestLazyLoading:
    def test_lazy_loading(self, provider_env):
        """After init, _model should be None (model not loaded yet)."""
        provider, _mocks, _mod = provider_env
        assert provider._model is None


class TestEmbed:
    @pytest.mark.asyncio
    async def test_embed_returns_correct_length(self, provider_env):
        """embed() returns a list of length 768."""
        provider, _mocks, _mod = provider_env
        expected = [0.1] * 768

        with patch.object(provider, "_infer_single", return_value=expected):
            result = await provider.embed(b"fake-image", "image/jpeg")

        assert isinstance(result, list)
        assert len(result) == 768
        assert result == expected

    @pytest.mark.asyncio
    async def test_embed_batch_empty(self, provider_env):
        """embed_batch with empty list returns []."""
        provider, _mocks, _mod = provider_env
        result = await provider.embed_batch([], "image/jpeg")
        assert result == []

    @pytest.mark.asyncio
    async def test_embed_batch_uses_batch_inference(self, provider_env):
        """embed_batch calls _infer_batch, not N individual calls."""
        provider, _mocks, _mod = provider_env
        batch_result = [[0.1] * 768, [0.2] * 768, [0.3] * 768]

        with patch.object(
            provider, "_infer_batch", return_value=batch_result
        ) as mock_batch:
            items = [b"img1", b"img2", b"img3"]
            result = await provider.embed_batch(items, "image/jpeg")

        mock_batch.assert_called_once_with(items, "image/jpeg")
        assert len(result) == 3
        for vec in result:
            assert len(vec) == 768


class TestDecodeImage:
    def test_decode_image_invalid_bytes(self, provider_env):
        """_decode_image raises ValueError on garbage bytes."""
        provider, mocks, _mod = provider_env

        # Make PIL.Image.open raise on garbage data
        mock_pil_image = mocks["pil_image"]
        mock_pil_image.open.return_value.convert.side_effect = Exception("bad image")

        with pytest.raises(ValueError, match="Cannot decode"):
            provider._decode_image(b"\x00\x01\x02not-an-image", "image/jpeg")


class TestDeviceConfig:
    def test_device_defaults_to_cpu(self, provider_env):
        """Without CLIP_DEVICE env var, device defaults to cpu."""
        provider, _mocks, _mod = provider_env
        assert provider._device == "cpu"

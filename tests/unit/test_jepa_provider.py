"""Unit tests for JEPAVisualProvider."""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Helpers: build mock modules for torch / torchvision / PIL / logos_config
# so we can import jepa_provider in a test environment that lacks them.
# ---------------------------------------------------------------------------


def _make_mock_torch() -> MagicMock:
    """Create a mock ``torch`` module with the subset the provider uses."""
    mock_torch = MagicMock(name="torch")

    # torch.device returns something whose str() gives the device name
    mock_torch.device = MagicMock(
        side_effect=lambda s: MagicMock(__str__=lambda _self: s)
    )

    # dtype sentinels
    mock_torch.float32 = "torch.float32"
    mock_torch.float16 = "torch.float16"
    mock_torch.bfloat16 = "torch.bfloat16"

    # torch.no_grad context manager
    mock_torch.no_grad.return_value.__enter__ = MagicMock(return_value=None)
    mock_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)

    # torch.hub.load
    mock_torch.hub = MagicMock()

    # torch.load
    mock_torch.load = MagicMock()

    return mock_torch


def _make_mock_torchvision() -> tuple[MagicMock, MagicMock]:
    """Create a mock ``torchvision`` module."""
    mock_tv = MagicMock(name="torchvision")
    mock_transforms = MagicMock(name="torchvision.transforms")
    mock_tv.transforms = mock_transforms
    return mock_tv, mock_transforms


def _make_mock_pil() -> tuple[MagicMock, MagicMock]:
    """Create a mock ``PIL`` module."""
    mock_pil = MagicMock(name="PIL")
    mock_pil_image = MagicMock(name="PIL.Image")
    mock_pil.Image = mock_pil_image
    return mock_pil, mock_pil_image


def _make_mock_logos_config() -> MagicMock:
    """Create a mock ``logos_config`` module."""
    mock_lc = MagicMock(name="logos_config")
    # get_env_value returns None by default (no env vars set)
    mock_lc.get_env_value = MagicMock(return_value=None)
    return mock_lc


def _inject_mocks() -> tuple[dict[str, ModuleType | MagicMock], dict[str, MagicMock]]:
    """Build all mock modules.

    Returns (modules_dict_for_sys_modules, mocks_by_name).
    """
    mock_torch = _make_mock_torch()
    mock_tv, mock_transforms = _make_mock_torchvision()
    mock_pil, mock_pil_image = _make_mock_pil()
    mock_lc = _make_mock_logos_config()

    modules: dict[str, ModuleType | MagicMock] = {
        "torch": mock_torch,
        "torchvision": mock_tv,
        "torchvision.transforms": mock_transforms,
        "PIL": mock_pil,
        "PIL.Image": mock_pil_image,
        "logos_config": mock_lc,
    }
    mocks = {
        "torch": mock_torch,
        "torchvision": mock_tv,
        "transforms": mock_transforms,
        "pil": mock_pil,
        "pil_image": mock_pil_image,
        "logos_config": mock_lc,
    }
    return modules, mocks


@pytest.fixture()
def provider_env():
    """Import JEPAVisualProvider with all heavy deps mocked.

    Yields (provider_instance, mocks_dict, module).
    """
    modules, mocks = _inject_mocks()

    # Remove cached module so it re-imports with our mocks
    mod_name = "hermes.visual_providers.jepa_provider"
    saved = sys.modules.pop(mod_name, None)

    with patch.dict(sys.modules, modules):
        # Force re-import under the mock-injected environment
        import importlib

        import hermes.visual_providers.jepa_provider as jp_mod

        importlib.reload(jp_mod)
        provider = jp_mod.JEPAVisualProvider()
        yield provider, mocks, jp_mod

    # Restore
    if saved is not None:
        sys.modules[mod_name] = saved
    else:
        sys.modules.pop(mod_name, None)


# ------------------------------------------------------------------
# Test classes
# ------------------------------------------------------------------


class TestProtocolCompliance:
    def test_protocol_compliance(self, provider_env):
        """JEPAVisualProvider has the required VisualEmbeddingProvider interface."""
        provider, _mocks, _mod = provider_env

        assert hasattr(provider, "dimension")
        assert hasattr(provider, "model_name")
        assert hasattr(provider, "embed")
        assert hasattr(provider, "embed_batch")
        assert callable(provider.embed)
        assert callable(provider.embed_batch)


class TestProperties:
    def test_dimension_is_1024(self, provider_env):
        provider, _mocks, _mod = provider_env
        assert provider.dimension == 1024

    def test_model_name(self, provider_env):
        provider, _mocks, _mod = provider_env
        assert isinstance(provider.model_name, str)
        assert len(provider.model_name) > 0


class TestLazyLoading:
    def test_lazy_loading_no_model_on_init(self, provider_env):
        """After init, _model should be None (lazy loading)."""
        provider, _mocks, _mod = provider_env
        assert provider._model is None


class TestEmbed:
    @pytest.mark.asyncio
    async def test_embed_returns_correct_length(self, provider_env):
        """embed() returns a list of length 1024."""
        provider, _mocks, _mod = provider_env
        expected = [0.1] * 1024

        with patch.object(provider, "_run_inference", return_value=expected):
            result = await provider.embed(b"fake-image", "image/jpeg")

        assert isinstance(result, list)
        assert len(result) == 1024
        assert result == expected

    @pytest.mark.asyncio
    async def test_embed_batch_empty(self, provider_env):
        """embed_batch with empty list returns []."""
        provider, _mocks, _mod = provider_env
        result = await provider.embed_batch([], "image/jpeg")
        assert result == []

    @pytest.mark.asyncio
    async def test_embed_batch_multiple(self, provider_env):
        """embed_batch with 3 items returns 3 results of correct length."""
        provider, _mocks, _mod = provider_env
        expected_single = [0.1] * 1024

        with patch.object(provider, "_run_inference", return_value=expected_single):
            items = [b"img1", b"img2", b"img3"]
            result = await provider.embed_batch(items, "image/jpeg")

        assert len(result) == 3
        for vec in result:
            assert len(vec) == 1024


class TestNanDetection:
    def test_nan_detection(self, provider_env):
        """_run_inference raises RuntimeError when model output has NaN."""
        provider, mocks, _mod = provider_env

        # Build a mock model whose forward pass returns a 1-D tensor with NaN
        mock_model = MagicMock()
        nan_values = [0.1] * 1023 + [float("nan")]
        mock_output = MagicMock()
        mock_output.dim.return_value = 1
        mock_output.cpu.return_value.float.return_value.tolist.return_value = nan_values
        mock_model.return_value = mock_output

        provider._model = mock_model

        with patch.object(provider, "_preprocess", return_value=MagicMock()):
            with pytest.raises(RuntimeError, match="non-finite"):
                provider._run_inference(b"fake", "image/jpeg")


class TestDeviceConfig:
    def test_device_defaults_to_cpu(self, provider_env):
        """Without JEPA_DEVICE env var, device defaults to cpu."""
        provider, mocks, _mod = provider_env
        mock_torch = mocks["torch"]
        mock_torch.device.assert_called_with("cpu")


class TestDtypeConfig:
    def test_dtype_defaults_to_fp32(self, provider_env):
        """Without JEPA_DTYPE env var, dtype defaults to torch.float32."""
        provider, mocks, _mod = provider_env
        mock_torch = mocks["torch"]
        assert provider._dtype == mock_torch.float32

    def test_dtype_env_var_fp16(self):
        """Setting JEPA_DTYPE=fp16 selects torch.float16."""
        modules, mocks = _inject_mocks()
        mock_lc = mocks["logos_config"]

        def _env_side_effect(key, **kwargs):
            if key == "JEPA_DTYPE":
                return "fp16"
            return None

        mock_lc.get_env_value.side_effect = _env_side_effect

        mod_name = "hermes.visual_providers.jepa_provider"
        saved = sys.modules.pop(mod_name, None)

        try:
            with patch.dict(sys.modules, modules):
                import importlib

                import hermes.visual_providers.jepa_provider as jp_mod

                importlib.reload(jp_mod)
                provider = jp_mod.JEPAVisualProvider()
                assert provider._dtype == mocks["torch"].float16
        finally:
            if saved is not None:
                sys.modules[mod_name] = saved
            else:
                sys.modules.pop(mod_name, None)

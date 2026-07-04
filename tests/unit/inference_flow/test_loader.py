"""Unit tests for harmonyspeech/modeling/loader.py — _get_model_cls."""

from unittest.mock import MagicMock, patch

import pytest

from harmonyspeech.common.config import DeviceConfig, ModelConfig
from harmonyspeech.modeling.loader import _check_ctranslate2_cuda_available, _get_model_cls


@pytest.fixture
def model_config_factory():
    """Factory to create ModelConfig with a given model_type."""

    def _make(model_type: str):
        return ModelConfig(
            name="test-model",
            model="path/to/model",
            model_type=model_type,
            max_batch_size=1,
            device_config=DeviceConfig("cpu"),
            dtype="float32",
        )

    return _make


def test_get_model_cls_raises_for_unsupported_type(model_config_factory):
    """Unknown model_type raises ValueError with type name in message."""
    with (
        patch("harmonyspeech.modeling.models.ModelRegistry.load_model_cls", return_value=None),
        patch(
            "harmonyspeech.modeling.models.ModelRegistry.get_supported_archs",
            return_value=["MeloTTSSynthesizer", "FasterWhisper"],
        ),
    ):
        cfg = model_config_factory("UnsupportedModelXYZ")
        with pytest.raises(ValueError, match="UnsupportedModelXYZ"):
            _get_model_cls(cfg)


def test_get_model_cls_returns_class_from_registry(model_config_factory):
    """When registry returns a class, _get_model_cls returns it unchanged."""
    mock_class = MagicMock()
    with patch("harmonyspeech.modeling.models.ModelRegistry.load_model_cls", return_value=mock_class):
        cfg = model_config_factory("MeloTTSSynthesizer")
        result = _get_model_cls(cfg)
        assert result is mock_class


def test_get_model_cls_passes_model_type_to_registry(model_config_factory):
    """Registry is called with the model_type string from ModelConfig."""
    mock_class = MagicMock()
    with patch("harmonyspeech.modeling.models.ModelRegistry.load_model_cls", return_value=mock_class) as mock_load:
        cfg = model_config_factory("FasterWhisper")
        _get_model_cls(cfg)
        mock_load.assert_called_once_with("FasterWhisper")


def test_get_model_cls_error_mentions_model_type(model_config_factory):
    """ValueError message includes the unsupported model_type name."""
    with patch("harmonyspeech.modeling.models.ModelRegistry.load_model_cls", return_value=None):
        with patch("harmonyspeech.modeling.models.ModelRegistry.get_supported_archs", return_value=[]):
            cfg = model_config_factory("NonExistentModel")
            with pytest.raises(ValueError) as exc_info:
                _get_model_cls(cfg)
            assert "NonExistentModel" in str(exc_info.value)


def test_get_model_cls_error_includes_supported_list(model_config_factory):
    """ValueError message includes list of supported architectures."""
    with (
        patch("harmonyspeech.modeling.models.ModelRegistry.load_model_cls", return_value=None),
        patch(
            "harmonyspeech.modeling.models.ModelRegistry.get_supported_archs",
            return_value=["MeloTTSSynthesizer", "FasterWhisper", "OpenVoiceV1Synthesizer"],
        ),
    ):
        cfg = model_config_factory("UnknownModel")
        with pytest.raises(ValueError) as exc_info:
            _get_model_cls(cfg)
        error_msg = str(exc_info.value)
        assert "MeloTTSSynthesizer" in error_msg
        assert "FasterWhisper" in error_msg


# ===========================================================================
# _check_ctranslate2_cuda_available — cuDNN pre-check
# ===========================================================================


@pytest.mark.unit
@patch("harmonyspeech.modeling.loader.ctypes.CDLL")
def test_cuda_check_false_when_no_cuda_devices(mock_cdll):
    """Returns False when ctranslate2 reports zero CUDA devices."""
    import harmonyspeech.modeling.loader as loader_mod

    with patch.dict("sys.modules", {"ctranslate2": MagicMock(get_cuda_device_count=lambda: 0)}):
        # Reload not needed — _check uses runtime import
        result = loader_mod._check_ctranslate2_cuda_available()
    assert result is False


@pytest.mark.unit
@patch("harmonyspeech.modeling.loader.ctypes.CDLL", side_effect=OSError("not found"))
def test_cuda_check_false_when_cudnn_missing(mock_cdll):
    """Returns False when cuDNN shared libraries cannot be loaded."""
    import harmonyspeech.modeling.loader as loader_mod

    with patch.dict("sys.modules", {"ctranslate2": MagicMock(get_cuda_device_count=lambda: 1)}):
        result = loader_mod._check_ctranslate2_cuda_available()
    assert result is False


@pytest.mark.unit
@patch("harmonyspeech.modeling.loader.ctypes.CDLL")
def test_cuda_check_true_when_devices_and_cudnn_present(mock_cdll):
    """Returns True when CUDA devices exist and cuDNN libs load successfully."""
    import harmonyspeech.modeling.loader as loader_mod

    with patch.dict("sys.modules", {"ctranslate2": MagicMock(get_cuda_device_count=lambda: 1)}):
        result = loader_mod._check_ctranslate2_cuda_available()
    assert result is True
    # Should have attempted to load the cuDNN libraries
    assert mock_cdll.call_count >= 1


@pytest.mark.unit
def test_cuda_check_false_on_ctranslate2_import_error():
    """Returns False if ctranslate2 cannot be imported at all."""
    import harmonyspeech.modeling.loader as loader_mod

    original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

    def _block_ctranslate2(name, *args, **kwargs):
        if name == "ctranslate2":
            raise ImportError("no ctranslate2")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=_block_ctranslate2):
        result = loader_mod._check_ctranslate2_cuda_available()
    assert result is False


# ===========================================================================
# FasterWhisper device fallback in get_model
# ===========================================================================


def _make_fasterwhisper_config(device: str = "cpu") -> ModelConfig:
    return ModelConfig(
        name="test-whisper",
        model="tiny",
        model_type="FasterWhisper",
        max_batch_size=4,
        device_config=DeviceConfig(device),
        dtype="float32",
    )


@pytest.mark.unit
@patch("harmonyspeech.modeling.loader.WhisperModel")
@patch("harmonyspeech.modeling.loader._check_ctranslate2_cuda_available", return_value=False)
def test_fasterwhisper_cuda_falls_back_to_cpu_when_cudnn_missing(mock_cuda_check, mock_whisper_cls):
    """When device='cuda' but cuDNN is unavailable, WhisperModel must be created
    with device='cpu' and compute_type='int8' — NOT device='cuda' (which would
    cause a native SIGABRT crash)."""
    from harmonyspeech.modeling.loader import get_model

    cfg = _make_fasterwhisper_config(device="cuda")
    get_model(cfg, DeviceConfig("cuda"))

    mock_whisper_cls.assert_called_once()
    _, kwargs = mock_whisper_cls.call_args
    assert kwargs["device"] == "cpu"
    assert kwargs["compute_type"] == "int8"


@pytest.mark.unit
@patch("harmonyspeech.modeling.loader.WhisperModel")
@patch("harmonyspeech.modeling.loader._check_ctranslate2_cuda_available", return_value=True)
def test_fasterwhisper_cuda_used_when_cudnn_available(mock_cuda_check, mock_whisper_cls):
    """When device='cuda' and cuDNN is available, WhisperModel uses cuda/float16."""
    from harmonyspeech.modeling.loader import get_model

    cfg = _make_fasterwhisper_config(device="cuda")
    get_model(cfg, DeviceConfig("cuda"))

    mock_whisper_cls.assert_called_once()
    _, kwargs = mock_whisper_cls.call_args
    assert kwargs["device"] == "cuda"
    assert kwargs["compute_type"] == "float16"


@pytest.mark.unit
@patch("harmonyspeech.modeling.loader.WhisperModel")
def test_fasterwhisper_cpu_uses_int8_compute_type(mock_whisper_cls):
    """When device='cpu', WhisperModel must be created with compute_type='int8'."""
    from harmonyspeech.modeling.loader import get_model

    cfg = _make_fasterwhisper_config(device="cpu")
    get_model(cfg, DeviceConfig("cpu"))

    mock_whisper_cls.assert_called_once()
    _, kwargs = mock_whisper_cls.call_args
    assert kwargs["device"] == "cpu"
    assert kwargs["compute_type"] == "int8"


@pytest.mark.unit
@patch("harmonyspeech.modeling.loader.WhisperModel")
@patch("harmonyspeech.modeling.loader._check_ctranslate2_cuda_available", return_value=False)
def test_fasterwhisper_cuda_fallback_does_not_pass_cuda_to_whispermodel(mock_cuda_check, mock_whisper_cls):
    """Critical safety: the string 'cuda' must NEVER appear in device kwarg when
    cuDNN is unavailable, as that would trigger the native abort() crash."""
    from harmonyspeech.modeling.loader import get_model

    cfg = _make_fasterwhisper_config(device="cuda")
    get_model(cfg, DeviceConfig("cuda"))

    _, kwargs = mock_whisper_cls.call_args
    assert "cuda" not in str(kwargs["device"])

"""Unit tests for harmonyspeech/modeling/loader.py — _get_model_cls."""
import pytest
from unittest.mock import patch, MagicMock
from harmonyspeech.common.config import DeviceConfig, ModelConfig
from harmonyspeech.modeling.loader import _get_model_cls


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
    with patch(
        "harmonyspeech.modeling.models.ModelRegistry.load_model_cls",
        return_value=None,
    ):
        with patch(
            "harmonyspeech.modeling.models.ModelRegistry.get_supported_archs",
            return_value=["MeloTTSSynthesizer", "FasterWhisper"],
        ):
            cfg = model_config_factory("UnsupportedModelXYZ")
            with pytest.raises(ValueError, match="UnsupportedModelXYZ"):
                _get_model_cls(cfg)


def test_get_model_cls_returns_class_from_registry(model_config_factory):
    """When registry returns a class, _get_model_cls returns it unchanged."""
    mock_class = MagicMock()
    with patch(
        "harmonyspeech.modeling.models.ModelRegistry.load_model_cls",
        return_value=mock_class,
    ):
        cfg = model_config_factory("MeloTTSSynthesizer")
        result = _get_model_cls(cfg)
        assert result is mock_class


def test_get_model_cls_passes_model_type_to_registry(model_config_factory):
    """Registry is called with the model_type string from ModelConfig."""
    mock_class = MagicMock()
    with patch(
        "harmonyspeech.modeling.models.ModelRegistry.load_model_cls",
        return_value=mock_class,
    ) as mock_load:
        cfg = model_config_factory("FasterWhisper")
        _get_model_cls(cfg)
        mock_load.assert_called_once_with("FasterWhisper")


def test_get_model_cls_error_mentions_model_type(model_config_factory):
    """ValueError message includes the unsupported model_type name."""
    with patch(
        "harmonyspeech.modeling.models.ModelRegistry.load_model_cls",
        return_value=None,
    ):
        with patch(
            "harmonyspeech.modeling.models.ModelRegistry.get_supported_archs",
            return_value=[],
        ):
            cfg = model_config_factory("NonExistentModel")
            with pytest.raises(ValueError) as exc_info:
                _get_model_cls(cfg)
            assert "NonExistentModel" in str(exc_info.value)


def test_get_model_cls_error_includes_supported_list(model_config_factory):
    """ValueError message includes list of supported architectures."""
    with patch(
        "harmonyspeech.modeling.models.ModelRegistry.load_model_cls",
        return_value=None,
    ):
        with patch(
            "harmonyspeech.modeling.models.ModelRegistry.get_supported_archs",
            return_value=["MeloTTSSynthesizer", "FasterWhisper", "OpenVoiceV1Synthesizer"],
        ):
            cfg = model_config_factory("UnknownModel")
            with pytest.raises(ValueError) as exc_info:
                _get_model_cls(cfg)
            error_msg = str(exc_info.value)
            assert "MeloTTSSynthesizer" in error_msg
            assert "FasterWhisper" in error_msg

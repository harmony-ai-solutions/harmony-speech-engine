"""Unit tests for Chatterbox model registry integration."""
import pytest

from harmonyspeech.modeling.models import ModelRegistry


@pytest.mark.unit
def test_chatterbox_tts_in_supported_archs():
    """Test that ChatterboxTTS is in the list of supported architectures."""
    archs = ModelRegistry.get_supported_archs()
    assert "ChatterboxTTS" in archs, "ChatterboxTTS should be in supported architectures"


@pytest.mark.unit
def test_chatterbox_turbo_tts_in_supported_archs():
    """Test that ChatterboxTurboTTS is in the list of supported architectures."""
    archs = ModelRegistry.get_supported_archs()
    assert "ChatterboxTurboTTS" in archs, "ChatterboxTurboTTS should be in supported architectures"


@pytest.mark.unit
def test_chatterbox_multilingual_tts_in_supported_archs():
    """Test that ChatterboxMultilingualTTS is in the list of supported architectures."""
    archs = ModelRegistry.get_supported_archs()
    assert "ChatterboxMultilingualTTS" in archs, "ChatterboxMultilingualTTS should be in supported architectures"


@pytest.mark.unit
def test_chatterbox_vc_in_supported_archs():
    """Test that ChatterboxVC is in the list of supported architectures."""
    archs = ModelRegistry.get_supported_archs()
    assert "ChatterboxVC" in archs, "ChatterboxVC should be in supported architectures"


@pytest.mark.unit
def test_load_model_cls_returns_native_for_chatterbox_tts():
    """Test that load_model_cls returns 'native' for ChatterboxTTS."""
    result = ModelRegistry.load_model_cls("ChatterboxTTS")
    assert result == "native", "ChatterboxTTS should return 'native' from load_model_cls"


@pytest.mark.unit
def test_load_model_cls_returns_native_for_chatterbox_vc():
    """Test that load_model_cls returns 'native' for ChatterboxVC."""
    result = ModelRegistry.load_model_cls("ChatterboxVC")
    assert result == "native", "ChatterboxVC should return 'native' from load_model_cls"
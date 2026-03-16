"""Unit tests verifying Chatterbox TTS package imports.

These tests ensure all required Chatterbox dependencies are installed
and their primary public API is accessible.
"""

import pytest


@pytest.mark.unit
def test_perth_importable():
    """perth package must be importable for audio watermarking."""
    import perth  # noqa: F401


@pytest.mark.unit
def test_pyloudnorm_importable():
    """pyloudnorm package must be importable for Turbo loudness normalization."""
    import pyloudnorm  # noqa: F401


@pytest.mark.unit
def test_chatterbox_tts_importable():
    """ChatterboxTTS class must be importable from chatterbox package."""
    from chatterbox import ChatterboxTTS  # noqa: F401


@pytest.mark.unit
def test_chatterbox_vc_importable():
    """ChatterboxVC class must be importable from chatterbox package."""
    from chatterbox import ChatterboxVC  # noqa: F401


@pytest.mark.unit
def test_chatterbox_multilingual_importable():
    """ChatterboxMultilingualTTS and SUPPORTED_LANGUAGES must be importable."""
    from chatterbox import SUPPORTED_LANGUAGES, ChatterboxMultilingualTTS  # noqa: F401

    assert isinstance(SUPPORTED_LANGUAGES, (list, tuple, set, dict))
    assert len(SUPPORTED_LANGUAGES) > 0


@pytest.mark.unit
def test_chatterbox_turbo_importable():
    """ChatterboxTurboTTS must be importable from chatterbox.tts_turbo submodule."""
    from chatterbox.tts_turbo import ChatterboxTurboTTS  # noqa: F401


@pytest.mark.unit
def test_chatterbox_supported_languages_count():
    """Multilingual model must advertise at least 23 supported languages."""
    from chatterbox import SUPPORTED_LANGUAGES

    assert len(SUPPORTED_LANGUAGES) >= 23

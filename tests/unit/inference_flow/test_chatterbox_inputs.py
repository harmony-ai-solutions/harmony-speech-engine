"""Unit tests for Chatterbox input preparation functions.

Tests are organized into three groups:
1. Data model extension tests (should PASS after Plan 01)
2. Prepare function behavior tests (will FAIL until Plan 02 implements them - RED state)
3. Language registration tests (should PASS after Plan 01 for SUPPORTED_LANGUAGES)
"""

import base64
import io
from dataclasses import fields
from unittest.mock import MagicMock, patch

import pytest

# --- Data model imports (always available after Task 1) ---
from harmonyspeech.common.inputs import TextToSpeechGenerationOptions
from harmonyspeech.endpoints.openai.protocol import GenerationOptions, LanguageOptions
from harmonyspeech.modeling.models.chatterbox.chatterbox import ChatterboxMultilingualTTSModel

# --- Prepare function imports (available after Plan 02) ---
try:
    from harmonyspeech.task_handler.inputs import (
        prepare_chatterbox_tts_inputs,
        prepare_chatterbox_turbo_tts_inputs,
        prepare_chatterbox_multilingual_tts_inputs,
        prepare_chatterbox_embedding_inputs,
        prepare_chatterbox_vc_inputs,
    )

    PREPARE_FUNCTIONS_AVAILABLE = True
except ImportError:
    PREPARE_FUNCTIONS_AVAILABLE = False

# ============================================================
# Group 1: Data model field tests (REQ-INPUT-01)
# ============================================================

CHATTERBOX_FIELD_NAMES = [
    "exaggeration",
    "cfg_weight",
    "temperature",
    "repetition_penalty",
    "top_p",
    "min_p",
    "top_k",
    "norm_loudness",
]


def test_generation_options_fields():
    """TextToSpeechGenerationOptions must have all 8 Chatterbox-specific fields."""
    field_names = {f.name for f in fields(TextToSpeechGenerationOptions)}
    for name in CHATTERBOX_FIELD_NAMES:
        assert name in field_names, f"Missing field: {name}"


def test_generation_options_chatterbox_fields_default_none():
    """All 8 Chatterbox fields on TextToSpeechGenerationOptions must default to None."""
    opts = TextToSpeechGenerationOptions(seed=None, style=None, speed=None, pitch=None, energy=None)
    for name in CHATTERBOX_FIELD_NAMES:
        assert getattr(opts, name) is None, f"Field {name} should default to None"


def test_protocol_generation_options_fields():
    """GenerationOptions Pydantic model must have all 8 Chatterbox-specific fields."""
    opts = GenerationOptions()
    for name in CHATTERBOX_FIELD_NAMES:
        assert hasattr(opts, name), f"Missing field: {name}"
        assert getattr(opts, name) is None, f"Field {name} should default to None"


# ============================================================
# Group 2: SUPPORTED_LANGUAGES constant tests (REQ-INPUT-06)
# ============================================================


def test_supported_languages_count():
    """ChatterboxMultilingualTTSModel.SUPPORTED_LANGUAGES must have exactly 23 entries."""
    assert len(ChatterboxMultilingualTTSModel.SUPPORTED_LANGUAGES) == 23


def test_supported_languages_contains_english():
    """SUPPORTED_LANGUAGES must include 'en' (English)."""
    assert "en" in ChatterboxMultilingualTTSModel.SUPPORTED_LANGUAGES


def test_supported_languages_is_dict_of_codes():
    """SUPPORTED_LANGUAGES keys must be short language codes (2-3 chars)."""
    for code in ChatterboxMultilingualTTSModel.SUPPORTED_LANGUAGES.keys():
        assert len(code) <= 3, f"Language code too long: {code}"


# ============================================================
# Helpers for prepare function tests
# ============================================================


def _make_tts_request(
    input_text="Hello world", input_embedding=None, input_audio=None, language_id=None, generation_options=None
):
    """Create a minimal mock TextToSpeechRequestInput."""
    req = MagicMock()
    req.input_text = input_text
    req.input_embedding = input_embedding
    req.input_audio = input_audio
    req.language_id = language_id
    req.generation_options = generation_options
    return req


def _make_vc_request(source_audio=None, target_audio=None, target_embedding=None):
    """Create a minimal mock VoiceConversionRequestInput."""
    req = MagicMock()
    req.source_audio = base64.b64encode(b"fake_audio_data").decode() if source_audio is None else source_audio
    req.target_audio = target_audio
    req.target_embedding = target_embedding
    return req


def _make_embedding_request(input_audio=None):
    """Create a minimal mock SpeechEmbeddingRequestInput."""
    req = MagicMock()
    req.input_audio = base64.b64encode(b"fake_audio_data").decode() if input_audio is None else input_audio
    return req


# ============================================================
# Group 3: TTS prepare function tests (REQ-INPUT-02, REQ-INPUT-05)
# ============================================================


@pytest.mark.skipif(not PREPARE_FUNCTIONS_AVAILABLE, reason="prepare functions not yet implemented")
def test_prepare_chatterbox_tts_inputs_basic():
    """prepare_chatterbox_tts_inputs returns correct tuple with text and defaults."""
    req = _make_tts_request()
    results = prepare_chatterbox_tts_inputs([req])
    assert len(results) == 1
    text, conditionals, exaggeration, cfg_weight, temperature, repetition_penalty, top_p, min_p = results[0]
    assert text == "Hello world"
    assert conditionals is None
    # Default values per CONTEXT.md
    assert exaggeration == 0.5
    assert cfg_weight == 0.5
    assert temperature == 0.8
    assert repetition_penalty == 1.2
    assert top_p == 1.0
    assert min_p == 0.05


@pytest.mark.skipif(not PREPARE_FUNCTIONS_AVAILABLE, reason="prepare functions not yet implemented")
def test_tts_rejects_top_k():
    """ChatterboxTTS must raise ValueError when top_k is non-None."""
    opts = GenerationOptions(top_k=500)
    req = _make_tts_request(generation_options=opts)
    with pytest.raises(ValueError, match="top_k"):
        prepare_chatterbox_tts_inputs([req])


@pytest.mark.skipif(not PREPARE_FUNCTIONS_AVAILABLE, reason="prepare functions not yet implemented")
def test_tts_rejects_norm_loudness():
    """ChatterboxTTS must raise ValueError when norm_loudness is non-None."""
    opts = GenerationOptions(norm_loudness=True)
    req = _make_tts_request(generation_options=opts)
    with pytest.raises(ValueError, match="norm_loudness"):
        prepare_chatterbox_tts_inputs([req])


@pytest.mark.skipif(not PREPARE_FUNCTIONS_AVAILABLE, reason="prepare functions not yet implemented")
def test_tts_conflict_both_audio_and_embedding():
    """ChatterboxTTS must raise ValueError when both input_audio and input_embedding provided."""
    req = _make_tts_request(
        input_audio=base64.b64encode(b"audio").decode(), input_embedding=base64.b64encode(b"embedding").decode()
    )
    with pytest.raises(ValueError):
        prepare_chatterbox_tts_inputs([req])


@pytest.mark.skipif(not PREPARE_FUNCTIONS_AVAILABLE, reason="prepare functions not yet implemented")
def test_validation_none_fields_no_error():
    """No ValueError when unsupported fields are None (backward compat — full options object with None fields)."""
    # Pass a full options object with all unsupported fields explicitly None
    opts = GenerationOptions(top_k=None, norm_loudness=None)
    req = _make_tts_request(generation_options=opts)
    # Must not raise
    results = prepare_chatterbox_tts_inputs([req])
    assert len(results) == 1


@pytest.mark.skipif(not PREPARE_FUNCTIONS_AVAILABLE, reason="prepare functions not yet implemented")
def test_prepare_turbo_inputs_basic():
    """prepare_chatterbox_turbo_tts_inputs returns correct tuple with turbo defaults."""
    req = _make_tts_request()
    results = prepare_chatterbox_turbo_tts_inputs([req])
    assert len(results) == 1
    text, conditionals, temperature, repetition_penalty, top_p, top_k, norm_loudness = results[0]
    assert text == "Hello world"
    assert conditionals is None
    # Turbo-specific defaults per CONTEXT.md
    assert temperature == 0.8
    assert repetition_penalty == 1.2
    assert top_p == 0.95
    assert top_k == 1000
    assert norm_loudness is True


@pytest.mark.skipif(not PREPARE_FUNCTIONS_AVAILABLE, reason="prepare functions not yet implemented")
def test_turbo_rejects_exaggeration():
    """ChatterboxTurboTTS must raise ValueError when exaggeration is non-None."""
    opts = GenerationOptions(exaggeration=0.8)
    req = _make_tts_request(generation_options=opts)
    with pytest.raises(ValueError, match="exaggeration"):
        prepare_chatterbox_turbo_tts_inputs([req])


@pytest.mark.skipif(not PREPARE_FUNCTIONS_AVAILABLE, reason="prepare functions not yet implemented")
def test_turbo_rejects_cfg_weight():
    """ChatterboxTurboTTS must raise ValueError when cfg_weight is non-None."""
    opts = GenerationOptions(cfg_weight=0.3)
    req = _make_tts_request(generation_options=opts)
    with pytest.raises(ValueError, match="cfg_weight"):
        prepare_chatterbox_turbo_tts_inputs([req])


@pytest.mark.skipif(not PREPARE_FUNCTIONS_AVAILABLE, reason="prepare functions not yet implemented")
def test_turbo_rejects_min_p():
    """ChatterboxTurboTTS must raise ValueError when min_p is non-None."""
    opts = GenerationOptions(min_p=0.05)
    req = _make_tts_request(generation_options=opts)
    with pytest.raises(ValueError, match="min_p"):
        prepare_chatterbox_turbo_tts_inputs([req])


@pytest.mark.skipif(not PREPARE_FUNCTIONS_AVAILABLE, reason="prepare functions not yet implemented")
def test_multilingual_defaults_to_en():
    """prepare_chatterbox_multilingual_tts_inputs defaults language_id to 'en' when absent."""
    req = _make_tts_request(language_id=None)
    results = prepare_chatterbox_multilingual_tts_inputs([req])
    assert len(results) == 1
    # language_id should be "en" in the tuple
    # Tuple: (text, language_id, conditionals, exaggeration, cfg_weight, temperature, repetition_penalty, top_p, min_p)
    text, language_id, conditionals, *_ = results[0]
    assert language_id == "en"


@pytest.mark.skipif(not PREPARE_FUNCTIONS_AVAILABLE, reason="prepare functions not yet implemented")
def test_multilingual_rejects_top_k():
    """ChatterboxMultilingualTTS must raise ValueError when top_k is non-None."""
    opts = GenerationOptions(top_k=500)
    req = _make_tts_request(generation_options=opts)
    with pytest.raises(ValueError, match="top_k"):
        prepare_chatterbox_multilingual_tts_inputs([req])


# ============================================================
# Group 4: Embedding prepare function tests (REQ-INPUT-03)
# ============================================================


@pytest.mark.skipif(not PREPARE_FUNCTIONS_AVAILABLE, reason="prepare functions not yet implemented")
def test_embedding_inputs_returns_bytes():
    """prepare_chatterbox_embedding_inputs returns raw audio bytes (base64-decoded)."""
    raw_audio = b"fake_wav_data"
    encoded = base64.b64encode(raw_audio).decode()
    req = _make_embedding_request(input_audio=encoded)
    results = prepare_chatterbox_embedding_inputs([req])
    assert len(results) == 1
    assert results[0] == raw_audio


# ============================================================
# Group 5: VC prepare function tests (REQ-INPUT-04)
# ============================================================


@pytest.mark.skipif(not PREPARE_FUNCTIONS_AVAILABLE, reason="prepare functions not yet implemented")
def test_vc_conflict_both():
    """prepare_chatterbox_vc_inputs raises ValueError when both target_audio and target_embedding provided."""
    req = _make_vc_request(
        target_audio=base64.b64encode(b"audio").decode(), target_embedding=base64.b64encode(b"embedding").decode()
    )
    with pytest.raises(ValueError, match="not both"):
        prepare_chatterbox_vc_inputs([req])


@pytest.mark.skipif(not PREPARE_FUNCTIONS_AVAILABLE, reason="prepare functions not yet implemented")
def test_vc_conflict_neither():
    """prepare_chatterbox_vc_inputs raises ValueError when neither target_audio nor target_embedding provided."""
    req = _make_vc_request(target_audio=None, target_embedding=None)
    with pytest.raises(ValueError, match="requires either"):
        prepare_chatterbox_vc_inputs([req])


@pytest.mark.skipif(not PREPARE_FUNCTIONS_AVAILABLE, reason="prepare functions not yet implemented")
def test_vc_with_target_audio_returns_correct_tuple():
    """prepare_chatterbox_vc_inputs returns (source_bytes, None, target_bytes) when target_audio provided."""
    source = b"source_audio"
    target = b"target_audio"
    req = _make_vc_request(
        source_audio=base64.b64encode(source).decode(),
        target_audio=base64.b64encode(target).decode(),
        target_embedding=None,
    )
    results = prepare_chatterbox_vc_inputs([req])
    assert len(results) == 1
    source_bytes, target_conditionals, target_audio_bytes = results[0]
    assert source_bytes == source
    assert target_conditionals is None
    assert target_audio_bytes == target

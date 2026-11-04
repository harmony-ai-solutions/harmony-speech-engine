"""Unit tests for output format / sample-rate honoring (Phase 1-2)."""

import base64

import numpy as np
import pytest

from harmonyspeech.common import audio_codec
from harmonyspeech.endpoints.openai.output_encoding import (
    apply_output_options,
    normalize_format,
    validate_output_options,
)
from harmonyspeech.endpoints.openai.protocol import AudioOutputOptions

SOURCE_SR = 44100


@pytest.fixture
def wav_b64(test_data_dir):
    return base64.b64encode((test_data_dir / "samples" / "codec_tone.wav").read_bytes()).decode("ascii")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validate_no_options_is_valid():
    assert validate_output_options(None) is None


def test_validate_default_wav_is_valid():
    assert validate_output_options(AudioOutputOptions()) is None


@pytest.mark.parametrize("fmt", ["wav", "flac", "ogg", "mp3", "pcm"])
def test_validate_supported_formats(fmt):
    assert validate_output_options(AudioOutputOptions(format=fmt)) is None


def test_validate_unknown_format_rejected():
    message = validate_output_options(AudioOutputOptions(format="wma"))
    assert message is not None
    assert "Unsupported output format" in message
    assert "wav" in message


def test_validate_format_is_case_insensitive():
    assert validate_output_options(AudioOutputOptions(format="FLAC")) is None
    assert normalize_format("MP3") == "mp3"


def test_validate_aac_is_capability_aware(monkeypatch):
    monkeypatch.setattr(audio_codec, "SUPPORTED_OUTPUT_FORMATS", {"wav", "flac", "ogg", "mp3", "pcm"})
    message = validate_output_options(AudioOutputOptions(format="aac"))
    assert message is not None
    assert "aac" in message


@pytest.mark.parametrize("rate", [8000, 16000, 22050, 24000, 32000, 44100, 48000])
def test_validate_supported_sample_rates(rate):
    assert validate_output_options(AudioOutputOptions(sample_rate=rate)) is None


@pytest.mark.parametrize("rate", [0, -1, 12345])
def test_validate_bad_sample_rate_rejected(rate):
    message = validate_output_options(AudioOutputOptions(sample_rate=rate))
    assert message is not None
    assert "sample_rate" in message


# ---------------------------------------------------------------------------
# Passthrough
# ---------------------------------------------------------------------------


def test_apply_no_options_is_byte_identical(wav_b64):
    encoded = apply_output_options(wav_b64, None)
    assert encoded.data is wav_b64
    assert encoded.format == "wav"
    assert encoded.sample_rate is None


def test_apply_default_wav_is_byte_identical(wav_b64):
    encoded = apply_output_options(wav_b64, AudioOutputOptions())
    assert encoded.data == wav_b64
    assert encoded.sample_rate is None


def test_apply_explicit_wav_without_rate_is_byte_identical(wav_b64):
    encoded = apply_output_options(wav_b64, AudioOutputOptions(format="wav"))
    assert encoded.data == wav_b64


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", ["flac", "ogg", "mp3"])
def test_apply_converts_and_resamples(wav_b64, fmt):
    if fmt not in audio_codec.SUPPORTED_OUTPUT_FORMATS:
        pytest.skip(f"{fmt} not encodable by this libsndfile build")

    encoded = apply_output_options(wav_b64, AudioOutputOptions(format=fmt, sample_rate=16000))
    assert encoded.format == fmt
    assert encoded.sample_rate == 16000

    decoded, decoded_sr = audio_codec.decode_audio(base64.b64decode(encoded.data))
    assert decoded_sr == 16000
    assert abs(decoded.size / 16000 - 1.0) < 0.05
    assert float(np.max(np.abs(decoded))) > 0.1


def test_apply_pcm_returns_s16le_at_model_rate(wav_b64):
    encoded = apply_output_options(wav_b64, AudioOutputOptions(format="pcm"))
    assert encoded.format == "pcm"
    assert encoded.sample_rate == SOURCE_SR

    raw = base64.b64decode(encoded.data)
    assert len(raw) == SOURCE_SR * 2  # 1 second, mono s16le
    samples = np.frombuffer(raw, dtype="<i2")
    assert float(np.max(np.abs(samples.astype(np.float32)))) > 1000


def test_apply_pcm_with_explicit_rate(wav_b64):
    encoded = apply_output_options(wav_b64, AudioOutputOptions(format="pcm", sample_rate=8000))
    assert encoded.sample_rate == 8000
    raw = base64.b64decode(encoded.data)
    assert abs(len(raw) / 2 / 8000 - 1.0) < 0.05


def test_apply_wav_resample_only(wav_b64):
    encoded = apply_output_options(wav_b64, AudioOutputOptions(sample_rate=22050))
    assert encoded.format == "wav"
    assert encoded.sample_rate == 22050
    decoded, decoded_sr = audio_codec.decode_audio(base64.b64decode(encoded.data))
    assert decoded_sr == 22050
    assert decoded.size > 0


def test_apply_raises_on_invalid_options(wav_b64):
    with pytest.raises(ValueError, match="Unsupported output format"):
        apply_output_options(wav_b64, AudioOutputOptions(format="wma"))


@pytest.mark.skipif(not audio_codec.FFMPEG_AVAILABLE, reason="ffmpeg not available on PATH")
def test_apply_aac_when_ffmpeg_available(wav_b64):
    encoded = apply_output_options(wav_b64, AudioOutputOptions(format="aac", sample_rate=24000))
    assert encoded.sample_rate == 24000
    decoded, decoded_sr = audio_codec.decode_audio(base64.b64decode(encoded.data))
    assert decoded_sr == 24000
    assert decoded.size > 0

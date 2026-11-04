"""Unit tests for the centralized boundary audio codec (Phase 1-1)."""

import base64
import io

import numpy as np
import pytest
import soundfile as sf

from harmonyspeech.common import audio_codec
from harmonyspeech.common.audio_codec import (
    AudioDecodeError,
    decode_audio,
    decode_wav_b64,
    encode_audio,
    ensure_librosa_readable_bytes,
    resample,
    sniff_format,
    to_wav_bytes,
)

SAMPLE_FILES = {
    "wav": "codec_tone.wav",
    "flac": "codec_tone.flac",
    "ogg": "codec_tone.ogg",
    "mp3": "codec_tone.mp3",
    "aac": "codec_tone.aac",
}

FFMPEG_ONLY = pytest.mark.skipif(
    not audio_codec.FFMPEG_AVAILABLE, reason="ffmpeg not available on PATH"
)


def _sample_bytes(test_data_dir, fmt: str) -> bytes:
    return (test_data_dir / "samples" / SAMPLE_FILES[fmt]).read_bytes()


# ---------------------------------------------------------------------------
# Sniffing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", ["wav", "flac", "ogg", "mp3", "aac"])
def test_sniff_format(test_data_dir, fmt):
    assert sniff_format(_sample_bytes(test_data_dir, fmt)) == fmt


def test_sniff_format_returns_none_for_garbage():
    assert sniff_format(b"not audio at all, definitely") is None


def test_sniff_format_returns_none_for_empty():
    assert sniff_format(b"") is None


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", ["wav", "flac", "ogg", "mp3"])
def test_decode_audio_native_formats(test_data_dir, fmt):
    waveform, sample_rate = decode_audio(_sample_bytes(test_data_dir, fmt))
    assert sample_rate == 44100
    assert waveform.dtype == np.float32
    assert waveform.ndim == 1
    assert 0.95 <= waveform.size / 44100 <= 1.05
    assert float(np.sqrt(np.mean(waveform**2))) > 0.05


@FFMPEG_ONLY
def test_decode_audio_aac_via_ffmpeg(test_data_dir):
    waveform, sample_rate = decode_audio(_sample_bytes(test_data_dir, "aac"))
    assert sample_rate == 44100
    assert waveform.ndim == 1
    assert 0.9 <= waveform.size / 44100 <= 1.1
    assert float(np.sqrt(np.mean(waveform**2))) > 0.03


@FFMPEG_ONLY
def test_decode_audio_aac_fails_cleanly_without_ffmpeg(test_data_dir, monkeypatch):
    """AAC must raise AudioDecodeError (not a bare RuntimeError) when ffmpeg is absent."""
    monkeypatch.setattr(audio_codec, "FFMPEG_AVAILABLE", False)
    with pytest.raises(AudioDecodeError) as excinfo:
        decode_audio(_sample_bytes(test_data_dir, "aac"))
    assert "aac" in str(excinfo.value)


def test_decode_audio_resamples_to_target(test_data_dir):
    waveform, sample_rate = decode_audio(_sample_bytes(test_data_dir, "wav"), target_sr=16000)
    assert sample_rate == 16000
    assert abs(waveform.size / 16000 - 1.0) < 0.02


def test_decode_audio_garbage_raises_typed_error_with_format_and_supported_list():
    with pytest.raises(AudioDecodeError) as excinfo:
        decode_audio(b"\x00\x01\x02 totally not audio")
    message = str(excinfo.value)
    assert "Unable to decode input audio" in message
    assert "probed format" in message
    assert "Supported input formats" in message
    assert "wav" in message


def test_decode_audio_empty_raises():
    with pytest.raises(AudioDecodeError, match="empty"):
        decode_audio(b"")


def test_decode_audio_rejects_invalid_target_sr(test_data_dir):
    with pytest.raises(ValueError, match="target_sr"):
        decode_audio(_sample_bytes(test_data_dir, "wav"), target_sr=0)


def test_decode_wav_b64_roundtrip(test_data_dir):
    payload = base64.b64encode(_sample_bytes(test_data_dir, "wav")).decode()
    waveform, sample_rate = decode_wav_b64(payload)
    assert sample_rate == 44100
    assert waveform.size == 44100


def test_decode_wav_b64_invalid_base64():
    with pytest.raises(ValueError):
        decode_wav_b64("!!!not base64!!!")


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------


def test_resample_matches_librosa_within_tolerance():
    librosa = pytest.importorskip("librosa")
    sample_rate = 44100
    t = np.arange(sample_rate) / sample_rate
    waveform = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    ours = resample(waveform, sample_rate, 16000)
    reference = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16000, res_type="soxr_hq")

    assert ours.dtype == np.float32
    assert abs(ours.size - reference.size) <= 2
    assert float(np.max(np.abs(ours - reference[: ours.size]))) < 0.05


def test_resample_noop_when_rates_match():
    waveform = np.zeros(100, dtype=np.float32)
    assert resample(waveform, 22050, 22050) is not None
    assert resample(waveform, 22050, 22050).size == 100


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", ["wav", "flac", "ogg", "mp3", "opus"])
def test_encode_audio_roundtrips(fmt):
    pytest.importorskip("soundfile")
    if fmt not in audio_codec.SUPPORTED_OUTPUT_FORMATS:
        pytest.skip(f"{fmt} not encodable by this libsndfile build")

    sample_rate = 24000
    t = np.arange(sample_rate) / sample_rate
    waveform = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    encoded = encode_audio(waveform, sample_rate, fmt)
    decoded, decoded_sr = decode_audio(encoded)
    assert decoded_sr == sample_rate
    assert abs(decoded.size - waveform.size) <= 2000  # codecs add tiny padding
    assert float(np.max(np.abs(decoded))) > 0.05


def test_encode_audio_resamples():
    sample_rate = 44100
    waveform = np.zeros(sample_rate, dtype=np.float32)
    encoded = encode_audio(waveform, sample_rate, "wav", target_sr=16000)
    _, decoded_sr = decode_audio(encoded)
    assert decoded_sr == 16000


def test_encode_audio_pcm_is_s16le():
    waveform = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
    pcm = encode_audio(waveform, 16000, "pcm")
    assert len(pcm) == waveform.size * 2
    samples = np.frombuffer(pcm, dtype="<i2")
    assert samples[0] == 0
    assert 16000 < samples[1] < 16500
    assert samples[3] == 32767
    assert samples[4] == -32767


def test_encode_audio_rejects_unknown_format():
    with pytest.raises(ValueError, match="Supported formats"):
        encode_audio(np.zeros(10, dtype=np.float32), 16000, "wma")


@FFMPEG_ONLY
def test_encode_audio_aac():
    sample_rate = 24000
    t = np.arange(sample_rate // 2) / sample_rate
    waveform = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    encoded = encode_audio(waveform, sample_rate, "aac")
    assert encoded[:2] in (b"\xff\xf1", b"\xff\xf9")
    decoded, decoded_sr = decode_audio(encoded)
    assert decoded_sr == sample_rate


def test_supported_output_formats_always_include_pcm():
    assert "pcm" in audio_codec.SUPPORTED_OUTPUT_FORMATS


# ---------------------------------------------------------------------------
# WAV normalisation helpers
# ---------------------------------------------------------------------------


def test_to_wav_bytes_produces_soundfile_readable_wav(test_data_dir):
    wav_bytes = to_wav_bytes(_sample_bytes(test_data_dir, "mp3"))
    assert wav_bytes[:4] == b"RIFF"
    waveform, sample_rate = decode_audio(wav_bytes)
    assert sample_rate == 44100
    assert waveform.size > 0


def test_ensure_librosa_readable_passes_wav_through(test_data_dir):
    data = _sample_bytes(test_data_dir, "wav")
    assert ensure_librosa_readable_bytes(data) is data


@FFMPEG_ONLY
def test_ensure_librosa_readable_transcodes_aac(test_data_dir):
    data = _sample_bytes(test_data_dir, "aac")
    normalized = ensure_librosa_readable_bytes(data)
    assert normalized[:4] == b"RIFF"
    with sf.SoundFile(io.BytesIO(normalized)) as handle:
        assert handle.samplerate == 44100
        assert len(handle) > 0

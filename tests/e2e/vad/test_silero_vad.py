"""E2E tests for SileroVAD voice activity detection model.

Exercises full pipeline: DetectVoiceActivityRequest → OpenAIServingVoiceActivityDetection →
AsyncHarmonySpeech → Scheduler → CPUExecutor → CPUWorker → CPUModelRunner →
SileroVAD model → DetectVoiceActivityResponse.
"""
import asyncio
import struct
import base64
import math
import pytest

from tests.e2e.conftest import load_sample_audio_b64
from harmonyspeech.endpoints.openai.protocol import (
    DetectVoiceActivityRequest, DetectVoiceActivityResponse
)


def _make_silent_wav_b64(sample_rate: int = 16000, duration_secs: float = 1.0) -> str:
    """Generate a silent WAV file as base64 string."""
    num_samples = int(sample_rate * duration_secs)
    pcm_data = struct.pack('<' + 'h' * num_samples, *([0] * num_samples))
    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF', 36 + len(pcm_data), b'WAVE',
        b'fmt ', 16, 1, 1, sample_rate, sample_rate * 2, 2, 16,
        b'data', len(pcm_data)
    )
    return base64.b64encode(header + pcm_data).decode()


def _make_speech_like_wav_b64(sample_rate: int = 16000, duration_secs: float = 1.0) -> str:
    """Generate a WAV with loud tones simulating speech-like signal."""
    num_samples = int(sample_rate * duration_secs)
    # Mix two frequencies to make it more speech-like
    pcm_data = struct.pack(
        '<' + 'h' * num_samples,
        *[
            int(16000 * (math.sin(2 * math.pi * 200 * i / sample_rate) +
                         math.sin(2 * math.pi * 600 * i / sample_rate)))
            for i in range(num_samples)
        ]
    )
    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF', 36 + len(pcm_data), b'WAVE',
        b'fmt ', 16, 1, 1, sample_rate, sample_rate * 2, 2, 16,
        b'data', len(pcm_data)
    )
    return base64.b64encode(header + pcm_data).decode()


@pytest.mark.e2e
@pytest.mark.slow
def test_silero_vad_silent_audio(vad_engine, mock_raw_request):
    """SileroVAD: silent audio produces a valid DetectVoiceActivityResponse (pipeline completes)."""
    engine, serving_vad = vad_engine
    audio_b64 = _make_silent_wav_b64()
    request = DetectVoiceActivityRequest(
        model="silero-vad",
        input_audio=audio_b64,
    )
    response = asyncio.run(serving_vad.check_voice_activity(request, mock_raw_request))
    assert isinstance(response, DetectVoiceActivityResponse), f"Expected DetectVoiceActivityResponse, got: {response}"
    assert hasattr(response, "speech_activity")
    assert response.speech_activity is False


@pytest.mark.e2e
@pytest.mark.slow
def test_silero_vad_with_timestamps(vad_engine, mock_raw_request):
    """SileroVAD: request with get_timestamps=True returns response without error."""
    engine, serving_vad = vad_engine
    audio_b64 = _make_silent_wav_b64()
    request = DetectVoiceActivityRequest(
        model="silero-vad",
        input_audio=audio_b64,
        get_timestamps=True,
        return_seconds=True,
    )
    response = asyncio.run(serving_vad.check_voice_activity(request, mock_raw_request))
    assert isinstance(response, DetectVoiceActivityResponse), f"Expected DetectVoiceActivityResponse, got: {response}"
    assert hasattr(response, "timestamps")
    assert isinstance(response.timestamps, list)


@pytest.mark.e2e
@pytest.mark.slow
def test_silero_vad_with_speech(vad_engine, mock_raw_request):
    """SileroVAD: audio with speech produces speech_activity=True."""
    engine, serving_vad = vad_engine
    audio_b64 = load_sample_audio_b64("wanda4")
    request = DetectVoiceActivityRequest(
        model="silero-vad",
        input_audio=audio_b64,
    )
    response = asyncio.run(serving_vad.check_voice_activity(request, mock_raw_request))
    assert isinstance(response, DetectVoiceActivityResponse), f"Expected DetectVoiceActivityResponse, got: {response}"
    assert hasattr(response, "speech_activity")
    assert response.speech_activity is True
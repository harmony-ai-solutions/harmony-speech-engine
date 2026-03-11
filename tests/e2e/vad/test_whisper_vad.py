"""E2E tests for Whisper-based voice activity detection.

Exercises full pipeline: DetectVoiceActivityRequest → OpenAIServingSpeechToText →
AsyncHarmonySpeech → Scheduler → CPUExecutor → CPUWorker → CPUModelRunner →
FasterWhisper (VAD mode) → DetectVoiceActivityResponse.
"""
import asyncio
import struct
import base64
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


@pytest.mark.e2e
@pytest.mark.slow
def test_whisper_vad_silent_audio(whisper_vad_engine, mock_raw_request):
    """Whisper VAD: silent audio produces speech_activity=False."""
    engine, serving_vad = whisper_vad_engine
    audio_b64 = _make_silent_wav_b64()
    request = DetectVoiceActivityRequest(
        model="faster-whisper",
        input_audio=audio_b64,
    )
    response = asyncio.run(serving_vad.check_voice_activity(request, mock_raw_request))
    assert isinstance(response, DetectVoiceActivityResponse), f"Expected DetectVoiceActivityResponse, got: {response}"
    assert hasattr(response, "speech_activity")
    assert response.speech_activity is False


@pytest.mark.e2e
@pytest.mark.slow
def test_whisper_vad_with_speech(whisper_vad_engine, mock_raw_request):
    """Whisper VAD: audio with speech produces speech_activity=True."""
    engine, serving_vad = whisper_vad_engine
    audio_b64 = load_sample_audio_b64("wanda4")
    request = DetectVoiceActivityRequest(
        model="faster-whisper",
        input_audio=audio_b64,
    )
    response = asyncio.run(serving_vad.check_voice_activity(request, mock_raw_request))
    assert isinstance(response, DetectVoiceActivityResponse), f"Expected DetectVoiceActivityResponse, got: {response}"
    assert hasattr(response, "speech_activity")
    assert response.speech_activity is True
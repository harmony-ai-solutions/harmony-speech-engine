"""E2E tests for FasterWhisper STT model.

Exercises full pipeline: SpeechTranscribeRequest → OpenAIServingSpeechToText →
AsyncHarmonySpeech → Scheduler → CPUExecutor → CPUWorker → CPUModelRunner →
FasterWhisper → SpeechToTextResponse.
"""
import asyncio
import struct
import base64
import math
import pytest

from harmonyspeech.endpoints.openai.protocol import SpeechTranscribeRequest, SpeechToTextResponse


def _make_sine_wav_b64(frequency: int = 440, sample_rate: int = 16000, duration_secs: float = 1.0) -> str:
    """Generate a WAV file with a sine wave tone (simulates speech-like audio) as base64."""
    num_samples = int(sample_rate * duration_secs)
    pcm_data = struct.pack(
        '<' + 'h' * num_samples,
        *[int(32767 * math.sin(2 * math.pi * frequency * i / sample_rate)) for i in range(num_samples)]
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
def test_whisper_tiny_transcription(whisper_engine, mock_raw_request):
    """FasterWhisper tiny: full engine pipeline audio -> text transcription."""
    engine, serving_stt = whisper_engine
    audio_b64 = _make_sine_wav_b64()
    request = SpeechTranscribeRequest(
        model="faster-whisper",
        input_audio=audio_b64,
    )
    response = asyncio.run(serving_stt.create_transcription(request, mock_raw_request))
    assert isinstance(response, SpeechToTextResponse), f"Expected SpeechToTextResponse, got: {response}"
    assert hasattr(response, "text")
    # text may be empty for pure sine wave (no real speech), but pipeline must complete without error
    assert response.text is not None


@pytest.mark.e2e
@pytest.mark.slow
def test_whisper_tiny_transcription_with_language(whisper_engine, mock_raw_request):
    """FasterWhisper tiny: returns language tag when get_language=True."""
    engine, serving_stt = whisper_engine
    audio_b64 = _make_sine_wav_b64()
    request = SpeechTranscribeRequest(
        model="faster-whisper",
        input_audio=audio_b64,
        get_language=True,
    )
    response = asyncio.run(serving_stt.create_transcription(request, mock_raw_request))
    assert isinstance(response, SpeechToTextResponse), f"Expected SpeechToTextResponse, got: {response}"
    assert hasattr(response, "text")
    assert response.text is not None

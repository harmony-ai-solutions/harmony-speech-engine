"""E2E tests for VoiceFixer audio restoration pipeline.

Exercises full pipeline:
AudioConversionRequest → OpenAIServingAudioConversion → AsyncHarmonySpeech →
Scheduler → CPUExecutor → CPUWorker → CPUModelRunner →
VoiceFixerRestorer → VoiceFixerVocoder → VoiceConversionResponse.
"""
import asyncio
import pytest

from harmonyspeech.endpoints.openai.protocol import AudioConversionRequest, VoiceConversionResponse
from tests.e2e.conftest import load_sample_audio_b64


@pytest.mark.e2e
@pytest.mark.slow
def test_voicefixer_restores_audio(voicefixer_engine, mock_raw_request):
    """VoiceFixer: full restorer→vocoder pipeline produces audio output."""
    engine, serving_audio = voicefixer_engine
    audio_b64 = load_sample_audio_b64("wanda4")
    request = AudioConversionRequest(
        model="voicefixer",
        source_audio=audio_b64,
    )
    response = asyncio.run(serving_audio.convert_audio(request, mock_raw_request))
    assert isinstance(response, VoiceConversionResponse), f"Expected VoiceConversionResponse, got: {response}"
    assert response.data is not None
    assert len(response.data) > 0
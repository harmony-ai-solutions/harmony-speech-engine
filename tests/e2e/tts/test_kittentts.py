"""E2E tests for KittenTTS model variants.

These tests exercise the complete application stack:
TextToSpeechRequest → OpenAIServingTextToSpeech → AsyncHarmonySpeech → 
Scheduler → CPUExecutor → CPUWorker → CPUModelRunner → KittenTTSSynthesizer → TextToSpeechResponse.
"""
import asyncio
import pytest

from harmonyspeech.endpoints.openai.protocol import TextToSpeechRequest, TextToSpeechResponse


# Test input text
TEXT_INPUT = "Hello, world."


@pytest.mark.e2e
@pytest.mark.slow
def test_kittentts_mini_single_speaker(kittentts_mini_engine, mock_raw_request):
    """KittenTTS mini: full engine pipeline text -> audio."""
    engine, serving_tts = kittentts_mini_engine
    request = TextToSpeechRequest(
        model="kitten-tts-mini",
        input=TEXT_INPUT,
        mode="single_speaker_tts",
        voice="Jasper",
    )
    response = asyncio.run(serving_tts.create_text_to_speech(request, mock_raw_request))
    assert isinstance(response, TextToSpeechResponse), f"Expected TextToSpeechResponse, got: {response}"
    assert response.data is not None
    assert len(response.data) > 0


@pytest.mark.e2e
@pytest.mark.slow
def test_kittentts_micro_single_speaker(kittentts_micro_engine, mock_raw_request):
    """KittenTTS micro: full engine pipeline text -> audio."""
    engine, serving_tts = kittentts_micro_engine
    request = TextToSpeechRequest(
        model="kitten-tts-micro",
        input=TEXT_INPUT,
        mode="single_speaker_tts",
        voice="Jasper",
    )
    response = asyncio.run(serving_tts.create_text_to_speech(request, mock_raw_request))
    assert isinstance(response, TextToSpeechResponse), f"Expected TextToSpeechResponse, got: {response}"
    assert response.data is not None
    assert len(response.data) > 0


@pytest.mark.e2e
@pytest.mark.slow
def test_kittentts_nano_single_speaker(kittentts_nano_engine, mock_raw_request):
    """KittenTTS nano: full engine pipeline text -> audio."""
    engine, serving_tts = kittentts_nano_engine
    request = TextToSpeechRequest(
        model="kitten-tts-nano",
        input=TEXT_INPUT,
        mode="single_speaker_tts",
        voice="Jasper",
    )
    response = asyncio.run(serving_tts.create_text_to_speech(request, mock_raw_request))
    assert isinstance(response, TextToSpeechResponse), f"Expected TextToSpeechResponse, got: {response}"
    assert response.data is not None
    assert len(response.data) > 0


@pytest.mark.e2e
@pytest.mark.slow
def test_kittentts_nano_int8_single_speaker(kittentts_nano_int8_engine, mock_raw_request):
    """KittenTTS nano-int8: full engine pipeline text -> audio."""
    engine, serving_tts = kittentts_nano_int8_engine
    request = TextToSpeechRequest(
        model="kitten-tts-nano-int8",
        input=TEXT_INPUT,
        mode="single_speaker_tts",
        voice="Jasper",
    )
    response = asyncio.run(serving_tts.create_text_to_speech(request, mock_raw_request))
    assert isinstance(response, TextToSpeechResponse), f"Expected TextToSpeechResponse, got: {response}"
    assert response.data is not None
    assert len(response.data) > 0

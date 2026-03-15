"""E2E tests for ChatterboxTurboTTS with real model inference.

Coverage added based on bugs discovered during manual testing:
  - ChatterboxTurboTTS: float64 dtype crash in s3tokenizer + voice_encoder
  - Engine crash on inference errors (too-short audio, etc.) leaking through the async loop

All tests require CUDA and load the real models from HuggingFace.
Tests are automatically skipped when CUDA is unavailable.

Test audio samples (tests/test-data/samples/):
  wanda4.wav               — short voice sample (< 5 seconds); used for negative-path embedding tests
  wanda5.wav               — short voice sample (< 5 seconds); spare short reference
  wanda6.wav               — short voice sample (< 5 seconds); spare short reference
  jerry_seinfeld_prompt.wav — long voice prompt (~32 seconds); used where >= 5 seconds is required
                              (ChatterboxTurbo embedding, voice cloning reference)
"""

import asyncio
import base64

import pytest

from harmonyspeech.endpoints.openai.protocol import (
    TextToSpeechRequest,
    TextToSpeechResponse,
    EmbedSpeakerRequest,
    EmbedSpeakerResponse,
    ErrorResponse,
)
from tests.e2e.conftest import load_sample_audio_b64

pytestmark = [pytest.mark.e2e, pytest.mark.slow, pytest.mark.cuda]

TEXT_INPUT = "Hello, world. This is a test of the Chatterbox voice cloning system."

# wanda4 is a short clip (< 5 seconds) — suitable for positive tests where audio length is not checked
# and for negative-path tests that require audio shorter than 5 seconds.
SHORT_REFERENCE_AUDIO = load_sample_audio_b64("wanda4")

# jerry_seinfeld_prompt is ~32 seconds — required for ChatterboxTurbo embedding (>= 5s enforced upstream).
LONG_REFERENCE_AUDIO = load_sample_audio_b64("jerry_seinfeld_prompt")


# ---------------------------------------------------------------------------
# ChatterboxTurboTTS
# ---------------------------------------------------------------------------


def test_chatterbox_turbo_tts_no_cloning(chatterbox_turbo_engine, mock_raw_request):
    """TTS direct: text → base64 WAV; non-empty output."""
    engine, serving_tts, _ = chatterbox_turbo_engine
    request = TextToSpeechRequest(model="chatterbox_turbo", input=TEXT_INPUT, mode="single_speaker_tts")
    response = asyncio.run(serving_tts.create_text_to_speech(request, mock_raw_request))
    assert isinstance(response, TextToSpeechResponse), f"Got: {response}"
    assert response.data
    assert len(base64.b64decode(response.data)) > 0


def test_chatterbox_turbo_embedding(chatterbox_turbo_engine, mock_raw_request):
    """Turbo embedding: long enough audio (>= 5s) → base64 Conditionals; non-empty.

    Uses jerry_seinfeld_prompt.wav (~32s) because upstream enforces a minimum 5-second audio prompt.
    """
    engine, _, serving_embed = chatterbox_turbo_engine
    request = EmbedSpeakerRequest(model="chatterbox_turbo_embedding", input_audio=LONG_REFERENCE_AUDIO)
    response = asyncio.run(serving_embed.create_voice_embedding(request, mock_raw_request))
    assert isinstance(response, EmbedSpeakerResponse), f"Got: {response}"
    assert response.data
    assert len(base64.b64decode(response.data)) > 0


def test_chatterbox_turbo_tts_with_precomputed_embedding(chatterbox_turbo_engine, mock_raw_request):
    """Turbo TTS + precomputed embedding: embed (long audio) → TTS; non-empty WAV."""
    engine, serving_tts, serving_embed = chatterbox_turbo_engine

    embed_request = EmbedSpeakerRequest(model="chatterbox_turbo_embedding", input_audio=LONG_REFERENCE_AUDIO)
    embed_response = asyncio.run(serving_embed.create_voice_embedding(embed_request, mock_raw_request))
    assert isinstance(embed_response, EmbedSpeakerResponse), f"Embed step got: {embed_response}"

    tts_request = TextToSpeechRequest(
        model="chatterbox_turbo", input=TEXT_INPUT, input_embedding=embed_response.data, mode="voice_cloning"
    )
    tts_response = asyncio.run(serving_tts.create_text_to_speech(tts_request, mock_raw_request))
    assert isinstance(tts_response, TextToSpeechResponse), f"TTS step got: {tts_response}"
    assert tts_response.data
    assert len(base64.b64decode(tts_response.data)) > 0


def test_chatterbox_turbo_embedding_short_audio_returns_error(chatterbox_turbo_engine, mock_raw_request):
    """Turbo embedding with audio < 5s returns 400 ErrorResponse; engine survives to serve next request.

    Regression test for:
      AssertionError: "Audio prompt must be longer than 5 seconds!"
    propagating out of _execute_chatterbox_embedding and crashing the async engine loop.

    wanda4.wav is a short voice sample (< 5 seconds) — suitable as the bad input here.
    """
    engine, serving_tts, serving_embed = chatterbox_turbo_engine

    bad_request = EmbedSpeakerRequest(
        model="chatterbox_turbo_embedding",
        input_audio=SHORT_REFERENCE_AUDIO,  # wanda4.wav < 5 seconds
    )
    bad_response = asyncio.run(serving_embed.create_voice_embedding(bad_request, mock_raw_request))

    # Must return an error, not crash
    assert isinstance(bad_response, ErrorResponse), (
        f"Expected ErrorResponse for too-short audio, got: {type(bad_response).__name__}: {bad_response}"
    )

    # Engine must still be alive — send a valid TTS request and confirm it succeeds
    recovery_request = TextToSpeechRequest(model="chatterbox_turbo", input=TEXT_INPUT, mode="single_speaker_tts")
    recovery_response = asyncio.run(serving_tts.create_text_to_speech(recovery_request, mock_raw_request))
    assert isinstance(recovery_response, TextToSpeechResponse), (
        f"Engine did not recover after bad request: {recovery_response}"
    )
    assert recovery_response.data

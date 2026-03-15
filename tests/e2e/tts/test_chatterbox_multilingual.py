"""E2E tests for ChatterboxMultilingualTTS with real model inference.

Coverage added based on bugs discovered during manual testing:
  - ChatterboxMultilingualTTS: sdpa attn_implementation incompatibility with AlignmentStreamAnalyzer
  - Embedding requests to multilingual models crashing _check_model (missing language field)
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
)
from tests.e2e.conftest import load_sample_audio_b64

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.slow,
    pytest.mark.cuda,
]

TEXT_INPUT = "Hello, world. This is a test of the Chatterbox voice cloning system."

# wanda4 is a short clip (< 5 seconds) — suitable for positive tests where audio length is not checked
# and for negative-path tests that require audio shorter than 5 seconds.
SHORT_REFERENCE_AUDIO = load_sample_audio_b64("wanda4")


# ---------------------------------------------------------------------------
# ChatterboxMultilingualTTS
# ---------------------------------------------------------------------------

def test_chatterbox_multilingual_tts_no_cloning_english(chatterbox_multilingual_engine, mock_raw_request):
    """Multilingual TTS in English without voice cloning: non-empty WAV.

    Regression test for:
      ValueError: The `output_attentions` attribute is not supported when using
      the `attn_implementation` set to sdpa
    """
    engine, serving_tts, _ = chatterbox_multilingual_engine
    request = TextToSpeechRequest(
        model="chatterbox_multilingual",
        input=TEXT_INPUT,
        language="en",
        mode="single_speaker_tts",
    )
    response = asyncio.run(serving_tts.create_text_to_speech(request, mock_raw_request))
    assert isinstance(response, TextToSpeechResponse), f"Got: {response}"
    assert response.data
    assert len(base64.b64decode(response.data)) > 0


def test_chatterbox_multilingual_embedding_no_language_check(chatterbox_multilingual_engine, mock_raw_request):
    """Multilingual embedding request must not be rejected by _check_model language validation.

    Regression test for:
      AttributeError: 'EmbedSpeakerRequest' object has no attribute 'language'
    in serving_engine._check_model when the model has languages but the request type
    (EmbedSpeakerRequest) has no language field.

    Unlike ChatterboxTurbo, the multilingual model does not enforce a minimum audio length,
    so the short wanda4.wav sample is sufficient here.
    """
    engine, _, serving_embed = chatterbox_multilingual_engine
    request = EmbedSpeakerRequest(
        model="chatterbox_multilingual_embedding",
        input_audio=SHORT_REFERENCE_AUDIO,
    )
    response = asyncio.run(serving_embed.create_voice_embedding(request, mock_raw_request))
    assert isinstance(response, EmbedSpeakerResponse), f"Got: {response}"
    assert response.data
    assert len(base64.b64decode(response.data)) > 0


def test_chatterbox_multilingual_tts_with_precomputed_embedding(chatterbox_multilingual_engine, mock_raw_request):
    """Multilingual TTS + precomputed embedding: embed (short audio) → TTS in English; non-empty WAV.

    Multilingual embedding does not enforce a 5-second minimum, so wanda4.wav is sufficient.
    """
    engine, serving_tts, serving_embed = chatterbox_multilingual_engine

    embed_request = EmbedSpeakerRequest(
        model="chatterbox_multilingual_embedding",
        input_audio=SHORT_REFERENCE_AUDIO,
    )
    embed_response = asyncio.run(serving_embed.create_voice_embedding(embed_request, mock_raw_request))
    assert isinstance(embed_response, EmbedSpeakerResponse), f"Embed step got: {embed_response}"

    tts_request = TextToSpeechRequest(
        model="chatterbox_multilingual",
        input=TEXT_INPUT,
        language="en",
        input_embedding=embed_response.data,
        mode="voice_cloning",
    )
    tts_response = asyncio.run(serving_tts.create_text_to_speech(tts_request, mock_raw_request))
    assert isinstance(tts_response, TextToSpeechResponse), f"TTS step got: {tts_response}"
    assert tts_response.data
    assert len(base64.b64decode(tts_response.data)) > 0
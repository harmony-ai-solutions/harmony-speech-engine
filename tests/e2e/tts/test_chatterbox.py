"""E2E tests for Chatterbox TTS with real model inference.

Supports both CPU and CUDA based on the device fixture.
"""

import asyncio
import base64

import pytest

from harmonyspeech.endpoints.openai.protocol import (
    EmbedSpeakerRequest,
    EmbedSpeakerResponse,
    TextToSpeechRequest,
    TextToSpeechResponse,
    VoiceConversionRequest,
    VoiceConversionResponse,
)
from tests.e2e.conftest import load_sample_audio_b64

pytestmark = [pytest.mark.e2e, pytest.mark.slow]

TEXT_INPUT = "Hello, world. This is a test of the Chatterbox voice cloning system."
REFERENCE_AUDIO = load_sample_audio_b64("wanda4")


def test_chatterbox_tts_no_cloning(chatterbox_engine, mock_raw_request):
    """TTS direct: text → base64 WAV returned; non-empty."""
    engine, serving_tts, _, _ = chatterbox_engine
    request = TextToSpeechRequest(model="chatterbox", input=TEXT_INPUT, mode="single_speaker_tts")
    response = asyncio.run(serving_tts.create_text_to_speech(request, mock_raw_request))
    assert isinstance(response, TextToSpeechResponse), f"Got: {response}"
    assert response.data
    decoded = base64.b64decode(response.data)
    assert len(decoded) > 0


def test_chatterbox_tts_with_precomputed_embedding(chatterbox_engine, mock_raw_request):
    """TTS + precomputed embedding: embed audio then use embedding in TTS; non-empty WAV."""
    engine, serving_tts, serving_embed, _ = chatterbox_engine

    # Step 1: Compute embedding
    embed_request = EmbedSpeakerRequest(model="chatterbox", input_audio=REFERENCE_AUDIO)
    embed_response = asyncio.run(serving_embed.create_voice_embedding(embed_request, mock_raw_request))
    assert isinstance(embed_response, EmbedSpeakerResponse), f"Got: {embed_response}"
    assert embed_response.data

    # Step 2: Use embedding in TTS
    tts_request = TextToSpeechRequest(
        model="chatterbox", input=TEXT_INPUT, input_embedding=embed_response.data, mode="voice_cloning"
    )
    tts_response = asyncio.run(serving_tts.create_text_to_speech(tts_request, mock_raw_request))
    assert isinstance(tts_response, TextToSpeechResponse), f"Got: {tts_response}"
    assert tts_response.data
    assert len(base64.b64decode(tts_response.data)) > 0


def test_chatterbox_tts_voice_cloning(chatterbox_engine, mock_raw_request):
    """Voice cloning: input_audio triggers embed step + TTS step; non-empty WAV."""
    engine, serving_tts, _, _ = chatterbox_engine
    request = TextToSpeechRequest(
        model="chatterbox", input=TEXT_INPUT, input_audio=REFERENCE_AUDIO, mode="voice_cloning"
    )
    response = asyncio.run(serving_tts.create_text_to_speech(request, mock_raw_request))
    assert isinstance(response, TextToSpeechResponse), f"Got: {response}"
    assert response.data
    assert len(base64.b64decode(response.data)) > 0


def test_chatterbox_standalone_embedding(chatterbox_engine, mock_raw_request):
    """Audio → base64 Conditionals embedding; non-empty."""
    engine, _, serving_embed, _ = chatterbox_engine
    request = EmbedSpeakerRequest(model="chatterbox", input_audio=REFERENCE_AUDIO)
    response = asyncio.run(serving_embed.create_voice_embedding(request, mock_raw_request))
    assert isinstance(response, EmbedSpeakerResponse), f"Got: {response}"
    assert response.data
    assert len(base64.b64decode(response.data)) > 0


def test_chatterbox_vc_with_target_audio(chatterbox_engine, mock_raw_request):
    """VC with target audio: source + target audio → converted audio; non-empty WAV."""
    engine, _, _, serving_vc = chatterbox_engine
    request = VoiceConversionRequest(
        model="chatterbox_vc", source_audio=REFERENCE_AUDIO, target_audio=load_sample_audio_b64("wanda5")
    )
    response = asyncio.run(serving_vc.convert_voice(request, mock_raw_request))
    assert isinstance(response, VoiceConversionResponse), f"Got: {response}"
    assert response.data
    assert len(base64.b64decode(response.data)) > 0


def test_chatterbox_vc_with_target_embedding(chatterbox_engine, mock_raw_request):
    """VC with pre-computed embedding: source audio + embedding → converted audio; non-empty WAV."""
    engine, _, serving_embed, serving_vc = chatterbox_engine

    # Pre-compute target embedding
    embed_request = EmbedSpeakerRequest(model="chatterbox", input_audio=load_sample_audio_b64("wanda5"))
    embed_response = asyncio.run(serving_embed.create_voice_embedding(embed_request, mock_raw_request))
    assert isinstance(embed_response, EmbedSpeakerResponse), f"Got: {embed_response}"
    assert embed_response.data

    # VC with pre-computed embedding
    vc_request = VoiceConversionRequest(
        model="chatterbox_vc", source_audio=REFERENCE_AUDIO, target_embedding=embed_response.data
    )
    vc_response = asyncio.run(serving_vc.convert_voice(vc_request, mock_raw_request))
    assert isinstance(vc_response, VoiceConversionResponse), f"Got: {vc_response}"
    assert vc_response.data
    assert len(base64.b64decode(vc_response.data)) > 0

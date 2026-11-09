"""Real-model standard voice pack coverage (Phase 13-1), gated on weights.

The same pack name (``Abigail``) must work on more than one voice-cloning
family: each family embeds the shipped reference clip in its own embedding
space, so the identity is consistent even though the rendered audio differs.
Skipped unless ``HSE_E2E_MODEL_TESTS=1`` is set — the hermetic stub coverage
(including the 7-2 cache ride) lives in
``tests/integration/test_tts_chunking.py`` and runs in the default suite.
"""

import asyncio
import base64
import os

import pytest

from harmonyspeech.endpoints.openai.protocol import (
    TextToSpeechRequest,
    TextToSpeechResponse,
)

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("HSE_E2E_MODEL_TESTS") != "1",
        reason="requires downloaded model weights (set HSE_E2E_MODEL_TESTS=1 to enable)",
    ),
]

VOICE = "Abigail"
TEXT = "The standard voice pack speaks this sentence."


def _decode(response):
    from harmonyspeech.common.audio_codec import decode_audio

    assert isinstance(response, TextToSpeechResponse), response
    return decode_audio(base64.b64decode(response.data))


def _speak_pack_voice(serving_tts, raw_request, model, *, language=None):
    return asyncio.run(
        serving_tts.create_text_to_speech(
            TextToSpeechRequest(
                model=model,
                input=TEXT,
                mode="voice_cloning",
                voice=VOICE,
                language=language,
            ),
            raw_request,
        )
    )


def test_standard_voice_pack_on_chatterbox(chatterbox_engine, mock_raw_request):
    _engine, serving_tts, _embed, _vc = chatterbox_engine
    wave, sample_rate = _decode(
        _speak_pack_voice(serving_tts, mock_raw_request, "chatterbox")
    )
    assert wave.size > 0 and sample_rate > 0


def test_standard_voice_pack_on_chatterbox_turbo(
    chatterbox_turbo_engine, mock_raw_request
):
    _engine, serving_tts, _embed = chatterbox_turbo_engine
    wave, sample_rate = _decode(
        _speak_pack_voice(serving_tts, mock_raw_request, "chatterbox_turbo")
    )
    assert wave.size > 0 and sample_rate > 0


def test_standard_voice_pack_on_chatterbox_multilingual(
    chatterbox_multilingual_engine, mock_raw_request
):
    _engine, serving_tts, _embed = chatterbox_multilingual_engine
    wave, sample_rate = _decode(
        _speak_pack_voice(
            serving_tts, mock_raw_request, "chatterbox_multilingual", language="en"
        )
    )
    assert wave.size > 0 and sample_rate > 0

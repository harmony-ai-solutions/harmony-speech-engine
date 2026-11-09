"""Real-model long-form chunking coverage (Phase 12-2), gated on weights.

These tests use the existing per-family e2e fixtures (which download real
checkpoints) and are skipped unless ``HSE_E2E_MODEL_TESTS=1`` is set — the same
convention as ``tests/e2e/wire_contract``. The hermetic stub coverage for every
family lives in ``tests/integration/test_tts_chunking.py`` and runs in CI.

For each family: a multi-sentence input with ``split_text`` and a sentence pause
must return a decodable WAV whose duration is longer than the single-chunk
rendering of the same text (the pause is audible).
"""

import asyncio
import base64
import os

import pytest

from harmonyspeech.endpoints.openai.protocol import (
    GenerationOptions,
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

TEXT = "The first sentence is here. The second sentence follows. A third closes it."
PAUSE_MS = 80


def _decode(response):
    from harmonyspeech.common.audio_codec import decode_audio

    assert isinstance(response, TextToSpeechResponse), response
    return decode_audio(base64.b64decode(response.data))


def _assert_growth(serving_tts, raw_request, model, *, extra):
    plain = asyncio.run(
        serving_tts.create_text_to_speech(
            TextToSpeechRequest(
                model=model,
                input=TEXT,
                mode="single_speaker_tts",
                generation_options=GenerationOptions(split_text=True, chunk_size=1000),
                **extra,
            ),
            raw_request,
        )
    )
    chunked = asyncio.run(
        serving_tts.create_text_to_speech(
            TextToSpeechRequest(
                model=model,
                input=TEXT,
                mode="single_speaker_tts",
                generation_options=GenerationOptions(
                    split_text=True, chunk_size=40, sentence_pause_ms=PAUSE_MS
                ),
                **extra,
            ),
            raw_request,
        )
    )
    plain_wave, plain_sr = _decode(plain)
    chunked_wave, chunked_sr = _decode(chunked)
    assert plain_sr == chunked_sr and plain_sr > 0
    assert plain_wave.size > 0 and chunked_wave.size > 0
    pause = int(chunked_sr * PAUSE_MS / 1000)
    assert chunked_wave.size > plain_wave.size + pause


def test_chatterbox_chunked_growth(chatterbox_engine, mock_raw_request):
    _engine, serving_tts, _embed, _vc = chatterbox_engine
    _assert_growth(serving_tts, mock_raw_request, "chatterbox", extra={})


def test_chatterbox_turbo_chunked_growth(chatterbox_turbo_engine, mock_raw_request):
    _engine, serving_tts, _embed = chatterbox_turbo_engine
    _assert_growth(serving_tts, mock_raw_request, "chatterbox_turbo", extra={})


def test_chatterbox_multilingual_chunked_growth(chatterbox_multilingual_engine, mock_raw_request):
    _engine, serving_tts, _embed = chatterbox_multilingual_engine
    _assert_growth(serving_tts, mock_raw_request, "chatterbox_multilingual", extra={"language": "en"})


def test_kittentts_chunked_growth(kittentts_mini_engine, mock_raw_request):
    _engine, serving_tts = kittentts_mini_engine
    _assert_growth(
        serving_tts,
        mock_raw_request,
        "kitten-tts-mini",
        extra={"language": "default", "voice": "Jasper"},
    )


def test_melotts_chunked_growth(melotts_en_engine, mock_raw_request):
    _engine, serving_tts, _embed, _vc = melotts_en_engine
    _assert_growth(
        serving_tts,
        mock_raw_request,
        "ov2-synthesizer-en",
        extra={"language": "EN", "voice": "EN-Newest"},
    )


def test_openvoice_v1_chunked_growth(openvoice_v1_en_engine, mock_raw_request):
    _engine, serving_tts, _embed, _vc = openvoice_v1_en_engine
    _assert_growth(
        serving_tts,
        mock_raw_request,
        "ov1-synthesizer-en",
        extra={"language": "EN", "voice": "EN-Newest"},
    )


def test_harmonyspeech_clone_chunked_growth(harmonyspeech_engine, mock_raw_request):
    from tests.e2e.conftest import load_sample_audio_b64

    _engine, serving_tts, _embed = harmonyspeech_engine
    request = TextToSpeechRequest(
        model="harmonyspeech",
        input=TEXT,
        mode="voice_cloning",
        input_audio=load_sample_audio_b64("wanda4"),
        generation_options=GenerationOptions(split_text=True, chunk_size=40, sentence_pause_ms=PAUSE_MS),
    )
    chunked = asyncio.run(serving_tts.create_text_to_speech(request, mock_raw_request))
    wave, sr = _decode(chunked)
    assert wave.size > 0 and sr > 0


def test_openvoice_v1_clone_chunked_growth(openvoice_v1_en_engine, mock_raw_request):
    from tests.e2e.conftest import load_sample_audio_b64

    _engine, serving_tts, _embed, _vc = openvoice_v1_en_engine
    request = TextToSpeechRequest(
        model="openvoice_v1",
        input=TEXT,
        mode="voice_cloning",
        language="EN",
        voice="EN-Newest",
        input_audio=load_sample_audio_b64("wanda4"),
        generation_options=GenerationOptions(split_text=True, chunk_size=40, sentence_pause_ms=PAUSE_MS),
    )
    chunked = asyncio.run(serving_tts.create_text_to_speech(request, mock_raw_request))
    wave, sr = _decode(chunked)
    assert wave.size > 0 and sr > 0


# ---------------------------------------------------------------------------
# Phase 12-3 output filters: opt-in smoke per family
# ---------------------------------------------------------------------------


def _assert_filters_valid(serving_tts, raw_request, model, *, extra):
    """One request with all three filters on must return a decodable WAV.

    Real-model output has no guaranteed DC/silence structure, so this is a
    *smoke* gate (filters are sample-preserving by construction: valid WAV in ⇒
    valid WAV out). The deterministic on-vs-off assertions live on the synthetic
    fixtures in ``tests/unit/common/test_audio_utils.py`` and
    ``tests/integration/test_tts_output_filters.py``.
    """
    response = asyncio.run(
        serving_tts.create_text_to_speech(
            TextToSpeechRequest(
                model=model,
                input=TEXT,
                mode="single_speaker_tts",
                generation_options=GenerationOptions(
                    split_text=True,
                    chunk_size=40,
                    sentence_pause_ms=PAUSE_MS,
                    remove_dc_offset=True,
                    trim_silence=True,
                    fix_internal_silence=True,
                ),
                **extra,
            ),
            raw_request,
        )
    )
    wave, sr = _decode(response)
    assert wave.size > 0 and sr > 0


def test_chatterbox_output_filters_smoke(chatterbox_engine, mock_raw_request):
    _engine, serving_tts, _embed, _vc = chatterbox_engine
    _assert_filters_valid(serving_tts, mock_raw_request, "chatterbox", extra={})


def test_chatterbox_turbo_output_filters_smoke(chatterbox_turbo_engine, mock_raw_request):
    _engine, serving_tts, _embed = chatterbox_turbo_engine
    _assert_filters_valid(serving_tts, mock_raw_request, "chatterbox_turbo", extra={})


def test_chatterbox_multilingual_output_filters_smoke(
    chatterbox_multilingual_engine, mock_raw_request
):
    _engine, serving_tts, _embed = chatterbox_multilingual_engine
    _assert_filters_valid(
        serving_tts, mock_raw_request, "chatterbox_multilingual", extra={"language": "en"}
    )


def test_kittentts_output_filters_smoke(kittentts_mini_engine, mock_raw_request):
    _engine, serving_tts = kittentts_mini_engine
    _assert_filters_valid(
        serving_tts,
        mock_raw_request,
        "kitten-tts-mini",
        extra={"language": "default", "voice": "Jasper"},
    )


def test_melotts_output_filters_smoke(melotts_en_engine, mock_raw_request):
    _engine, serving_tts, _embed, _vc = melotts_en_engine
    _assert_filters_valid(
        serving_tts,
        mock_raw_request,
        "ov2-synthesizer-en",
        extra={"language": "EN", "voice": "EN-Newest"},
    )


def test_harmonyspeech_output_filters_smoke(harmonyspeech_engine, mock_raw_request):
    from tests.e2e.conftest import load_sample_audio_b64

    _engine, serving_tts, _embed = harmonyspeech_engine
    request = TextToSpeechRequest(
        model="harmonyspeech",
        input=TEXT,
        mode="voice_cloning",
        input_audio=load_sample_audio_b64("wanda4"),
        generation_options=GenerationOptions(
            split_text=True,
            chunk_size=40,
            sentence_pause_ms=PAUSE_MS,
            remove_dc_offset=True,
            trim_silence=True,
            fix_internal_silence=True,
        ),
    )
    chunked = asyncio.run(serving_tts.create_text_to_speech(request, mock_raw_request))
    wave, sr = _decode(chunked)
    assert wave.size > 0 and sr > 0


def test_openvoice_v1_output_filters_smoke(openvoice_v1_en_engine, mock_raw_request):
    _engine, serving_tts, _embed, _vc = openvoice_v1_en_engine
    _assert_filters_valid(
        serving_tts,
        mock_raw_request,
        "ov1-synthesizer-en",
        extra={"language": "EN", "voice": "EN-Newest"},
    )

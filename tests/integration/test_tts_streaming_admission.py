"""Phase 14-1 streaming admission through the real serving object.

The pure matrix lives in ``tests/unit/wire_contract/test_streaming_protocol.py``.
Here the same rules run through the real ``OpenAIServingTextToSpeech`` (with the
12-2 ``EngineHarness`` and stub models), proving that every rejection happens
**before synthesis**: the stub model never runs and the engine never sees the
request. Native-speed vs non-native ``speed`` is resolved from the engine's
configured model types, exactly as the chunked path does.
"""

import pytest

from harmonyspeech.endpoints.openai.protocol import AudioOutputOptions, ErrorResponse
from harmonyspeech.endpoints.openai.serving_text_to_speech import StreamingAudioResponse
from harmonyspeech.endpoints.openai.streaming import (
    HEADER_AUDIO_FORMAT,
    HEADER_AUDIO_SAMPLE_RATE,
)
from tests.integration.test_tts_chunking import (
    _chatterbox_harness,
    _kitten_harness,
    _request,
    _speak,
)


def _with_output_options(request, **output):
    return request.model_copy(update={"output_options": AudioOutputOptions(**output)})


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "generation_options",
    [
        {"trim_silence": True},
        {"fix_internal_silence": True},
        {"speed": 1.5},
    ],
)
async def test_chatterbox_whole_buffer_options_rejected_before_synthesis(generation_options):
    harness, model = _chatterbox_harness()
    async with harness:
        request = _with_output_options(
            _request("AAAA.", **generation_options), stream=True, format="wav"
        )
        response = await _speak(harness, request)

    assert isinstance(response, ErrorResponse), response
    assert response.code == 400
    assert "whole-buffer" in response.message
    assert model.generate_calls == []


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("fmt", ["mp3", "flac"])
async def test_whole_buffer_format_with_stream_rejected_before_synthesis(fmt):
    harness, model = _chatterbox_harness()
    async with harness:
        request = _with_output_options(_request("AAAA."), stream=True, format=fmt)
        response = await _speak(harness, request)

    assert isinstance(response, ErrorResponse), response
    assert response.code == 400
    assert "wav" in response.message and "pcm" in response.message
    assert model.generate_calls == []


@pytest.mark.integration
@pytest.mark.asyncio
async def test_native_speed_family_admits_speed_while_streaming():
    """KittenTTS consumes ``speed`` itself, so streaming does not reject it."""
    harness, model = _kitten_harness()
    async with harness:
        request = _with_output_options(
            _request("AAAA.", model="kitten-mini", speed=2.0), stream=True, format="wav"
        )
        response = await _speak(harness, request)
        body = b"".join([frame async for frame in response.body])

    # Admission passed and the 14-3 binary envelope served the audio.
    assert isinstance(response, StreamingAudioResponse), response
    assert response.headers[HEADER_AUDIO_FORMAT] == "wav"
    assert body[:4] == b"RIFF" and len(body) > 44
    assert model.generate_calls == ["AAAA."]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_remove_dc_offset_and_sample_rate_are_admitted_and_stream():
    harness, _model = _chatterbox_harness()
    async with harness:
        request = _with_output_options(
            _request("AAAA.", remove_dc_offset=True),
            stream_format="audio",
            format="pcm",
            sample_rate=16000,
        )
        response = await _speak(harness, request)
        body = b"".join([frame async for frame in response.body])

    assert isinstance(response, StreamingAudioResponse), response
    assert response.headers[HEADER_AUDIO_FORMAT] == "pcm"
    assert response.headers[HEADER_AUDIO_SAMPLE_RATE] == "16000"
    assert body and not body.startswith(b"RIFF")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sse_envelope_streams_after_admission():
    """14-4: an admitted SSE request streams events, never a 501/buffered body."""
    harness, model = _chatterbox_harness()
    async with harness:
        response = await _speak(
            harness,
            _with_output_options(
                _request("AAAA."), stream_format="sse", format="pcm", sample_rate=8000
            ),
        )
        body = "".join([frame async for frame in response.body])

    assert isinstance(response, StreamingAudioResponse), response
    assert response.headers[HEADER_AUDIO_FORMAT] == "pcm"
    assert response.headers[HEADER_AUDIO_SAMPLE_RATE] == "8000"
    assert body.startswith("event: speech.audio.delta")
    assert body.endswith("data: [DONE]\n\n")
    assert model.generate_calls == ["AAAA."]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_alias_and_stream_format_audio_are_equivalent():
    harness, _model = _chatterbox_harness()
    async with harness:
        alias = await _speak(
            harness,
            _with_output_options(_request("AAAA.", speed=2.0), stream=True, format="wav"),
        )
        explicit = await _speak(
            harness,
            _with_output_options(
                _request("AAAA.", speed=2.0), stream_format="audio", format="wav"
            ),
        )

    assert isinstance(alias, ErrorResponse) and isinstance(explicit, ErrorResponse)
    assert alias.code == explicit.code == 400
    assert alias.message == explicit.message


@pytest.mark.integration
@pytest.mark.asyncio
async def test_non_streaming_path_is_unchanged_by_the_streaming_gate():
    """The same buffered request with and without explicit defaults is identical."""
    harness, _model = _chatterbox_harness()
    async with harness:
        omitted = await _speak(harness, _request("AAAA."))
        explicit = await _speak(
            harness, _with_output_options(_request("AAAA."), stream=False, format="wav")
        )

    assert omitted.data == explicit.data
    assert omitted.sample_rate == explicit.sample_rate is None

"""Phase 14-1: streaming protocol, admission matrix and response headers.

The envelopes themselves land in 14-3/14-4; this file pins the contract they
will consume:

* ``stream_format`` / deprecated ``stream`` alias resolution (setting
  ``stream_format`` selects streaming; ``stream: true`` means ``"audio"``);
* the pre-synthesis admission matrix (wav/pcm only; whole-buffer-only
  ``trim_silence`` / ``fix_internal_silence`` / non-native ``speed``;
  ``sample_rate`` and ``remove_dc_offset`` allowed);
* the canonical response-header set for both envelopes;
* that the buffered (non-streaming) wire path is untouched.
"""

import base64
import json
import struct
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from harmonyspeech.common.inputs import TextToSpeechAudioOutputOptions, TextToSpeechRequestInput
from harmonyspeech.endpoints.openai.protocol import (
    AudioOutputOptions,
    GenerationOptions,
    TextToSpeechRequest,
)
from harmonyspeech.endpoints.openai.serving_text_to_speech import streaming_admission_error
from harmonyspeech.endpoints.openai.streaming import (
    HEADER_ACCEL_BUFFERING,
    HEADER_AUDIO_BITS,
    HEADER_AUDIO_CHANNELS,
    HEADER_AUDIO_FORMAT,
    HEADER_AUDIO_SAMPLE_RATE,
    HEADER_CACHE_CONTROL,
    HEADER_REQUEST_ID,
    SSE_EVENT_DELTA,
    SSE_EVENT_DONE,
    SSE_EVENT_ERROR,
    SSE_MEDIA_TYPE,
    STREAM_FORMATS,
    STREAMING_AUDIO_BITS,
    STREAMING_AUDIO_CHANNELS,
    STREAMING_AUDIO_FORMATS,
    STREAMING_RESPONSE_HEADERS,
    resolve_stream_format,
    streaming_response_headers,
)
from tests.wire_contract_support import WireApp, wav_b64


def _request(*, output_options=None, generation_options=None, model="test-tts"):
    return TextToSpeechRequest(
        model=model,
        input="hello",
        mode="single_speaker_tts",
        output_options=output_options,
        generation_options=generation_options,
    )


# ---------------------------------------------------------------------------
# Protocol fields
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_audio_output_options_stream_defaults_are_buffered():
    options = AudioOutputOptions()
    assert options.stream_format is None
    assert options.stream is False


@pytest.mark.unit
@pytest.mark.parametrize("stream_format", ["audio", "sse"])
def test_audio_output_options_accepts_both_envelopes(stream_format):
    assert AudioOutputOptions(stream_format=stream_format).stream_format == stream_format


@pytest.mark.unit
def test_audio_output_options_rejects_unknown_envelope():
    with pytest.raises(ValidationError):
        AudioOutputOptions(stream_format="pcm")


@pytest.mark.unit
def test_common_input_layer_carries_stream_format():
    """12-1 precedent: the plain input-layer type mirrors the wire field."""
    options = TextToSpeechAudioOutputOptions()
    assert options.stream_format is None
    options.stream_format = "sse"
    assert resolve_stream_format(options) == "sse"


@pytest.mark.unit
def test_request_input_passes_output_options_through_unchanged():
    wire_request = _request(output_options=AudioOutputOptions(stream_format="sse"))
    request_input = TextToSpeechRequestInput.from_openai("tts-1", wire_request)
    assert request_input.output_options.stream_format == "sse"


# ---------------------------------------------------------------------------
# Envelope resolution
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_resolve_stream_format_defaults_to_buffered():
    assert resolve_stream_format(None) is None
    assert resolve_stream_format(AudioOutputOptions()) is None
    assert resolve_stream_format(AudioOutputOptions(stream=False)) is None


@pytest.mark.unit
def test_stream_alias_maps_to_audio():
    assert resolve_stream_format(AudioOutputOptions(stream=True)) == "audio"


@pytest.mark.unit
@pytest.mark.parametrize("stream_format", ["audio", "sse"])
def test_stream_format_selects_streaming_on_its_own(stream_format):
    """A named format streams even with ``stream`` left at its false default."""
    options = AudioOutputOptions(stream_format=stream_format)
    assert resolve_stream_format(options) == stream_format


@pytest.mark.unit
def test_explicit_stream_format_wins_over_the_alias():
    options = AudioOutputOptions(stream=True, stream_format="sse")
    assert resolve_stream_format(options) == "sse"


@pytest.mark.unit
def test_stream_format_constants():
    assert {"audio", "sse"} == STREAM_FORMATS
    assert {"wav", "pcm"} == STREAMING_AUDIO_FORMATS


# ---------------------------------------------------------------------------
# Response headers
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_streaming_headers_are_the_canonical_set():
    headers = streaming_response_headers(
        audio_format="wav", sample_rate=24000, request_id="tts-abc"
    )
    assert tuple(headers) == STREAMING_RESPONSE_HEADERS
    assert headers[HEADER_AUDIO_FORMAT] == "wav"
    assert headers[HEADER_AUDIO_SAMPLE_RATE] == "24000"
    assert headers[HEADER_AUDIO_CHANNELS] == str(STREAMING_AUDIO_CHANNELS)
    assert headers[HEADER_AUDIO_BITS] == str(STREAMING_AUDIO_BITS)
    assert headers[HEADER_REQUEST_ID] == "tts-abc"
    assert headers[HEADER_ACCEL_BUFFERING] == "no"
    assert headers[HEADER_CACHE_CONTROL] == "no-cache, no-transform"


@pytest.mark.unit
def test_streaming_headers_report_the_target_rate():
    """``X-Audio-Sample-Rate`` is the effective/target rate, not the source rate."""
    headers = streaming_response_headers(
        audio_format="pcm", sample_rate=8000, request_id="tts-tel"
    )
    assert headers[HEADER_AUDIO_FORMAT] == "pcm"
    assert headers[HEADER_AUDIO_SAMPLE_RATE] == "8000"


@pytest.mark.unit
def test_streaming_headers_reject_whole_buffer_formats():
    with pytest.raises(ValueError, match="'wav' and 'pcm'"):
        streaming_response_headers(audio_format="mp3", sample_rate=24000, request_id="tts-x")


# ---------------------------------------------------------------------------
# Admission matrix (pure)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("fmt", ["wav", "pcm"])
def test_admission_accepts_wav_and_pcm(fmt):
    request = _request(output_options=AudioOutputOptions(format=fmt, stream=True))
    assert streaming_admission_error(request, uses_native_speed=False) is None


@pytest.mark.unit
def test_admission_rejects_whole_buffer_formats_naming_wav_pcm():
    request = _request(output_options=AudioOutputOptions(format="mp3", stream=True))
    message = streaming_admission_error(request, uses_native_speed=False)
    assert message is not None
    assert "wav" in message and "pcm" in message
    assert "mp3" in message


@pytest.mark.unit
@pytest.mark.parametrize("option", ["trim_silence", "fix_internal_silence"])
def test_admission_rejects_whole_buffer_filters(option):
    request = _request(
        output_options=AudioOutputOptions(stream=True),
        generation_options=GenerationOptions(**{option: True}),
    )
    message = streaming_admission_error(request, uses_native_speed=False)
    assert message is not None
    assert option in message
    assert "whole-buffer" in message


@pytest.mark.unit
def test_admission_rejects_non_native_speed_naming_whole_buffer():
    request = _request(
        output_options=AudioOutputOptions(stream=True),
        generation_options=GenerationOptions(speed=1.5),
    )
    message = streaming_admission_error(request, uses_native_speed=False)
    assert message is not None
    assert "speed" in message
    assert "whole-buffer" in message


@pytest.mark.unit
@pytest.mark.parametrize("speed", [None, 1, 1.0])
def test_admission_allows_unity_speed_on_non_native_families(speed):
    request = _request(
        output_options=AudioOutputOptions(stream=True),
        generation_options=GenerationOptions(speed=speed),
    )
    assert streaming_admission_error(request, uses_native_speed=False) is None


@pytest.mark.unit
def test_admission_allows_speed_on_native_families():
    request = _request(
        output_options=AudioOutputOptions(stream=True),
        generation_options=GenerationOptions(speed=2.0),
    )
    assert streaming_admission_error(request, uses_native_speed=True) is None


@pytest.mark.unit
def test_admission_allows_sample_rate_while_streaming():
    request = _request(
        output_options=AudioOutputOptions(format="pcm", sample_rate=8000, stream=True)
    )
    assert streaming_admission_error(request, uses_native_speed=False) is None


@pytest.mark.unit
def test_admission_allows_remove_dc_offset_while_streaming():
    request = _request(
        output_options=AudioOutputOptions(stream=True),
        generation_options=GenerationOptions(remove_dc_offset=True),
    )
    assert streaming_admission_error(request, uses_native_speed=False) is None


@pytest.mark.unit
def test_admission_ignores_every_restriction_when_not_streaming():
    request = _request(
        generation_options=GenerationOptions(
            trim_silence=True, fix_internal_silence=True, speed=2.0
        ),
    )
    assert streaming_admission_error(request, uses_native_speed=False) is None
    explicit_off = _request(
        output_options=AudioOutputOptions(stream=False),
        generation_options=GenerationOptions(trim_silence=True, speed=2.0),
    )
    assert streaming_admission_error(explicit_off, uses_native_speed=False) is None


@pytest.mark.unit
def test_sse_envelope_uses_the_same_admission_rules():
    accepted = _request(output_options=AudioOutputOptions(format="pcm", stream_format="sse"))
    assert streaming_admission_error(accepted, uses_native_speed=False) is None

    rejected = _request(output_options=AudioOutputOptions(format="mp3", stream_format="sse"))
    message = streaming_admission_error(rejected, uses_native_speed=False)
    assert message is not None and "wav" in message and "pcm" in message


@pytest.mark.unit
def test_stream_alias_and_stream_format_audio_are_equivalent():
    alias = _request(output_options=AudioOutputOptions(format="mp3", stream=True))
    explicit = _request(
        output_options=AudioOutputOptions(format="mp3", stream_format="audio")
    )
    assert streaming_admission_error(alias, uses_native_speed=False) == streaming_admission_error(
        explicit, uses_native_speed=False
    )


# ---------------------------------------------------------------------------
# HTTP admission (wire client, no synthesis)
# ---------------------------------------------------------------------------


def _tts_payload(**overrides):
    payload = {"model": "test-tts", "mode": "single_speaker_tts", "input": "hello"}
    payload.update(overrides)
    return payload


@pytest.mark.unit
def test_wire_rejects_mp3_stream_before_synthesis(wire_client):
    before = len(wire_client.engine.requests)
    response = wire_client.client.post(
        "/v1/audio/speech",
        json=_tts_payload(output_options={"format": "mp3", "stream_format": "audio"}),
    )
    assert response.status_code == 400
    message = response.json()["message"]
    assert "wav" in message and "pcm" in message
    assert len(wire_client.engine.requests) == before


@pytest.mark.unit
def test_wire_audio_envelope_streams_binary_after_admission(wire_client):
    """14-3: the admitted ``audio`` envelope streams a framed binary body.

    One request reaches the engine (the stream's first chunk, primed before
    headers), the response carries the canonical header set, and a ``wav``
    stream opens with the streaming RIFF header.
    """
    before = len(wire_client.engine.requests)
    response = wire_client.client.post(
        "/v1/audio/speech",
        json=_tts_payload(output_options={"stream": True, "format": "wav"}),
    )
    assert response.status_code == 200
    assert response.headers[HEADER_AUDIO_FORMAT] == "wav"
    assert response.headers[HEADER_AUDIO_SAMPLE_RATE] == "44100"
    assert response.headers[HEADER_AUDIO_CHANNELS] == "1"
    assert response.headers[HEADER_AUDIO_BITS] == "16"
    assert response.headers[HEADER_ACCEL_BUFFERING] == "no"
    assert response.headers[HEADER_CACHE_CONTROL] == "no-cache, no-transform"
    assert response.content[:4] == b"RIFF"
    assert response.content[8:12] == b"WAVE"
    assert struct.unpack("<I", response.content[40:44])[0] == 0xFFFFFFFF
    assert len(response.content) > 44
    assert len(wire_client.engine.requests) == before + 1


@pytest.mark.unit
def test_wire_pcm_envelope_omits_the_header(wire_client):
    response = wire_client.client.post(
        "/v1/audio/speech",
        json=_tts_payload(
            output_options={"stream_format": "audio", "format": "pcm", "sample_rate": 8000}
        ),
    )
    assert response.status_code == 200
    assert response.headers[HEADER_AUDIO_FORMAT] == "pcm"
    assert response.headers[HEADER_AUDIO_SAMPLE_RATE] == "8000"
    assert response.content[:4] != b"RIFF"
    # 1 s of the fake engine's tone at the target rate, mono s16le.
    assert len(response.content) == pytest.approx(8000 * 2, abs=64)


@pytest.mark.unit
def test_wire_sse_envelope_streams_events(wire_client):
    """14-4: the admitted ``sse`` envelope streams the event contract.

    The route wraps the serving response unchanged, so this pins the wire
    surface: ``text/event-stream``, the 14-1 header set naming PCM, deltas whose
    ``audio`` is base64 PCM, one ``done`` and ``data: [DONE]`` as the last line.
    """
    before = len(wire_client.engine.requests)
    response = wire_client.client.post(
        "/v1/audio/speech",
        json=_tts_payload(output_options={"stream_format": "sse", "format": "pcm"}),
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == SSE_MEDIA_TYPE
    assert response.headers[HEADER_AUDIO_FORMAT] == "pcm"
    assert response.headers[HEADER_AUDIO_SAMPLE_RATE] == "44100"
    assert response.headers[HEADER_AUDIO_CHANNELS] == "1"
    assert response.headers[HEADER_AUDIO_BITS] == "16"
    assert response.headers[HEADER_ACCEL_BUFFERING] == "no"
    assert response.headers[HEADER_CACHE_CONTROL] == "no-cache, no-transform"
    assert len(wire_client.engine.requests) == before + 1

    body = response.text
    assert body.endswith("data: [DONE]\n\n")
    lines = body.split("\n")
    assert lines[-3:] == ["data: [DONE]", "", ""]
    assert lines.count(f"event: {SSE_EVENT_DONE}") == 1
    assert f"event: {SSE_EVENT_ERROR}" not in lines

    payloads = [
        json.loads(line[len("data: ") :])
        for line in lines
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    deltas = [payload for payload in payloads if payload["type"] == SSE_EVENT_DELTA]
    done = [payload for payload in payloads if payload["type"] == SSE_EVENT_DONE]
    assert len(deltas) >= 1
    assert len(done) == 1
    assert done[0]["format"] == "pcm"
    assert done[0]["sample_rate"] == 44100
    assert done[0]["chunks"] == len(deltas)
    assert done[0]["id"] == response.headers[HEADER_REQUEST_ID]
    assert done[0]["usage"]["audio_seconds"] == pytest.approx(done[0]["duration_s"])
    # The delta audio is headerless s16le PCM, not a container.
    pcm = b"".join(base64.b64decode(delta["audio"]) for delta in deltas)
    assert pcm[:4] != b"RIFF"
    assert len(pcm) % 2 == 0 and len(pcm) > 0


@pytest.mark.unit
@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("trim_silence", True),
        ("fix_internal_silence", True),
        ("speed", 1.5),
    ],
)
def test_wire_rejects_whole_buffer_options_with_stream(wire_client, option, value):
    before = len(wire_client.engine.requests)
    response = wire_client.client.post(
        "/v1/audio/speech",
        json=_tts_payload(
            output_options={"stream": True, "format": "wav"},
            generation_options={option: value},
        ),
    )
    assert response.status_code == 400
    message = response.json()["message"]
    assert option in message
    assert "whole-buffer" in message
    assert len(wire_client.engine.requests) == before


@pytest.mark.unit
@pytest.mark.parametrize(
    "generation_options",
    [{"remove_dc_offset": True}, {"speed": 1}],
)
def test_wire_accepts_composable_options_with_stream(wire_client, generation_options):
    response = wire_client.client.post(
        "/v1/audio/speech",
        json=_tts_payload(
            output_options={"stream": True, "format": "pcm", "sample_rate": 8000},
            generation_options=generation_options,
        ),
    )
    # Admission passed and 14-3 streams the body: never a 400, never a buffered 200 JSON.
    assert response.status_code == 200
    assert response.headers[HEADER_AUDIO_FORMAT] == "pcm"
    assert response.content[:1] != b"{"


@pytest.mark.unit
def test_wire_alias_and_stream_format_have_the_same_outcome(wire_client):
    alias = wire_client.client.post(
        "/v1/audio/speech",
        json=_tts_payload(output_options={"stream": True, "format": "mp3"}),
    )
    explicit = wire_client.client.post(
        "/v1/audio/speech",
        json=_tts_payload(output_options={"stream_format": "audio", "format": "mp3"}),
    )
    assert alias.status_code == explicit.status_code == 400
    assert alias.json()["message"] == explicit.json()["message"]


@pytest.mark.unit
def test_wire_native_speed_family_accepts_speed_while_streaming():
    """A native-speed family consumes ``speed`` itself, so streaming admits it."""
    with WireApp(
        decode_inputs=True,
        model_configs=[
            SimpleNamespace(name="test-tts", model_type="KittenTTSSynthesizer")
        ],
    ) as wire:
        response = wire.client.post(
            "/v1/audio/speech",
            json=_tts_payload(
                output_options={"stream": True, "format": "wav"},
                generation_options={"speed": 2.0},
            ),
        )
    assert response.status_code == 200
    assert response.content[:4] == b"RIFF"

    # Control: the same request on a non-native (Chatterbox) family is a 400.
    with WireApp(
        decode_inputs=True,
        model_configs=[SimpleNamespace(name="test-tts", model_type="ChatterboxTTS")],
    ) as wire:
        rejected = wire.client.post(
            "/v1/audio/speech",
            json=_tts_payload(
                output_options={"stream": True, "format": "wav"},
                generation_options={"speed": 2.0},
            ),
        )
    assert rejected.status_code == 400
    assert "whole-buffer" in rejected.json()["message"]


@pytest.mark.unit
@pytest.mark.parametrize(
    "output_options",
    [
        None,
        {"format": "wav"},
        {"format": "wav", "stream": False},
        {"format": "wav", "stream_format": None},
    ],
)
def test_wire_non_streaming_output_is_byte_identical(wire_client, output_options):
    """Buffered requests (including explicit non-streaming fields) are unchanged."""
    response = wire_client.client.post(
        "/v1/audio/speech", json=_tts_payload(output_options=output_options)
    )
    assert response.status_code == 200
    body = response.json()
    assert body["data"] == wav_b64()
    assert body["sample_rate"] is None

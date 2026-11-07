"""Wire-contract matrix: input formats x transports x output formats (Phase 1-4).

The transport/format matrix runs against the real FastAPI app with a
boundary-decoding FakeEngine, so it needs no model downloads. It verifies:

* every accepted input container (wav/flac/mp3, plus aac when ffmpeg is
  installed) works over both JSON base64 and multipart uploads;
* every advertised output format/sample-rate combination produces decodable
  audio at the requested rate, and the default path stays byte-identical;
* error behaviour (400 on unsupported options / garbage audio, 413 on
  oversized uploads).

Tests that need real model weights are marked ``model`` and only run when
``HSE_E2E_MODEL_TESTS=1`` is set (they download from HuggingFace).
"""

import asyncio
import base64
import os

import numpy as np
import pytest

from harmonyspeech.common import audio_codec
from tests.wire_contract_support import (
    TRANSCRIPTION_PAYLOAD,
    WireApp,
    sample_b64,
    sample_bytes,
    wav_b64,
)

pytestmark = pytest.mark.e2e

FFMPEG_AVAILABLE = audio_codec.FFMPEG_AVAILABLE
NEEDS_MODELS = pytest.mark.skipif(
    os.environ.get("HSE_E2E_MODEL_TESTS") != "1",
    reason="requires downloaded model weights (set HSE_E2E_MODEL_TESTS=1 to enable)",
)

INPUT_FORMATS = ["wav", "flac", "mp3"] + (["aac"] if FFMPEG_AVAILABLE else [])
INPUT_FILES = {
    "wav": "codec_tone.wav",
    "flac": "codec_tone.flac",
    "mp3": "codec_tone.mp3",
    "aac": "codec_tone.aac",
}
TRANSPORTS = ["json", "multipart"]

OUTPUT_CASES = [
    (None, None),
    ("wav", None),
    (None, 16000),
    ("wav", 44100),
    ("flac", None),
    ("flac", 16000),
    ("ogg", 16000),
    ("mp3", 44100),
    ("pcm", None),
    ("pcm", 16000),
]
if FFMPEG_AVAILABLE:
    OUTPUT_CASES.append(("aac", 24000))


@pytest.fixture(scope="module")
def wire():
    with WireApp(decode_inputs=True) as app:
        yield app


@pytest.fixture(scope="module")
def wire_small_limit():
    with WireApp(decode_inputs=True, max_upload_bytes=4096) as app:
        yield app


def _tts_payload(**overrides):
    payload = {"model": WireApp.MODEL_IDS["tts"], "mode": "single_speaker_tts", "input": "hello world"}
    payload.update(overrides)
    return payload


# ---------------------------------------------------------------------------
# Input format x transport
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("transport", TRANSPORTS)
@pytest.mark.parametrize("input_format", INPUT_FORMATS)
def test_input_format_and_transport_matrix(wire, test_data_dir, input_format, transport):
    payload = (test_data_dir / "samples" / INPUT_FILES[input_format]).read_bytes()
    expected_b64 = base64.b64encode(payload).decode()
    url = "/v1/audio/transcriptions"
    model = WireApp.MODEL_IDS["stt"]

    if transport == "json":
        response = wire.client.post(url, json={"model": model, "input_audio": expected_b64})
    else:
        response = wire.client.post(
            url,
            files={"file": (f"tone.{input_format}", payload, f"audio/{input_format}")},
            data={"model": model},
        )

    assert response.status_code == 200, response.text
    assert response.json()["text"] == TRANSCRIPTION_PAYLOAD["text"]
    assert wire.engine.requests[-1].input_audio == expected_b64


@pytest.mark.parametrize("transport", TRANSPORTS)
@pytest.mark.parametrize("input_format", INPUT_FORMATS)
def test_tts_reference_audio_format_and_transport(wire, test_data_dir, input_format, transport):
    payload = (test_data_dir / "samples" / INPUT_FILES[input_format]).read_bytes()
    expected_b64 = base64.b64encode(payload).decode()

    if transport == "json":
        response = wire.client.post(
            "/v1/audio/speech",
            json=_tts_payload(mode="voice_cloning", input_audio=expected_b64),
        )
    else:
        response = wire.client.post(
            "/v1/audio/speech",
            files={"input_audio": (f"ref.{input_format}", payload, f"audio/{input_format}")},
            data={
                "model": WireApp.MODEL_IDS["tts"],
                "mode": "voice_cloning",
                "input": "hello world",
            },
        )

    assert response.status_code == 200, response.text
    assert wire.engine.requests[-1].input_audio == expected_b64


# ---------------------------------------------------------------------------
# STT response formats (Phase 10-4): both endpoints x {json, text, srt, vtt}
# ---------------------------------------------------------------------------

STT_ENDPOINTS = ["/v1/audio/transcriptions", "/v1/audio/translations"]
STT_RESPONSE_CASES = [
    ("json", "application/json"),
    ("text", "text/plain"),
    ("srt", "text/plain"),
    ("vtt", "text/plain"),
]

_SRT_DOCUMENT = "1\n00:00:00,000 --> 00:00:01,000\nhello world\n"
_VTT_DOCUMENT = "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nhello world\n"


@pytest.mark.parametrize("endpoint", STT_ENDPOINTS)
@pytest.mark.parametrize(("response_format", "media_type"), STT_RESPONSE_CASES)
def test_stt_response_format_matrix(wire, endpoint, response_format, media_type):
    response = wire.client.post(
        endpoint,
        json={
            "model": WireApp.MODEL_IDS["stt"],
            "input_audio": sample_b64("codec_tone.wav"),
            "response_format": response_format,
        },
    )
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith(media_type)

    if response_format == "json":
        body = response.json()
        assert set(body) == {"id", "model", "created", "text", "language", "timestamps"}
        assert body["text"] == TRANSCRIPTION_PAYLOAD["text"]
    elif response_format == "text":
        assert response.text == TRANSCRIPTION_PAYLOAD["text"]
    elif response_format == "srt":
        assert response.text == _SRT_DOCUMENT
    else:
        assert response.text == _VTT_DOCUMENT


@pytest.mark.parametrize("endpoint", STT_ENDPOINTS)
def test_stt_response_format_via_multipart(wire, test_data_dir, endpoint):
    response = wire.client.post(
        endpoint,
        files={"file": ("tone.wav", (test_data_dir / "samples" / "codec_tone.wav").read_bytes(), "audio/wav")},
        data={"model": WireApp.MODEL_IDS["stt"], "response_format": "vtt"},
    )
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/plain")
    assert response.text == _VTT_DOCUMENT


@pytest.mark.parametrize("endpoint", STT_ENDPOINTS)
def test_unknown_stt_response_format_is_rejected(wire, endpoint):
    before = len(wire.engine.requests)
    response = wire.client.post(
        endpoint,
        json={
            "model": WireApp.MODEL_IDS["stt"],
            "input_audio": sample_b64("codec_tone.wav"),
            "response_format": "ass",
        },
    )
    assert response.status_code == 400
    assert "Unsupported response_format" in response.json()["message"]
    assert len(wire.engine.requests) == before, "admission must reject before inference"


# ---------------------------------------------------------------------------
# Output format x sample rate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("output_format", "sample_rate"), OUTPUT_CASES)
def test_output_format_matrix(wire, output_format, sample_rate):
    options = {}
    if output_format is not None:
        options["format"] = output_format
    if sample_rate is not None:
        options["sample_rate"] = sample_rate

    response = wire.client.post("/v1/audio/speech", json=_tts_payload(output_options=options or None))
    assert response.status_code == 200, response.text
    body = response.json()

    if (output_format in (None, "wav")) and sample_rate is None:
        # Default path: byte-identical passthrough of the executor WAV.
        assert body["data"] == wav_b64()
        assert body["sample_rate"] is None
        return

    raw = base64.b64decode(body["data"])
    if output_format == "pcm":
        assert body["sample_rate"] == (sample_rate or 44100)
        samples = np.frombuffer(raw, dtype="<i2")
        assert samples.size > 0
        return

    decoded, decoded_sr = audio_codec.decode_audio(raw)
    expected_sr = sample_rate or 44100
    assert decoded_sr == expected_sr
    assert abs(decoded.size / expected_sr - 1.0) < 0.1
    assert float(np.max(np.abs(decoded))) > 0.05
    assert body["sample_rate"] == expected_sr


@pytest.mark.parametrize("output_format", ["flac", "mp3", "pcm"])
@pytest.mark.parametrize("endpoint", ["/v1/voice/convert", "/v1/audio/convert"])
def test_conversion_endpoint_output_matrix(wire, endpoint, output_format):
    if output_format != "pcm" and output_format not in audio_codec.SUPPORTED_OUTPUT_FORMATS:
        pytest.skip(f"{output_format} not encodable by this libsndfile build")

    volume_models = {"/v1/voice/convert": "test-vc", "/v1/audio/convert": "test-ac"}
    response = wire.client.post(
        endpoint,
        json={
            "model": volume_models[endpoint],
            "source_audio": sample_b64("codec_tone.wav"),
            "output_options": {"format": output_format, "sample_rate": 16000},
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["sample_rate"] == 16000
    raw = base64.b64decode(body["data"])
    if output_format == "pcm":
        assert np.frombuffer(raw, dtype="<i2").size > 0
    else:
        _, decoded_sr = audio_codec.decode_audio(raw)
        assert decoded_sr == 16000


def test_multipart_mp3_and_aac_through_decoder(wire, test_data_dir):
    """Uploading mp3 (and aac with ffmpeg) exercises the boundary decoder end-to-end."""
    for name in ["codec_tone.mp3"] + (["codec_tone.aac"] if FFMPEG_AVAILABLE else []):
        response = wire.client.post(
            "/v1/audio/vad",
            files={"file": (name, sample_bytes(name), "application/octet-stream")},
            data={"model": WireApp.MODEL_IDS["vad"]},
        )
        assert response.status_code == 200, f"{name}: {response.text}"
        assert response.json()["speech_activity"] is True


# ---------------------------------------------------------------------------
# Contract errors
# ---------------------------------------------------------------------------


def test_unsupported_output_format_is_rejected(wire):
    response = wire.client.post("/v1/audio/speech", json=_tts_payload(output_options={"format": "wma"}))
    assert response.status_code == 400
    assert "Unsupported output format" in response.json()["message"]


def test_unsupported_sample_rate_is_rejected(wire):
    response = wire.client.post("/v1/audio/speech", json=_tts_payload(output_options={"sample_rate": 12345}))
    assert response.status_code == 400
    assert "sample_rate" in response.json()["message"]


def test_garbage_audio_is_a_typed_error_over_both_transports(wire):
    json_response = wire.client.post(
        "/v1/audio/transcriptions",
        json={"model": "test-stt", "input_audio": base64.b64encode(b"garbage").decode()},
    )
    multipart_response = wire.client.post(
        "/v1/audio/transcriptions",
        files={"file": ("bad.bin", b"garbage", "application/octet-stream")},
        data={"model": "test-stt"},
    )
    for response in (json_response, multipart_response):
        assert response.status_code == 400
        assert "Unable to decode input audio" in response.json()["message"]


def test_oversized_upload_is_rejected(wire_small_limit):
    response = wire_small_limit.client.post(
        "/v1/audio/transcriptions",
        files={"file": ("big.wav", b"0" * 8192, "audio/wav")},
        data={"model": "test-stt"},
    )
    assert response.status_code == 413
    assert response.json()["type"] == "PayloadTooLargeError"


# ---------------------------------------------------------------------------
# True e2e cases (real models; skipped unless weights are available)
# ---------------------------------------------------------------------------


@NEEDS_MODELS
def test_real_whisper_accepts_base64_mp3(test_data_dir, whisper_engine, mock_raw_request):
    """FasterWhisper decodes an uploaded MP3 through the boundary codec."""
    from harmonyspeech.endpoints.openai.protocol import SpeechTranscribeRequest

    engine, serving = whisper_engine
    request = SpeechTranscribeRequest(
        model="faster-whisper",
        input_audio=base64.b64encode((test_data_dir / "samples" / "codec_tone.mp3").read_bytes()).decode(),
    )
    result = asyncio.run(serving.create_transcription(request, mock_raw_request))
    assert not getattr(result, "code", None), getattr(result, "message", result)


@NEEDS_MODELS
@pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg not available on PATH")
def test_real_whisper_accepts_base64_aac(test_data_dir, whisper_engine, mock_raw_request):
    from harmonyspeech.endpoints.openai.protocol import SpeechTranscribeRequest

    engine, serving = whisper_engine
    request = SpeechTranscribeRequest(
        model="faster-whisper",
        input_audio=base64.b64encode((test_data_dir / "samples" / "codec_tone.aac").read_bytes()).decode(),
    )
    result = asyncio.run(serving.create_transcription(request, mock_raw_request))
    assert not getattr(result, "code", None), getattr(result, "message", result)


@NEEDS_MODELS
def test_real_tts_honors_output_format(kittentts_mini_engine, mock_raw_request):
    """KittenTTS output is converted to FLAC@16k at the API boundary."""
    from harmonyspeech.endpoints.openai.protocol import TextToSpeechRequest

    engine, serving = kittentts_mini_engine
    request = TextToSpeechRequest(
        model="kitten-tts-mini",
        mode="single_speaker_tts",
        input="Hello from the wire contract test.",
        voice="Bella",
        output_options={"format": "flac", "sample_rate": 16000},
    )
    result = asyncio.run(serving.create_text_to_speech(request, mock_raw_request))
    assert not getattr(result, "code", None), getattr(result, "message", result)
    assert result.sample_rate == 16000
    decoded, decoded_sr = audio_codec.decode_audio(base64.b64decode(result.data))
    assert decoded_sr == 16000
    assert decoded.size > 0
    assert isinstance(result.data, str)

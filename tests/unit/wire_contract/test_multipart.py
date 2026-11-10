"""Endpoint-level tests for the multipart envelope and output options (Phase 1-2/1-3)."""

import base64
import json
from types import SimpleNamespace

import pytest
from starlette.requests import Request

from harmonyspeech.common import audio_codec
from harmonyspeech.endpoints.openai.protocol import SpeechTranscribeRequest
from harmonyspeech.endpoints.openai.request_envelope import (
    UploadTooLargeError,
    parse_dual_request,
    resolve_max_upload_bytes,
)
from tests.wire_contract_support import sample_b64, sample_bytes, wav_b64


def _wav(test_data_dir):
    return (test_data_dir / "samples" / "codec_tone.wav").read_bytes()


# ---------------------------------------------------------------------------
# Multipart vs JSON parity on every audio-input endpoint
# ---------------------------------------------------------------------------


def test_transcription_multipart_matches_json(wire_client, test_data_dir):
    client = wire_client.client
    payload = _wav(test_data_dir)
    b64 = base64.b64encode(payload).decode()

    json_response = client.post("/v1/audio/transcriptions", json={"model": "test-stt", "input_audio": b64})
    json_request = wire_client.engine.requests[-1]

    multipart_response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("tone.wav", payload, "audio/wav")},
        data={"model": "test-stt"},
    )
    multipart_request = wire_client.engine.requests[-1]

    assert json_response.status_code == multipart_response.status_code == 200
    assert json_response.json()["text"] == multipart_response.json()["text"] == "hello world"
    assert json_request.input_audio == multipart_request.input_audio == b64


def test_transcription_multipart_accepts_audio_alias(wire_client, test_data_dir):
    response = wire_client.client.post(
        "/v1/audio/transcriptions",
        files={"audio": ("tone.wav", _wav(test_data_dir), "audio/wav")},
        data={"model": "test-stt"},
    )
    assert response.status_code == 200


def test_vad_multipart(wire_client, test_data_dir):
    response = wire_client.client.post(
        "/v1/audio/vad",
        files={"file": ("tone.wav", _wav(test_data_dir), "audio/wav")},
        data={"model": "test-vad"},
    )
    assert response.status_code == 200
    assert response.json()["speech_activity"] is True


def test_embed_multipart(wire_client, test_data_dir):
    response = wire_client.client.post(
        "/v1/embed/speaker",
        files={"file": ("tone.wav", _wav(test_data_dir), "audio/wav")},
        data={"model": "test-embed"},
    )
    assert response.status_code == 200
    assert response.json()["data"] == "ZGF0YQ=="


def test_voice_convert_multipart_two_files(wire_client, test_data_dir):
    source = _wav(test_data_dir)
    target = sample_bytes("codec_tone.flac")
    response = wire_client.client.post(
        "/v1/voice/convert",
        files={
            "source_audio": ("source.wav", source, "audio/wav"),
            "target_audio": ("target.flac", target, "audio/flac"),
        },
        data={"model": "test-vc"},
    )
    assert response.status_code == 200
    request = wire_client.engine.requests[-1]
    assert request.source_audio == base64.b64encode(source).decode()
    assert request.target_audio == base64.b64encode(target).decode()


def test_audio_convert_multipart_file_alias(wire_client, test_data_dir):
    response = wire_client.client.post(
        "/v1/audio/convert",
        files={"file": ("tone.mp3", sample_bytes("codec_tone.mp3"), "audio/mpeg")},
        data={"model": "test-ac"},
    )
    assert response.status_code == 200


def test_tts_multipart_reference_audio(wire_client, test_data_dir):
    b64 = base64.b64encode(_wav(test_data_dir)).decode()
    response = wire_client.client.post(
        "/v1/audio/speech",
        files={"input_audio": ("ref.wav", _wav(test_data_dir), "audio/wav")},
        data={"model": "test-tts", "mode": "voice_cloning", "input": "hello"},
    )
    assert response.status_code == 200
    assert wire_client.engine.requests[-1].input_audio == b64


def test_tts_multipart_accepts_nested_output_options(wire_client, test_data_dir):
    """Nested options travel as JSON strings in multipart forms."""
    response = wire_client.client.post(
        "/v1/audio/speech",
        files={"input_audio": ("ref.wav", _wav(test_data_dir), "audio/wav")},
        data={
            "model": "test-tts",
            "mode": "voice_cloning",
            "input": "hello",
            "output_options": '{"format": "flac", "sample_rate": 16000}',
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["sample_rate"] == 16000
    _, decoded_sr = audio_codec.decode_audio(base64.b64decode(body["data"]))
    assert decoded_sr == 16000


def test_tts_multipart_invalid_nested_json_returns_400(wire_client, test_data_dir):
    response = wire_client.client.post(
        "/v1/audio/speech",
        files={"input_audio": ("ref.wav", _wav(test_data_dir), "audio/wav")},
        data={
            "model": "test-tts",
            "mode": "voice_cloning",
            "input": "hello",
            "output_options": "{not json",
        },
    )
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# Typed errors through the envelope
# ---------------------------------------------------------------------------


def test_multipart_garbage_audio_returns_400(wire_client):
    response = wire_client.client.post(
        "/v1/audio/transcriptions",
        files={"file": ("bad.bin", b"this is not audio at all", "application/octet-stream")},
        data={"model": "test-stt"},
    )
    assert response.status_code == 400
    assert "Unable to decode input audio" in response.json()["message"]


def test_json_garbage_audio_returns_400(wire_client):
    response = wire_client.client.post(
        "/v1/audio/transcriptions",
        json={"model": "test-stt", "input_audio": base64.b64encode(b"garbage bytes").decode()},
    )
    assert response.status_code == 400
    assert "Unable to decode input audio" in response.json()["message"]


def test_malformed_json_body_returns_400(wire_client):
    response = wire_client.client.post(
        "/v1/audio/transcriptions",
        content=b"{not json",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# Upload size limit
# ---------------------------------------------------------------------------


def test_oversized_multipart_upload_returns_413(wire_client_small_limit):
    response = wire_client_small_limit.client.post(
        "/v1/audio/transcriptions",
        files={"file": ("big.wav", b"0" * 8192, "audio/wav")},
        data={"model": "test-stt"},
    )
    assert response.status_code == 413
    assert response.json()["type"] == "PayloadTooLargeError"


def test_oversized_json_body_returns_413(wire_client_small_limit):
    response = wire_client_small_limit.client.post(
        "/v1/audio/speech",
        json={"model": "test-tts", "mode": "single_speaker_tts", "input": "x" * 10000},
    )
    assert response.status_code == 413


async def test_parser_enforces_limit_without_content_length(monkeypatch):
    """Chunked-ish requests without Content-Length are still bounded by the parser."""
    boundary = "----hseboundary"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="big.wav"\r\n'
        "Content-Type: audio/wav\r\n\r\n"
    ).encode() + b"0" * 2048 + f"\r\n--{boundary}--\r\n".encode()

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/audio/transcriptions",
        "headers": [
            (b"content-type", f"multipart/form-data; boundary={boundary}".encode()),
        ],
        "app": SimpleNamespace(state=SimpleNamespace(max_upload_bytes=1024)),
    }

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    request = Request(scope, receive)
    with pytest.raises(UploadTooLargeError):
        await parse_dual_request(
            request,
            SpeechTranscribeRequest,
            upload_fields={"input_audio": ("file", "audio", "input_audio")},
        )


def test_resolve_max_upload_bytes_env(monkeypatch):
    monkeypatch.delenv("HARMONYSPEECH_MAX_UPLOAD_MB", raising=False)
    assert resolve_max_upload_bytes(None) == 50 * 1024 * 1024
    monkeypatch.setenv("HARMONYSPEECH_MAX_UPLOAD_MB", "7")
    assert resolve_max_upload_bytes(None) == 7 * 1024 * 1024


# ---------------------------------------------------------------------------
# OpenAPI documents both content types
# ---------------------------------------------------------------------------


def test_garbage_audio_is_400_on_every_audio_endpoint(wire_client):
    """A garbage payload never 500s or hangs: every audio endpoint returns a typed 400."""
    bad = base64.b64encode(b"definitely not audio").decode()
    cases = [
        (
            "/v1/audio/speech",
            {"model": "test-tts", "mode": "voice_cloning", "input": "hi", "input_audio": bad},
        ),
        ("/v1/audio/transcriptions", {"model": "test-stt", "input_audio": bad}),
        ("/v1/audio/vad", {"model": "test-vad", "input_audio": bad}),
        ("/v1/embed/speaker", {"model": "test-embed", "input_audio": bad}),
        ("/v1/voice/convert", {"model": "test-vc", "source_audio": bad, "target_audio": bad}),
        ("/v1/audio/convert", {"model": "test-ac", "source_audio": bad}),
    ]
    for path, payload in cases:
        response = wire_client.client.post(path, json=payload)
        assert response.status_code == 400, f"{path}: {response.text}"
        assert "Unable to decode input audio" in response.json()["message"], path


@pytest.mark.parametrize(
    "path",
    [
        "/v1/audio/speech",
        "/v1/audio/transcriptions",
        "/v1/audio/vad",
        "/v1/embed/speaker",
        "/v1/voice/convert",
        "/v1/audio/convert",
    ],
)
def test_openapi_advertises_json_and_multipart(wire_client, path):
    schema = wire_client.client.get("/openapi.json").json()
    request_body = schema["paths"][path]["post"]["requestBody"]
    assert "application/json" in request_body["content"]
    assert "multipart/form-data" in request_body["content"]


# ---------------------------------------------------------------------------
# Output options: admission validation
# ---------------------------------------------------------------------------


def _tts_payload(**overrides):
    payload = {"model": "test-tts", "mode": "single_speaker_tts", "input": "hello"}
    payload.update(overrides)
    return payload


def test_unsupported_output_format_rejected_before_synthesis(wire_client):
    before = len(wire_client.engine.requests)
    response = wire_client.client.post(
        "/v1/audio/speech", json=_tts_payload(output_options={"format": "wma"})
    )
    assert response.status_code == 400
    assert "Unsupported output format" in response.json()["message"]
    assert len(wire_client.engine.requests) == before  # engine never saw the request


@pytest.mark.parametrize("rate", [0, -16000, 12345])
def test_invalid_output_sample_rate_rejected_before_synthesis(wire_client, rate):
    before = len(wire_client.engine.requests)
    response = wire_client.client.post(
        "/v1/audio/speech", json=_tts_payload(output_options={"sample_rate": rate})
    )
    assert response.status_code == 400
    assert "sample_rate" in response.json()["message"]
    assert len(wire_client.engine.requests) == before


def test_stream_output_alias_is_admitted_not_the_old_rejection(wire_client):
    """Phase 14-1/14-3: `stream: true` is the deprecated alias for stream_format="audio".

    The admission contract is final and 14-3 serves the binary envelope, so an
    admitted streaming request streams (never the old 400 rejection, never a
    silent buffered 200 JSON body).
    """
    response = wire_client.client.post(
        "/v1/audio/speech", json=_tts_payload(output_options={"stream": True})
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/")
    assert response.content[:4] == b"RIFF"


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        ("/v1/voice/convert", {"model": "test-vc", "source_audio": "YXVkaW8="}),
        ("/v1/audio/convert", {"model": "test-ac", "source_audio": "YXVkaW8="}),
    ],
)
def test_conversion_endpoints_validate_output_options_at_admission(wire_client, path, payload):
    before = len(wire_client.engine.requests)
    response = wire_client.client.post(path, json={**payload, "output_options": {"format": "wma"}})
    assert response.status_code == 400
    assert len(wire_client.engine.requests) == before


# ---------------------------------------------------------------------------
# Output options: response conversion
# ---------------------------------------------------------------------------


def test_tts_default_output_is_byte_identical(wire_client):
    response = wire_client.client.post("/v1/audio/speech", json=_tts_payload())
    assert response.status_code == 200
    body = response.json()
    assert body["data"] == wav_b64()
    assert body["sample_rate"] is None


def test_tts_explicit_wav_output_is_byte_identical(wire_client):
    response = wire_client.client.post(
        "/v1/audio/speech", json=_tts_payload(output_options={"format": "wav"})
    )
    assert response.json()["data"] == wav_b64()


@pytest.mark.parametrize("fmt", ["flac", "ogg", "mp3"])
def test_tts_converted_output(wire_client, fmt):
    if fmt not in audio_codec.SUPPORTED_OUTPUT_FORMATS:
        pytest.skip(f"{fmt} not encodable by this libsndfile build")
    response = wire_client.client.post(
        "/v1/audio/speech", json=_tts_payload(output_options={"format": fmt, "sample_rate": 16000})
    )
    assert response.status_code == 200
    body = response.json()
    assert body["sample_rate"] == 16000
    decoded, decoded_sr = audio_codec.decode_audio(base64.b64decode(body["data"]))
    assert decoded_sr == 16000
    assert decoded.size > 0


def test_tts_pcm_output(wire_client):
    response = wire_client.client.post(
        "/v1/audio/speech", json=_tts_payload(output_options={"format": "pcm", "sample_rate": 8000})
    )
    body = response.json()
    assert body["sample_rate"] == 8000
    raw = base64.b64decode(body["data"])
    assert abs(len(raw) / 2 / 8000 - 1.0) < 0.05


def test_voice_conversion_applies_output_options(wire_client):
    response = wire_client.client.post(
        "/v1/voice/convert",
        json={
            "model": "test-vc",
            "source_audio": sample_b64("codec_tone.wav"),
            "output_options": {"format": "flac", "sample_rate": 22050},
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["sample_rate"] == 22050
    _, decoded_sr = audio_codec.decode_audio(base64.b64decode(body["data"]))
    assert decoded_sr == 22050


def test_audio_conversion_applies_output_options(wire_client):
    response = wire_client.client.post(
        "/v1/audio/convert",
        json={
            "model": "test-ac",
            "source_audio": sample_b64("codec_tone.wav"),
            "output_options": {"format": "ogg"},
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["sample_rate"] == 44100
    _, decoded_sr = audio_codec.decode_audio(base64.b64decode(body["data"]))
    assert decoded_sr == 44100


def test_default_response_has_no_sample_rate_for_stt(wire_client):
    """Non-audio responses keep their exact previous shape (no stray fields)."""
    response = wire_client.client.post(
        "/v1/audio/transcriptions",
        json={"model": "test-stt", "input_audio": sample_b64("codec_tone.wav")},
    )
    body = response.json()
    assert "sample_rate" not in body
    assert body["text"] == "hello world"
    assert json.dumps(body)  # serialisable

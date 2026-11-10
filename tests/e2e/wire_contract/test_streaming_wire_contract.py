"""Phase 14-5: streaming wire contract against a **live** ASGI server.

The unit suites drive the serving objects directly and the ``TestClient``
transport, which cannot observe transport-level framing (``TestClient`` buffers
the body, so ``Transfer-Encoding`` never appears). These tests boot the real
FastAPI app under uvicorn on a loopback socket with the boundary-decoding
``FakeEngine`` — hermetic, CPU-only, no model weights — and assert the bytes a
real HTTP client sees:

* binary envelope — ``Transfer-Encoding: chunked``, **no** ``Content-Length``,
  the canonical ``X-Audio-*`` set (target rate under resampling), the streaming
  WAV header fields (``0xFFFFFFFF`` sizes, target rate) and a single-chunk
  payload equal to the buffered path's canonical PCM;
* SSE envelope — ``text/event-stream; charset=utf-8``, delta ordering/shape
  (``audio`` = base64 s16le PCM), exactly one terminal event, ``data: [DONE]``
  last, and a keepalive tolerated by the parser (a real one over the wire);
* admission — the full 14-1 owner matrix through the HTTP surface;
* the 14-5 metrics series are present on ``/metrics`` and move when a stream
  completes.
"""

import base64
import json
import socket
import struct
import threading
import time

import httpx
import numpy as np
import pytest

from harmonyspeech.common import audio_codec
from harmonyspeech.endpoints.openai import serving_text_to_speech as serving_module
from harmonyspeech.endpoints.openai.streaming import (
    HEADER_ACCEL_BUFFERING,
    HEADER_AUDIO_BITS,
    HEADER_AUDIO_CHANNELS,
    HEADER_AUDIO_FORMAT,
    HEADER_AUDIO_SAMPLE_RATE,
    HEADER_CACHE_CONTROL,
    HEADER_REQUEST_ID,
    SSE_DONE_FRAME,
    SSE_EVENT_DELTA,
    SSE_EVENT_DONE,
    SSE_EVENT_ERROR,
    SSE_KEEPALIVE,
    SSE_MEDIA_TYPE,
    pcm16_frame,
)
from harmonyspeech.endpoints.openai.streaming_metrics import (
    get_stream_stats,
    reset_stream_metrics,
)
from tests.wire_contract_support import WireApp

pytestmark = pytest.mark.e2e

SAMPLE_RATE = 44100  # FakeEngine's codec_tone.wav


@pytest.fixture(scope="module")
def live_wire():
    """Run the wire-contract app under a real uvicorn server on loopback."""
    import uvicorn

    with WireApp(decode_inputs=True) as wire:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        sock.listen(2048)
        port = sock.getsockname()[1]
        config = uvicorn.Config(
            wire.app,
            host="127.0.0.1",
            port=port,
            log_level="warning",
            lifespan="off",
        )
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
        thread.start()
        deadline = time.time() + 15
        while not server.started and time.time() < deadline:
            time.sleep(0.01)
        assert server.started, "uvicorn did not start"
        try:
            yield wire, f"http://127.0.0.1:{port}"
        finally:
            server.should_exit = True
            thread.join(timeout=15)
            sock.close()


@pytest.fixture
def client(live_wire):
    _wire, base_url = live_wire
    with httpx.Client(base_url=base_url, timeout=30.0) as http:
        yield http


def _tts_payload(**overrides):
    payload = {
        "model": WireApp.MODEL_IDS["tts"],
        "mode": "single_speaker_tts",
        "input": "hello world",
    }
    payload.update(overrides)
    return payload


def _parse_sse(text: str) -> list[tuple[str, object]]:
    """Split an SSE body into ``(kind, payload)`` blocks, tolerating comments."""
    blocks = []
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        lines = block.split("\n")
        if lines[0].startswith(":"):
            blocks.append(("keepalive", None))
            continue
        data = "\n".join(line[len("data: ") :] for line in lines if line.startswith("data: "))
        if data == "[DONE]":
            blocks.append(("done-frame", None))
            continue
        event = next(
            (line[len("event: ") :] for line in lines if line.startswith("event: ")), None
        )
        blocks.append((event, json.loads(data)))
    return blocks


def _deltas(events) -> list[dict]:
    return [payload for event, payload in events if event == SSE_EVENT_DELTA]


def _buffered_pcm(client, output_options=None) -> tuple[bytes, int]:
    """The buffered path's canonical s16le PCM (pre-watermark, no watermarker)."""
    response = client.post("/v1/audio/speech", json=_tts_payload(output_options=output_options))
    assert response.status_code == 200, response.text
    raw = base64.b64decode(response.json()["data"])
    waveform, rate = audio_codec.decode_audio(raw)
    return pcm16_frame(waveform, rate), rate


def _metric_value(text: str, name: str, **labels) -> float | None:
    wanted = ""
    if labels:
        wanted = "{" + ",".join(f'{key}="{value}"' for key, value in labels.items()) + "}"
    prefix = f"{name}{wanted} "
    for line in text.splitlines():
        if line.startswith(prefix):
            return float(line[len(prefix) :])
    return None


# ---------------------------------------------------------------------------
# Binary envelope
# ---------------------------------------------------------------------------


def test_binary_wav_stream_is_chunked_without_content_length(client):
    with client.stream(
        "POST",
        "/v1/audio/speech",
        json=_tts_payload(output_options={"stream_format": "audio", "format": "wav"}),
    ) as response:
        assert response.status_code == 200
        assert response.headers.get("transfer-encoding", "").lower() == "chunked"
        assert "content-length" not in response.headers
        body = b"".join(response.iter_raw())

        assert response.headers[HEADER_AUDIO_FORMAT] == "wav"
        assert response.headers[HEADER_AUDIO_SAMPLE_RATE] == str(SAMPLE_RATE)
        assert response.headers[HEADER_AUDIO_CHANNELS] == "1"
        assert response.headers[HEADER_AUDIO_BITS] == "16"
        assert response.headers[HEADER_ACCEL_BUFFERING] == "no"
        assert response.headers[HEADER_CACHE_CONTROL] == "no-cache, no-transform"
        assert response.headers[HEADER_REQUEST_ID].startswith("tts-")

    # Streaming WAV header, field by field.
    assert body[:4] == b"RIFF"
    assert struct.unpack("<I", body[4:8])[0] == 0xFFFFFFFF
    assert body[8:12] == b"WAVE"
    assert body[12:16] == b"fmt "
    assert struct.unpack("<I", body[16:20])[0] == 16
    assert struct.unpack("<H", body[20:22])[0] == 1  # PCM
    assert struct.unpack("<H", body[22:24])[0] == 1  # mono
    assert struct.unpack("<I", body[24:28])[0] == SAMPLE_RATE
    assert struct.unpack("<I", body[28:32])[0] == SAMPLE_RATE * 2  # byte rate
    assert struct.unpack("<H", body[32:34])[0] == 2  # block align
    assert struct.unpack("<H", body[34:36])[0] == 16
    assert body[36:40] == b"data"
    assert struct.unpack("<I", body[40:44])[0] == 0xFFFFFFFF

    # Single-chunk text: the payload is the buffered path's canonical PCM.
    expected, _rate = _buffered_pcm(client)
    assert body[44:] == expected


def test_binary_pcm_stream_is_headerless_at_the_target_rate(client):
    with client.stream(
        "POST",
        "/v1/audio/speech",
        json=_tts_payload(
            output_options={"stream_format": "audio", "format": "pcm", "sample_rate": 16000}
        ),
    ) as response:
        assert response.status_code == 200
        assert response.headers.get("transfer-encoding", "").lower() == "chunked"
        assert "content-length" not in response.headers
        assert response.headers[HEADER_AUDIO_FORMAT] == "pcm"
        assert response.headers[HEADER_AUDIO_SAMPLE_RATE] == "16000"
        body = b"".join(response.iter_raw())

    assert body[:4] != b"RIFF"
    assert len(body) % 2 == 0
    assert len(body) == pytest.approx(16000 * 2, abs=64)

    # The buffered path at the same rate is the reference (soxr is shared).
    expected, rate = _buffered_pcm(client, {"format": "wav", "sample_rate": 16000})
    assert rate == 16000
    streamed = np.frombuffer(body, dtype="<i2").astype(np.int32)
    reference = np.frombuffer(expected, dtype="<i2").astype(np.int32)
    assert streamed.size == reference.size
    assert np.max(np.abs(streamed - reference)) <= 1


def test_binary_wav_header_carries_the_target_rate_under_resampling(client):
    with client.stream(
        "POST",
        "/v1/audio/speech",
        json=_tts_payload(
            output_options={"stream_format": "audio", "format": "wav", "sample_rate": 8000}
        ),
    ) as response:
        assert response.status_code == 200
        body = b"".join(response.iter_raw())

    assert response.headers[HEADER_AUDIO_SAMPLE_RATE] == "8000"
    assert struct.unpack("<I", body[24:28])[0] == 8000
    assert struct.unpack("<I", body[28:32])[0] == 8000 * 2


# ---------------------------------------------------------------------------
# SSE envelope
# ---------------------------------------------------------------------------


def test_sse_stream_event_contract(client):
    with client.stream(
        "POST",
        "/v1/audio/speech",
        json=_tts_payload(output_options={"stream_format": "sse", "format": "pcm"}),
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"] == SSE_MEDIA_TYPE
        assert response.headers[HEADER_AUDIO_FORMAT] == "pcm"
        assert response.headers[HEADER_AUDIO_SAMPLE_RATE] == str(SAMPLE_RATE)
        body = "".join(response.iter_text())

    assert body.endswith(SSE_DONE_FRAME)
    events = _parse_sse(body)
    kinds = [event for event, _payload in events]

    # Ordering: deltas first, exactly one terminal done, [DONE] last.
    assert kinds[0] == SSE_EVENT_DELTA
    assert kinds[-1] == "done-frame"
    assert kinds[-2] == SSE_EVENT_DONE
    assert kinds.count(SSE_EVENT_DONE) == 1
    assert SSE_EVENT_ERROR not in kinds

    deltas = _deltas(events)
    assert deltas
    assert [delta["chunk_index"] for delta in deltas] == list(range(len(deltas)))
    for delta in deltas:
        assert set(delta) == {"type", "audio", "chunk_index"}
        assert delta["type"] == SSE_EVENT_DELTA
        pcm = base64.b64decode(delta["audio"])
        assert pcm and len(pcm) % 2 == 0
        assert pcm[:4] != b"RIFF"

    done = [payload for event, payload in events if event == SSE_EVENT_DONE][0]
    assert done["format"] == "pcm"
    assert done["sample_rate"] == SAMPLE_RATE
    assert done["chunks"] == len(deltas)
    assert done["id"] == response.headers[HEADER_REQUEST_ID]
    assert done["usage"]["audio_seconds"] == pytest.approx(done["duration_s"])


def test_sse_parser_tolerates_a_keepalive_comment(client):
    """A ``:`` comment anywhere must not disturb event parsing."""
    with client.stream(
        "POST",
        "/v1/audio/speech",
        json=_tts_payload(output_options={"stream_format": "sse", "format": "pcm"}),
    ) as response:
        body = "".join(response.iter_text())

    plain = _parse_sse(body)
    with_keepalive = _parse_sse(SSE_KEEPALIVE + body)
    assert [event for event, _ in with_keepalive] == ["keepalive", *[event for event, _ in plain]]
    assert _deltas(with_keepalive) == _deltas(plain)


def test_sse_emits_a_real_keepalive_before_the_first_delta(client, live_wire, monkeypatch):
    """A slow first chunk over the real server produces a keepalive, then audio."""
    wire, _base_url = live_wire
    monkeypatch.setattr(serving_module, "SSE_KEEPALIVE_TIMEOUT_S", 0.01)
    original_add_request = wire.engine.add_request

    async def slow_add_request(*args, **kwargs):
        import asyncio

        await asyncio.sleep(0.05)
        return await original_add_request(*args, **kwargs)

    monkeypatch.setattr(wire.engine, "add_request", slow_add_request)

    with client.stream(
        "POST",
        "/v1/audio/speech",
        json=_tts_payload(output_options={"stream_format": "sse", "format": "pcm"}),
    ) as response:
        body = "".join(response.iter_text())

    events = _parse_sse(body)
    kinds = [event for event, _payload in events]
    assert kinds[0] == "keepalive"
    assert kinds.count("keepalive") >= 1
    assert len(_deltas(events)) >= 1
    assert kinds.count(SSE_EVENT_DONE) == 1
    assert kinds[-1] == "done-frame"


# ---------------------------------------------------------------------------
# Admission matrix (through the live HTTP surface)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", ["mp3", "flac"])
def test_stream_with_a_whole_buffer_format_is_a_400(client, fmt):
    response = client.post(
        "/v1/audio/speech",
        json=_tts_payload(output_options={"stream_format": "audio", "format": fmt}),
    )
    assert response.status_code == 400
    message = response.json()["message"]
    assert "wav" in message and "pcm" in message


def test_alias_and_explicit_stream_format_have_identical_admission(client):
    alias = client.post(
        "/v1/audio/speech",
        json=_tts_payload(output_options={"stream": True, "format": "mp3"}),
    )
    explicit = client.post(
        "/v1/audio/speech",
        json=_tts_payload(output_options={"stream_format": "audio", "format": "mp3"}),
    )
    assert alias.status_code == explicit.status_code == 400
    assert alias.json()["message"] == explicit.json()["message"]


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("trim_silence", True),
        ("fix_internal_silence", True),
        ("speed", 1.5),
    ],
)
def test_whole_buffer_generation_options_are_400s_naming_the_restriction(client, option, value):
    response = client.post(
        "/v1/audio/speech",
        json=_tts_payload(
            output_options={"stream_format": "audio", "format": "wav"},
            generation_options={option: value},
        ),
    )
    assert response.status_code == 400
    message = response.json()["message"]
    assert option in message
    assert "whole-buffer" in message


@pytest.mark.parametrize(
    ("output_options", "generation_options"),
    [
        ({"stream_format": "audio", "format": "wav"}, {"remove_dc_offset": True}),
        ({"stream_format": "audio", "format": "pcm", "sample_rate": 16000}, {}),
        ({"stream": True, "format": "wav"}, {"speed": 1}),
    ],
)
def test_composable_options_are_admitted_and_stream(client, output_options, generation_options):
    with client.stream(
        "POST",
        "/v1/audio/speech",
        json=_tts_payload(output_options=output_options, generation_options=generation_options),
    ) as response:
        assert response.status_code == 200
        assert response.headers.get("transfer-encoding", "").lower() == "chunked"
        first = next(response.iter_raw())
        assert not first.startswith(b"{")


# ---------------------------------------------------------------------------
# Metrics surface
# ---------------------------------------------------------------------------


def test_metrics_series_are_present_and_move_after_a_stream(client):
    reset_stream_metrics()
    idle = client.get("/metrics").text
    for outcome in ("completed", "aborted", "error"):
        assert _metric_value(idle, "hse_tts_stream_chunks_total", outcome=outcome) == 0
    assert _metric_value(idle, "hse_tts_stream_first_frame_seconds_count") == 0
    assert _metric_value(idle, "hse_tts_stream_bytes_total") == 0

    with client.stream(
        "POST",
        "/v1/audio/speech",
        json=_tts_payload(output_options={"stream_format": "audio", "format": "pcm"}),
    ) as response:
        assert response.status_code == 200
        b"".join(response.iter_raw())

    stats = get_stream_stats()
    assert stats["chunks"] == {"completed": 1, "aborted": 0, "error": 0}
    assert stats["first_frame"]["count"] == 1
    assert stats["bytes"] > 0

    after = client.get("/metrics").text
    assert _metric_value(after, "hse_tts_stream_chunks_total", outcome="completed") == 1
    assert _metric_value(after, "hse_tts_stream_first_frame_seconds_count") == 1
    assert _metric_value(after, "hse_tts_stream_bytes_total") > 0

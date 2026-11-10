"""Phase 14-4: the SSE envelope through the real serving path and engine loop.

Every test drives the real ``OpenAIServingTextToSpeech`` and the real
``AsyncHarmonySpeech`` loop through the 12-2 ``EngineHarness`` (stub models,
CPU-only, hermetic). What is pinned:

* the event stream is the *same audio* as the binary envelope — deltas carry
  base64 s16le PCM that round-trips to the exact ``pcm16_frame`` bytes, and a
  multi-chunk stream equals ``crossfade_stitch`` of the same chunk set;
* ordering: deltas → one ``speech.audio.done`` → ``data: [DONE]`` as the last
  line (header priming included);
* exactly one terminal event, ``done`` XOR ``error``: a failure after N deltas
  is one ``speech.audio.error`` (no ``done``, no ``[DONE]``), a failure before
  the first byte is an ordinary JSON error response;
* ``done`` metadata: ``duration_s`` from the frame generator's emitted-sample
  counter (never the resampled wire length), the **target** sample rate,
  ``format: "pcm"`` always, ``chunks`` = delta frames, extended ``usage``;
* keepalives with an injected threshold, before the first delta and between
  deltas (never a 15 s sleep);
* a client disconnect cancels through the SSE branch exactly like the binary
  one: the in-flight chunk is aborted and the rest is never submitted.
"""

import asyncio
import base64
import json

import numpy as np
import pytest

from harmonyspeech.common import audio_codec
from harmonyspeech.common.audio_utils import crossfade_stitch
from harmonyspeech.endpoints.openai import serving_text_to_speech as serving_module
from harmonyspeech.endpoints.openai.protocol import AudioOutputOptions, ErrorResponse
from harmonyspeech.endpoints.openai.serving_text_to_speech import (
    StreamingAudioResponse,
)
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
from tests.integration.test_tts_chunking import (
    RAW,
    SAMPLE_RATE,
    ChatterboxStubModel,
    EngineHarness,
    _chatterbox_harness,
    _model_config,
    _request,
    _speak,
)
from tests.integration.test_tts_streaming import (
    CHUNK_SIZE,
    SENTENCE_A,
    SENTENCE_B,
    SENTENCE_C,
    TEXT_3,
    _chunk_waveform,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sse_request(text: str, *, model: str = "chatterbox", output: dict | None = None, **options):
    """A TTS request that selects the SSE envelope (default: pcm)."""
    output_options = {"stream_format": "sse", "format": "pcm"}
    output_options.update(output or {})
    request = _request(text, model=model, **options)
    return request.model_copy(update={"output_options": AudioOutputOptions(**output_options)})


async def _collect(response) -> str:
    assert isinstance(response, StreamingAudioResponse), response
    return "".join([frame async for frame in response.body])


def _parse_sse(text: str) -> list[tuple[str, object]]:
    """Split an SSE body into ``(kind, payload)`` blocks."""
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


def _done(events) -> dict:
    done = [payload for event, payload in events if event == SSE_EVENT_DONE]
    assert len(done) == 1, events
    return done[0]


def _delta_pcm(events) -> bytes:
    return b"".join(base64.b64decode(delta["audio"]) for delta in _deltas(events))


# ---------------------------------------------------------------------------
# Ordering, headers and the round-trip
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sse_stream_orders_deltas_done_then_done_frame():
    """Header priming → deltas → one done → ``data: [DONE]`` as the last line."""
    harness, _model = _chatterbox_harness()
    text = "A short single-chunk sentence."
    async with harness:
        response = await _speak(harness, _sse_request(text))
        body = await _collect(response)

    assert response.media_type == SSE_MEDIA_TYPE
    # The full 14-1 header set rides along; SSE always names headerless PCM.
    assert response.headers[HEADER_AUDIO_FORMAT] == "pcm"
    assert response.headers[HEADER_AUDIO_SAMPLE_RATE] == str(SAMPLE_RATE)
    assert response.headers[HEADER_AUDIO_CHANNELS] == "1"
    assert response.headers[HEADER_AUDIO_BITS] == "16"
    assert response.headers[HEADER_ACCEL_BUFFERING] == "no"
    assert response.headers[HEADER_CACHE_CONTROL] == "no-cache, no-transform"

    events = _parse_sse(body)
    kinds = [event for event, _payload in events]
    assert kinds[0] == SSE_EVENT_DELTA  # first byte = first audio frame
    assert kinds[-1] == "done-frame"
    assert kinds[-2] == SSE_EVENT_DONE
    assert kinds.count(SSE_EVENT_DONE) == 1
    assert SSE_EVENT_ERROR not in kinds
    assert body.endswith(SSE_DONE_FRAME)

    done = _done(events)
    assert response.headers[HEADER_REQUEST_ID] == done["id"]
    assert done["id"].startswith("tts-")
    assert done["model"] == "chatterbox"
    assert done["format"] == "pcm"
    assert done["sample_rate"] == SAMPLE_RATE
    assert done["chunks"] == len(_deltas(events))
    assert [delta["chunk_index"] for delta in _deltas(events)] == list(
        range(len(_deltas(events)))
    )
    # ``usage`` is extended additively.
    assert done["usage"]["input_characters"] == len(text)
    assert done["usage"]["audio_seconds"] == pytest.approx(done["duration_s"])


@pytest.mark.integration
@pytest.mark.asyncio
async def test_delta_audio_round_trips_to_the_exact_pcm_bytes():
    """One chunk: the concatenated deltas are the buffered path's PCM exactly."""
    harness, _model = _chatterbox_harness()
    text = "A short single-chunk sentence."
    async with harness:
        response = await _speak(harness, _sse_request(text))
        body = await _collect(response)

    events = _parse_sse(body)
    assert _delta_pcm(events) == pcm16_frame(_chunk_waveform(text), SAMPLE_RATE)
    assert _done(events)["duration_s"] == pytest.approx(
        _chunk_waveform(text).size / SAMPLE_RATE
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_multi_chunk_sse_stream_matches_the_whole_buffer_stitch():
    """The envelope wraps the same generator: ordering + slicing are identical."""
    harness, model = _chatterbox_harness()
    request = _sse_request(
        TEXT_3,
        split_text=True,
        chunk_size=CHUNK_SIZE,
        crossfade_ms=8,
        sentence_pause_ms=40,
    )
    async with harness:
        response = await _speak(harness, request)
        body = await _collect(response)

    assert model.generate_calls == [SENTENCE_A, SENTENCE_B, SENTENCE_C]
    expected = crossfade_stitch(
        [_chunk_waveform(text) for text in model.generate_calls],
        crossfade_ms=8,
        sentence_pause_ms=40,
        sample_rate=SAMPLE_RATE,
    )
    events = _parse_sse(body)
    assert _delta_pcm(events) == pcm16_frame(expected, SAMPLE_RATE)
    assert response.state is not None
    assert response.state.chunks == 3
    done = _done(events)
    # The stitcher's flush tail is one more delta than synthesis chunks.
    assert done["chunks"] == len(_deltas(events)) == 4
    assert done["duration_s"] == pytest.approx(expected.size / SAMPLE_RATE)
    assert response.state.duration_s == pytest.approx(done["duration_s"])


# ---------------------------------------------------------------------------
# Resampling: target rate in metadata, source-rate duration
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_done_reports_target_rate_and_source_sample_duration_under_resampling():
    harness, model = _chatterbox_harness()
    target = 16000
    # ``format: "wav"`` is admitted for streaming, but SSE still emits PCM.
    request = _sse_request(
        TEXT_3,
        split_text=True,
        chunk_size=CHUNK_SIZE,
        crossfade_ms=8,
        sentence_pause_ms=0,
        output={"stream_format": "sse", "format": "wav", "sample_rate": target},
    )
    async with harness:
        response = await _speak(harness, request)
        body = await _collect(response)

    assert response.headers[HEADER_AUDIO_FORMAT] == "pcm"
    assert response.headers[HEADER_AUDIO_SAMPLE_RATE] == str(target)

    events = _parse_sse(body)
    done = _done(events)
    assert done["sample_rate"] == target
    assert done["format"] == "pcm"

    stitched = crossfade_stitch(
        [_chunk_waveform(text) for text in model.generate_calls],
        crossfade_ms=8,
        sentence_pause_ms=0,
        sample_rate=SAMPLE_RATE,
    )
    resampled = audio_codec.resample(stitched, SAMPLE_RATE, target)
    pcm = _delta_pcm(events)
    reference = pcm16_frame(resampled, target)
    # Frames are at the target rate (bit-identical to one-shot soxr, 1 LSB).
    assert len(pcm) == len(reference)
    streamed = np.frombuffer(pcm, dtype="<i2").astype(np.int32)
    expected = np.frombuffer(reference, dtype="<i2").astype(np.int32)
    assert np.max(np.abs(streamed - expected)) <= 1

    # duration_s is the emitted *source* samples ÷ source rate — the resampled
    # wire length differs, so it can never be the source of the number.
    assert done["duration_s"] == pytest.approx(stitched.size / SAMPLE_RATE)
    assert response.state is not None
    assert response.state.emitted_samples == stitched.size
    assert len(pcm) // 2 != response.state.emitted_samples
    assert done["usage"]["audio_seconds"] == pytest.approx(done["duration_s"])


# ---------------------------------------------------------------------------
# Terminal-event semantics (done XOR error)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_error_after_deltas_emits_exactly_one_error_and_no_done():
    """A mid-stream failure is an event, not a truncated 200 body."""
    harness, model = _chatterbox_harness(fail_on="FAIL")
    # chunk 0 = SENTENCE_A (long enough to clear the 40 ms carry), chunk 1
    # = "FAIL." — it fails only after chunk 0's delta is on the wire.
    text = f"{SENTENCE_A} FAIL. {SENTENCE_B}"
    async with harness:
        response = await _speak(
            harness, _sse_request(text, split_text=True, chunk_size=87)
        )
        assert isinstance(response, StreamingAudioResponse), response
        body = await _collect(response)

    events = _parse_sse(body)
    kinds = [event for event, _payload in events]
    assert kinds.count(SSE_EVENT_DELTA) >= 1
    assert kinds.count(SSE_EVENT_ERROR) == 1
    assert SSE_EVENT_DONE not in kinds
    assert not any(kind == "done-frame" for kind in kinds)
    assert kinds[-1] == SSE_EVENT_ERROR

    error = [payload for event, payload in events if event == SSE_EVENT_ERROR][0]
    assert error["error"]["type"] == "StreamSynthesisError"
    assert "synthetic failure" in error["error"]["message"]
    assert error["error"]["code"] is None
    # Chunk 1 failed; chunk 2 was never submitted.
    assert model.generate_calls == [SENTENCE_A]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_first_chunk_failure_is_a_plain_json_error():
    """Nothing on the wire yet ⇒ headers never commit, normal HTTP error."""
    harness, model = _chatterbox_harness(fail_on="FAIL")
    async with harness:
        response = await _speak(harness, _sse_request("FAIL."))

    assert isinstance(response, ErrorResponse), response
    assert response.code == 400
    assert "synthetic failure" in response.message
    assert model.generate_calls == []


# ---------------------------------------------------------------------------
# Keepalive
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_keepalive_before_the_first_delta_and_between_deltas(monkeypatch):
    """An injected threshold keeps the stream warm; no 15 s sleep in tests."""
    monkeypatch.setattr(serving_module, "SSE_KEEPALIVE_TIMEOUT_S", 0.01)

    config = _model_config("chatterbox", "ChatterboxTTS")
    model = ChatterboxStubModel()
    harness = EngineHarness([config], {"chatterbox": model}, executor_delay_s=0.05)
    async with harness:
        response = await _speak(
            harness,
            _sse_request(TEXT_3, split_text=True, chunk_size=CHUNK_SIZE),
        )
        assert isinstance(response, StreamingAudioResponse), response
        # Priming: the first frame is a keepalive, so the connection is already
        # warm (and committed) while chunk 0 synthesises.
        first = await anext(response.body)
        rest = [frame async for frame in response.body]

    assert first == SSE_KEEPALIVE
    events = _parse_sse(first + "".join(rest))
    kinds = [event for event, _payload in events]
    assert kinds[0] == "keepalive"

    first_delta = kinds.index(SSE_EVENT_DELTA)
    assert "keepalive" in kinds[1:first_delta]
    # A keepalive also lands between two deltas (a later chunk's wait).
    deltas = [index for index, kind in enumerate(kinds) if kind == SSE_EVENT_DELTA]
    assert len(deltas) >= 2
    assert "keepalive" in kinds[deltas[0] + 1 : deltas[1]]
    # The keepalives never consume or drop a frame, and the stream completes.
    assert len(_deltas(events)) == 4
    assert kinds[-2:] == [SSE_EVENT_DONE, "done-frame"]


# ---------------------------------------------------------------------------
# Disconnect
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_disconnect_aborts_in_flight_chunk_through_the_sse_branch():
    """A closed tab stops synthesis: abort the in-flight chunk, submit no more."""
    config = _model_config("chatterbox", "ChatterboxTTS")
    model = ChatterboxStubModel()
    harness = EngineHarness([config], {"chatterbox": model}, executor_delay_s=0.15)
    async with harness:
        serving = harness.serving
        aborts: list[str] = []
        submissions: list[str] = []

        original_abort = harness.engine.abort

        async def spy_abort(request_id):
            aborts.append(request_id)
            return await original_abort(request_id)

        original_submit = serving._submit_chunk

        async def spy_submit(request_id, request_data, artifacts):
            submissions.append(request_id)
            return await original_submit(request_id, request_data, artifacts)

        harness.engine.abort = spy_abort
        serving._submit_chunk = spy_submit

        request = _sse_request(TEXT_3, split_text=True, chunk_size=CHUNK_SIZE)
        response = await serving.create_text_to_speech(request, RAW)

        # Priming synthesised chunk 0 only: the first delta is already in hand.
        assert isinstance(response, StreamingAudioResponse), response
        assert len(submissions) == 1
        first = await anext(response.body)
        assert first.startswith(f"event: {SSE_EVENT_DELTA}")

        # Resume: chunk 1 is submitted and its synthesis is in flight.
        pending = asyncio.ensure_future(anext(response.body))
        await asyncio.sleep(0.02)
        assert len(submissions) == 2
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        await response.body.aclose()

    # The in-flight chunk was aborted, and chunks 2.. were never submitted.
    assert aborts == [submissions[1]]
    assert submissions[1].endswith("-chunk-1")
    assert len(submissions) == 2
    assert model.generate_calls == [SENTENCE_A]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_polled_disconnect_stops_before_the_next_chunk_is_submitted():
    """The per-chunk ``is_disconnected`` poll is the second line of defence."""

    class _DisconnectAfterFirstCheck:
        def __init__(self):
            self.calls = 0

        async def is_disconnected(self):
            self.calls += 1
            return self.calls > 1

    harness, model = _chatterbox_harness()
    raw = _DisconnectAfterFirstCheck()
    async with harness:
        response = await harness.serving.create_text_to_speech(
            _sse_request(TEXT_3, split_text=True, chunk_size=CHUNK_SIZE), raw
        )
        assert isinstance(response, StreamingAudioResponse), response
        frames = [frame async for frame in response.body]

    events = _parse_sse("".join(frames))
    # Chunk 0 was in hand when the poll turned true; chunk 1 was never
    # submitted. The frame source ends *normally* here (it polls, it does not
    # raise), so the envelope cannot tell this apart from a completed stream
    # and still writes its one terminal frame — to a connection that is gone.
    assert model.generate_calls == [SENTENCE_A]
    assert len(_deltas(events)) == 1
    assert not any(event == SSE_EVENT_ERROR for event, _payload in events)
    assert [event for event, _payload in events].count(SSE_EVENT_DONE) == 1

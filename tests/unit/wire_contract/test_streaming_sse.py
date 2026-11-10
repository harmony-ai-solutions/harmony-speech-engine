"""Phase 14-4: the SSE envelope's event stream (``stream_format: "sse"``).

The engine-facing half (real chunks through the real engine loop) is pinned by
``tests/integration/test_tts_streaming_sse.py``; here the pure envelope is
driven with synthetic float32 segments. What is pinned:

* ordering — deltas in frame order, one ``speech.audio.done``, then
  ``data: [DONE]`` as the very last line (header priming is the caller's job);
* the delta ``audio`` field is base64 s16le PCM that round-trips to the exact
  bytes the one canonical ``pcm16_frame`` produces, with a 0-based
  ``chunk_index``;
* exactly one terminal event — ``done`` XOR ``error`` — and a pre-first-frame
  failure is re-raised so the caller can still answer with a plain JSON error;
* ``done`` fields: ``duration_s`` from the emitted-sample counter, the
  **target** rate under resampling, ``format`` always ``"pcm"``, ``chunks`` =
  delta frames sent, ``usage`` extended additively;
* keepalives — a timeout race around the frame wait, before the first delta
  too, with the module-level threshold overridable;
* cancellation (client disconnect) is not converted into an error event.
"""

import asyncio
import base64
import json
import threading

import numpy as np
import pytest

from harmonyspeech.endpoints.openai import serving_text_to_speech as serving_module
from harmonyspeech.endpoints.openai.serving_text_to_speech import (
    SSE_KEEPALIVE_TIMEOUT_S,
    StreamState,
    StreamSynthesisError,
    _encode_sse_frames,
    _pcm16_base64,
)
from harmonyspeech.endpoints.openai.streaming import (
    SSE_DONE_FRAME,
    SSE_EVENT_DELTA,
    SSE_EVENT_DONE,
    SSE_EVENT_ERROR,
    SSE_KEEPALIVE,
    SSE_MEDIA_TYPE,
    pcm16_frame,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _segments(items):
    for item in items:
        yield item


async def _failing_after(items, message="synthetic stream failure"):
    for item in items:
        yield item
    raise StreamSynthesisError(message)


async def _failing_immediately(message="synthetic first-frame failure"):
    raise StreamSynthesisError(message)
    yield np.empty(0, dtype=np.float32)  # pragma: no cover - unreachable


def _parse_sse(text: str) -> list[tuple[str, object]]:
    """Split an SSE body into ``(kind, payload)`` blocks.

    ``kind`` is the ``event:`` name, ``"keepalive"`` for a ``:`` comment line,
    or ``"done-frame"`` for ``data: [DONE]``.
    """
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


def _events(frames) -> list[tuple[str, object]]:
    return _parse_sse("".join(frames))


async def _render(
    segments,
    *,
    state: StreamState | None = None,
    request_id: str = "tts-sse",
    model: str = "chatterbox",
    created: int = 1_700_000_000,
    usage=None,
    keepalive_timeout: float | None = None,
) -> list[tuple[str, object]]:
    state = state if state is not None else StreamState(source_rate=24000, target_rate=24000)
    frames = [
        frame
        async for frame in _encode_sse_frames(
            segments,
            state=state,
            request_id=request_id,
            model=model,
            created=created,
            usage=usage,
            keepalive_timeout=keepalive_timeout,
        )
    ]
    return _events(frames)


def _payloads(events, kind):
    return [payload for event, payload in events if event == kind]


# ---------------------------------------------------------------------------
# Deltas + ordering
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_delta_audio_round_trips_to_the_exact_pcm_bytes():
    first = np.full(8, 0.25, dtype=np.float32)
    second = np.full(4, -0.5, dtype=np.float32)
    state = StreamState(source_rate=24000, target_rate=24000, emitted_samples=12, chunks=2)

    events = await _render(_segments([first, second]), state=state)
    deltas = _payloads(events, SSE_EVENT_DELTA)

    assert [delta["chunk_index"] for delta in deltas] == [0, 1]
    assert base64.b64decode(deltas[0]["audio"]) == pcm16_frame(first, 24000)
    assert base64.b64decode(deltas[1]["audio"]) == pcm16_frame(second, 24000)
    assert all(set(delta) == {"type", "audio", "chunk_index"} for delta in deltas)
    assert all(delta["type"] == SSE_EVENT_DELTA for delta in deltas)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_event_ordering_is_deltas_then_one_done_then_done_frame():
    segments = [np.full(4, 0.1 * (index + 1), dtype=np.float32) for index in range(3)]
    state = StreamState(source_rate=8000, target_rate=8000, emitted_samples=12, chunks=3)

    frames = [
        frame
        async for frame in _encode_sse_frames(
            _segments(segments), state=state, request_id="tts-order", model="m", created=7
        )
    ]

    assert frames[-1] == SSE_DONE_FRAME
    events = _events(frames)
    kinds = [event for event, _payload in events]
    assert kinds == [SSE_EVENT_DELTA, SSE_EVENT_DELTA, SSE_EVENT_DELTA, SSE_EVENT_DONE, "done-frame"]
    # No keepalive on a fast stream: the threshold is far above a test's runtime.
    assert SSE_KEEPALIVE not in "".join(frames)
    assert kinds.count(SSE_EVENT_DONE) == 1
    assert SSE_EVENT_ERROR not in kinds


@pytest.mark.unit
@pytest.mark.asyncio
async def test_empty_segments_are_skipped_and_do_not_consume_a_chunk_index():
    empty = np.empty(0, dtype=np.float32)
    first = np.full(4, 0.25, dtype=np.float32)
    second = np.full(4, -0.25, dtype=np.float32)
    state = StreamState(source_rate=8000, target_rate=8000, emitted_samples=8, chunks=2)

    events = await _render(_segments([empty, first, empty, second]), state=state)
    deltas = _payloads(events, SSE_EVENT_DELTA)

    assert [delta["chunk_index"] for delta in deltas] == [0, 1]
    assert base64.b64decode(deltas[0]["audio"]) == pcm16_frame(first, 8000)
    assert base64.b64decode(deltas[1]["audio"]) == pcm16_frame(second, 8000)
    # ``chunks`` counts the frames the client actually received.
    assert _payloads(events, SSE_EVENT_DONE)[0]["chunks"] == 2


# ---------------------------------------------------------------------------
# Done payload
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_done_payload_reports_target_rate_and_emitted_sample_duration():
    segment = np.full(4, 0.1, dtype=np.float32)
    state = StreamState(
        source_rate=24000, target_rate=16000, emitted_samples=24000, chunks=3
    )

    events = await _render(
        _segments([segment]),
        state=state,
        request_id="tts-done",
        model="chatterbox-multilingual",
        created=1_700_000_123,
        usage={"input_characters": 11},
    )
    done = _payloads(events, SSE_EVENT_DONE)[0]

    assert done["type"] == SSE_EVENT_DONE
    assert done["id"] == "tts-done"
    assert done["model"] == "chatterbox-multilingual"
    assert done["created"] == 1_700_000_123
    # Target rate under resampling, never the source rate.
    assert done["sample_rate"] == 16000
    # SSE audio is headerless PCM even when the request asked for wav.
    assert done["format"] == "pcm"
    # One delta frame was sent, not the planned synthesis chunk count.
    assert done["chunks"] == 1
    # duration_s is the emitted-sample counter ÷ source rate (24k samples at
    # 24 kHz = 1 s), NOT the resampled wire length ÷ target rate.
    assert done["duration_s"] == pytest.approx(1.0)
    assert done["usage"] == {"input_characters": 11, "audio_seconds": pytest.approx(1.0)}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_done_usage_is_extended_with_zero_input_characters_when_absent():
    state = StreamState(source_rate=8000, target_rate=8000, emitted_samples=8000, chunks=1)
    events = await _render(_segments([np.full(4, 0.1, dtype=np.float32)]), state=state)
    done = _payloads(events, SSE_EVENT_DONE)[0]

    assert done["usage"] == {"audio_seconds": pytest.approx(1.0), "input_characters": 0}
    assert done["duration_s"] == pytest.approx(1.0)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_done_never_mutates_the_caller_usage_mapping():
    usage = {"input_characters": 5}
    state = StreamState(source_rate=8000, target_rate=8000, emitted_samples=8000, chunks=1)
    await _render(_segments([np.full(4, 0.1, dtype=np.float32)]), state=state, usage=usage)

    assert usage == {"input_characters": 5}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_done_duration_is_zero_without_a_source_rate():
    state = StreamState()
    events = await _render(_segments([np.full(4, 0.1, dtype=np.float32)]), state=state)
    done = _payloads(events, SSE_EVENT_DONE)[0]

    assert done["duration_s"] == 0.0
    assert done["sample_rate"] == 0


# ---------------------------------------------------------------------------
# Terminal-event semantics (done XOR error)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_error_after_a_delta_is_one_terminal_error_event_and_no_done():
    segment = np.full(4, 0.1, dtype=np.float32)
    state = StreamState(source_rate=8000, target_rate=8000, emitted_samples=4, chunks=3)

    events = await _render(_failing_after([segment], "chunk 1 failed"), state=state)
    kinds = [event for event, _payload in events]

    assert kinds == [SSE_EVENT_DELTA, SSE_EVENT_ERROR]
    assert kinds.count(SSE_EVENT_ERROR) == 1
    assert SSE_EVENT_DONE not in kinds
    error = _payloads(events, SSE_EVENT_ERROR)[0]
    assert error == {
        "error": {
            "message": "chunk 1 failed",
            "type": "StreamSynthesisError",
            "code": None,
        }
    }


@pytest.mark.unit
@pytest.mark.asyncio
async def test_error_event_is_not_followed_by_the_done_frame():
    events = await _render(_failing_after([np.full(4, 0.1, dtype=np.float32)]))
    assert events[-1][0] == SSE_EVENT_ERROR
    assert not any(event == "done-frame" for event, _payload in events)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_failure_before_the_first_frame_is_raised_not_an_event():
    state = StreamState(source_rate=8000, target_rate=8000)
    generator = _encode_sse_frames(
        _failing_immediately("nothing on the wire yet"),
        state=state,
        request_id="tts-x",
        model="m",
        created=0,
    )

    with pytest.raises(StreamSynthesisError, match="nothing on the wire yet"):
        await anext(generator)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_a_keepalive_counts_as_the_first_byte_for_error_semantics():
    """After a keepalive the stream is committed, so a failure is an event."""

    async def _slow_then_fail():
        await asyncio.sleep(0.05)
        raise StreamSynthesisError("late failure")
        yield np.empty(0, dtype=np.float32)  # pragma: no cover - unreachable

    events = await _render(_slow_then_fail(), keepalive_timeout=0.01)

    assert events[0] == ("keepalive", None)
    assert events[-1][0] == SSE_EVENT_ERROR
    assert {event for event, _payload in events} == {"keepalive", SSE_EVENT_ERROR}
    assert _payloads(events, SSE_EVENT_ERROR)[0]["error"]["message"] == "late failure"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cancellation_propagates_and_is_never_an_error_event():
    entered = asyncio.Event()

    async def _one_then_block():
        yield np.full(4, 0.1, dtype=np.float32)
        entered.set()
        await asyncio.sleep(30)
        yield np.full(4, 0.1, dtype=np.float32)  # pragma: no cover - cancelled

    state = StreamState(source_rate=8000, target_rate=8000)
    generator = _encode_sse_frames(
        _one_then_block(), state=state, request_id="tts-cancel", model="m", created=0
    )
    first = await anext(generator)
    assert first.startswith(f"event: {SSE_EVENT_DELTA}")

    pending = asyncio.ensure_future(anext(generator))
    await entered.wait()
    pending.cancel()
    # A disconnect is a cancellation, not a mid-stream error event.
    with pytest.raises(asyncio.CancelledError):
        await pending
    await generator.aclose()


# ---------------------------------------------------------------------------
# Keepalive
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_keepalive_threshold_default_is_15_seconds():
    assert SSE_KEEPALIVE_TIMEOUT_S == 15.0
    assert SSE_KEEPALIVE == ": keepalive\n\n"
    assert SSE_MEDIA_TYPE == "text/event-stream; charset=utf-8"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_keepalive_is_emitted_before_the_first_delta():
    async def _slow_first():
        await asyncio.sleep(0.06)
        yield np.full(4, 0.1, dtype=np.float32)

    state = StreamState(source_rate=8000, target_rate=8000, emitted_samples=4, chunks=1)
    events = await _render(_slow_first(), state=state, keepalive_timeout=0.01)
    kinds = [event for event, _payload in events]

    assert kinds[0] == "keepalive"
    assert kinds.count("keepalive") >= 1
    assert kinds[-2:] == [SSE_EVENT_DONE, "done-frame"]
    assert kinds.index(SSE_EVENT_DELTA) > 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_keepalive_is_emitted_mid_stream_and_never_drops_a_frame():
    async def _slow_pair():
        yield np.full(4, 0.1, dtype=np.float32)
        await asyncio.sleep(0.06)
        yield np.full(4, -0.1, dtype=np.float32)

    state = StreamState(source_rate=8000, target_rate=8000, emitted_samples=8, chunks=2)
    events = await _render(_slow_pair(), state=state, keepalive_timeout=0.01)
    kinds = [event for event, _payload in events]

    first_delta = kinds.index(SSE_EVENT_DELTA)
    second_delta = kinds.index(SSE_EVENT_DELTA, first_delta + 1)
    # A keepalive sits between the two deltas, and both frames arrived intact.
    assert "keepalive" in kinds[first_delta + 1 : second_delta]
    assert len(_payloads(events, SSE_EVENT_DELTA)) == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_module_level_threshold_is_resolved_at_call_time(monkeypatch):
    """The test seam: patching the module constant needs no 15 s sleep."""
    monkeypatch.setattr(serving_module, "SSE_KEEPALIVE_TIMEOUT_S", 0.01)

    async def _slow_first():
        await asyncio.sleep(0.06)
        yield np.full(4, 0.1, dtype=np.float32)

    state = StreamState(source_rate=8000, target_rate=8000, emitted_samples=4, chunks=1)
    events = await _render(_slow_first(), state=state)  # keepalive_timeout omitted

    assert [event for event, _payload in events][0] == "keepalive"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_a_non_positive_threshold_disables_keepalives():
    async def _slow_first():
        await asyncio.sleep(0.02)
        yield np.full(4, 0.1, dtype=np.float32)

    state = StreamState(source_rate=8000, target_rate=8000, emitted_samples=4, chunks=1)
    events = await _render(_slow_first(), state=state, keepalive_timeout=0)

    assert not any(event == "keepalive" for event, _payload in events)
    assert [event for event, _payload in events] == [SSE_EVENT_DELTA, SSE_EVENT_DONE, "done-frame"]


# ---------------------------------------------------------------------------
# Worker-thread offload
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_pcm16_base64_is_the_one_canonical_quantizer():
    segment = np.full(4, 0.25, dtype=np.float32)
    assert _pcm16_base64(segment, 8000) == base64.b64encode(pcm16_frame(segment, 8000)).decode(
        "ascii"
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_quantization_runs_on_a_worker_thread(monkeypatch):
    calls: list[int] = []
    real = serving_module.pcm16_frame

    def recording(audio, sample_rate=None):
        calls.append(threading.get_ident())
        return real(audio, sample_rate)

    monkeypatch.setattr(serving_module, "pcm16_frame", recording)
    state = StreamState(source_rate=8000, target_rate=8000, emitted_samples=4, chunks=1)

    await _render(_segments([np.full(4, 0.25, dtype=np.float32)]), state=state)

    assert calls and all(ident != threading.get_ident() for ident in calls)

"""Phase 14-5: streaming-TTS metrics (renderer + observation points).

The series follow the 7-3/9-2/10-1 house style: a pure text renderer with every
series present on every scrape (zeros when idle), fed by a process-global
recorder that the streaming serving layer updates.

What is pinned:

* the idle shape — all three series, all three outcome labels, zeros;
* the first-frame histogram bucket/sum/count arithmetic;
* outcome and byte counters;
* the encoders observe the first **audio** frame (a keepalive is not audio),
  count completed streams on normal exhaustion, ``aborted`` on a cancelled /
  closed generator or a polled disconnect (``StreamState.aborted``), and
  ``error`` on a mid-stream failure;
* SSE byte accounting is raw s16le, not the larger base64 event text.
"""

import asyncio

import numpy as np
import pytest

from harmonyspeech.endpoints.openai.serving_text_to_speech import (
    StreamState,
    StreamSynthesisError,
    _encode_sse_frames,
    _encode_stream_frames,
)
from harmonyspeech.endpoints.openai.streaming_metrics import (
    STREAM_FIRST_FRAME_BUCKETS,
    STREAM_OUTCOMES,
    get_stream_stats,
    observe_first_frame,
    record_stream_bytes,
    record_stream_outcome,
    render_stream_metrics,
    reset_stream_metrics,
)


@pytest.fixture(autouse=True)
def _clean_metrics():
    reset_stream_metrics()
    yield
    reset_stream_metrics()


async def _segments(items):
    for item in items:
        yield item


async def _failing_after(items, message="synthetic stream failure"):
    for item in items:
        yield item
    raise StreamSynthesisError(message)


async def _one_then_block():
    yield np.full(8, 0.25, dtype=np.float32)
    await asyncio.sleep(30)
    yield np.full(8, 0.25, dtype=np.float32)  # pragma: no cover - cancelled


def _metric_value(text: str, name: str, **labels) -> float | None:
    """Return a sample value from a Prometheus text body (exact label match)."""
    wanted = ""
    if labels:
        wanted = "{" + ",".join(f'{key}="{value}"' for key, value in labels.items()) + "}"
    prefix = f"{name}{wanted} "
    for line in text.splitlines():
        if line.startswith(prefix):
            return float(line[len(prefix) :])
    return None


# ---------------------------------------------------------------------------
# Renderer / recorder
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_idle_renderer_has_every_series_and_zero_values():
    body = render_stream_metrics(None)

    assert "# TYPE hse_tts_stream_first_frame_seconds histogram" in body
    assert "# TYPE hse_tts_stream_chunks_total counter" in body
    assert "# TYPE hse_tts_stream_bytes_total counter" in body
    assert _metric_value(body, "hse_tts_stream_first_frame_seconds_count") == 0
    assert _metric_value(body, "hse_tts_stream_first_frame_seconds_sum") == 0
    assert _metric_value(body, 'hse_tts_stream_first_frame_seconds_bucket', le="+Inf") == 0
    for outcome in STREAM_OUTCOMES:
        assert _metric_value(body, "hse_tts_stream_chunks_total", outcome=outcome) == 0
    assert _metric_value(body, "hse_tts_stream_bytes_total") == 0
    # Every histogram bucket is present even with no observations.
    assert body.count("hse_tts_stream_first_frame_seconds_bucket{") == len(
        STREAM_FIRST_FRAME_BUCKETS
    ) + 1


@pytest.mark.unit
def test_first_frame_histogram_accumulates_buckets_sum_and_count():
    observe_first_frame(0.04)  # <= 0.05
    observe_first_frame(0.3)  # <= 0.5
    observe_first_frame(12.0)  # <= 15.0

    body = render_stream_metrics(get_stream_stats())

    assert _metric_value(body, "hse_tts_stream_first_frame_seconds_count") == 3
    assert _metric_value(body, "hse_tts_stream_first_frame_seconds_sum") == pytest.approx(12.34)
    # Cumulative bucket counts.
    assert _metric_value(body, "hse_tts_stream_first_frame_seconds_bucket", le="0.05") == 1
    assert _metric_value(body, "hse_tts_stream_first_frame_seconds_bucket", le="0.5") == 2
    assert _metric_value(body, "hse_tts_stream_first_frame_seconds_bucket", le="10") == 2
    assert _metric_value(body, "hse_tts_stream_first_frame_seconds_bucket", le="15") == 3
    assert _metric_value(body, "hse_tts_stream_first_frame_seconds_bucket", le="+Inf") == 3


@pytest.mark.unit
def test_outcome_and_byte_counters():
    record_stream_outcome("completed")
    record_stream_outcome("completed")
    record_stream_outcome("aborted")
    record_stream_outcome("error")
    record_stream_bytes(1024)
    record_stream_bytes(24)

    body = render_stream_metrics(get_stream_stats())

    assert _metric_value(body, "hse_tts_stream_chunks_total", outcome="completed") == 2
    assert _metric_value(body, "hse_tts_stream_chunks_total", outcome="aborted") == 1
    assert _metric_value(body, "hse_tts_stream_chunks_total", outcome="error") == 1
    assert _metric_value(body, "hse_tts_stream_bytes_total") == 1048


@pytest.mark.unit
def test_unknown_outcome_is_rejected():
    with pytest.raises(ValueError, match="unknown streaming outcome"):
        record_stream_outcome("finished")


@pytest.mark.unit
def test_get_stream_stats_returns_a_copy():
    observe_first_frame(1.0)
    snapshot = get_stream_stats()
    snapshot["chunks"]["completed"] = 99
    snapshot["first_frame"]["buckets"][0] = 99

    fresh = get_stream_stats()
    assert fresh["chunks"]["completed"] == 0
    assert fresh["first_frame"]["buckets"][0] == 0


# ---------------------------------------------------------------------------
# Observation points: binary envelope
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_binary_encoder_records_completed_first_frame_and_bytes():
    state = StreamState(source_rate=8000, target_rate=8000)
    segment = np.full(8, 0.25, dtype=np.float32)

    frames = [
        frame
        async for frame in _encode_stream_frames(
            _segments([segment]), audio_format="wav", state=state
        )
    ]

    stats = get_stream_stats()
    assert state.first_frame_observed is True
    assert stats["first_frame"]["count"] == 1
    assert stats["chunks"] == {"completed": 1, "aborted": 0, "error": 0}
    assert stats["bytes"] == sum(len(frame) for frame in frames)
    # The WAV header is included in the wire byte count.
    assert stats["bytes"] > len(frames[0]) - 44


@pytest.mark.unit
@pytest.mark.asyncio
async def test_binary_encoder_records_aborted_when_closed_early():
    state = StreamState(source_rate=8000, target_rate=8000)
    generator = _encode_stream_frames(
        _one_then_block(), audio_format="pcm", state=state
    )
    first = await anext(generator)
    assert first

    await generator.aclose()

    stats = get_stream_stats()
    assert state.aborted is True
    assert stats["chunks"] == {"completed": 0, "aborted": 1, "error": 0}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_binary_encoder_records_a_polled_abort_as_aborted():
    """The frame source ends normally after the poll; the flag reclassifies it."""
    state = StreamState(source_rate=8000, target_rate=8000)
    state.aborted = True

    [_frame async for _frame in _encode_stream_frames(
        _segments([np.full(8, 0.25, dtype=np.float32)]),
        audio_format="pcm",
        state=state,
    )]

    assert get_stream_stats()["chunks"] == {"completed": 0, "aborted": 1, "error": 0}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_binary_encoder_records_error_after_the_first_frame():
    state = StreamState(source_rate=8000, target_rate=8000)

    with pytest.raises(StreamSynthesisError):
        async for _frame in _encode_stream_frames(
            _failing_after([np.full(8, 0.25, dtype=np.float32)]),
            audio_format="pcm",
            state=state,
        ):
            pass

    assert get_stream_stats()["chunks"] == {"completed": 0, "aborted": 0, "error": 1}


# ---------------------------------------------------------------------------
# Observation points: SSE envelope
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_sse_encoder_records_completed_and_counts_raw_pcm_bytes():
    state = StreamState(source_rate=8000, target_rate=8000)
    segment = np.full(8, 0.25, dtype=np.float32)

    frames = [
        frame
        async for frame in _encode_sse_frames(
            _segments([segment]), state=state, request_id="tts-m", model="m", created=0
        )
    ]

    stats = get_stream_stats()
    assert state.first_frame_observed is True
    assert stats["first_frame"]["count"] == 1
    assert stats["chunks"] == {"completed": 1, "aborted": 0, "error": 0}
    # 8 s16le samples = 16 raw bytes, not the base64 text length.
    assert stats["bytes"] == 16
    assert len("".join(frames)) > stats["bytes"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_sse_keepalive_is_not_a_first_frame_observation():
    state = StreamState(source_rate=8000, target_rate=8000)

    async def _slow_first():
        await asyncio.sleep(0.05)
        yield np.full(4, 0.1, dtype=np.float32)

    events = [
        frame
        async for frame in _encode_sse_frames(
            _slow_first(),
            state=state,
            request_id="tts-k",
            model="m",
            created=0,
            keepalive_timeout=0.01,
        )
    ]

    assert events[0] == ": keepalive\n\n"
    # One audio delta, so exactly one TTFA observation and one completed stream.
    assert get_stream_stats()["first_frame"]["count"] == 1
    assert get_stream_stats()["chunks"] == {"completed": 1, "aborted": 0, "error": 0}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_sse_encoder_records_aborted_when_closed_early():
    state = StreamState(source_rate=8000, target_rate=8000)
    generator = _encode_sse_frames(
        _one_then_block(), state=state, request_id="tts-a", model="m", created=0
    )
    first = await anext(generator)
    assert first.startswith("event: speech.audio.delta")

    await generator.aclose()

    assert state.aborted is True
    assert get_stream_stats()["chunks"] == {"completed": 0, "aborted": 1, "error": 0}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_sse_encoder_records_error_after_a_delta():
    state = StreamState(source_rate=8000, target_rate=8000)
    events = [
        frame
        async for frame in _encode_sse_frames(
            _failing_after([np.full(4, 0.1, dtype=np.float32)]),
            state=state,
            request_id="tts-e",
            model="m",
            created=0,
        )
    ]

    assert events[-1].startswith("event: speech.audio.error")
    assert get_stream_stats()["chunks"] == {"completed": 0, "aborted": 0, "error": 1}

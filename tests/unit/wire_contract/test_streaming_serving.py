"""Phase 14-3: the binary envelope's frame encoder and deadline rule.

The generator's engine-facing half is pinned by
``tests/integration/test_tts_streaming.py`` (it needs the real engine loop).
Here the pure pieces are pinned:

* ``_encode_stream_frames`` — one ``pcm16_frame`` per segment, the streaming WAV
  header on the **first** yield only, headerless for ``pcm``, and quantization
  always on a worker thread (never the event loop);
* ``_prepend_frames`` — the primed first frame is re-yielded once, in order;
* the sequential streaming deadline rule (``base`` per chunk) next to the
  buffered ``base * (chunks + 1)`` rule it deliberately does not use.
"""

import threading
from types import SimpleNamespace

import numpy as np
import pytest

from harmonyspeech.endpoints.openai import serving_text_to_speech as serving_module
from harmonyspeech.endpoints.openai.serving_text_to_speech import (
    OpenAIServingTextToSpeech,
    StreamState,
    _encode_stream_frames,
    _prepend_frames,
)
from harmonyspeech.endpoints.openai.streaming import pcm16_frame, wav_stream_header


async def _segments(items):
    for item in items:
        yield item


def _serving(request_timeout_s):
    serving = OpenAIServingTextToSpeech.__new__(OpenAIServingTextToSpeech)
    serving.engine = SimpleNamespace(request_timeout_s=request_timeout_s)
    return serving


# ---------------------------------------------------------------------------
# Frame encoder
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_wav_header_rides_on_the_first_frame_only():
    first = np.full(8, 0.25, dtype=np.float32)
    second = np.full(4, -0.5, dtype=np.float32)
    state = StreamState(source_rate=24000, target_rate=24000)

    frames = [
        frame
        async for frame in _encode_stream_frames(
            _segments([first, second]), audio_format="wav", state=state
        )
    ]

    assert frames[0] == wav_stream_header(24000) + pcm16_frame(first, 24000)
    assert frames[1] == pcm16_frame(second, 24000)
    assert len(frames) == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_pcm_envelope_never_carries_a_header():
    segment = np.full(8, 0.25, dtype=np.float32)
    state = StreamState(source_rate=8000, target_rate=8000)

    frames = [
        frame
        async for frame in _encode_stream_frames(
            _segments([segment]), audio_format="pcm", state=state
        )
    ]

    assert frames == [pcm16_frame(segment, 8000)]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_empty_segments_are_skipped_and_do_not_consume_the_header():
    first = np.full(8, 0.25, dtype=np.float32)
    empty = np.empty(0, dtype=np.float32)
    state = StreamState(source_rate=16000, target_rate=16000)

    frames = [
        frame
        async for frame in _encode_stream_frames(
            _segments([empty, first]), audio_format="wav", state=state
        )
    ]

    assert len(frames) == 1
    assert frames[0] == wav_stream_header(16000) + pcm16_frame(first, 16000)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_header_uses_the_target_rate_not_the_source_rate():
    segment = np.full(4, 0.1, dtype=np.float32)
    state = StreamState(source_rate=24000, target_rate=8000)

    frames = [
        frame
        async for frame in _encode_stream_frames(
            _segments([segment]), audio_format="wav", state=state
        )
    ]

    assert frames[0][:44] == wav_stream_header(8000)
    assert frames[0][44:] == pcm16_frame(segment, 8000)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_quantization_runs_on_a_worker_thread(monkeypatch):
    """``pcm16_frame`` must never run on the event loop thread."""
    calls: list[int] = []
    real = serving_module.pcm16_frame

    def recording(audio, sample_rate=None):
        calls.append(threading.get_ident())
        return real(audio, sample_rate)

    monkeypatch.setattr(serving_module, "pcm16_frame", recording)
    state = StreamState(source_rate=24000, target_rate=24000)
    segment = np.full(8, 0.25, dtype=np.float32)

    [_frame async for _frame in _encode_stream_frames(
        _segments([segment]), audio_format="pcm", state=state
    )]

    assert calls and all(ident != threading.get_ident() for ident in calls)


# ---------------------------------------------------------------------------
# Primed-first-frame wrapper
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_prepend_frames_yields_the_primed_frame_first():
    body = [frame async for frame in _prepend_frames(b"first", _segments([b"a", b"b"]))]
    assert body == [b"first", b"a", b"b"]


# ---------------------------------------------------------------------------
# Deadline rule
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_sequential_streaming_uses_the_base_deadline_not_the_burst_rule():
    serving = _serving(30.0)
    assert serving._stream_chunk_request_timeout_s() == 30.0
    # The buffered burst rule is unchanged and deliberately different.
    assert serving._chunk_request_timeout_s(4) == 150.0


@pytest.mark.unit
def test_disabled_engine_timeout_stays_disabled_for_streaming():
    serving = _serving(None)
    assert serving._stream_chunk_request_timeout_s() is None
    assert serving._chunk_request_timeout_s(4) is None
    serving = _serving(0)
    assert serving._stream_chunk_request_timeout_s() is None


@pytest.mark.unit
def test_stream_state_duration_is_source_rate_based():
    state = StreamState()
    assert state.duration_s == 0.0
    state.source_rate = 24000
    state.target_rate = 8000
    state.emitted_samples = 24000
    assert state.duration_s == 1.0

"""Phase 14-3: the binary streaming envelope (``stream_format: "audio"``).

Every test drives the real ``OpenAIServingTextToSpeech`` and the real
``AsyncHarmonySpeech`` loop through the 12-2 ``EngineHarness`` (stub models,
CPU-only, hermetic). What is pinned here:

* a one-chunk stream is sample-identical to the buffered single-shot path's
  **pre-watermark** PCM (the 44-byte header differs by design, and per-segment
  watermarking diverges from the buffered whole-clip call by owner decision);
* a multi-chunk stream equals ``crossfade_stitch`` of the same chunk set
  sample-for-sample (proves the generator's chunk ordering + frame slicing; the
  stitcher primitive itself is pinned in ``tests/unit/common/test_audio_utils.py``);
* the carry is continuous across frame boundaries (no step at a frame seam);
* client disconnect aborts the chunk in flight and never submits the rest;
* a resampled stream reports the **target** rate and emits frames at it;
* sequential submission stamps the base deadline per chunk (not
  ``base * (chunks + 1)``, which assumes the buffered all-at-once burst);
* the generator never receives a request the admission gate rejects;
* TTFA under co-batching is a **structural** property: chunk 0 alone emits its
  first frame as soon as chunk 0 is done; chunk 0 co-batched behind a slower
  row waits for the slowest row (12-8's deferred vocode). The real-GPU TTFA
  histogram is the orchestrator's benchmark pass (14-5) — this file only pins
  the mechanism.
"""

import asyncio
import struct

import numpy as np
import pytest

from harmonyspeech.common import audio_codec
from harmonyspeech.common.audio_utils import crossfade_stitch, remove_dc_offset
from harmonyspeech.common.inputs import TextToSpeechRequestInput
from harmonyspeech.endpoints.openai.protocol import (
    AudioOutputOptions,
    ErrorResponse,
    GenerationOptions,
    TextToSpeechRequest,
)
from harmonyspeech.endpoints.openai.serving_text_to_speech import (
    StreamingAudioResponse,
    StreamState,
)
from harmonyspeech.endpoints.openai.streaming import (
    HEADER_AUDIO_SAMPLE_RATE,
    pcm16_frame,
    wav_stream_header,
)
from tests.integration.test_tts_chunking import (
    RAW,
    SAMPLE_RATE,
    SAMPLES_PER_CHAR,
    ChatterboxStubModel,
    EngineHarness,
    _chatterbox_harness,
    _decode,
    _model_config,
    _request,
    _speak,
)

#: One sentence per chunk. Each is longer than two 40 ms carry windows at the
#: stub's 8 kHz (704 samples vs 640), so every join is a real crossfade and the
#: first frame is produced by chunk 0 alone. ``chunk_size=99`` keeps exactly one
#: sentence per chunk (two would not fit).
CHUNK_SIZE = 99


def _sentence(letter: str, words: int = 16) -> str:
    return f"{letter}{letter.lower() * 5} " + " ".join([f"{letter.lower()}ord"] * words) + "."


SENTENCE_A = _sentence("A")  # first char 'A' -> stub amplitude 0.16
SENTENCE_B = _sentence("B")
SENTENCE_C = _sentence("C")
SENTENCE_Z = _sentence("Z")  # first char 'Z' -> stub amplitude 0.41
TEXT_3 = f"{SENTENCE_A} {SENTENCE_B} {SENTENCE_C}"
#: Amplitude contrast between neighbours: a naive concat steps by 0.25.
TEXT_CONTRAST = f"{SENTENCE_A} {SENTENCE_Z} {SENTENCE_A}"
SLOW_TEXT = "ZZZZ slow row."


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stream_request(
    text: str,
    *,
    model: str = "chatterbox",
    output: dict | None = None,
    **generation_options,
) -> TextToSpeechRequest:
    """A TTS request that selects the binary envelope (default: wav)."""
    options = {"stream_format": "audio", "format": "wav"}
    options.update(output or {})
    request = _request(text, model=model, **generation_options)
    return request.model_copy(update={"output_options": AudioOutputOptions(**options)})


async def _collect(response) -> bytes:
    """Consume a streaming response body into one bytes object."""
    assert isinstance(response, StreamingAudioResponse), response
    return b"".join([frame async for frame in response.body])


def _header_rate(header: bytes) -> int:
    """Sample rate field of a canonical 44-byte PCM RIFF header (offset 24)."""
    return struct.unpack("<I", header[24:28])[0]


def _chunk_waveform(text: str) -> np.ndarray:
    """The stub Chatterbox's audio for one chunk (see ``ChatterboxStubModel``)."""
    samples = max(1, len(text)) * SAMPLES_PER_CHAR
    value = (ord(text[0]) % 50 + 1) / 100.0
    return np.full((samples,), value, dtype=np.float32)


# ---------------------------------------------------------------------------
# Streamed vs buffered
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("remove_dc", [False, True])
async def test_one_chunk_stream_matches_buffered_pre_watermark_pcm(remove_dc):
    """One chunk: the streamed samples equal the buffered path's pre-watermark PCM.

    The stub's watermarker is the identity, so the buffered path's samples are
    exactly the stitch/DC output — the axis the owner decision scoped equality
    to (per-segment vs whole-clip perth diverges by design). The reference is
    the stub's own waveform after the same per-chunk ``remove_dc_offset`` the
    buffered path applies, quantized through the one canonical ``pcm16_frame``.

    The buffered WAV body is compared separately with a 1-LSB tolerance: a WAV
    container is written by libsndfile (x32768, truncating), while headerless
    PCM goes through ``_to_s16le`` (x32767, rounding) — a pre-existing 1-LSB
    difference between the two *containers*, not a streaming difference.
    """
    harness, _model = _chatterbox_harness()
    text = "A short single-chunk sentence."
    async with harness:
        buffered = await _speak(harness, _request(text, remove_dc_offset=remove_dc))
        streamed = await _speak(
            harness, _stream_request(text, remove_dc_offset=remove_dc)
        )
        body = await _collect(streamed)

    expected = _chunk_waveform(text)
    if remove_dc:
        expected = remove_dc_offset(expected)

    assert body[:44] == wav_stream_header(SAMPLE_RATE)
    assert body[44:] == pcm16_frame(expected, SAMPLE_RATE)

    # Cross-check against the buffered wire body (1 LSB: container encoders).
    buffered_waveform, sample_rate = _decode(buffered)
    assert sample_rate == SAMPLE_RATE
    buffered_pcm = np.frombuffer(pcm16_frame(buffered_waveform, sample_rate), dtype="<i2")
    streamed_pcm = np.frombuffer(body[44:], dtype="<i2")
    assert streamed_pcm.size == buffered_pcm.size
    assert np.max(np.abs(streamed_pcm.astype(np.int32) - buffered_pcm.astype(np.int32))) <= 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_multi_chunk_stream_matches_whole_buffer_stitch_sample_for_sample():
    """The generator's ordering + frame slicing reproduce ``crossfade_stitch``."""
    harness, model = _chatterbox_harness()
    request = _stream_request(
        TEXT_3,
        split_text=True,
        chunk_size=CHUNK_SIZE,
        crossfade_ms=8,
        sentence_pause_ms=40,
    )
    async with harness:
        response = await _speak(harness, request)
        body = await _collect(response)

    # Sequential submission in chunk-index order: the model saw the chunks in order.
    assert model.generate_calls == [SENTENCE_A, SENTENCE_B, SENTENCE_C]
    expected = crossfade_stitch(
        [_chunk_waveform(text) for text in model.generate_calls],
        crossfade_ms=8,
        sentence_pause_ms=40,
        sample_rate=SAMPLE_RATE,
    )
    assert body[:44] == wav_stream_header(SAMPLE_RATE)
    assert body[44:] == pcm16_frame(expected, SAMPLE_RATE)
    # The response carries the stream's own accounting (14-4's done payload).
    assert response.state is not None
    assert response.state.chunks == 3
    assert response.state.duration_s == pytest.approx(expected.size / SAMPLE_RATE)


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("pause_ms", [0, 120])
async def test_carry_continuity_across_frame_boundaries(pause_ms):
    """No discontinuity at a frame seam beyond the crossfade's own ramp.

    The 40 ms carry (320 samples @ 8 kHz) holds back each chunk's tail, so a
    frame boundary lands inside continuous, already-crossfaded audio. A naive
    concatenation would step by the 0.25 amplitude difference between the
    ``SENTENCE_A`` and ``SENTENCE_Z`` chunks; the crossfade/fade ramps never do.
    """
    harness, _model = _chatterbox_harness()
    request = _stream_request(
        TEXT_CONTRAST,
        split_text=True,
        chunk_size=CHUNK_SIZE,
        crossfade_ms=40,
        sentence_pause_ms=pause_ms,
    )
    async with harness:
        response = await _speak(harness, request)
        frames = [frame async for frame in response.body]

    assert isinstance(response, StreamingAudioResponse)
    assert frames[0][:44] == wav_stream_header(SAMPLE_RATE)
    # Sample offsets where one yielded frame ends and the next begins.
    boundaries = []
    offset = 0
    for index, frame in enumerate(frames):
        pcm = frame[44:] if index == 0 else frame
        offset += len(pcm) // 2
        boundaries.append(offset)
    boundaries.pop()  # the last frame's end is not a seam

    pcm = np.frombuffer(b"".join([frames[0][44:], *frames[1:]]), dtype="<i2")
    samples = pcm.astype(np.float64) / 32768.0
    steps = np.abs(np.diff(samples))
    # A naive concat of the 0.16/0.41/0.16 blocks would show a 0.25 step.
    assert steps.max() < 0.05, steps.max()
    for boundary in boundaries:
        assert abs(samples[boundary] - samples[boundary - 1]) < 0.05, boundary


# ---------------------------------------------------------------------------
# Disconnect
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_disconnect_aborts_in_flight_chunk_and_never_submits_the_rest():
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

        request = _stream_request(TEXT_3, split_text=True, chunk_size=CHUNK_SIZE)
        response = await serving.create_text_to_speech(request, RAW)

        # Priming synthesised chunk 0 only: the first frame is already in hand.
        assert isinstance(response, StreamingAudioResponse), response
        assert len(submissions) == 1
        first = await anext(response.body)
        assert first[:4] == b"RIFF"

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
            _stream_request(TEXT_3, split_text=True, chunk_size=CHUNK_SIZE), raw
        )
        assert isinstance(response, StreamingAudioResponse), response
        frames = [frame async for frame in response.body]

    # Chunk 0 was in hand when the poll turned true; chunk 1 was never submitted
    # and no stitcher/resampler tail was flushed to a client that is gone.
    assert model.generate_calls == [SENTENCE_A]
    assert len(frames) == 1


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_resampled_stream_header_carries_the_target_rate():
    """``sample_rate`` ⇒ header + frames at the target rate, continuous across chunks."""
    harness, model = _chatterbox_harness()
    target = 16000
    request = _stream_request(
        TEXT_3,
        split_text=True,
        chunk_size=CHUNK_SIZE,
        crossfade_ms=8,
        sentence_pause_ms=0,
        output={"stream_format": "audio", "format": "wav", "sample_rate": target},
    )
    async with harness:
        response = await _speak(harness, request)
        body = await _collect(response)

    assert isinstance(response, StreamingAudioResponse), response
    assert response.headers[HEADER_AUDIO_SAMPLE_RATE] == str(target)
    assert _header_rate(body[:44]) == target

    expected = audio_codec.resample(
        crossfade_stitch(
            [_chunk_waveform(text) for text in model.generate_calls],
            crossfade_ms=8,
            sentence_pause_ms=0,
            sample_rate=SAMPLE_RATE,
        ),
        SAMPLE_RATE,
        target,
    )
    streamed = np.frombuffer(body[44:], dtype="<i2")
    reference = np.frombuffer(pcm16_frame(expected, target), dtype="<i2")
    # Frames are at the target rate: same length, same samples (the 14-2
    # resampler is bit-identical to one-shot soxr resampling; allow 1 LSB).
    assert streamed.size == reference.size
    assert np.max(np.abs(streamed.astype(np.int32) - reference.astype(np.int32))) <= 1


# ---------------------------------------------------------------------------
# Deadlines
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sequential_stream_stamps_the_base_deadline_per_chunk():
    """Sequential submission ⇒ ``base`` per chunk, not ``base * (chunks + 1)``."""
    config = _model_config("chatterbox", "ChatterboxTTS")
    model = ChatterboxStubModel()
    harness = EngineHarness(
        [config], {"chatterbox": model}, engine_kwargs={"request_timeout_s": 60.0}
    )
    async with harness:
        serving = harness.serving
        deadlines: list[float | None] = []
        original_submit = serving._submit_chunk

        async def spy_submit(request_id, request_data, artifacts):
            deadlines.append(request_data.request_timeout_s)
            return await original_submit(request_id, request_data, artifacts)

        serving._submit_chunk = spy_submit

        request = _stream_request(TEXT_3, split_text=True, chunk_size=CHUNK_SIZE)
        response = await _speak(harness, request)
        await _collect(response)

        # The buffered rule (the all-at-once burst) is unchanged and different.
        assert serving._chunk_request_timeout_s(3) == 240.0
        assert serving._stream_chunk_request_timeout_s() == 60.0

    assert deadlines == [60.0, 60.0, 60.0]


# ---------------------------------------------------------------------------
# Admission cross-check
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_generator_never_receives_a_request_the_admission_gate_rejects():
    """Whole-buffer options and non-streamable formats stop before any submission."""
    harness, model = _chatterbox_harness()
    async with harness:
        submissions: list[str] = []
        original_submit = harness.serving._submit_chunk

        async def spy_submit(request_id, request_data, artifacts):
            submissions.append(request_id)
            return await original_submit(request_id, request_data, artifacts)

        harness.serving._submit_chunk = spy_submit

        rejected = [
            _stream_request("AAAA.", trim_silence=True),
            _stream_request("AAAA.", fix_internal_silence=True),
            _stream_request("AAAA.", speed=1.5),
            _stream_request("AAAA.", output={"stream_format": "audio", "format": "mp3"}),
        ]
        for request in rejected:
            response = await _speak(harness, request)
            assert isinstance(response, ErrorResponse), response
            assert response.code == 400
            assert "whole-buffer" in response.message or "wav" in response.message

    assert submissions == []
    assert model.generate_calls == []


# ---------------------------------------------------------------------------
# TTFA under co-batching (structural)
# ---------------------------------------------------------------------------


class _BatchModelExecutor:
    """Executor that models 12-8's deferred vocode: a batch releases together.

    Every row in a batch is emitted only when the slowest row is done — exactly
    the rule that makes a streaming request's TTFA depend on batch composition
    (``_pending_vocode`` holds a finished AR row's vocode until its co-batched
    siblings finish). A row whose text starts with ``Z`` is the slow one.
    """

    def __init__(self, runner, events: list):
        self.runner = runner
        self.events = events
        self.batches: list[list[tuple[str, str]]] = []

    @staticmethod
    def _row_duration(text: str) -> float:
        return 0.25 if text.startswith("Z") else 0.02

    async def execute_model_async(self, requests_to_batch):
        rows = [
            (entry.request_id, getattr(entry.request_data, "input_text", "") or "")
            for entry in requests_to_batch
        ]
        self.batches.append(rows)
        await asyncio.sleep(max(self._row_duration(text) for _, text in rows))
        outputs = self.runner.execute_model(requests_to_batch)
        for _request_id, text in rows:
            self.events.append(("generated", text))
        return outputs

    def execute_model(self, requests_to_batch):
        return self.runner.execute_model(requests_to_batch)

    def check_health(self):
        return {"status": "ok"}

    def release_request(self, request_id):  # noqa: ARG002
        return None


def _slow_request_input() -> TextToSpeechRequestInput:
    return TextToSpeechRequestInput(
        request_id="slow-row",
        requested_model="chatterbox",
        model="chatterbox",
        input_text=SLOW_TEXT,
        mode="single_speaker_tts",
        generation_options=GenerationOptions(),
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ttfa_structural_chunk_zero_alone_emits_before_the_slow_row():
    """Chunk 0 submitted alone ⇒ first frame after chunk 0, not after others."""
    config = _model_config("chatterbox", "ChatterboxTTS")
    model = ChatterboxStubModel()
    harness = EngineHarness([config], {"chatterbox": model})
    events: list[tuple] = []
    async with harness:
        executor = _BatchModelExecutor(
            harness.engine.engine.model_executors["chatterbox"].runner, events
        )
        harness.engine.engine.model_executors["chatterbox"] = executor

        response = await harness.serving.create_text_to_speech(
            _stream_request(TEXT_3, split_text=True, chunk_size=CHUNK_SIZE), RAW
        )
        events.append(("first-frame",))

        # The other client's slow row arrives only afterwards.
        slow_stream = await harness.engine.add_request(
            "slow-row", _slow_request_input(), artifacts={}
        )
        await _collect(response)
        async for _output in slow_stream:
            pass

    assert isinstance(response, StreamingAudioResponse)
    # Chunk 0 ran alone (sequential submission): its first frame did not wait.
    assert len(executor.batches[0]) == 1
    assert [text for _rid, text in executor.batches[0]] == [SENTENCE_A]
    assert events.index(("first-frame",)) < events.index(("generated", SLOW_TEXT))


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ttfa_structural_cobatched_chunk_zero_waits_for_the_slowest_row():
    """Chunk 0 co-batched behind a slower row ⇒ first frame waits for that row."""
    config = _model_config("chatterbox", "ChatterboxTTS")
    model = ChatterboxStubModel()
    harness = EngineHarness([config], {"chatterbox": model})
    events: list[tuple] = []
    async with harness:
        executor = _BatchModelExecutor(
            harness.engine.engine.model_executors["chatterbox"].runner, events
        )
        harness.engine.engine.model_executors["chatterbox"] = executor

        # The slow row is queued first; chunk 0 is submitted in the same engine
        # iteration (add_request never yields), so both land in one batch.
        slow_stream = await harness.engine.add_request(
            "slow-row", _slow_request_input(), artifacts={}
        )
        response = await harness.serving.create_text_to_speech(
            _stream_request(TEXT_3, split_text=True, chunk_size=CHUNK_SIZE), RAW
        )
        events.append(("first-frame",))

        await _collect(response)
        async for _output in slow_stream:
            pass

    assert isinstance(response, StreamingAudioResponse)
    cobatch = [batch for batch in executor.batches if len(batch) == 2]
    assert cobatch, executor.batches
    assert ("slow-row", SLOW_TEXT) in cobatch[0]
    assert any(text == SENTENCE_A for _rid, text in cobatch[0])
    # The mechanism: chunk 0's audio (its first frame) waited for the slow row.
    assert events.index(("generated", SLOW_TEXT)) < events.index(("first-frame",))


# ---------------------------------------------------------------------------
# Duration accounting (the 14-4 ``done`` payload's source)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_stream_state_counts_emitted_source_samples():
    """``duration_s`` is the generator's own counter, not the wire length."""
    harness, _model = _chatterbox_harness()
    state = StreamState()
    chunks = [SENTENCE_A, SENTENCE_B, SENTENCE_C]
    async with harness:
        # Drive the shared frame source directly so the accounting is visible.
        request = _stream_request(TEXT_3, split_text=True, chunk_size=CHUNK_SIZE)
        request_data = TextToSpeechRequestInput.from_openai("tts-state", request)
        segments = [
            segment
            async for segment in harness.serving._stream_audio_segments(
                raw_request=RAW,
                request_id="tts-state",
                request_data=request_data,
                artifacts={},
                chunks=chunks,
                state=state,
                requested_rate=None,
            )
        ]

    assert state.chunks == 3
    assert state.source_rate == SAMPLE_RATE
    assert state.target_rate == SAMPLE_RATE
    assert state.segments == len(segments)
    # The counter equals the whole-buffer stitch length (sum of feed + flush).
    stitched = crossfade_stitch(
        [_chunk_waveform(text) for text in chunks],
        crossfade_ms=40,
        sentence_pause_ms=120,
        sample_rate=SAMPLE_RATE,
    )
    assert state.emitted_samples == stitched.size
    assert state.duration_s == pytest.approx(stitched.size / SAMPLE_RATE)


@pytest.mark.integration
def test_stream_state_duration_math_is_honest():
    assert StreamState().duration_s == 0.0
    state = StreamState(source_rate=24000, emitted_samples=48000)
    assert state.duration_s == 2.0
    # target rate never enters the duration: it counts source-rate samples.
    state.target_rate = 16000
    assert state.duration_s == 2.0

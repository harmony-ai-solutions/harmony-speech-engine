"""Phase 12-3 output filters composed with the 12-2 chunking path.

Drives the real ``OpenAIServingTextToSpeech`` + ``AsyncHarmonySpeech`` loop with
a stub synthesizer that emits a known DC offset and known silence structure, so
the serving-layer wiring (not just the pure functions) is pinned:

* DC removal runs **per chunk, before stitching** — two chunks with different
  offsets must both come out at zero, which a whole-buffer mean cannot do;
* lead/trail trim runs on the **final stitched buffer**, so the silence between
  two chunks is not mistaken for the buffer's ends;
* internal-silence fixing runs **after stitching**, where the inter-chunk pause
  actually exists (per chunk there is nothing to fix);
* the single-shot path applies the same three filters to its final buffer;
* all flags off keeps the pre-12 output byte-identical.
"""

import numpy as np
import pytest
import torch

from harmonyspeech.endpoints.openai.protocol import TextToSpeechResponse
from tests.integration.test_tts_chunking import (
    SAMPLE_RATE,
    ChatterboxStubModel,
    EngineHarness,
    _decode,
    _model_config,
    _request,
    _speak,
)

#: The stub's sample rate is 8 kHz, so the 32 ms analysis frame is 256 samples.
#: The fixtures below keep silence stretches and active runs comfortably longer
#: than one frame so the energy gate is unambiguous.
LEAD = 800  # 100 ms
ACTIVE = 1600  # 200 ms
TRAIL = 800  # 100 ms


class StructuredChatterboxModel(ChatterboxStubModel):
    """Chatterbox stand-in with a controlled DC offset and silence structure.

    ``values`` maps the chunk's first character to the constant sample value;
    ``lead``/``trail`` zeros are prepended/appended to every chunk. The watermark
    is still routed through ``self.watermarker`` so the 12-2 suppression
    mechanism is exercised (and the serving layer re-applies it once).
    """

    def __init__(self, *, values=None, lead=0, active=ACTIVE, trail=0, model_type="ChatterboxTTS"):
        super().__init__(model_type=model_type)
        self.values = values or {}
        self.lead = lead
        self.active = active
        self.trail = trail

    def generate(self, text, **kwargs):  # noqa: ARG002 - family signature superset
        self.generate_calls.append(text)
        value = float(self.values.get(text[:1], 0.3))
        wav = np.concatenate(
            [
                np.zeros(self.lead, dtype=np.float32),
                np.full(self.active, value, dtype=np.float32),
                np.zeros(self.trail, dtype=np.float32),
            ]
        )
        wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)
        return torch.from_numpy(np.asarray(wav, dtype=np.float32)).unsqueeze(0)


def _harness(model):
    config = _model_config("chatterbox", "ChatterboxTTS")
    return EngineHarness([config], {"chatterbox": model})


TWO_CHUNKS = "AAAA. BBBB."
CHUNK_OPTIONS = {"split_text": True, "chunk_size": 6}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_dc_offset_removed_per_chunk_before_stitching():
    """Different per-chunk offsets both reach zero => DC ran before stitching."""
    model = StructuredChatterboxModel(values={"A": 0.5, "B": 0.2}, lead=0, active=400, trail=0)
    harness = _harness(model)
    async with harness:
        response = await _speak(harness, _request(TWO_CHUNKS, remove_dc_offset=True, **CHUNK_OPTIONS))

    waveform, sr = _decode(response)
    assert sr == SAMPLE_RATE
    assert model.generate_calls == ["AAAA.", "BBBB."]
    # A whole-buffer mean (0.35) would leave -0.15 / +0.15 blocks. Per-chunk
    # means remove each block exactly.
    assert float(np.max(np.abs(waveform))) < 2e-3


@pytest.mark.integration
@pytest.mark.asyncio
async def test_edge_conditioning_bounds_chunk_edges_and_trim_is_a_no_op_there():
    """Phase 12-5 interplay: conditioning owns the joins, trim is whole-output.

    The stitcher trims each chunk's 100 ms lead/trail to the 10 ms guard before
    joining, so the stitched buffer's outer edges and its internal boundary
    carry only guard silence. ``trim_silence`` then sees silence shorter than
    its 32 ms analysis frame and leaves the buffer unchanged — it stays an
    explicit whole-output option, not a per-chunk step.
    """
    model = StructuredChatterboxModel(values={"A": 0.3, "B": 0.3}, lead=LEAD, active=ACTIVE, trail=TRAIL)
    harness = _harness(model)
    plain_options = dict(CHUNK_OPTIONS, crossfade_ms=0, sentence_pause_ms=0)
    async with harness:
        untrimmed = await _speak(harness, _request(TWO_CHUNKS, **plain_options))
        trimmed = await _speak(
            harness, _request(TWO_CHUNKS, trim_silence=True, **plain_options)
        )

    plain, _ = _decode(untrimmed)
    cut, _ = _decode(trimmed)
    guard = int(SAMPLE_RATE * 10 / 1000)  # 10 ms guard at 8 kHz
    assert plain.size == 2 * (guard + ACTIVE + guard)
    # 10 ms is below the 32 ms analysis frame: trim_silence has nothing to do.
    assert cut.size == plain.size
    # The inter-chunk boundary keeps both guard pads (20 ms total).
    boundary = guard + ACTIVE
    assert np.allclose(cut[boundary : boundary + 2 * guard], 0.0, atol=2e-3)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_internal_silence_fixing_runs_after_stitching():
    """The only long pause exists between chunks, so the fix must be post-stitch."""
    model = StructuredChatterboxModel(values={"A": 0.3, "B": 0.3}, lead=0, active=ACTIVE, trail=0)
    harness = _harness(model)
    pause_ms = 900  # above the 700 ms minimum
    async with harness:
        unfixed = await _speak(
            harness,
            _request(
                TWO_CHUNKS, sentence_pause_ms=pause_ms, crossfade_ms=0, **CHUNK_OPTIONS
            ),
        )
        fixed = await _speak(
            harness,
            _request(
                TWO_CHUNKS,
                sentence_pause_ms=pause_ms,
                crossfade_ms=0,
                fix_internal_silence=True,
                **CHUNK_OPTIONS,
            ),
        )

    plain, _ = _decode(unfixed)
    collapsed, _ = _decode(fixed)
    pause = int(SAMPLE_RATE * pause_ms / 1000)
    assert plain.size == 2 * ACTIVE + pause
    # Collapsed to ~300 ms; allow one analysis frame (32 ms) of boundary slack.
    kept = int(SAMPLE_RATE * 300 / 1000)
    frame = int(SAMPLE_RATE * 32 / 1000)
    assert abs(collapsed.size - (2 * ACTIVE + kept)) <= frame
    assert collapsed.size < plain.size


@pytest.mark.integration
@pytest.mark.asyncio
async def test_single_shot_path_applies_output_filters():
    model = StructuredChatterboxModel(values={"A": 0.5}, lead=0, active=800, trail=0)
    harness = _harness(model)
    async with harness:
        plain = await _speak(harness, _request("AAAA."))
        filtered = await _speak(harness, _request("AAAA.", remove_dc_offset=True))

    plain_wave, sr = _decode(plain)
    filtered_wave, filtered_sr = _decode(filtered)
    assert sr == filtered_sr == SAMPLE_RATE
    assert plain_wave.size == filtered_wave.size == 800
    assert np.allclose(plain_wave, 0.5, atol=2e-3)
    assert float(np.max(np.abs(filtered_wave))) < 2e-3


@pytest.mark.integration
@pytest.mark.asyncio
async def test_filters_on_chunked_path_produce_a_valid_wav():
    model = StructuredChatterboxModel(values={"A": 0.3, "B": 0.3}, lead=LEAD, active=ACTIVE, trail=TRAIL)
    harness = _harness(model)
    async with harness:
        response = await _speak(
            harness,
            _request(
                TWO_CHUNKS,
                remove_dc_offset=True,
                trim_silence=True,
                fix_internal_silence=True,
                sentence_pause_ms=900,
                **CHUNK_OPTIONS,
            ),
        )

    assert isinstance(response, TextToSpeechResponse), response
    waveform, sr = _decode(response)
    assert sr == SAMPLE_RATE
    assert waveform.size > 0
    assert np.isfinite(waveform).all()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_filters_run_before_the_post_stitch_speed_stretch():
    """A 900 ms pause collapses only if measured before the stretch.

    At speed 2.0 the stretched pause would be 450 ms, below the 700 ms minimum,
    so a filter running *after* the stretch would leave it alone. Running before
    the stretch collapses 900 ms -> 300 ms and the stretch then halves both the
    speech and the kept pause: (3200 + 2400) / 2 = 2800 samples.
    """
    model = StructuredChatterboxModel(values={"A": 0.3, "B": 0.3}, lead=0, active=ACTIVE, trail=0)
    harness = _harness(model)
    async with harness:
        response = await _speak(
            harness,
            _request(
                TWO_CHUNKS,
                sentence_pause_ms=900,
                crossfade_ms=0,
                fix_internal_silence=True,
                speed=2.0,
                **CHUNK_OPTIONS,
            ),
        )

    waveform, _sr = _decode(response)
    expected = (2 * ACTIVE + int(SAMPLE_RATE * 300 / 1000)) // 2
    assert abs(waveform.size - expected) <= 200
    # An after-stretch filter would leave the 450 ms pause and yield ~5200.
    assert waveform.size < 4000


class RecordingWatermarker:
    def __init__(self):
        self.lengths = []

    def apply_watermark(self, wav, sample_rate=None):  # noqa: ARG002 - upstream signature
        self.lengths.append(int(np.asarray(wav).size))
        return np.asarray(wav, dtype=np.float32)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_filters_run_before_the_single_watermark():
    """The watermark must receive the already-filtered buffer, not the raw stitch."""
    model = StructuredChatterboxModel(values={"A": 0.3, "B": 0.3}, lead=0, active=ACTIVE, trail=0)
    watermarker = RecordingWatermarker()
    model.watermarker = watermarker
    harness = _harness(model)
    async with harness:
        response = await _speak(
            harness,
            _request(
                TWO_CHUNKS,
                sentence_pause_ms=900,
                crossfade_ms=0,
                fix_internal_silence=True,
                **CHUNK_OPTIONS,
            ),
        )

    waveform, _sr = _decode(response)
    # Chunk synthesis is watermark-suppressed, so the only recorded call is the
    # serving-side final watermark, and it saw the filtered (collapsed-pause)
    # length — not the raw 900 ms stitch.
    assert watermarker.lengths == [waveform.size]
    frame = int(SAMPLE_RATE * 32 / 1000)
    assert abs(waveform.size - (2 * ACTIVE + int(SAMPLE_RATE * 300 / 1000))) <= frame


@pytest.mark.integration
@pytest.mark.asyncio
async def test_all_filters_off_is_byte_identical_to_omitting_them():
    model = StructuredChatterboxModel(values={"A": 0.3, "B": 0.3}, lead=LEAD, active=ACTIVE, trail=TRAIL)
    harness = _harness(model)
    async with harness:
        omitted = await _speak(harness, _request(TWO_CHUNKS, **CHUNK_OPTIONS))
        explicit_off = await _speak(
            harness,
            _request(
                TWO_CHUNKS,
                remove_dc_offset=False,
                trim_silence=False,
                fix_internal_silence=False,
                **CHUNK_OPTIONS,
            ),
        )
    assert omitted.data == explicit_off.data


@pytest.mark.integration
@pytest.mark.asyncio
async def test_filter_flags_reach_chunk_sub_requests_without_double_applying():
    """Filters are copied onto chunk options but the runner must not act on them."""
    model = StructuredChatterboxModel(values={"A": 0.5, "B": 0.2}, lead=0, active=400, trail=0)
    harness = _harness(model)
    seen: list[tuple] = []
    original = harness.prepare

    def prepare(model_config, requests_to_batch):
        for entry in requests_to_batch:
            options = getattr(entry.request_data, "generation_options", None)
            if options is not None:
                seen.append(
                    (
                        getattr(options, "remove_dc_offset", False),
                        getattr(options, "trim_silence", False),
                        getattr(options, "fix_internal_silence", False),
                    )
                )
        return original(model_config, requests_to_batch)

    harness.prepare = prepare
    async with harness:
        response = await _speak(
            harness,
            _request(TWO_CHUNKS, remove_dc_offset=True, trim_silence=True, **CHUNK_OPTIONS),
        )

    assert isinstance(response, TextToSpeechResponse), response
    # The flags are carried on each chunk sub-request (internal options clone),
    # and the runner ignores them; the serving layer applied DC once per chunk.
    assert seen == [(True, True, False), (True, True, False)]

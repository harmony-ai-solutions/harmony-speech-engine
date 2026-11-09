# Phase 12-3: Phase B Audio Filters (DC offset, silence trim, internal silence)

> **Plan:** [Engine Modernization](summary.md) · **Phase:** 12 — Long-form audio
> **Depends on:** 12-1 (audio_utils home), 12-2 (single + chunked paths to wire into)
> **Codebase mapping consulted:** March plan 6-1/7-1/8-1 sub-task files (superseded); upstream `utils.py` reference implementations
> **State reference:** verified 2026-11-08 — upstream prior art: `trim_lead_trail_silence` (`utils.py:580`), `fix_internal_silence` (`:649`), plus an extra `remove_long_unvoiced_segments` (`:758`) the March plan didn't have; **no** `remove_dc_offset` upstream (March's own addition — simple mean/one-pole-highpass removal).

## Objective

Three (optionally four) opt-in output filters, model-agnostic, wired into **both** the plain and chunked synthesis paths: `remove_dc_offset`, `trim_silence` (leading/trailing), `fix_internal_silence` (collapse overlong internal pauses). Evaluate upstream's `remove_long_unvoiced_segments` as a fourth while here — adopt only if it composes cleanly with `fix_internal_silence` (overlapping intent; if unclear, skip with a dated note).

## Context (self-contained)

- **Ordering matters and is pinned by test:** per March design — DC removal **per chunk before stitching** (offsets differ per synthesis run), trim + internal-silence **on the final stitched output** (a chunk boundary must not be mistaken for lead/trail silence). In the single-shot path all three run on the final buffer.
- **Honesty of effect:** filters change samples by construction; the audio-regression harness (4-2) must treat them as an explicit opt-in axis, not as drift — assert filter-on vs filter-off on a fixture with known DC/silence structure (synthesized), not on real-model output alone.
- Defaults all `false` ⇒ byte-identical today behavior; wire contracts unchanged.
- Thresholds: expose only coarse knobs if any (the March plan shipped none — keep it threshold-free/preset; document the chosen defaults in `docs/api.md`).

## Implementation steps

1. Filters in `harmonyspeech/common/audio_utils.py`: `remove_dc_offset` (subtract mean, or one-pole highpass ~20 Hz — pick, document), `trim_silence(wav, sr)` (energy-gated), `fix_internal_silence(wav, sr, max_pause)` (collapse pauses beyond a documented bound).
2. Protocol fields (`remove_dc_offset`, `trim_silence`, `fix_internal_silence`: `bool = False`) in the three input layers (12-1 precedent).
3. Wiring in `serving_text_to_speech.py`: plain path (final buffer) + chunked path (DC per chunk pre-stitch; trim/internal post-stitch, before/after crossfade — decide from the fixture evidence and pin it).
4. Tests: unit (all-silence, no-silence, DC-heavy, mono/stereo, boundary pauses at chunk edges); integration (filters composed with chunking); e2e smoke per family (flag-on ⇒ valid WAV, duration shrinks for trim fixture); audio-regression opt-in axis note.
5. Docs: `docs/api.md`, `CHANGELOG.md` (single Phase 12 user-visible block if 12-2 already shipped one — extend it).

## Files to create/modify

| File | Change |
|---|---|
| `harmonyspeech/common/audio_utils.py` | three filters (+ optional fourth) |
| protocol/inputs layers ×3 | opt-in bool fields |
| `harmonyspeech/endpoints/openai/serving_text_to_speech.py` | wiring both paths |
| `tests/unit/common/test_audio_utils.py` (extend), integration, e2e | gates above |

## Tests / validation

- Default-off byte-identity gate re-run; full suites green.

## Checklist

- [x] Filters landed + ordering pinned (per-chunk DC, final trim/internal)
- [x] Upstream fourth filter evaluated (adopted or skipped with dated note)
- [ ] Tests + docs + CHANGELOG; suites green; `summary.md` ticked

## Deviations / verification log

Dated 2026-11-09 (Phase 12-3 implementation).

**Filter algorithms and thresholds (pure numpy, no new dependency)**

- `remove_dc_offset(wav)`: subtracts the **per-channel mean**. Chosen over a
  one-pole ~20 Hz high-pass because mean subtraction is exact, stateless and
  phase-free — it removes only the 0 Hz term and leaves every other bin intact;
  a high-pass would also attenuate legitimate low frequencies and shift phase.
  Per-channel means (not one global mean) preserve a stereo image whose channels
  carry different offsets.
- `trim_silence(wav, sample_rate)`: short-time **RMS** gate. A frame is silent
  when its RMS is more than **40 dB below the loudest frame** (the upstream
  `top_db` convention, relative not absolute, so uniform low-level content is not
  mistaken for silence). Defaults: **32 ms** analysis frames, **8 ms** hop,
  **50 ms** padding kept at each end (the upstream reference used 2048/512
  samples and 50 ms padding; frame length is expressed in *time* here so the
  gate behaves the same at 8 kHz and 44.1 kHz — a fixed 2048-sample frame is
  256 ms at 8 kHz and far too coarse). Frame boundaries are refined to the exact
  sample crossing the same gate's sample floor, so a trim is not quantised to a
  whole frame. All-silent input trims to an empty waveform (documented); no
  silence returns an unchanged copy.
- `fix_internal_silence(wav, sample_rate)`: same RMS gate; only silent runs
  strictly **between** two active regions are touched. A run is shortened when
  it lasts at least **700 ms**; the first **300 ms** are kept (upstream defaults
  `min_silence_to_fix_ms=700`, `max_allowed_silence_ms=300`). Leading/trailing
  silence is deliberately left to `trim_silence` (a chunk edge must never be
  mistaken for the buffer's ends). Run boundaries are refined to the exact
  sample crossing the gate, so a 699 ms pause is not collapsed because it
  overlapped a frame.
- Frames are computed with a cumulative-sum integral image: `O(n)` time and
  memory regardless of overlap, no window function, mono or `(samples, channels)`
  (channels averaged for the gate; the same sample range is cut from every
  channel).
- Thresholds are function-level keyword arguments for tests/future callers; the
  request layer exposes only the on/off booleans (no wire knobs), per the spec.

**Pinned ordering**

- Chunked path: **DC per chunk before `crossfade_stitch`**, then
  `trim_silence` and `fix_internal_silence` on the **final stitched buffer**,
  then `time_stretch` (non-native speed), then the once-per-request watermark.
  Pinned by integration tests:
  `test_dc_offset_removed_per_chunk_before_stitching` (two chunks at 0.5/0.2
  both reach zero; a whole-buffer mean would leave ±0.15 blocks),
  `test_trim_runs_on_the_final_stitched_buffer_not_per_chunk` (per-chunk trim
  would delete the inter-chunk silence; the final-buffer result keeps it), and
  `test_internal_silence_fixing_runs_after_stitching` (the only 900 ms pause
  exists only after stitching, and is collapsed there), plus
  `test_filters_run_before_the_post_stitch_speed_stretch` (at speed 2.0 a 900 ms
  pause would be 450 ms — below the 700 ms gate — if filters ran after the
  stretch; the observed 300 ms-then-stretched result proves filters run first)
  and `test_filters_run_before_the_single_watermark` (a recording watermarker
  sees the trimmed length, not the raw stitch).
- Single-shot path: DC → trim → internal on the model's final buffer. Note the
  honest asymmetry: families that watermark inside `generate` (Chatterbox) have
  already marked the single-shot buffer when these filters run, whereas the
  chunked path filters *before* its serving-side watermark. No runner change was
  made (the spec's wiring point is the serving layer).
- `speed` is applied after the filters, so pause durations are defined on the
  synthesized timeline and the stretch scales them uniformly with the speech;
  the watermark stays last so trim cannot remove marked samples.

**Upstream fourth filter — `remove_long_unvoiced_segments`: SKIPPED (2026-11-09)**

- Upstream implements it with **Parselmouth/Praat pitch analysis** (`to_pitch`,
  `get_VoicedVoicelessUnvoiced`), which would be a new heavy dependency; this
  task forbids new dependencies.
- Its intent overlaps `fix_internal_silence`: both collapse long non-speech
  stretches. The difference is the detector (pitch/voicing vs short-time
  energy). `fix_internal_silence` already removes long *silent* pauses without
  touching voiced or unvoiced-but-loud speech (fricatives, breaths are above the
  energy floor); a pitch-based remover would additionally cut long *quiet but
  non-silent* unvoiced regions, which is a different, more aggressive product
  decision. Composing the two would need a precedence rule and a new dependency
  for no clear gain, so it is not adopted. Revisit only if a "remove long
  unvoiced runs" capability is requested on its own, with an explicit energy
  fallback instead of Parselmouth.

**Deviations**

- **`task_handler/inputs.py` needed no change** (same as 12-1): `from_openai`
  passes the pydantic `GenerationOptions` object through whole and the serving
  layer reads the new flags with `getattr`, so the two code layers that carry
  fields are `protocol.py` and `common/inputs.py`; the third layer is the
  pass-through. Pinned by wire-contract tests.
- **`post_generation_filters` was not used.** It is parsed and copied onto chunk
  sub-requests but is a declarative future pipeline concept (6-A), not the
  three boolean filters; wiring it would be a different feature.
- **`peak_normalize` was not wired** (explicitly out of this task's field list).
- **All-silent trim returns an empty waveform.** Documented; an opt-in filter on
  digital silence has nothing to keep. Real models do not emit digital silence,
  and `to_wire_audio`'s artifact guard falls back cleanly if a caller ever hits
  it.
- **Audio-regression harness:** no new axis was added. Its existing axes compare
  two configurations of the *same* computation; the filters are an intentional
  sample change, and on real-model output a "must differ" rule is
  non-deterministic (a model may legitimately have no DC/long silence). The
  honest on-vs-off + valid-WAV gate lives on synthesized fixtures
  (unit + integration). A dated note was added to the harness docstring
  (`scripts/audio_regression.py`) recording why.
- **`summary.md` untouched per task scope**, so the last checklist box stays
  unticked (tests, docs and CHANGELOG are green). 12-2 precedent unchanged.
- **12-4 cost profile (for the benchmark task):** measured on CPU (this host,
  `hse` env, 2026-11-09) on **60 s of 44.1 kHz mono** random audio with a
  0.5 s lead/trail and a 1 s internal gap: `remove_dc_offset` ~**5 ms**,
  `trim_silence` ~**17 ms**, `fix_internal_silence` ~**16 ms** (~38 ms total per
  audio-minute, i.e. ~0.06 % of real time). Each filter is a small number of
  `O(n)` passes; the gate builds one float64 power array plus one float64
  cumulative-sum array (≈16 bytes/sample transient, ~42 MB for 60 s, ~420 MB for
  10 min at 44.1 kHz) before the final float32 buffer. That transient is the only
  thing worth watching on memory-constrained hosts; if 12-4 sees multi-minute
  buffers, a float32 accumulator or chunked gate is the escape hatch (not done
  here because float64 avoids drift on long sums). The chunked path pays DC once
  per chunk (still `O(total samples)`) then trim/internal once on the stitched
  buffer; the non-native speed stretch remains the only expensive post-step.
  Worth measuring the chunked path with DC on vs off to confirm the per-chunk
  pass does not perturb the 12-4 RTF budget.

**Full-suite result:** `1450 passed, 5 skipped` (`tests/unit tests/integration
tests/e2e/wire_contract`, 2026-11-09; baseline 1413 + 37 new tests: 23 filter
unit tests in `tests/unit/common/test_audio_utils.py`, 9 filter/chunking
integration tests in the new `tests/integration/test_tts_output_filters.py`, and
5 wire-contract additions in `tests/unit/wire_contract/test_long_form_fields.py`).
The gated real-model smoke tests were added to
`tests/e2e/tts/test_long_form_chunking.py` (outside the suite command, skipped
unless `HSE_E2E_MODEL_TESTS=1`).


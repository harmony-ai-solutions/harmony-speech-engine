# Phase 8-1: Batching Core Extensions

> **Plan:** [Engine Modernization](summary.md) · **Phase:** 8 — Batching across all steps
> **Depends on:** Phase 5-2 infra (`task_handler/batching.py`); independent of Phase 7 (composes: cache hits shrink batches)
> **Codebase mapping consulted:** `.planning/codebase/ARCHITECTURE.md`, `docs/batching.md` (5-3)
> **State reference:** verified 2026-11-05 at `b81352d`+`d28b81d`

## Objective

Generalize the Phase 5-2 padded-batch machinery from the phone-sequence axis (synthesizers) to the two axes the remaining steps need — **mel/feature frames** (tone converters, restorers, vocoders) and **audio samples** (embedders) — so 8-2…8-4 are mostly wiring. Also generalize the config semantics so every newly batched model gets the same `batched_inference` + `batch_token_budget` contract the synthesizers already have.

## Context (self-contained)

`harmonyspeech/task_handler/batching.py` today (verified): `supports_batched_inference(model_config)`, `batched_inference_enabled(model_config)`, `pad_phone_batch(sequences, pad_id, device) -> (padded, lengths)`, `pad_feature_batch(features, pad_value, device) -> padded`, `plan_batches(...)` (length-similar grouping `T_max/T_min ≤ 4` + token budget), `default_token_budget(lengths, max_batch_size)`, `collapse_per_sample(values, tol)` (broadcast a shared scalar param to per-sample), `trim_decoder_output(...)`.

Currently `supports_batched_inference` is allow-listed to the Phase 5-2 model types (OpenVoice V1 synth + Melo synth). Handlers for everything else still loop per item with honest `max_batch_size: 1` (values verified in `config.gpu.yml`: hs1-* = 1, ov*-tone-converter* = 1, voicefixer-* = 1).

**Scope guard (product decision 2026-11-05):** every step whose architecture supports batching gets it, **except the Chatterbox family entirely** (AR TTS now; its non-AR parts — VC, embedding — wait for the dedicated later Chatterbox workstream). KittenTTS and Silero VAD are expected to remain loops (single-stream upstream APIs) — 8-5 documents the verdicts.

## Implementation steps

### 1. Axis-general helpers in `batching.py`

- `pad_feature_batch` already pads a list of `[bins, frames]` features to `[B, bins, T_max]` — verify it returns lengths too (add if not: callers need `frame_lengths` for trimming and for `plan_batches`).
- Add `trim_feature_output(outputs, frame_lengths, dim=-1)` — the mel-frame sibling of `trim_decoder_output` (split a `[B, bins, T_max]` batch back into per-request slices, padding-aware).
- Audio axis for embedders (8-2): `pad_waveform_batch(waveforms)` → `[B, 1, T_max]` + `sample_lengths` (waveforms are all one sample rate after the boundary decode + `resampled()` views, Phase 2 — assert it).
- `plan_batches` + `default_token_budget` already take generic `lengths` — document the three axis units ("token" = phones for synths, mel frames for TC/restorer/vocoder, samples for embedders) and confirm no phone-specific assumptions leak.

### 2. Config semantics

- Extend `supports_batched_inference` to the new model types **as each of 8-2/8-3/8-4 lands** (not upfront — the flag must not advertise batching before the handler can do it). `batched_inference` (bool override) and `batch_token_budget` (axis-unit cap) apply unchanged.
- Keep the Phase 5-2 conventions: `len(inputs) == 1` takes the single-sample fast path; any batched-path exception falls back to the per-item loop for that request (error isolation per Phase 0-3).

### 3. Gate plumbing

The `batching:1-vs-N` axis in `scripts/audio_regression.py` (5-2) is family-keyed; extend its family map as each step lands so 8-2…8-4 can gate immediately (B=1 output vs each sample synthesized inside a B=4 batch; deterministic settings — fixed noise params via `generation_options`, seeds where exposed; tolerance per the numeric-noise note in `docs/benchmarks/audio-regression.md`: sub-LSB float32 accumulation deltas pass, structural differences fail).

## Files to create/modify

| File | Change |
|---|---|
| `harmonyspeech/task_handler/batching.py` | lengths-aware feature padding, `trim_feature_output`, `pad_waveform_batch`, axis docs |
| `harmonyspeech/common/config.py` | docstring: axis unit per model type (no schema change) |
| `scripts/audio_regression.py` | family-map extension points for new batched steps |
| `tests/unit/inference_flow/test_batching_extensions.py` (new) | helper tests |

## Tests

- Unit: feature padding + trim round-trip (known lengths → original slices recovered exactly); waveform padding round-trip; `plan_batches` grouping on frame/sample axes (mixed lengths split per `T_max/T_min ≤ 4`).
- Unit: budget enforcement on each axis.

## Checklist

- [x] Feature/waveform padding + trimming helpers with lengths semantics
- [x] Axis units documented; budget applies to all three axes
- [x] Harness family map ready for new steps
- [x] Helper unit tests green
- [x] `summary.md` Implementation Status updated

## Deviations

### API design (implemented 2026-11-06)

**Lengths-aware feature padding — flag, not a sibling function.**
`pad_feature_batch(features, pad_value=0.0, device=None, return_lengths=False)`.
The Phase 5-2 MeloTTS caller (two call sites, four `padded_bert` /
`padded_ja_bert` constructions) destructures only the tensor, so the default
return stays `padded` and that source is untouched. With `return_lengths=True`
it returns `(padded [B, bins, T_max], frame_lengths long[B])`. One code path
means one set of shape/validation messages; a sibling function would have
duplicated the channel-check and padding loop. Both return shapes are spelled
out in the docstring.

**`trim_feature_output(outputs, frame_lengths, dim=-1)`.**
Accepts both ranks the handlers produce rather than asserting 3-D:
`[B, T_max]` (an embedding over frames) and `[B, bins, T_max]` (mel/feature
maps), generalising to any rank ≥ 2. The batch axis is always `0`; `dim` is the
padded time axis (normalised, so positive indices work too). Each returned
tensor drops the batch axis and keeps every other axis, i.e. `[bins, T_i]` /
`[T_i]` — the same "drop batch, keep channels" contract as
`trim_decoder_output`, so a `[B, 1, T]` waveform trims to `[1, T_i]` and the
sample axis reuses this helper instead of a fourth trimmer. Validation mirrors
`trim_decoder_output`: length count must equal `outputs.shape[0]`, no row may
ask for more frames than the padded axis holds, negative lengths clamp to 0.

**`pad_waveform_batch(waveforms, sample_rates=None, pad_value=0.0, device=None)`
→ `(padded [B, 1, T_max], sample_lengths long[B])`.**
Two accepted forms so the 8-2 embedder handlers can pass whatever they already
hold: `[(waveform, sample_rate), ...]` rows (artifact-style) or bare tensors
plus a parallel `sample_rates=` list. Rates are mandatory in effect — if they
come as rows, the rows carry them — because the equal-rate invariant cannot be
checked without them; a mismatch raises `ValueError` naming the distinct rates.
Integer (PCM) rows are widened to `float32`; `pad_value` defaults to silence
(0.0).

**No family added to `BATCHED_INFERENCE_FAMILIES`.**
Left as `{"OpenVoiceV1Synthesizer", "MeloTTSSynthesizer"}` per the scope guard:
8-2/8-3/8-4 add their model types when the handler can actually honour the
flag.

**Axis units.** The module docstring now carries the three-axis table
(`token` = phones for synthesizers, `frame` = mel/feature frames for tone
converters/restorers/vocoders/HarmonySpeech encoder, `sample` = audio samples
for embedders) and states that `plan_batches` / `default_token_budget` /
`ModelConfig.batch_token_budget` see only a length vector, so they work in
whichever unit the step passes and no phone-specific assumption leaks. The
`ModelConfig` docstrings for `batched_inference` and `batch_token_budget`
document the unit per model type (docstring-only, no schema change).

### `scripts/audio_regression.py` extension mechanism

* **Allow-list + pending map** (the one-line activation): `BATCHING_AXIS_ENABLED`
  is the existing family gate (unchanged membership: `openvoice_v1`,
  `melotts`). `BATCHING_AXIS_PENDING` names the families whose case sets are
  registered but whose handlers do not batch yet, together with the task that
  activates them (`openvoice_v2` → 8-2/8-3, `harmonyspeech` / `voicefixer` →
  8-4). `_cases_for_axis` now refuses a pending family with that task name
  instead of the old "no batching_cases" message; enabling a family later is a
  single edit to `BATCHING_AXIS_ENABLED`.
* **Case sets registered now**: `openvoice_v2` gets the same four mixed-length
  `clone_voice_*` cases as `openvoice_v1`; `harmonyspeech` gets four
  mixed-length clone cases (`batch_1..4`); `voicefixer` gets four `restore_*`
  cases over the four distinct reference clips (a gate needs ≥2 distinct
  inputs; `restore_voice_a` is the primary case 8-4 names).
* **Worker generalized by kind**: request construction moved into
  `_build_batching_request_input(case, request_id, text_table, deterministic)`,
  which maps `tts` → `TextToSpeechRequestInput`, `vc` →
  `VoiceConversionRequestInput`, `restore` → `AudioConversionRequestInput`.
  Previously the worker hard-coded TTS requests, so a restore family could not
  have been gated at all. The drive loop (add all → `step_async`) and the
  deterministic `GenerationOptions(noise_scale=0, noise_scale_w=0, seed=0)` are
  unchanged, so existing families/axes behave exactly as before.
* Tests for the plumbing live in `tests/unit/benchmarks/test_audio_regression.py`
  (allow-list disjointness, case-set well-formedness, refusal message, the
  per-kind request builder) rather than in the new batching-helper test file.


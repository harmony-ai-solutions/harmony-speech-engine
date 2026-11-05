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

- [ ] Feature/waveform padding + trimming helpers with lengths semantics
- [ ] Axis units documented; budget applies to all three axes
- [ ] Harness family map ready for new steps
- [ ] Helper unit tests green
- [ ] `summary.md` Implementation Status updated

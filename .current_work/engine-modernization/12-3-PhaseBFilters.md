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

- [ ] Filters landed + ordering pinned (per-chunk DC, final trim/internal)
- [ ] Upstream fourth filter evaluated (adopted or skipped with dated note)
- [ ] Tests + docs + CHANGELOG; suites green; `summary.md` ticked

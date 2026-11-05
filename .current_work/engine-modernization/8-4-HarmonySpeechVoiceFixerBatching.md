# Phase 8-4: HarmonySpeech & VoiceFixer Batching

> **Plan:** [Engine Modernization](summary.md) · **Phase:** 8 — Batching across all steps
> **Depends on:** 8-1
> **Codebase mapping consulted:** `.planning/codebase/ARCHITECTURE.md`; vendored trees `harmonyspeech/modeling/models/{harmonyspeech,voicefixer}/`; Phase 4 leftovers (VoiceFixer GPU placement bug)
> **State reference:** handler names verified 2026-11-05

## Objective

Batch the remaining one-shot, fixed-graph steps: the HarmonySpeech pipeline (synthesizer + vocoder; its encoder is handled in 8-2) and the VoiceFixer pipeline (restorer + vocoder). Also fix the pre-existing **VoiceFixer GPU device-placement bug** (`input cuda vs weight cpu`, flagged since Phase 4) — it blocks this family's GPU gate entirely.

## Context (self-contained)

- `_execute_harmonyspeech_synthesizer` produces a `MelArtifact` (+ `breaks`) per request; `_execute_harmonyspeech_vocoder` turns mels into audio. The synth is expected to be VITS/`SynthesizerTrn`-family (the 5-2 pattern should transfer almost verbatim); **verify-first** the vendored signature (`modeling/models/harmonyspeech/`) — if it is already the same `SynthesizerTrn` used by OpenVoice V1, this may be a near-copy of 5-2's handler change.
- `_execute_voicefixer_restorer` runs a UNet over mel (input via `MelArtifact`/audio per Phase 2 — no JSON path remains); `_execute_voicefixer_vocoder` is a conv vocoder over mel. UNets and conv vocoders are batch-friendly with frame padding; **verify-first** both vendored forwards for `[B, bins, T]` acceptance and length semantics.
- VoiceFixer GPU bug: the restorer's weights stay on CPU while its input lands on CUDA (loader moves only part of the model). Fix in the loader/runner init (consistent with how the other GPU runners place their models) — it's a prerequisite for running this family's gate on GPU; CPU runs are unaffected.
- Both families currently run honest `max_batch_size: 1` loops.

## Implementation steps

1. **VoiceFixer device fix**: locate the partial placement (loader `modeling/loader.py` VoiceFixer branch vs runner init), move the full model consistently, add a unit test asserting every parameter of the loaded (stubbed) model sits on the configured device. Re-run the family on GPU — it should now work at all (it hasn't since before Phase 4).
2. **Vendored audits** (hs1 synth/vocoder, voicefixer restorer/vocoder): document signatures; add batched variants only where missing, mirroring single-sample math exactly, with B=1 bit-equality tests (5-2 precedent).
3. **Handler batching** (four handlers): rows → pad on the mel-frame axis → `plan_batches` → batched forward → trim/unpack → per-request artifacts. Keep per-family quirks bit-faithful:
   - HarmonySpeech vocoder consumes `MelArtifact.breaks` (silence insertion points) — batching must not disturb the per-request breaks pairing (re-read the Phase 2 handler before editing; the breaks apply **after** decode, so pad/trim concerns don't touch them).
   - VoiceFixer restorer output mel feeds its vocoder as `MelArtifact` — trim per-request frames exactly.
4. **Config + gates**: `supports_batched_inference` += the four types; defaults on after gates; `batching:1-vs-4` harness axis extended to `harmonyspeech` (tts cases) and `voicefixer` (`restore_voice_a`) families.

## Files to create/modify

| File | Change |
|---|---|
| `harmonyspeech/modeling/loader.py` (or runner init) | VoiceFixer full-device placement fix + test |
| `harmonyspeech/modeling/models/harmonyspeech/…`, `…/voicefixer/…` | batched forwards if missing (+ B=1 equality tests) |
| `harmonyspeech/task_handler/model_runner_base.py` | four batched handlers |
| `harmonyspeech/task_handler/batching.py` | supports-entries |
| `tests/unit/modeling/test_batched_hs_voicefixer.py` (new) | tests |

## Tests

- Unit: VoiceFixer placement (all params on configured device, stubbed model); B=1 bit-equality ×4 models; breaks pairing preserved under batching (hs1 vocoder).
- Integration: multi-request restoration + hs1 TTS with mixed-length inputs → per-request output lengths + pairing invariants.
- Harness: `batching:1-vs-4` green for `harmonyspeech` and `voicefixer` (GPU; the placement fix makes the VoiceFixer GPU axis runnable for the first time — record the family's dtype axis too while it's finally testable, appending to `docs/benchmarks/audio-regression.md`).

## Checklist

- [ ] VoiceFixer GPU placement bug fixed + regression-tested (family runs on GPU)
- [ ] Four handlers batched (or evidence-based skip documented) with loop fallback
- [ ] B=1 gates + family harness axes green
- [ ] VoiceFixer dtype axis recorded now that GPU runs work
- [ ] `summary.md` Implementation Status updated

# Phase 8-2: Embedder Batching

> **Plan:** [Engine Modernization](summary.md) · **Phase:** 8 — Batching across all steps
> **Depends on:** 8-1; composes with 7-2 (cache hits leave the batch)
> **Codebase mapping consulted:** `.planning/codebase/ARCHITECTURE.md`, vendored model trees under `harmonyspeech/modeling/models/{openvoice,harmonyspeech}/`
> **State reference:** handler names verified 2026-11-05

## Objective

Batch the speaker-embedding steps so a flood of clone requests with **distinct** voices (the cache-miss path) also moves through the embed step in one forward: `_execute_openvoice_tone_converter_encoder` (V1+V2) and `_execute_harmonyspeech_encoder` move from per-item loops to padded batched calls.

## Context (self-contained)

- Both handlers consume audio (the OpenVoice TCE consumes **VAD-segmented** audio — per request there may be multiple segments, mirroring the `(request, sentence)` rows of 5-2's synthesizers; HarmonySpeech encoder consumes the full clip) and emit `SpeakerArtifact.embedding`.
- **Verify-first (mandatory):** re-read the vendored encoder forwards before planning surgery —
  - OpenVoice `SpeakerEncoder`/TCE encoder (`modeling/models/openvoice/`): a `resnet`/`lstm`-style ref encoder over mel frames. Determine whether `forward` accepts `[B, T]` (batch-first mel) or is written single-sample; whether segment batching is supported upstream or must be added (our code — same pattern as 5-2's `_infer_impl` extraction).
  - HarmonySpeech encoder (`modeling/models/harmonyspeech/`): same questions. If its checkpoint graph is single-sample-only and surgery is disproportionate, record the verdict and leave it a documented loop (honest value stays 1) — an explicit, evidence-based skip is an acceptable outcome.
- Embedders are small; the win is **step latency under flood** (20 requests × per-item forward → 1–2 padded forwards), not single-request latency.

## Implementation steps

1. **Vendored forward audit** (both families): document the exact signature, whether lengths are accepted, and the batch-capable path. Add a batched variant only if missing, mirroring the single-sample implementation exactly (no math changes), with B=1 bit-equality unit tests (5-2 precedent).
2. **Handler batching**: build `(request, segment)` rows → `pad_feature_batch`/`pad_waveform_batch` (per 8-1, on the axis the model wants) → `plan_batches` with the appropriate lengths → single forward per sub-batch → split outputs by lengths → one `SpeakerArtifact.embedding` per request (OpenVoice: aggregate per-request segment embeddings exactly the way the loop does today — re-read the aggregation (mean/stack) and reproduce it bit-for-bit).
3. **Config**: extend `supports_batched_inference` for the two model types; `batched_inference` default **on** only after the gate passes; fallback loop for `len==1` and on exceptions.
4. **Cache composition** (7-2): resolve cache hits first; only misses enter the batch; outputs reassembled in request order.
5. **Gate**: `batching:1-vs-4` axis for the full clone pipelines whose embed step changed (openvoice_v1, openvoice_v2 — clone cases; the embedding output feeds synthesis, so the end-to-end audio comparison covers it; add an embedding-tensor equality assert for the embed-only path).

## Files to create/modify

| File | Change |
|---|---|
| `harmonyspeech/modeling/models/openvoice/…`, `harmonyspeech/modeling/models/harmonyspeech/…` | batched forward variants if missing (+ B=1 equality tests) |
| `harmonyspeech/task_handler/model_runner_base.py` | batched `_execute_openvoice_tone_converter_encoder`, `_execute_harmonyspeech_encoder` |
| `harmonyspeech/task_handler/batching.py` | `supports_batched_inference` entries |
| `tests/unit/modeling/test_batched_embedders.py`, `tests/integration/test_batched_synthesis.py` (extend) | tests |

## Tests

- Unit: vendored batched forward B=1 == loop path (bit-identical); segment aggregation equivalence (N segments sequentially vs padded batch → same embedding within tolerance / bitwise if the op order allows).
- Integration: 4 clone requests with distinct references → correct per-request embeddings (pairing invariants), embeddings equal to loop-computed reference values.
- Harness: `batching:1-vs-4` green for openvoice_v1/v2 clone cases.

## Checklist

- [ ] Vendored forward audit documented (both families; skip verdicts recorded if any)
- [ ] Both handlers batched (or evidence-based skip documented) with loop fallback
- [ ] Segment aggregation bit-faithful; B=1 gates green
- [ ] Cache-hit composition verified (hits never enter the batch)
- [ ] `summary.md` Implementation Status updated

# Phase 8-3: Tone-Converter Batching

> **Plan:** [Engine Modernization](summary.md) · **Phase:** 8 — Batching across all steps
> **Depends on:** 8-1 (feature-axis helpers); 8-2 beneficial (clone flood composes) but not required
> **Codebase mapping consulted:** `.planning/codebase/ARCHITECTURE.md`; 5-2 precedents (`_infer_impl` extraction, `infer_batched`, `collapse_per_sample`)
> **State reference:** handler names verified 2026-11-05

## Objective

Batch the tone-conversion step — `_execute_openvoice_tone_converter` (V1 + V2 share the handler; distinct model configs) — from a per-item loop to padded mel-frame batches. This is the step that made the OV2 clone path throughput **flat** in the 5-3 benchmark (1.86 → 1.90 req/s at c=8: synthesis batches at 8, then tone-converts 8× sequentially). With 8-2 + this, the full clone pipeline (VAD→embed→synth→TC) batches end to end.

## Context (self-contained)

- The tone converter consumes the **synthesizer's output audio for this request** (converted to mel by the TC's own mel frontend — re-read the handler + `prepare_openvoice_tone_converter_inputs` in `task_handler/inputs.py` for the current artifact flow: audio in → mel → flow/decoder → audio out with the target speaker embedding) plus the target `SpeakerArtifact.embedding`.
- It is a **one-shot flow + decoder** (VITS-family `SynthesizerTrn`-adjacent architecture — the same class of model 5-2 already batched on the synthesis side). The expectation is a batched path exists or is trivially added in the vendored tree (`modeling/models/openvoice/`), accepting `[B, bins, T_max]` mel + `x_lengths` + `[B, D]` speaker embeddings.
- **Verify-first (mandatory):** re-read the vendored TC forward. Confirm: batched input acceptance, lengths semantics, per-sample noise/scale params (vectorizable via `collapse_per_sample` — the 5-2 broadcast pattern `[B,1,1]`), output frame-count semantics for unpadding. If the vendored forward is single-sample, extract the shared math and add `infer_batched` exactly like 5-2 did for the synthesizers (no math changes; B=1 bit-equality tests).
- Phase 7 interaction: the **target embedding** may be a cache hit (7-2); the TC's *input audio* is per-request synthesis output and is never cached.

## Implementation steps

1. Vendored TC audit + (if needed) `infer_batched` with B=1 bit-equality unit tests.
2. Handler: build `(request, …)` rows from the forwarded synth artifacts → mel frontend per request → `pad_feature_batch` (frame lengths) → `plan_batches` → single `infer_batched` per sub-batch → `trim_feature_output`-style unpack → per-request `AudioArtifact` outputs (device moves explicit, Phase 2 discipline).
3. Config: `supports_batched_inference` += OpenVoice V1/V2 tone-converter types; `batched_inference` default on after the gate; loop fallback (`len==1`, exceptions).
4. Gate: `batching:1-vs-4` on the **full clone pipelines** (openvoice_v1, openvoice_v2 `clone_voice_a`/`clone_voice_b` cases) — B=1 vs in-batch end-to-end audio comparison, deterministic noise settings; plus a TC-only unit comparison (same input audio + embedding, loop vs batched → byte-identical or sub-LSB per the numeric-noise note).

## Files to create/modify

| File | Change |
|---|---|
| `harmonyspeech/modeling/models/openvoice/…` | TC `infer_batched` if missing (+ B=1 equality tests) |
| `harmonyspeech/task_handler/model_runner_base.py` | batched `_execute_openvoice_tone_converter` |
| `harmonyspeech/task_handler/batching.py` | supports-entry |
| `tests/unit/modeling/test_batched_tone_converter.py` (new), integration extensions | tests |

## Tests

- Unit: B=1 bit-equality; mixed-frame-length unpack (known lengths → correct output lengths per request); per-sample params broadcast (`collapse_per_sample`).
- Integration: 4 clone requests end-to-end (with 8-2: batched embed + synth + TC) → pairing invariants, per-request audio lengths correct.
- Harness: `batching:1-vs-4` green for both clone pipelines.

## Checklist

- [ ] Vendored TC audit documented; `infer_batched` added if needed with B=1 proof
- [ ] Handler batched with frame-length planning + loop fallback
- [ ] Gates green (TC-level + end-to-end clone pipelines)
- [ ] `summary.md` Implementation Status updated

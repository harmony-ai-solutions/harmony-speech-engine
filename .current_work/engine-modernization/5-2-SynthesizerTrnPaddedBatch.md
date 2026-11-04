# Phase 5-2: Padded Batch on SynthesizerTrn

> **Plan:** [Engine Modernization](summary.md) · **Phase:** 5 — Honest batching (HSE only)
> **Depends on:** 5-1, 4-2 (regression gate for output equivalence)
> **Codebase mapping consulted:** `.planning/codebase/ARCHITECTURE.md`, research §5.3(d)

## Objective

Implement **real batched inference** for the one-shot synthesis models — OpenVoice V1 `SynthesizerTrn` and MeloTTS synthesizers — replacing the per-item loop in their `_execute_*` handlers with a single padded-batch forward per step. These are fixed-graph one-shot models (no autoregression): text → phones → mel/vocoder in one pass, so `[B, T_max]` padding is the standard, safe technique. Expected: near-linear throughput scaling per step up to memory limits on the 4060 Ti.

Out of scope by decision (see plan summary): Chatterbox TTS (autoregressive, needs a KV-cache story — separate workstream), Whisper (already genuinely batched via `BatchedInferencePipeline`, `model_runner_base.py:371-394`), KittenTTS (CPU ONNX, per-item generate API).

## Context (self-contained)

- Today: `_execute_harmonyspeech_synthesizer` (`model_runner_base.py:185-231`), `_execute_openvoice_synthesizer` (`:397` region), Melo handler (`:456` region) all carry `FIXME: not properly batched` and loop per item.
- Upstream model APIs: `SynthesizerTrn.forward(x, x_lengths, ...)`/`infer` accept batched `x [B, T]` with per-sample lengths (verify exact signature in `harmonyspeech/modeling/models/openvoice/` + `melo/` trees — the vendored copies in HSE are the source of truth, not the original repos).
- Per-sample generation params (speed/`alpha`, noise scales) are vectors at batch level in the VITS-family API — confirm; if any param is scalar-only in the vendored code, extend the vendored forward (it's our code) rather than sharding the batch.

## Implementation steps

### 1. Re-read the vendored synthesizer forwards

Document the exact batched signature(s) for OpenVoice V1 synth and Melo synth (input shapes, lengths semantics, per-sample param support, output unpacking). If a batched `infer` variant is absent, add one to the vendored model file mirroring the single-sample path (pad → forward → unpad using `x_lengths`), covered by unit tests against the single-sample path for B=1 equality.

### 2. Batched handler

In each synthesizer handler: build the padded batch from the prepared per-request inputs (phone-id sequences padded to `T_max` with the model's pad id; `x_lengths`; stacked speaker embeddings `[B, 256]`; per-sample params), single `self.model.infer(...)` call, split outputs by lengths (mel frames per sample = f(x_length)), then run the existing per-sample post-processing (per 2-3: produce `AudioArtifact`/`MelArtifact` per request). Keep the loop as fallback for `len(inputs) == 1` and behind a per-model config toggle if it proves delicate:

```yaml
- name: "ov2-synthesizer-en"
  model_type: "MeloTTSSynthesizer"
  batched_inference: true   # default true after gate; false = legacy loop
```

### 3. Memory-aware batch cap

Padded-batch memory scales with `B × T_max`; a pathological mix (one long + many short texts) wastes compute on padding. Guard: split the step's request list into sub-batches where `T_max / T_min > 4` (length-similar grouping, greedy), and cap sub-batch size so `B × T_max ≤` a per-model token budget (start: `max_batch_size × median_text_len`, measured and tuned in 5-3).

### 4. Equivalence gate

4-2 harness axis `batching:1-vs-N`: B=1 outputs must be bit-identical (deterministic families) or threshold-equal (noise-scale families) to each sample synthesized inside a B=4 batch. Any divergence = padding/unpacking bug.

## Files to create/modify

| File | Change |
|---|---|
| `harmonyspeech/modeling/models/openvoice/…`, `…/melo/…` | batched infer (if missing) + B=1 equivalence units |
| `harmonyspeech/task_handler/model_runner_base.py` | batched handlers + length-similar sub-batching + cap |
| `harmonyspeech/common/config.py` | `batched_inference` per-model flag |
| `tests/integration/` | batched synthesis integration tests |

## Tests

- Unit: vendored batched infer B=1 == single path (bit-identical).
- Unit: unpacking — synthetic batch with known lengths yields correct per-sample mel shapes.
- Harness (GPU): `batching` axis within thresholds.
- Integration: 4 mixed-length requests → 4 correct-length outputs, ids paired (5-1 invariants hold).

## Checklist

- [x] Batched infer available (vendored code) with B=1 equivalence proof
- [x] Handlers batched with length-similar grouping + memory cap; config toggle
- [x] 4-2 batching axis passes
- [x] Integration tests green
- [x] `summary.md` Implementation Status updated

## Implementation notes (2026-11-04)

* **Vendored forwards re-read, not assumed.** Both `SynthesizerTrn.infer`
  implementations already accepted `[B, T]`/`x_lengths`; what they did not
  accept was *per-sample* `length_scale`/`noise_scale`/`noise_scale_w` (scalars
  only, i.e. one speed for the whole batch). The shared math moved into
  `_infer_impl` (unchanged for the single-sample path) and a new
  `infer_batched` normalises those params to `[B, 1, 1]` tensors, forwards once
  and returns the per-sample decoder frame counts (`y_lengths`) so the caller can
  unpad without guessing. `infer` still returns its original 4-tuple.
* **Batching happens over sentences, not requests.** The existing handlers
  already split a request into sentences and concatenate them with a 50 ms gap,
  so the step is flattened into `(request, sentence)` rows, batched, and
  reassembled per request in input order. A request with several sentences
  therefore benefits even when it is the only request in the step.
* **Memory guard.** `plan_batches` caps a group at `max_batch_size`, at
  `T_max / T_min <= 4` and at `batch_token_budget` (default
  `max_batch_size × median length`, overridable per model).
* **Sampling parameters moved onto the request.** The gate needs a deterministic
  comparison, and the models can already do that; `generation_options` gained
  `noise_scale`, `noise_scale_w`, `sdp_ratio` (None = current hard-coded
  defaults), used by both the loop and the batched path. This is a small,
  backward-compatible API addition rather than a test-only escape hatch.
* **Gate method (4-2 `batching:1-vs-N`).** Side `1` = `max_batch_size: 1` +
  `batched_inference: false` (legacy loop), side `N` = `max_batch_size: N` +
  batched, four different-length texts, both sides noise-free and with TF32
  disabled (TF32's kernel choice depends on the batch shape and would swamp the
  signal). Result: OpenVoice V1 renders byte-identical WAVs, MeloTTS differs by
  1.6e-5 max (below one 16-bit LSB). Details in `docs/benchmarks/throughput.md`.
* **Known, documented boundary effect.** The convolutional vocoder's last frame
  sees a different right-hand context inside a padded batch (explicit zeros plus
  conv bias) than alone; measured ≤1e-4 peak with the shipped weights, and the
  gate's 2e-3 bound covers it. Unit tests assert the exact parts (B=1 bit
  identity, predicted durations and post-attention prior) and a looser interior
  waveform match.
* **Deviations.** Full OpenVoice V2 clone throughput does not improve (per-item
  tone-converter hops dominate; those handlers are out of scope and stay at
  `max_batch_size: 1`). Long-text synthesis shows no gain (compute-bound). Both
  recorded in `docs/benchmarks/throughput.md`; `max_batch_size: 8` is the
  measured sweet spot for short texts (still rising at 8, no knee observed).


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

## Vendored tone-converter audit

> Audited 2026-11-06 against the working tree (line numbers are post-change). The
> converter graph is batch-capable through the encoder and the flow, but **not**
> through the decoder: a padded row's waveform drifted by up to **0.30 peak** until
> the decoder's padding region was held at zero per stage. A shared-impl
> extraction plus a batched entry point was required — no evidenced skip.

### Class and configuration

- `OpenVoiceV1ToneConverter` / `OpenVoiceV2ToneConverter` are both registered as
  `("openvoice", "SynthesizerTrn")` (`harmonyspeech/modeling/models/__init__.py`).
  The converter configs have no speaker table (`n_speakers: 0`), so the class
  builds `ref_enc` instead of `enc_p`/`sdp`/`dp`/`emb_g`
  (`openvoice.py:559-568`); `voice_conversion` only touches `enc_q`, `flow` and
  `dec`, so the speakerless build is irrelevant to this step.
- `spec_channels = filter_length // 2 + 1`, `gin_channels = 256` for the shipped
  checkpoints; `zero_g` is `false`.

### Exact signature (before this task)

`SynthesizerTrn.voice_conversion(self, y, y_lengths, sid_src, sid_tgt, tau=1.0)`
(`openvoice.py:672`):

```python
z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g_src, tau=tau)
z_p = self.flow(z, y_mask, g=g_src)
z_hat = self.flow(z_p, y_mask, g=g_tgt, reverse=True)
o_hat = self.dec(z_hat * y_mask, g=g_tgt)
return o_hat, y_mask, (z, z_p, z_hat)
```

`y` is `[B, bins, T_max]`, `y_lengths` is `[B]`, `sid_src`/`sid_tgt` are
`[B, gin_channels, 1]`, and the return is `(waveform [B, 1, T], y_mask
[B, 1, T_frames], extras)`.

### End-to-end batch acceptance

- **Encoder (`enc_q`)** — batch-safe. `PosteriorEncoder.forward`
  (`openvoice.py:234`) builds `x_mask = sequence_mask(x_lengths, x.size(2))` on the
  **time** axis, masks `pre` and `proj`, and `WN.forward` re-masks `x` after every
  layer (`modules.py:167`) and zeroes its output (`modules.py:171`). A padded row's
  conv inputs past its length are therefore zeros — exactly what a solo conv sees
  implicitly — so the bias never feeds the next layer's edge frames.
- **Flow (`ResidualCouplingBlock`)** — batch-safe for the same reason:
  `ResidualCouplingLayer.forward` masks `pre`/`post` and runs the masked `WN`; the
  coupling update multiplies by `x_mask`. `Flip` only reverses channels.
- **Noise** — `z = (m + randn_like(m) * tau * exp(logs)) * x_mask`
  (`openvoice.py:240`). `randn_like` draws on the full padded shape, so each row
  gets independent noise (padding positions are masked), but not the solo run's
  RNG stream: with `tau > 0` a row inside a batch is statistically, not
  bit-for-bit, the loop's output. `tau` is a plain scalar multiplier, so a
  per-sample `[B, 1, 1]` tensor broadcasts with no math change.
- **Decoder (`Generator.forward(x, g=None)`)** — **not** padding-safe. `conv_pre`
  has a bias (`openvoice.py:260`), `self.cond(g)` adds the speaker bias to every
  position, and the first `ConvTranspose1d` spreads the padding frames (which hold
  bias, not zeros) back into the valid tail. The resblock convs (kernel 3,
  dilation 1/3/5) then spread the contamination much further inward. Measured on
  the tiny vendored class: a 17-frame row inside a 3-row padded batch differed
  from its solo run by 0.295 peak, with the first differing sample at index 3 of
  68 — not a one-sample edge effect. The fix is the 8-2 lesson applied to 1-D:
  hold each row's padding region at zero **after every stage**, re-expanding the
  mask by each transposed conv's stride, and pass it to the resblocks through
  their existing `x_mask` parameter (which already multiplies before each conv).
- **Output frame-count semantics** — the decoder output length is exactly
  `T_frames × prod(upsample_rates)` (with `padding=(k-u)//2` and `k = u + 2·padding`
  the transposed convs preserve `T × u`). The solo path returns that whole tensor,
  so the batched unpad is **sample-domain**: `y_lengths × prod(upsample_rates)`,
  i.e. `trim_decoder_output(output, y_lengths, samples_per_frame)`. For both
  shipped converters `prod(upsample_rates) = 256 = hop_length` (V1 and V2 both
  `[8, 8, 2, 2]`); the handler reads the rate off `model.dec.upsample_rates` and
  only falls back to `hf_config.data.hop_length` when a stub has no decoder rates.

### Chosen path

- `_voice_conversion_impl(self, y, y_lengths, sid_src, sid_tgt, tau=1.0,
  mask_decoder_padding=False)` shares the math; `voice_conversion` calls it with
  the flag off (identical ops to the historical method) and
  `voice_conversion_batched(self, y, y_lengths, sid_src, sid_tgt, tau=1.0)` calls
  it with the flag on after broadcasting a per-sample `tau` through
  `_batch_param` (the 5-2 `[B, 1, 1]` pattern).
- `Generator.forward(self, x, g=None, x_lengths=None)` gained the **opt-in**
  per-stage mask (`_length_mask`, then `mask.repeat_interleave(stride, dim=2)`
  per upsample). Default `None` leaves `infer`/`infer_batched` (5-2) untouched,
  including their documented bounded decoder edge tail.
- The handler's mel frontend stays per clip (reflect-padded STFT, 8-2 lesson) and
  the batch is built on the frames it produces.

### Evidence summary

| Check | Result |
|---|---|
| TC `voice_conversion_batched` `B=1` vs `voice_conversion` | bit-identical (`torch.equal`) |
| Mixed lengths (`23/17/11` frames), `tau=0`, masked decoder | ≤1e-7 peak vs each solo run |
| Same batch with the decoder mask **off** (the trap) | 0.30 peak vs solo |
| Per-sample `tau=[0.3, 0.3]` vs scalar `tau=0.3` (same seed) | bit-identical |
| Decoder output length | `T_frames × prod(upsample_rates)`, trims to `y_lengths × rate` |

Tests: `tests/unit/modeling/test_batched_tone_converter.py` (B=1 bit-equality,
mixed-length no-leak, unmasked-leak contrast, sample-domain unpad, per-sample
tau, distinct speaker pairing, `zero_g`), the handler tier in
`tests/unit/inference_flow/test_batched_tone_converter_handler.py`, and the
4-request engine test in `tests/integration/test_batched_synthesis.py`.

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

- [x] Vendored TC audit documented; `infer_batched` added if needed with B=1 proof — encoder/flow were batch-safe, the decoder was not; `_voice_conversion_impl` + `voice_conversion_batched` + an opt-in per-stage decoder mask; B=1 bit-identical
- [x] Handler batched with frame-length planning + loop fallback — `_execute_openvoice_tone_converter` pads mel frames, plans length-similar sub-batches, trims in the sample domain with the model's own frame counts; `len == 1`, `batched_inference: false`, and per-sub-batch failures fall back to the loop
- [x] Gates green (TC-level + end-to-end clone pipelines) — TC-level unit/integration tier green; the real-model GPU `batching:1-vs-N` clone-pipeline run is deferred to the benchmark host (see Deviations 1)
- [x] `summary.md` Implementation Status updated

## Deviations

1. **No real-model GPU gate run in this task.** The `batching:1-vs-N` harness
   axis already allow-lists `openvoice_v1`/`openvoice_v2` (8-2), and the harness
   tests are green, but the actual GPU run against downloaded weights is deferred
   to the benchmark host exactly like Phase 7-3/8-2. The unit tier backs it: B=1
   bit-equality, mixed-length rows equal to their solo runs (≤1e-7 at `tau=0`),
   the handler-level loop-vs-batch checks, and the 4-request engine test.
2. **The shipped configs keep `max_batch_size: 1` for both tone converters.**
   They are now in `BATCHED_INFERENCE_FAMILIES` (so `batched_inference` defaults
   on and a multi-request step batches), but the truthful admission capacity is
   owned by 8-6 — same deferral as 8-2's embedders, and the config files are
   deliberately untouched.
3. **`Generator.forward` gained an opt-in `x_lengths` parameter.** It is `None`
   for every existing call, so `infer`/`infer_batched` (5-2) keep their exact
   arithmetic and their documented bounded decoder edge tail. The 5-2 synthesis
   path was intentionally **not** retrofitted to pass `y_lengths`: that would
   change landed batched-synthesis output for no task requirement.
4. **`tau` stays the historical constant `0.3`.** The vendored batched entry
   point accepts a per-sample vector (tested), but no request field exposes
   per-request conversion noise, so behavior is unchanged.
5. **Unpad is sample-domain, not `trim_feature_output`.** The decoder emits a
   waveform whose solo-path output is the whole tensor, so the correct unpad is
   `y_lengths × prod(upsample_rates)` (`trim_decoder_output`), not a frame-axis
   slice.
6. **Handler isolation is per sub-batch, not per step.** 5-2/8-2 fall back to the
   loop for the whole step on a batched failure; here a poisoned or mismatched
   row degrades only its own sub-batch, so a single bad request cannot de-batch
   its neighbours. `test_tone_converter_poisoned_request_errors_alone` pins it.


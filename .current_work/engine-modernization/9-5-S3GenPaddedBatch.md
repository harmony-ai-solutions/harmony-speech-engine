# Phase 9-5: S3Gen Padded Batch (the non-AR Chatterbox hop)

> **Plan:** [Engine Modernization](summary.md) · **Phase:** 9 — Chatterbox AR batching
> **Depends on:** 9-1 (seam), 8-1 helpers (landed); parallelizable with 9-3/9-4 — it consumes AR rows but does not need the runner to exist (the sequential loop produces the same token streams)
> **Codebase mapping consulted:** `.planning/codebase/ARCHITECTURE.md`; design input: study #14 D9; upstream facts `chatterbox-tts==0.1.6` (line numbers from study #14 — re-verify in the installed package)
> **State reference:** the S3Gen hop is upstream, not vendored (same situation as T3 — see 9-1). The Phase 8-4 pattern (`MelGANGenerator.inference_batched`, `run_masked` in `modeling/models/batch_padding.py`) is the template.

## Objective

Batch the S3Gen token→waveform hop (speech tokens + mels → audio) with the Phase 8 pattern, so a finished AR batch's rows vocode in one forward instead of N sequential calls. Today upstream S3Gen is "designed for batch_size=1 only" (`s3gen.py:173-229`) — but its graph already accepts `speech_token_lens` per row, and the only hard batch-1 artifact the study found is the noise tensor `torch.randn(1, 80, …)` (`s3gen.py:313-316`).

## Context (self-contained)

- **Verify-first padding audit (the 8-2/8-3/8-4 discipline):** re-read the upstream forward for the token/time axis — flow/conv stages that consume padded positions must either be length-masked per stage (the 8-3 decoder treatment, via `run_masked`) or confirmed padding-safe by construction (the 8-3 encoder treatment). Record the audit in this file with measured leak numbers (tiny instance, mixed-length rows vs solo runs) before adding the batched entry point.
- **Noise determinism:** upstream draws one noise tensor; the batched path must draw **per-row seeded noise** (per-row `torch.Generator` from the request seed) so a row's audio is identical whether it vocodes alone or in a batch — the same reproducibility contract as 9-3's samplers. B=1 with the same seed must be bit-identical to the sequential path.
- **HSE-side implementation** through the 9-1 seam (subclass/extension module in `harmonyspeech/modeling/models/chatterbox/`), not a site-packages edit; upstream `inference()` stays untouched for the fallback loop.
- **Consumers:** the TTS path (tokens+mels from 9-3 rows or the sequential loop) and the VC path's vocode step if it shares S3Gen (re-read `_execute_chatterbox_vc`; batch only what audits clean).
- Turbo: meanflow S3Gen is a different graph — separate verify-first audit; document as skip if padding-hostile, in `docs/batching.md` (the turbo **AR decode** is committed scope per 9-3; only this non-AR vocode hop may be a documented skip).

## Implementation steps

1. Padding audit of upstream S3Gen forward (token + mel axes), with leak measurements on a tiny deterministic config — recorded here.
2. `inference_batched` (HSE-side): per-row seeded noise, per-stage masks where the audit demands, lengths-driven unpad; B=1/all-equal-rows fast path calls the upstream graph unchanged (bit-identity).
3. Handler wiring: collect finished rows per step → `plan_batches` (token axis unit, `batch_token_budget` reused) → one forward per sub-batch → per-request `AudioArtifact`s; `len==1`/`batched_inference: false`/failure → loop fallback; poisoned row errors alone.
4. `docs/batching.md` matrix rows updated when gates pass.

## Files to create/modify

| File | Change |
|---|---|
| `harmonyspeech/modeling/models/chatterbox/` (seam module) | `inference_batched` + masks |
| `harmonyspeech/task_handler/model_runner_base.py` | batched S3Gen hop in the chatterbox handlers |
| `harmonyspeech/task_handler/batching.py` | supports-entry + axis unit |
| `tests/unit/modeling/test_chatterbox_s3gen_batch.py`, handler tests (new/extend) | gates below |

## Tests / validation

- Unit: B=1 bit-identity (same seed); mixed-length rows ≤1e-6 vs solo runs; per-row noise reproducibility; padding-independence (poisoned padding values).
- Handler: loop-vs-batch equality on seeded inputs; poisoned row isolation; pairing invariants.
- GPU gate (`batching:1-vs-N`, benchmark host): family hop equality + the throughput delta recorded in 9-6.

## Upstream padding audit (2026-11-07)

> Re-verified in the installed `chatterbox-tts==0.1.6`
> (`…\site-packages\chatterbox\models\s3gen`). Measured on a tiny, seeded
> instance of the **real** upstream classes: a 512-dim `UpsampleConformerEncoder`
> (1 block, 2 heads, no CNN/macaron), a `ConditionalDecoder(in_channels=320,
> out_channels=80, channels=[32], n_blocks=1, num_mid_blocks=1)` under
> `CausalConditionalCFM`, and a `HiFTGenerator` (`base_channels=8`,
> `upsample_rates=[2,2]`, `n_fft=8`, `hop_len=2`, `harmonics=2`). "Leak" is the
> max |Δ| on a short row's own valid span between a padded batch and its solo
> run; the batch-kernel numerical floor is ~2–6e-6.

### Exact entry points and batch-1 artifacts

| Location | What it does | Batch-1 assumption |
|---|---|---|
| `s3gen.py:173-229` (`S3Token2Mel.forward`) | ref_dict cast → `flow.inference` | docstring "designed for batch_size=1 only" (`:193`) |
| `s3gen.py:300-321` (`S3Token2Wav.flow_inference`) | picks `n_cfm_timesteps`, draws meanflow noise | `noise = torch.randn(1, 80, speech_tokens.size(-1) * 2, …)` (`:313-316`) |
| `s3gen.py:329-362` (`S3Token2Wav.inference`) | flow → `hift_inference` → `trim_fade` | fade applied to `output_wavs[:, :len(trim_fade)]` |
| `flow.py:131-198` (`CausalMaskedDiffWithXvec.inference`) | concat prompt+token, encoder, conds, decoder | `mel_len1 = prompt_feat.shape[1]` / `mel_len2 = h.shape[1] - mel_len1` (`:174`); `mask = (~make_pad_mask(h_lengths)).unsqueeze(1)` builds **max(valid)** columns, not the padded width (`:182-185`) |
| `flow_matching.py:195-233` (`CausalConditionalCFM.forward`) | `z = randn_like(mu)` (`:216`), Euler solver | one start-noise tensor for the batch |
| `decoder.py:243-333` (`ConditionalDecoder.forward`) | per-stage `* mask`, causal convs | mask *length* only (above) |
| `transformer/upsample_encoder.py:237-304` | embed → `pre_lookahead_layer` (`:280`) → `forward_layers` → `up_layer` (`:285`) → `up_embed` → `forward_up_layers` | `LinearNoSubsampling` adds a bias to padding; the causal lookahead reads it forward |
| `hifigan.py:462-474` (`HiFTGenerator.inference`) | f0 → source → `decode` (`:412-444`) | `SourceModuleHnNSF`/`SineGen` (`:169-283`) draw phase + two noises from the global RNG; STFT/ISTFT at the batch end |
| `f0_predictor.py:52-55` (`ConvRNNF0Predictor.forward`) | 5 symmetric convs + linear | padding leaks through the convs |

### Stage-by-stage verdict + measured leak

| Stage | Verdict | Measured leak (unmasked) | After the fix |
|---|---|---|---|
| `UpsampleConformerEncoder` | **not safe by construction** — mask after `embed`, after the lookahead, and before `up_layer` (8-3 decoder treatment) | short row (4 tokens in a 7-token batch): **0.4075 peak** | 3.6e-6 (batch-kernel floor) |
| `ConditionalDecoder` / CFM | **safe by construction** (per-stage masks, causal convs) — only the mask *length* is rebuilt over the padded axis (8-3 encoder treatment) | same-length control: 2.1e-6 (numerical) | 2.1e-6 |
| Full flow (encoder+decoder) | lengths-driven per-row prompt/token padding + per-row noise | mixed rows 3.3e-6 / 5.9e-6; `B=1` orchestration **bit-identical** to upstream | — |
| `HiFTGenerator` | **not safe** — f0 convs, transposed ups, source branch and resblocks all leak; the STFT/ISTFT reflect/overlap at the batch end | short row: **0.44 peak** | 1.5e-7 |
| Full S3Gen hop | per-row seeded noise + per-stage masks + per-row STFT/ISTFT + sample-domain trim | `B=1` **bit-identical**; mixed rows ≤2.1e-7; reproducibility 1.5e-7; poisoned padding exact | — |

**Noise determinism.** Upstream draws four tensors from the global RNG: the CFM
start noise (`flow_matching.py:216`), the `SineGen` phase (`Uniform.sample`) and
noise (`randn_like(sine_waves)`), and the `SourceModule` noise (`randn_like(uv)`).
Each is re-drawn per row at the row's own valid length from
`torch.Generator().manual_seed(request seed)`; a fresh generator seeded like the
global RNG produces the same stream, so `B=1` is bit-identical and a row is
identical alone or batched.

### Chosen path

* HSE-side `inference_batched` in `harmonyspeech/modeling/models/chatterbox/s3gen_batch.py`
  (the 9-1 seam discipline: the installed package is untouched). It reuses the
  upstream submodules but owns the orchestration, the corrected mask, the
  per-stage masking and the per-row RNG.
* `B=1` **and unseeded** calls upstream `s3gen.inference` unchanged. Every other
  shape runs the padded graph; with one seeded row the padded graph is
  bit-identical to upstream (proven by the gate).
* Turbo (meanflow) is refused by `supports_batched_s3gen`; the handler keeps the
  per-row upstream call.

### Turbo (meanflow) S3Gen verdict — documented hop skip

The meanflow path shares the audited `ConditionalDecoder` and `HiFTGenerator`
(those stages are padding-safe with the same masks), but its solver has two
batch-1 assumptions: `flow_inference` hard-codes a `[1, 80, T·2]` noise tensor
(`s3gen.py:313-316`) and `CausalConditionalCFM.forward` places `noised_mels` at a
single `prompt_len = mu.size(2) - noised_mels.size(2)` (`flow_matching.py:218-220`),
which is wrong for per-row prompt lengths. Batching it would require
re-implementing `basic_euler` to inject per-row noise. Per the spec this is the
**only allowed documented skip**; the Turbo **AR decode** remains 9-3's committed
scope. Recorded in `docs/batching.md`.

## Checklist

- [x] S3Gen padding audit recorded here (with leak numbers)
- [x] `inference_batched` landed; B=1 bit-identity + mixed-length gates green
- [x] Handlers batched with fallbacks; docs matrix updated
- [x] Turbo S3Gen audited (included or documented skip)
- [x] `summary.md` Implementation Status updated

## Deviations

1. **The meanflow (Turbo) vocode hop is the documented skip** (audit above); the
   standard graph is fully batched.
2. **`B=1` with a seed runs the padded orchestrator, not literally upstream
   `inference`.** It is bit-identical to upstream (the equality gate), and the
   unseeded `B=1` case still calls upstream unchanged. This keeps one code path
   for the reproducibility contract.
3. **The handler's per-row fallback is seeded** (`torch.manual_seed(row.seed)` with
   the previous RNG state restored), so a batch failure is a performance
   fallback, not an audio change. `ARRow` gained the resolved `seed` field to make
   that possible.
4. **VC is batched only when more than one request is admitted.** `ChatterboxVC`
   stays `loop`-class with `max_batch_size: 1` in the Phase 10-5 table (a test
   pins `batched_inference_enabled(ChatterboxVC) is False`), so the batched vocode
   hop is dormant in the shipped configs and ready for a 9-6 capacity change. The
   branch is gated by the `batched_inference` flag directly (an explicit `false`
   keeps the loop).
5. **The mixed-length gate is ≤1e-6 at the waveform**, not at the CFM mel. The
   tiny decoder's mel has a ~2–6e-6 batch-kernel floor on CPU float32; the full
   hop's waveform gate measured ≤2.1e-7.
6. **No real-model / GPU gate in this task.** The `batching:1-vs-N` real-model
   run and throughput delta are 9-6's benchmark-host session (`_tbd_`).

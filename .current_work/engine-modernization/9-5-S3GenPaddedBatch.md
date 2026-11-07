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

## Checklist

- [ ] S3Gen padding audit recorded here (with leak numbers)
- [ ] `inference_batched` landed; B=1 bit-identity + mixed-length gates green
- [ ] Handlers batched with fallbacks; docs matrix updated
- [ ] Turbo S3Gen audited (included or documented skip)
- [ ] `summary.md` Implementation Status updated

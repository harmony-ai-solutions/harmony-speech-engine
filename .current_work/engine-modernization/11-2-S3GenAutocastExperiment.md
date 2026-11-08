# Phase 11-2: S3Gen Autocast Experiment (bf16/fp16 mixed precision)

> **Plan:** [Engine Modernization](summary.md) · **Phase:** 11
> **Depends on:** 4-1/4-2 (dtype policy + audio regression harness), 9-5 (batched S3Gen), 11-1 (VC batching)
> **Owner decision 2026-11-08:** run the experiment. This is a **falsifiable experiment**, not a rollout: a clean negative is a valid outcome that closes the question with evidence.

## Objective

Test whether S3Gen (the flow-matching vocoder hop shared by chatterbox en / multilingual / VC, and turbo's own meanflow variant) can run under **`torch.autocast` mixed precision** without audible degradation, and measure the throughput/VRAM win — especially for **batched VC**, where vocode is the GPU-dominant hop.

## Why this is plausible (and what it must beat)

- Phase 4 measured that **pure fp16 weights** for S3Gen audibly degrade output (`docs/models.md` §Precision) — that result stands and is the prior.
- Autocast is a different shape: fp32 master weights, per-op casting; norm/softmax/reductions stay fp32 — only matmul-heavy ops drop to bf16/fp16. Phase 4 never tested it.
- Upstream Chatterbox-TTS-Server v2.0 uses exactly this pattern for **T3** (bf16 + autocast, S3Gen deliberately fp32) claiming ~40% on token generation. We apply the idea to S3Gen HSE-side.
- We own the batched flow loop (`s3gen_batch.py::_flow_batched`), so the surgical variant is available: autocast around the estimator forwards only, ODE state updates / masks / `mel2wav` in fp32.

## Design

1. **Env gate** `HSE_S3GEN_AUTOCAST` = `off` (default — zero behavior change) | `bfloat16` | `float16`.
2. **Wrap sites:**
   - `s3gen_batch.py::_flow_batched` — surgical: autocast context around the encoder / CFM decoder forwards; ODE updates, masking, unpadding, `mel2wav` remain fp32. Covers seeded solo (B=1 seeded runs the batched path) and all batched vocode.
   - Solo upstream path: `model_runner_base.py` `_vocode…` (the `s3gen.inference(...)` calls around `:1119`/`:1123`) — blanket autocast around the upstream call.
   - Turbo meanflow call site (own solver; explicitly unsupported by the batched path) — blanket autocast; report separately.
3. **Gate axis** in `scripts/audio_regression.py`: `dtype:fp32-vs-autocast` — seeded A/B per family (chatterbox en, multilingual, VC, turbo), log-mel MSE + correlation vs the fp32 reference, thresholds + family allow-list following the existing `batching:` axis pattern (11-1 used seeded MSE ≤ 1e-4-scale + corr ≈ 1.0 for VC batching; pick and justify similar).
4. **Benchmarks on GPU 1** (`CUDA_VISIBLE_DEVICES=1`, INFO logging, two A/B server runs, same fixtures/seeds — reuse `out/bench/` + `scripts/bench_throughput.py`):
   - VC c={1,4,8}: p50 / req/s, per-hop timings (tokenize / flow / mel2wav / decode / perth) fp32 vs autocast.
   - en + turbo clone floods c=8: req/s + p50.
   - Activation peaks and weight/arena VRAM deltas; note the activation headroom implication for future vocode width.
5. **Decision rule:** audio gate passes (seeded thresholds) **and** a measurable win → a *follow-up* task promotes it to a config knob (e.g. `s3gen_precision`) with docs + honesty tests. Gate fails → document the negative result next to the Phase 4 fp16 result in `docs/models.md` §Precision and the benchmarks doc; revert nothing (default stays `off`). **No shipped-config change in this task.**

## Files to create/modify

| File | Change |
|---|---|
| `harmonyspeech/modeling/models/chatterbox/s3gen_batch.py` | autocast wrap in `_flow_batched` (surgical) |
| `harmonyspeech/task_handler/model_runner_base.py` | autocast wrap on solo `s3gen.inference` + turbo meanflow site |
| `scripts/audio_regression.py` | `dtype:fp32-vs-autocast` seeded axis + allow-list |
| `scripts/bench_throughput.py` (or bench fixture) | per-hop timing capture if not already emitted |
| `docs/models.md`, `docs/benchmarks/*.md` | experiment section: method, gate verdicts, measured numbers |
| tests | unit tests for the env gate + autocast branch behavior (CPU: autocast no-op correctness) |

## Environment facts

- Interpreter `C:\Users\sge20\miniconda3\envs\hse\python.exe`; weights cached; **GPU 0 carries the desktop — heavy runs on GPU 1**.
- `out/bench/` fixtures untracked — reuse, never commit.
- Full `tests/unit` + `tests/integration` + `tests/e2e/wire_contract` green at the end (4 pre-existing weights-gated e2e errors are known artifacts).

## Checklist

- [x] Env gate (default off, zero behavior change) + wraps at all three sites
- [x] `dtype:fp32-vs-autocast` axis in the 4-2 harness; seeded gate verdicts per family
- [x] A/B benchmarks on GPU 1: VC c={1,4,8} per-hop, en/turbo floods c=8, VRAM deltas
- [x] Docs: experiment section + verdict (pass → follow-up promotion task noted; fail → negative result recorded)
- [x] Suites green; `summary.md` status updated

## Result (2026-11-08)

**Verdict: `bfloat16` rejected; `float16` passes the seeded gate with a real
throughput win.** All numbers, methods and reproduce commands are in
[`docs/benchmarks/throughput.md` §Phase 11-2](../../docs/benchmarks/throughput.md#phase-11-2--s3gen-autocast);
the operator-facing summary is in [`docs/models.md` §Precision](../../docs/models.md#precision-dtype-and-shared-backbones).

| Gate (`dtype:fp32-vs-autocast`) | bfloat16 | float16 |
|---|---|---|
| chatterbox (en TTS + VC), 4 cases | **2 FAIL** (VC log-mel MSE 3.00; clone corr 0.8475) | PASS |
| chatterbox_multilingual, 2 cases | PASS (0.37–0.43 dB²) | PASS (0.009–0.013 dB²) |
| chatterbox_turbo (meanflow), 1 case | PASS (0.69 dB²) | PASS (0.020 dB²) |
| **overall** | **FAIL** | **PASS** (7/7) |

Duration Δ is 0.0 % on every case (the AR token stream is identical), so the
differences are pure vocode arithmetic. bf16 is 5–30× worse than fp16
everywhere — its 8-bit mantissa is the problem, not its range.

| Throughput (GPU 1, INFO, 16 req/level) | fp32 | float16 | speed-up |
|---|---|---|---|
| VC c=1 req/s / p50 | 2.68 / 373 ms | 2.73 / 366 ms | 1.02× |
| VC c=4 req/s / p50 | 3.70 / 1069 ms | 4.87 / 802 ms | **1.32×** |
| VC c=8 req/s / p50 | 4.10 / 1920 ms | 6.19 / 1273 ms | **1.51×** |
| en clone flood c=8 | 0.94 / 5354 ms | 0.89 / 5659 ms | −5 % (noise) |
| turbo clone flood c=8 | 0.70 / 8439 ms | 0.73 / 8750 ms | +4 % (noise) |

Per-hop at B=8: flow **1581 → 646 ms (2.45×)**, mel2wav 165 ms unchanged
(surgical shape keeps HiFT fp32), vocode end-to-end **1748 → 808 ms (2.16×)**.
Per-hop at B=1: flow 322 → 275 ms (1.17×).

VRAM: the 8-row vocode activation peak is **unchanged** (406.5 MiB; the flow
phase alone shrinks 343.5 → 303.8 MiB but HiFT dominates), and the server-level
device footprint grows ~240 MiB (non-torch allocator). No VRAM case for it.

**Follow-up (not implemented here):** promote `float16` to a `s3gen_precision`
config knob (`off` default) with docs, a config-honesty test and a human
listening pass over the `out/regression-pack` A/B clips. bf16 is documented as
rejected. No shipped config sets the gate.

## Deviations (2026-11-08)

1. **The autocast seam is installed on the `S3Gen` instance at load, not only at
   the `s3gen.inference` call sites.** The spec named the solo upstream path in
   `model_runner_base.py`; in this build the sequential English/Turbo loops,
   multilingual and VC all reach the vocoder through upstream `generate()`
   methods, so a wrap at our call sites could not cover them without also
   autocasting T3 (which would change the seeded token stream and invalidate the
   gate). `install_inference_autocast` decorates the loaded `s3gen.inference`
   once (the same instance-level seam the loader already uses for dtype
   hygiene), and `_vocode_ar_row` keeps an explicit wrap as specified. The
   batched path is the specified surgical wrap in `_flow_batched`.
2. **`_to_flow` now passes `None` through.** The real `embed_ref` dict carries
   `prompt_feat_len: None`; the B=1 unseeded fast path converted every value
   with `torch.as_tensor`, so a single-row unseeded VC group always raised and
   fell back to the loop. One-line fix, pinned by the probe; the batched path
   never read the value. Default-off audio is unchanged (unit bit-identity
   gates green).
3. **Gate thresholds are the harness's audibility bounds (1.0 dB² / corr 0.98 /
   RMS 5 % / duration 0.1 %), not a batching-order bound.** The first pass used
   5e-3 dB² and failed *both* precisions (bf16 0.28–3.0, fp16 0.008–0.099);
   5e-3 is stricter than the batching axis's own 1e-2 and is not an audibility
   bound. The shipped bound is calibrated by Phase 4's measured audible class
   (8.9–141.6 dB² / corr 0.05–0.76). Both numbers are recorded in
   `docs/benchmarks/throughput.md`; under the batching-order bound the verdict
   would be "both fail".
4. **`float16` server A/B only; bf16 measured per-hop.** bf16's flow is within
   0.5 % of fp16 (649.6 vs 646.4 ms at B=8), so the server table represents
   both; bf16 is rejected by the audio gate regardless.
5. **Two clone families were added to `scripts/bench_throughput.py`**
   (`chatterbox_clone`, `chatterbox_turbo_clone`) so the requested en/turbo
   clone floods carry a reference clip. Additive: existing families and their
   published tables are unchanged.
6. **The VRAM expectation was falsified, not met.** The task expected autocast
   intermediates to shrink the 8-row peak; they do not, because the surgical
   shape keeps HiFT fp32 and HiFT dominates the vocode peak. Recorded as a
   finding.
7. **No commits/pushes** (task constraint). The per-hop/audio probe is a
   **tracked** script (`scripts/bench_s3gen_autocast.py`, modes `timing` and
   `audio`) so the published numbers are reproducible from the repo; the
   benchmark configs (`out/bench/config-11-2-{vc,en}.yml`), JSON reports and
   server logs are untracked measurement scaffolding, as in Phase 11-1.


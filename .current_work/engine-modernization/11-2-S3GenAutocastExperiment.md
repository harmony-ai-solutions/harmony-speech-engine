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

- [ ] Env gate (default off, zero behavior change) + wraps at all three sites
- [ ] `dtype:fp32-vs-autocast` axis in the 4-2 harness; seeded gate verdicts per family
- [ ] A/B benchmarks on GPU 1: VC c={1,4,8} per-hop, en/turbo floods c=8, VRAM deltas
- [ ] Docs: experiment section + verdict (pass → follow-up promotion task noted; fail → negative result recorded)
- [ ] Suites green; `summary.md` status updated

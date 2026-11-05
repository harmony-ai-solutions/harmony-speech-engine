# Phase 8-6: Honest Config, Pipeline Benchmark & Phase Exit

> **Plan:** [Engine Modernization](summary.md) · **Phase:** 8 — Batching across all steps
> **Depends on:** 8-2, 8-3, 8-4, 8-5
> **Codebase mapping consulted:** `.planning/codebase/TESTING.md`; benchmark precedents `docs/benchmarks/throughput.md` (5-3, 7-3)

## Objective

Set the newly batched models' `max_batch_size` to measured truths, and prove the phase's headline claim with the **flood scenario**: 20 (and 200) concurrent voice-clone requests moving through a pipeline where **every step batches**. Then close the phase.

## Context (self-contained)

- The motivating scenario (product decision 2026-11-05): "if I get 20 or 200 voice cloning requests in parallel, we already benefit a lot from parallel processing and can move through the pipeline faster." Post 8-2/8-3 the OpenVoice clone pipeline (VAD→embed→synth→TC) batches at every step; Phase 7's cache removes the embed compute entirely for repeated voices.
- Honesty note for the benchmark report: within one engine step, models still execute **sequentially** — the win is per-step batched forwards (fewer kernel launches, better GPU utilization, fewer host round-trips), not cross-step parallelism (that's the separate worker-concurrency workstream). State this in `docs/benchmarks/throughput.md` so nobody misreads the numbers.
- Baseline to beat (5-3, GPU 1): OV2 clone path 1.86 → 1.90 req/s at c=8 (flat — TC/embed loops bound it); MeloTTS synth 14.43 req/s at c=8.

## Implementation steps

### 1. Truthful config values

After gates pass, set measured values (start 8, tune on data) for every newly batched model type in **both** configs: OpenVoice V1/V2 TCEs + TCs, hs1 synthesizer/vocoder, voicefixer restorer/vocoder (+ whisper stays 16 if 8-5 unchanged). Loops that stayed loops keep 1 (hs1-encoder if 8-2 skipped it, silero 16 as cheap CPU concurrency per 8-5 verdict, kitten 4, Chatterbox family 1). Keep the config-honesty test (5-3) in sync with every change.

### 2. The flood benchmark

Extend `scripts/bench_throughput.py`:

- `--concurrency 1,4,20` (and one manual 200 run for the report — HTTP client-side pooling permitting; if the client saturates below 200, record the ceiling honestly).
- Scenario matrix for the OV2 clone pipeline: `{same-voice, distinct-voices} × {cache on, cache off}` (reusing 7-3's `--clone-reference`), plus hs1 TTS and voicefixer restoration curves at the new batch sizes, plus a **KittenTTS lanes curve** (c=8 with lanes vs the old sequential loop — the message-queue behavior proof for a non-batchable model, using the fixed `intra_op` from 8-5).
- Report per-step timings (engine DEBUG already logs sub-batch composition — reuse the 5-3 mechanism) + end-to-end p50/p95/req/s.

Expected shape (record actuals): distinct-voices flood shows the full 8-2+8-3+5-2 stacking (each hop ~1–2 sub-batches instead of N sequential calls); same-voice + cache collapses the embed step (7-3 numbers) on top.

### 3. Documentation & exit

- `docs/benchmarks/throughput.md`: Phase 8 section — all curves, engine commit, GPU, cache state, the sequential-steps honesty note, and the 200-connection observation.
- `docs/batching.md` final matrix cross-checked against final config values (must agree — 8-5's consistency lint).
- Knowledge updates: `memory-bank/progress.md` + `activeContext.md`; `CONCERNS.md` (retire/annotate the "clone path throughput flat" observation with the new numbers); `CHANGELOG.md` (user-visible: higher concurrent throughput + new cache knobs — no internals).
- Full suites green (`tests/unit` ≥ 593 + new, `tests/integration`, `tests/e2e/wire_contract`, real-model e2e for the touched families: openvoice v1/v2, harmonyspeech, voicefixer, whisper STT).
- Tick Phase 8 in [summary.md](summary.md); deviations noted in the 8-* files.

## Files to create/modify

| File | Change |
|---|---|
| `config.gpu.yml`, `config.yml` | truthful values for newly batched models |
| `scripts/bench_throughput.py` | concurrency sweeps + per-step reporting |
| `docs/benchmarks/throughput.md`, `docs/batching.md` | results + final matrix |
| `tests/unit/initialization/test_config.py` (extend) | config-honesty sync |

## Tests / validation

- Config-honesty test updated and green.
- Flood benchmark recorded (all four clone-path combinations + hs1 + voicefixer curves + the 200-connection observation).
- All suites green.

## Checklist

- [ ] All newly batched models at measured truthful `max_batch_size`; honesty test in sync
- [ ] Flood benchmark recorded (20 + 200 concurrency; per-step timings; cache on/off × same/distinct)
- [ ] Docs: throughput section + final batching matrix + honesty note
- [ ] Knowledge updates (memory-bank, CONCERNS, CHANGELOG) done
- [ ] Full suites green; Phase 8 ticked in summary.md

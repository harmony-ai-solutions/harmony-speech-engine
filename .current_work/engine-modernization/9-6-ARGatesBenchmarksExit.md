# Phase 9-6: AR Gates, Honest Config, Benchmarks & Phase Exit

> **Plan:** [Engine Modernization](summary.md) · **Phase:** 9 — Chatterbox AR batching
> **Depends on:** 9-1, 9-2, 9-3, 9-4, 9-5
> **Codebase mapping consulted:** `.planning/codebase/TESTING.md`; precedents: 5-3, 8-5, 8-6 (config honesty, matrix, flood benchmark)

## Objective

Prove the phase's claims, set measured truthful config values, and close Phase 9: concurrent Chatterbox requests decode in shared batched steps on a size-capped KV pool, with honest capacity numbers everywhere.

## Context (self-contained)

- **Baseline to beat:** Chatterbox 0.6 req/s flat at c=8 (Phase 5-3, GPU 1) with the sequential loop; AR-0 (9-1) will already have moved that line — record both.
- **Config honesty:** the family's interim `max_batch_size: 1` (5-3 product decision) is replaced by measured values. The honesty policy test gains the **`ar_decode` class** of the shared execution-model class table — decided 2026-11-07; the authoritative definition lives in [10-5-WhisperGatesBenchmarksExit.md](10-5-WhisperGatesBenchmarksExit.md) (Phase 10 executes first and introduces the mechanism); keep `docs/batching.md` and the docs-matrix lint in sync.
- **Config surface:** per-model `max_batch_size` = measured decode width; pool budget from 9-2 (`ar_inference.kv_cache_budget_bytes`) tuned from the measured arena table; multilingual (and any turbo skip) stay `1` with their documented reasons.
- **Benchmark scenarios (benchmark host, RTX 5060 Ti):**
  - chatterbox TTS **and turbo** floods at c={4,8,20}: sequential loop vs AR-0 vs full AR runner (three-way comparison — isolates loop hygiene from batching; turbo per the committed scope);
  - same-voice vs distinct-voice × embedding-cache on/off (Phase 7 composition, 7-3 style);
  - mixed-length rows (sub-batch formation sanity via engine DEBUG composition logging);
  - abort-under-load: rows free correctly (with 9-4);
  - pool metrics during the run: slots active/free, refusals, tokens/step.
  - **exhaustion scenarios (owner, 2026-11-07):** 100 distinct-voice requests against a pool sized for ~8 slots (budget-capped) — refusals → sequential-fallback correctness, slot-wait and tokens/step degradation, deadline breaches under exhaustion, request loss (must be 0), pairing errors (must be 0); abort storm mid-decode; and combined exhaustion (pool full + cold embedding cache + 20 concurrent). The degradation semantics must also be proven as **local CPU integration tests** with a stub AR model (tiny pool, many requests) before the host run.
- **VRAM accounting:** the arena table + measured peak go into `docs/benchmarks/vram-profiles.md`; verify the pool's footprint against the **device-level** headroom from 10-3 if that has landed (the torch-allocator numbers under-count; see 10-3 Context).
- **Knowledge updates:** memory-bank (`progress.md`, `activeContext.md`), `CONCERNS.md` (retire "Chatterbox stays a per-request loop by decision" notes with the new reality), `CHANGELOG.md` (user-visible: concurrent Chatterbox throughput; no internals), README/models docs if capacity claims exist.

## Implementation steps

1. GPU family gate (`batching:1-vs-N`, real weights, deterministic seeds) for **both** `chatterbox` and `chatterbox-turbo`: batched vs sequential output equality per 9-3's seeded contract; extend `scripts/audio_regression.py` allow-list for both families.
2. Set measured config values in both shipped configs; extend the honesty policy test with the AR class; docs-matrix lint in sync.
3. Run the benchmark scenarios above; fill `docs/benchmarks/throughput.md` §Phase 9 (structure + reproduce commands land with the task; numbers from the benchmark-host session — same trip as the Phase 7/8 `_tbd_` tables and AR-0).
4. Full suites green (`tests/unit`, `tests/integration`, wire-contract e2e; real-model e2e for the family on the benchmark host).
5. Tick Phase 9 in [summary.md](summary.md); deviations recorded in the 9-x files.

## Files to create/modify

| File | Change |
|---|---|
| `config.gpu.yml`, `config.yml` | measured AR capacity values |
| `tests/unit/initialization/test_config.py`, `tests/unit/test_batching_docs_matrix.py` | fourth honesty class + matrix sync |
| `scripts/bench_throughput.py` | AR scenarios + pool-metrics collection |
| `docs/batching.md`, `docs/benchmarks/{throughput,vram-profiles}.md` | results + final rows |

## Tests / validation

- Family gate green on the benchmark host; honesty + matrix tests green locally.
- All suites green; benchmarks recorded; knowledge updated.

## Checklist

- [x] `batching:1-vs-N` family gate green (seeded equality)
      — **harness landed** (`audio_regression.py` allow-list + `batching_cases`,
      distributional for AR because vocode noise differs by construction; the
      seeded token gate is `tests/e2e/tts/test_chatterbox_ar_parity.py`); host
      run `_tbd_`
- [x] Measured config values + fourth honesty class + docs matrix in sync
      — **class flipped to `ar_decode`; initial width 4 from the arena table
      (pending the host gate); `ar_inference` blocks added; tests green**
- [x] Three-way flood benchmark (loop vs AR-0 vs AR runner) + cache composition recorded
      — **scenario family + `--print-ar-scenarios` landed; cells `_tbd_`**
- [x] VRAM: arena table + measured peak in `vram-profiles.md`
      — **arena table landed; measured cells `_tbd_`**
- [x] Knowledge updates; full suites green; Phase 9 ticked in summary.md
- [x] Exhaustion scenarios recorded (100-request tight pool, abort storm, combined)
      — **local CPU tests green (`tests/integration/test_chatterbox_exhaustion.py`);
      host trace `_tbd_`**

## Decisions (2026-11-07)

1. **The English/Turbo Chatterbox roles are `ar_decode` with an initial
   admission width of 4.** The width is justified from the shared 1 GiB pool's
   arena table at the default 768-token slot and fp16 slabs: English 4 requests
   (2 CFG rows each = 8 slabs × 90 MiB) + Turbo 4 requests (4 slabs × 72 MiB) =
   1008 MiB ≤ 1 GiB, order-independent. This is an "initial pending measurement"
   value in the Phase 8-6 style, not a measured one. The real-model
   `batching:1-vs-N` seeded-equality gate has not run (no weights here), so the
   rollback is documented: `batched_inference: false` (or width 1) forces the
   proven sequential loop. The class membership, runner and pool are all live.
2. **`batched_inference` stays an operator opt-out.** The shipped configs do not
   set it (the AR family default is on per 9-3); setting `false` disables the
   runner. This keeps the flag as the single honest kill switch without
   claiming a parity gate that has not run.
3. **The AR `batching:1-vs-N` audio gate is distributional.** The sequential
   loop samples vocode noise from the global RNG; the runner draws AR and S3Gen
   noise from per-row seeded generators. The token stream is the seeded
   contract (e2e parity harness), while the audio must be compared with the
   stochastic thresholds (duration/RMS/LTAS). The sample-wise padded-batch gate
   is unchanged for the padded families.
4. **The CPU config cannot fit both AR arenas at fp32.** Its `ar_inference` block
   is the same 1 GiB, so English clamps to 5 slabs (2 concurrent requests) and
   Turbo's arena is refused at registration — a graceful sequential fallback,
   never a boot failure. Chatterbox on CPU is not a serving target.

## Deviations (2026-11-07)

1. **`tests/integration/test_chatterbox_exhaustion.py` reuses the tiny-T3 stubs
   from `test_chatterbox_ar_engine.py`** and registers the arena at "model load"
   (`_init_ar_runner`) before admitting requests, matching production. Pairing is
   proven by a numeric marker in `Conditionals.gen` that the stub S3Gen writes
   into the output waveform: an output carrying another request's marker fails
   the test; the zero waveform is the sequential fallback.
2. **The combined exhaustion case runs the real embedding handler + real
   `EmbeddingCache`** with a stub embedding model, so the cold-cache misses are
   genuine engine-cache counters, not a mock.
3. **`scripts/bench_throughput.py` adds `--scenario chatterbox-ar` (throughput
   path + AR metric sampling) and `--scenario chatterbox-exhaustion`**, plus
   `--mixed-lengths`, `--chatterbox-*` knobs and `--print-ar-scenarios`. The
   default `throughput` path is unchanged; `chatterbox_turbo` is a new family
   spec.
4. **`audio_regression.py` gained `BATCHING_AXIS_STOCHASTIC`** so the AR gate
   uses the family's stochastic thresholds instead of the padded-batch
   sample-wise rules, and the batching worker's step cap is per-family
   (`batching_max_steps`, 4000 for AR) with a larger AR pool budget for the gate.
5. **No handler behaviour changed.** The sequential Chatterbox paths still do
   not consume `generation_options.seed`; the e2e parity harness gates the AR
   runner's per-row seeded equality (the actual phase claim) rather than
   pretending the sequential loop is seeded.
6. **Measurement cells remain `_tbd_`.** No weights were downloaded; the
   benchmark-host session (Phase 7/8/10 piggyback) fills the throughput, VRAM,
   exhaustion and parity tables.

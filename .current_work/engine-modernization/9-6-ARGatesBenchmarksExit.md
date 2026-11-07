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

- [ ] `batching:1-vs-N` family gate green (seeded equality)
- [ ] Measured config values + fourth honesty class + docs matrix in sync
- [ ] Three-way flood benchmark (loop vs AR-0 vs AR runner) + cache composition recorded
- [ ] VRAM: arena table + measured peak in `vram-profiles.md`
- [ ] Knowledge updates; full suites green; Phase 9 ticked in summary.md

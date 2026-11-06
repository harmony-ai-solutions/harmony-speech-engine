# Phase 7-3: Cache Metrics, Benchmark & Phase Exit

> **Plan:** [Engine Modernization](summary.md) · **Phase:** 7 — Cross-request embedding cache
> **Depends on:** 7-1, 7-2
> **Codebase mapping consulted:** `.planning/codebase/TESTING.md`, benchmark precedents `docs/benchmarks/throughput.md` (5-3)

## Objective

Expose the cache's behavior to operators (`/metrics` + `/health`), prove its value with the clone-flood benchmark scenario, and close the phase (docs, knowledge updates, exit review).

## Implementation steps

### 1. Metrics + health

Extend the existing `/metrics` Prometheus-text endpoint (added in 4-5) and `/health` payload (extended in 4-5/0-5):

```
hse_embedding_cache_entries
hse_embedding_cache_bytes
hse_embedding_cache_hits_total
hse_embedding_cache_misses_total
hse_embedding_cache_stores_total
hse_embedding_cache_ttl_expirations_total
hse_embedding_cache_fifo_evictions_total
hse_embedding_cache_hit_ratio            # gauge: hits / (hits + misses), 0 when no traffic
```

Series names follow the existing `hse_*` convention; empty/disabled cache emits zeros (never absent — dashboards depend on series presence). Unit test the exposition format with a warm and a disabled cache.

### 2. Clone-flood benchmark

Extend `scripts/bench_throughput.py` with a reference-audio mode:

```
--clone-reference same      # all concurrent requests clone the SAME reference voice (worker flood pattern)
--clone-reference distinct  # each request a different reference (cache-miss path — worst case)
```

Run on an idle GPU (measurements precedent: GPU 1), OpenVoice V2 clone pipeline (VAD→embed→synth→TC), concurrency {1, 4, 20}, **cache on vs off** (off via `--no-embedding-cache` boot flag). Expectation: `same` + cache ≈ removes the embed step entirely from steady-state latency; `distinct` ≈ parity with cache-off (a hash computation per request, ~µs, is noise). Record all four combinations in `docs/benchmarks/throughput.md` as the Phase 7 section (engine commit, GPU, cache settings).

### 3. Ops documentation

`docs/` (operations/cache section): what is cached (embeddings + Chatterbox conditionals derived from user audio), the knobs (`embedding_cache_enabled/ttl_s/max_entries/max_bytes`), the eviction policy (TTL 300 s default, FIFO beyond 1000 entries, byte ceiling), and the **privacy note**: derived voice biometrics are retained in process memory up to the TTL — deployments with strict data-retention requirements should disable the cache or lower the TTL. This is user-visible behavior → CHANGELOG entry (no internals).

### 4. Phase exit

- Full suites green (`tests/unit` ≥ current 593 baseline + new, `tests/integration`, `tests/e2e/wire_contract`; real-model spot-check of a clone e2e with cache on and off).
- Knowledge updates: `memory-bank/progress.md` + `activeContext.md`; `.planning/codebase/CONCERNS.md` — add-and-resolve a note documenting that repeated-voice recomputation is addressed; CHANGELOG.
- Tick Phase 7 in [summary.md](summary.md); note any deviations in the 7-* files.

## Files to create/modify

| File | Change |
|---|---|
| `harmonyspeech/endpoints/openai/api_server.py` (+ engine `get_*` surface) | cache metrics series |
| `harmonyspeech/engine/harmonyspeech_engine.py` | health payload extension |
| `scripts/bench_throughput.py` | `--clone-reference` mode |
| `docs/benchmarks/throughput.md` | Phase 7 results table |
| `docs/` (ops) + `CHANGELOG.md` | cache semantics + privacy note |

## Tests

- Unit: metrics exposition (warm/disabled), hit-ratio gauge edge cases (no traffic → 0).
- Benchmark validation run recorded (the four combinations).

## Checklist

- [x] `/metrics` + `/health` cache series live (zeros when disabled)
- [ ] Clone-flood benchmark recorded (same/distinct × cache on/off, c={1,4,20}) — **tooling + doc structure done; GPU benchmark run deferred to benchmark host**
- [x] Ops docs incl. privacy note; CHANGELOG entry
- [x] Full suites green (unit 655 / integration 37 / wire-contract e2e 38 + 3); knowledge docs updated
- [x] Phase 7 ticked in summary.md

## Deviations & notes (7-3)

- **Metrics surface is slightly wider than the plan's list.** Beyond the eight
  required series the exporter also emits `hse_embedding_cache_enabled` (0/1) and
  `hse_embedding_cache_byte_evictions_total`, so "cache disabled" is
  distinguishable from "cache cold" and every eviction reason is visible. Series
  names, types and semantics are in `docs/embedding-cache.md`.
- **`/health` + `/metrics` telemetry wiring was fixed, not just extended.** The
  serving layer holds `AsyncHarmonySpeech`, which previously exposed neither
  `get_vram_stats()` nor the `check_health()` payload to the endpoints, so the
  Phase 4-5 VRAM block/series were being dropped at runtime (only the unit test
  shell engine exercised them). 7-3 adds `get_vram_stats()` /
  `get_embedding_cache_stats()` proxies and returns the payload from
  `AsyncHarmonySpeech.check_health()`; the dead-loop `AsyncEngineDeadError`
  behaviour is unchanged.
- **Benchmark reference variants.** `--clone-reference distinct` derives its
  per-request clips deterministically from the base reference (rotation by
  `index × 37` samples + a 0.1 % gain change per index, re-encoded to WAV), which
  keeps duration/sample rate and speech shape while guaranteeing a unique
  `AudioArtifact.content_hash`. Request 0 is the untouched base clip so `same`
  and `distinct` share a baseline. Families without reference audio ignore the
  flag (one payload reused).
- **Cache hits are across requests, not within one batch.** A handler resolves
  every cache slot before computing any of them, so two identical requests that
  land in the *same* engine step both miss (and both store). The engine-level
  test therefore drives request 1 to completion before admitting request 2 —
  that is the workload the cache targets.
- **`docs/benchmarks/throughput.md` numbers are `_tbd_`.** Methodology, the four
  tables (`same`/`distinct` × cache on/off), the scrape fields to record and the
  reproduce commands are in place; the GPU run was out of scope for this task.

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

- [ ] `/metrics` + `/health` cache series live (zeros when disabled)
- [ ] Clone-flood benchmark recorded (same/distinct × cache on/off, c={1,4,20})
- [ ] Ops docs incl. privacy note; CHANGELOG entry
- [ ] Full suites green; knowledge docs updated
- [ ] Phase 7 ticked in summary.md

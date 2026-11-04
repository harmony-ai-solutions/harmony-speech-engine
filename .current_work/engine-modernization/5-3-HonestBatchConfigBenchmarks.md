# Phase 5-3: Honest Batch Config & Benchmarks

> **Plan:** [Engine Modernization](summary.md) · **Phase:** 5 — Honest batching (HSE only)
> **Depends on:** 5-1, 5-2
> **Codebase mapping consulted:** `.planning/codebase/ARCHITECTURE.md`, research §1.6 #3, §5.3(d)

## Objective

Make `max_batch_size` tell the truth — a capacity signal the scheduler and operators can trust — and record per-family throughput before/after the phase so the numbers exist for capacity planning.

## Context (selfensible — current config values, verified 2026-11-04)

| Config entries | `max_batch_size` today | Reality |
|---|---|---|
| `faster-whisper-*` | 16 | **True** — `BatchedInferencePipeline(batch_size=16)` |
| OpenVoice/Melo synthesizers + TC/TCE | 10 | Sequential loop → true concurrency 1 (until 5-2) |
| `hs1-*` (HarmonySpeech) | 10 | Sequential loop |
| `voicefixer-*` | 4 | Sequential loop |
| Chatterbox family (5 entries) | 10 (re-read exact values at implementation) | Sequential loop **and** autoregressive → honest value 1 |
| `kitten-tts-*`, `silero-vad` | 4 / 16 | Cheap CPU loops — concurrency admission is honest-ish (latency scales linearly, throughput fine) |

## Implementation steps

### 1. Truthful values in `config.gpu.yml` (+ `config.yml`)

- Chatterbox entries → `max_batch_size: 1` (agreed plan; placeholder until a possible future AR-batching workstream).
- Post-5-2 synthesizer entries: set to the **measured** sweet spot from step 2 (start: 8; adjust on data). If 5-2 shipped with `batched_inference: false` defaults, set 1 and note why.
- VoiceFixer / HarmonySpeech loops: set 1 (sequential reality) unless 5-2 was extended to them — it was not (scope decision).
- Whisper/Kitten/Silero: keep (real batching / cheap CPU loops).
- Document the semantics next to the config field in `docs/`: `max_batch_size` = max requests admitted per model per engine step **and actually processed together**.

### 2. Throughput benchmark `scripts/bench_throughput.py`

- Against a running engine (config flag), fires N concurrent requests per family via the HTTP API (aiohttp/httpx pool): warmup, then measure p50/p95 per-request latency + requests/s at concurrency levels {1, 2, 4, 8}.
- Families: melo (batched), openvoice_v2 full clone pipeline, chatterbox (bs=1 honesty check), whisper STT.
- Output: table appended to `docs/benchmarks/throughput.md` with engine commit, GPU, dtype/sharing state. Run before Phase 5 (from a pre-phase git state) and after — the delta is the phase's evidence.

### 3. Capacity-signal docs

`docs/` (operations section): what `max_batch_size` means post-truthification, expected per-family concurrency on a 4060 Ti from the benchmark, and the note that Chatterbox scales by replicas, not batch (per the modernization decision).

### 4. Phase exit + knowledge updates

- Record results; tick Phase 5 in [summary.md](summary.md); update `memory-bank/progress.md`, `CONCERNS.md` (retire "misleading batch values"), `CHANGELOG.md`.

## Files to create/modify

| File | Change |
|---|---|
| `config.gpu.yml`, `config.yml` | truthful `max_batch_size` values |
| `scripts/bench_throughput.py` (new) | concurrency benchmark |
| `docs/benchmarks/throughput.md` (new) | recorded runs |
| `docs/` | capacity/ops documentation |

## Tests / validation

- Benchmark runs recorded (pre/post).
- Config lint (if any) or boot test with new values — scheduler budgets consistent with executor behavior (no assertion regressions in 5-1 suite).

## Checklist

- [ ] All `max_batch_size` values match executor reality (post-5-2)
- [ ] Chatterbox = 1 everywhere
- [ ] `bench_throughput.py` committed; pre/post tables in `docs/benchmarks/throughput.md`
- [ ] Ops docs on concurrency semantics + replica-scaling note
- [ ] Phase 5 ticked in summary.md; knowledge docs updated

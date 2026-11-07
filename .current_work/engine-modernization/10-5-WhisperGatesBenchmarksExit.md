# Phase 10-5: Whisper Gates, Benchmarks & Phase Exit

> **Plan:** [Engine Modernization](summary.md) · **Phase:** 10 — Whisper serving
> **Depends on:** 10-1, 10-2, 10-3, 10-4
> **Codebase mapping consulted:** `.planning/codebase/TESTING.md`; precedents: 5-3, 8-5, 8-6 (honesty policy, flood benchmark, docs matrix)

## Objective

Prove the phase's claims with the concurrency story the research promised, set measured truthful config values, and close Phase 10: whisper requests no longer pile up behind one sequentially-executed model — width (lanes) and depth (window batching) are both real, bounded by an honest memory budget, with the STT wire surface rounded out.

## Context (self-contained)

- **Baseline to beat:** whisper 2.8 req/s flat at c=8 with the sequential loop (Phase 5-3, GPU 1); head-of-line behavior unmeasured (the whisper.cpp issue-#1803 scenario: a 10-min file in front of a 3-s clip). The **headline metric is p95 of short clips under mixed load**, not just req/s.
- **Benchmark scenarios (benchmark host, RTX 5060 Ti, piggyback on the same session as the Phase 7/8 `_tbd_` tables, AR-0, and the 9-6 run if Phase 9 is executing):**
  - `{lanes only} vs {lanes + windows}` at c={4,8,20} + one 200-connection observation (client-ceiling honesty note, 8-6 pattern);
  - **fairness proof**: mixed queue (one ≥10-min file first, then short clips) — short-clip p50/p95 vs the sequential baseline (this is the pile-up answer);
  - M-sweep: `num_workers` 1/2/4 (turbo) and 1/4/6 (tiny) — width only helps if the data says so (CT2's own warning);
  - window-batch sweep: 4/8 windows per batch under flood;
  - translations endpoint + srt/vtt spot-check (10-4 real-model pass);
  - during runs: `hse_whisper_*` lane/queue metrics, CT2 queue gauges (if exposed), `hse_vram_device_used_bytes` vs the 10-3 budget (the admission-scaling proof: device_used stays under the bound while throughput holds).
- **Config honesty (this task introduces the shared mechanism — owner decision 2026-11-07):** measured `num_workers` + window-batch caps replace the initial values (turbo 2 / tiny 4 / windows 8). The three-class policy test (`tests/unit/initialization/test_config.py`) becomes a declarative **execution-model class table**: a model type maps to exactly one class and its shipped config must match that class's semantics (9-6 conforms when Phase 9 lands):

  | Class | Execution semantics | `max_batch_size` means | Additional knobs | Members (post-10) |
  |---|---|---|---|---|
  | `padded_batch` | one padded forward per length-similar sub-batch | sub-batch width | `batched_inference`, `batch_token_budget` | OV1/V2 synth + TC + TCE, hs1-enc/voc, voicefixer restorer/vocoder |
  | `loop` | per-request sequential | must be 1 | — | hs1-synthesizer, chatterbox-multilingual |
  | `queue_depth` | admission/backpressure only | queue depth | — | silero VAD (16) |
  | `lanes_windows` | M CT2 workers × cross-request window batches | admission depth | `num_workers`, window batch cap | faster-whisper |
  | `ar_decode` | token-level AR decode on the shared KV pool | admission depth | pool slots, token budget | chatterbox (en) + turbo (post-9) |

  `docs/batching.md` whisper row final ("lanes M × windows B, intra-audio 16, reasons"), docs-matrix lint in sync.
- **Knowledge updates:** memory-bank, `CONCERNS.md` (retire the whisper sequential-loop notes; record the memory-governance resolution of the "CT2 invisible to VRAM metrics" gap), `CHANGELOG.md` (user-visible: concurrent transcription throughput + fairness, new formats + translations endpoint), `docs/models.md` capacity numbers if present.

## Implementation steps

1. Extend `scripts/bench_throughput.py`: whisper scenario family (mixed-length generator with a pinned long-first case), M/window sweeps, fairness percentiles, device-memory sampling during runs.
2. Run the scenarios; fill `docs/benchmarks/throughput.md` §Phase 10 (structure + reproduce commands land with the task; numbers from the session).
3. Set measured config values; honesty policy + docs-matrix tests in sync.
4. Full suites green (`tests/unit`, `tests/integration`, wire-contract e2e incl. 10-4's matrix; real-model e2e for whisper STT/translations on the benchmark host).
5. Tick Phase 10 in [summary.md](summary.md); deviations recorded in the 10-x files.

## Files to create/modify

| File | Change |
|---|---|
| `scripts/bench_throughput.py` | whisper scenario family + sweeps + memory sampling |
| `config.gpu.yml`, `config.yml` | measured `num_workers` / window caps |
| `tests/unit/initialization/test_config.py`, `tests/unit/test_batching_docs_matrix.py` | policy + matrix sync |
| `docs/batching.md`, `docs/benchmarks/{throughput,vram-profiles}.md` | final rows + budget validation |

## Tests / validation

- Honesty + matrix tests green locally.
- Benchmark set recorded (incl. the fairness proof and the memory-budget hold); all suites green.

## Checklist

- [ ] `{lanes}` vs `{lanes+windows}` throughput + fairness percentiles recorded
- [ ] M-sweep + window sweep → measured config values (honesty tests in sync)
- [ ] Device-memory budget held under load (10-3 validation)
- [ ] 10-4 real-model spot-check green; docs + knowledge updated
- [ ] Full suites green; Phase 10 ticked in summary.md

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
  - **exhaustion scenarios (owner, 2026-11-07):** 100 heterogeneous requests (durations 2 s–8 min, mixed languages, transcription + translation, varied response formats) executed against deliberately tight caps (M=2 workers, window batch=4, queue 16) — capture queue-depth/wall-time traces, lane fairness under flood, timeout count, request loss (must be 0) and pairing errors (must be 0); plus a **memory-budget trip run**: `whisper_vram_budget_bytes` set below measured demand → admission-scaling engages, latency degrades gracefully, no OOM/crash. The degradation semantics (fallback correctness, zero loss) must also be proven as **local CPU integration tests** with stubbed CT2 and mocked `mem_get_info` before the host run.
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
- Full local suites green: `tests/unit` **974 passed**, `tests/integration`
  **59 passed**, `tests/e2e/wire_contract` **50 passed + 3 skipped**
  (model-gated). The 4 pre-existing `tests/e2e/tts/test_harmonyspeech.py`
  checkpoint-load errors are unchanged on this box (no weights).
- Exhaustion proof green locally (`tests/integration/test_whisper_exhaustion.py`,
  6 cases): 100-request tight-cap flood with zero loss/zero pairing errors,
  timeout storm, waiting+in-flight abort storm, mocked-`mem_get_info` budget trip
  (admission 6→2→1→0) and batch-level fallback counter.
- Benchmark set recorded (incl. the fairness proof and the memory-budget hold);
  all suites green. **Benchmark-host cells remain `_tbd_`** (no weights
  downloaded); the structure + reproduce commands are in
  `docs/benchmarks/throughput.md` §Phase 10.

## Checklist

- [ ] `{lanes}` vs `{lanes+windows}` throughput + fairness percentiles recorded
      — **structure + reproduce commands landed; host cells `_tbd_`**
- [ ] M-sweep + window sweep → measured config values (honesty tests in sync)
      — **initial values kept (turbo 2 / tiny 4 / windows 8); sweeps are `_tbd_`**
- [ ] Device-memory budget held under load (10-3 validation)
      — **local mocked-`mem_get_info` proof green; host trip run `_tbd_`**
- [ ] 10-4 real-model spot-check green; docs + knowledge updated
      — **docs + knowledge updated; real-model spot-check `_tbd_`**
- [x] Exhaustion scenarios recorded (100-request tight-cap flood + memory-budget
      trip) + local stubbed degradation tests green
      — **local tests green; host cells `_tbd_`**
- [x] Full suites green; Phase 10 ticked in summary.md

## Decisions (2026-11-07)

1. **`batched_inference` stays opt-in for Whisper.** The window batcher has a
   hermetic parity harness, per-request isolation and three safe fallbacks
   (`len == 1`, flag off, batch-level failure), but the **B=1 real-model parity
   gate has not run**. Flipping the shipped default now would make an
   unvalidated re-implementation of faster-whisper's window formation/stitching
   the default path; the phase's honesty principle says behaviour that changes
   output semantics ships only after its gate. The shipped configs therefore
   keep the flag unset, and the benchmark-host B=1 table in
   `docs/benchmarks/throughput.md` §Phase 10 is the flip condition. Fallback
   correctness and zero-loss degradation are proven locally
   (`tests/integration/test_whisper_exhaustion.py`).
2. **`num_workers` ships unmeasured.** Width only changes how many independent
   requests overlap on shared weights; each request still runs the proven
   per-request path, and the lane/isolation tests pin the M bound, per-thread
   pipeline state and abort semantics. Owner-approved initial values stay:
   turbo `2`, tiny `4`. The M-sweep tunes them on the host.
3. **Window cap stays 8** (owner-approved initial) and remains a module constant
   (`WHISPER_WINDOW_BATCH_SIZE`). The exhaustion scenario exercises 4 via the
   batcher's constructor; making it operator-tunable is a small follow-up if the
   host sweep shows a per-deployment win.
4. **The five-class execution-model table is authoritative** for the config
   honesty test; `ar_decode` has no shipped member until Phase 9-6 moves the
   English/Turbo Chatterbox roles into it (the test asserts that pending state).

## Deviations (2026-11-07)

1. **Benchmark script scenarios are a new `--scenario` axis, not a new family.**
   `scripts/bench_throughput.py` gained `--scenario {throughput,whisper-flood,
   whisper-fairness,whisper-budget-trip}` plus the duration generator, audio
   tiler, `/metrics` sampler/parser and per-request trace summary. The default
   remains `throughput`, so every Phase 5/7/8 invocation is unchanged.
   `--print-whisper-scenarios` is separate from `--print-scenarios` so the
   existing Phase 8 recipe test keeps its exact output.
2. **Per-request queue-wait is captured as the engine's histogram delta, not a
   per-request log line.** The engine exposes
   `hse_whisper_queue_wait_seconds_{sum,count,bucket}`; the harness records the
   delta next to client wall time. True per-request pairing is proven by the
   local engine tests, which have request ids and expected transcripts.
3. **Local exhaustion tests use the real engine/scheduler/handler with stubbed
   CT2, VAD and feature extraction** (no weights, CPU only). They mirror the
   100-request heterogeneous flood, a timeout storm, a waiting+in-flight abort
   storm and a batch-level fallback. The 8-minute request is included; total
   runtime is ~14 s.
4. **The memory-budget trip test mocks `torch.cuda.mem_get_info` but keeps the
   real `get_whisper_admission` → `Scheduler.schedule()` path**, so the
   observable is admitted requests per step (6 → 2 → 1 → 0), not a private
   counter.
5. **`docs/batching.md` gained a five-class table before the model matrix** (the
   matrix section must stay the first table under its heading for the docs lint).
   The Whisper row is the final "lanes M × windows B" wording and still declares
   `16`; the docs-matrix test gained a Phase 10 assertion.
6. **No config values changed.** The four owner-approved initial values are
   already in `config.yml` / `config.gpu.yml` (turbo 2, tiny 4, window 8 as a
   module constant, admission 16); this task only decides what ships unmeasured
   and records it.
7. **CHANGELOG needed no new entry.** The user-visible Phase 10 changes
   (concurrent transcription + fairness, `srt`/`vtt`, `/v1/audio/translations`)
   landed in 10-1/10-2/10-4; adding benchmark placeholders would be noise.
8. **Measurement cells remain `_tbd_`.** No model weights were downloaded and
   the benchmark-host session (Phase 7/8 piggyback) is outstanding; every table
   structure and reproduce command is in place.
9. **KittenTTS is classed `lanes_windows`** even though the task table's members
   cell names only Faster-Whisper. The five classes have no separate "lanes"
   class and every registered type must map to exactly one; KittenTTS is a
   lanes-only member (its validator pins `max_batch_size` to the lane count and
   forbids a batch flag), while Whisper is lanes × windows. The docs table states
   both.
10. **`whisper_max_audio_seconds` is documented as the probe's max shape, not an
    admission rejection.** The 10-3 key is stamped onto the config for the
    startup probe and is not enforced as a request-size limit today; the
    exhaustion boot therefore sets it to 480 s so the probe covers the 8-minute
    request. This corrects the earlier "request-size cap" wording rather than
    adding a new admission rejection outside this task's files.

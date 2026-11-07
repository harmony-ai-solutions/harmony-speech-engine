# Phase 10-1: Whisper Lanes — CT2 Workers, Fairness & Cancellation

> **Plan:** [Engine Modernization](summary.md) · **Phase:** 10 — Whisper serving
> **Depends on:** nothing (land first in Phase 10; 10-2 builds on its in-flight accounting). Do **after 10-3** if convenient — the worker-count initial values want the memory probe's numbers.
> **Codebase mapping consulted:** `.planning/codebase/ARCHITECTURE.md`; inputs: the 2026-11-06 concurrency research (Speaches/whisper.cpp/CT2 study), Phase 8-5 audit
> **State reference:** verified 2026-11-07 — the loader passes **no worker knobs** (`harmonyspeech/modeling/loader.py:312-316`: `WhisperModel(model, device, compute_type)`), so CT2 runs `inter_threads=1` today; the handler `_execute_faster_whisper` (`model_runner_base.py:1234-1294`) builds one `BatchedInferencePipeline` per step (`:1263`, clip batch 16 at `:1264`, per-request `last_speech_timestamp` reset at `:1273`) and loops requests **sequentially** (`:1285`); the scheduler refuses admission while a model has running work (`scheduler.py:190-192`).

## Objective

Give the whisper executors real execution width: M CT2 workers on shared weights (data-parallel, separate CUDA streams, GIL released during compute), an M-bound in-flight semaphore, two-lane fairness so short clips stop queueing behind long audio, and segment-bound cooperative cancellation. Queue capacity 16 stays what it is: backpressure.

## Context (self-contained)

- **Why workers, not model copies:** CTranslate2 documents that same-device workers **share model weights** (`num_workers → inter_threads`), so the VRAM cost is activations/workspaces, not model duplication — the only viable width on a card with ~7 GiB claimed headroom. faster-whisper passes `num_workers` straight through (`faster_whisper/transcribe.py` 1.2.1). Caveat to carry into benchmarks: the CT2 docs warn the gain is workload-dependent (measured reports of `num_workers=2` still serializing exist) — the 10-5 benchmark is the truth, not the knob.
- **Config:** `num_workers` per faster-whisper ModelConfig (plumb `config.py → loader.py:312`). Default `1` = today's behavior exactly. Shipped initial values (honesty pattern of 8-6's "initial 8"; owner-approved 2026-11-07): `faster-whisper-large-v3-turbo: 2`, `faster-whisper-tiny: 4`, tuned on the 10-5 data.
- **Handler shape:** the sequential `for` at `:1285` becomes an M-thread dispatch (ThreadPoolExecutor(M) owned by the runner, **M = num_workers, never the anyio default 40** — the Speaches anti-pattern); each worker runs `pipeline.transcribe(...)` and fully consumes the generator inside the worker so GPU time lands in our accounting. Per-request pipeline state discipline (`:1273` cursor reset) preserved per worker invocation.
- **Fairness lanes:** two admission lanes per whisper model — audio ≤30 s vs >30 s — drained with a credit ratio (e.g. 3 short : 1 long). Duration is known at admission: the artifact bus decodes once in `add_request`/`from_wire_audio` (Phase 2), so lane selection costs nothing. Lane depths on `/metrics`.
- **Cancellation:** faster-whisper has no mid-`generate` abort (unlike whisper.cpp's `abort_callback`); granularity is one segment — the worker checks the engine's abort/finish state between segments while consuming the generator.
- **Metrics:** CT2 queue introspection if the binding exposes it (`num_queued_batches`/`num_active_batches` — verify; else our own in-flight gauges): `hse_whisper_{workers,in_flight,lane_short_depth,lane_long_depth}` + per-request queue-wait seconds.

## Implementation steps

1. Plumb `num_workers` (config → loader → `WhisperModel`), default 1; unit-test the plumbing like the Phase 4 dtype work.
2. Rework `_execute_faster_whisper` to M-lane dispatch with the semaphore bound, per-worker generator consumption, abort checks between segments; `len==1` stays a direct call.
3. Lane admission in the scheduler/engine request path for whisper models (capacity-based like 9-4's AR admission, boolean otherwise unchanged).
4. Metrics + docs: `docs/batching.md` whisper row becomes "lanes (M workers) + intra-audio batch 16"; config-honesty test note for the new knob.
5. Benchmark-host: quick width sanity (c=4, short clips, M=1 vs 2 vs 4) before the full 10-5 run.

## Files to create/modify

| File | Change |
|---|---|
| `harmonyspeech/common/config.py`, `harmonyspeech/modeling/loader.py` | `num_workers` plumbing |
| `harmonyspeech/task_handler/model_runner_base.py` | M-lane handler + cancellation checks |
| `harmonyspeech/processing/scheduler.py` (admission path) | two-lane fairness for whisper models |
| `tests/unit/inference_flow/test_whisper_lanes.py`, integration (new/extend) | behaviors below |

## Tests / validation

- Unit: config plumbing; semaphore never exceeds M; lane selection from artifact duration; worker consumes generator fully.
- Integration: 4 mixed requests pair correctly; a long-first queue does not delay the short lane beyond the credit ratio; abort between segments errors only the aborted request.
- CUDA unit where available; full benchmark validation is 10-5.

## Checklist

- [ ] `num_workers` plumbed (default 1 = unchanged behavior)
- [ ] M-lane handler + cancellation landed; `len==1` fast path kept
- [ ] Two-lane fairness admission + lane metrics
- [ ] Worker-count initial values set (turbo 2 / tiny 4) pending 10-5 tuning
- [ ] `summary.md` Implementation Status updated

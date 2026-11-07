# Phase 9-4: Scheduler "Unfinished" Support (multi-step decode)

> **Plan:** [Engine Modernization](summary.md) · **Phase:** 9 — Chatterbox AR batching
> **Depends on:** nothing in Phase 9 (land before 9-3's handler integration; parallelizable with 9-1/9-2)
> **Codebase mapping consulted:** `.planning/codebase/ARCHITECTURE.md`; design input: study #14 D4
> **State reference:** verified 2026-11-07 against `harmonyspeech/processing/scheduler.py` — `waiting`/`running` deques (`:81-83`), `_schedule_waiting_requests` **refuses to admit while the model has running requests** (`:190-192`), `free_finished_requests()` drops finished requests from `running` each step (`:270-271`); engine `execute_model()` returns final results only.

## Objective

Teach the engine loop and scheduler that a request can be **RUNNING and unfinished across many engine steps** — the precondition for cross-request AR decode. Today one step == one full request execution; without this task, 9-3's runner would collapse back to serializing whole requests (and the `:190-192` boolean would starve admission for the whole family while anything decodes).

## Context (self-contained)

- **What changes semantically:** an executor may return a *partial* result ("row still decoding, N tokens done") instead of a final one. Unfinished requests stay in `running` across steps; the client still receives exactly one final response (no streaming — that stays the separate deferred workstream).
- **Admission becomes capacity-based for AR models:** replace the "model has running work → admit nothing" boolean with "admit while `active_rows < slots`" for models flagged AR. **Non-AR families keep today's boolean behavior byte-for-byte** — no scheduling change may leak to Phase 5/8 families (regression suites pin this).
- **Deadlines:** Phase 0-5's per-request deadline (`request_timeout_s`) must keep bounding total decode time — verify the enforcement point still trips for a request that survives many steps (it checks per step; confirm and test).
- **Abort/disconnect:** `engine.abort()` and client disconnect must free the row + its cache slot mid-decode (9-3 test depends on it).
- **Metrics:** per-AR-model `active_rows`, slot occupancy (from 9-2), tokens/step — the operator's live view of decode progress.

## Implementation steps

1. `EngineRequest`: unfinished flag + partial-progress field (request-scoped, cleared on finish; artifacts untouched).
2. `common/outputs.py` / executor result contract: an `unfinished` executor output variant carrying per-request progress, no client-visible payload.
3. `scheduler.py`: keep unfinished requests in `running`; capacity-based admission for AR-flagged models (slots come from the runner/9-2 via the model runner), boolean unchanged otherwise.
4. `harmonyspeech_engine.py` + `async_harmonyspeech.py`: propagate partial results into the running set; deadlines checked per step for unfinished requests; abort frees rows.
5. Feature-flag by model type (`ar_inference` family set) so a config without AR models behaves exactly as today.

## Files to create/modify

| File | Change |
|---|---|
| `harmonyspeech/processing/scheduler.py` | unfinished-aware running set + capacity admission |
| `harmonyspeech/engine/harmonyspeech_engine.py`, `engine/async_harmonyspeech.py` | partial-result propagation, deadline re-check, abort |
| `harmonyspeech/common/outputs.py` (or request type) | unfinished result variant |
| `tests/integration/test_ar_scheduling.py` (new) | behaviors below |

## Tests / validation

- Engine-level integration (stub AR model): unfinished request survives steps and finishes; a second request is admitted while the first decodes; abort mid-decode frees the row and the slot; deadline expiry errors the request (not the loop — 0-5 invariant); a poisoned decoding request errors alone.
- Regression: full Phase 5/8 scheduling suites green — zero behavior change for non-AR families (the batching-invariants suite is the canary).

## Checklist

- [x] Unfinished result variant + running-set semantics landed
- [x] Capacity-based admission for AR models; boolean preserved elsewhere (regression green)
- [x] Deadline + abort semantics verified for multi-step requests
- [x] Progress metrics exposed
- [x] `summary.md` Implementation Status updated

## Implementation notes (2026-11-07)

**Result contract.** `common/outputs.py` gains an explicit `UnfinishedRequestOutput`
(`unfinished=True`, `progress={"tokens", "step"}`; `finish_reason=None`, so
`finished()` stays `False`) plus `unfinished`/`progress` on the base
`RequestOutput`. The engine keys on the marker, never on a null finish reason, so
an error result cannot be mistaken for progress. `EngineRequest` gains
`unfinished` + `progress`, set by `set_progress()` and cleared by
`clear_progress()` on every terminal transition (alongside `clear_artifacts`).

**Running-set semantics.** `Scheduler._schedule_default()` reschedules a
`running` request when `_continues_across_steps()` says it is an AR-family row
that returned a partial result. The row is not duplicated in `running`; it is
prepended to that model's step list before newly admitted requests. Non-AR
families never have unfinished running rows (each step completes them), so their
schedule is byte-for-byte unchanged.

**Admission composition.** `AR_INFERENCE_MODEL_TYPES = {ChatterboxTTS,
ChatterboxTurboTTS}` (mirrors `docs/batching.md` / 10-5's
`FUTURE_AR_DECODE_MEMBERS`). For those models the width is
`min(max_batch_size - running, pool.slots_free)` via the injected
`ar_admission` hook (`HarmonySpeechEngine.get_ar_cache_capacity`, the same
precedent as the Whisper governor). A disabled pool, an unregistered arena
(`KeyError`, expected until 9-3) or a hook failure falls back to the historical
boolean rule; every non-AR model always takes the boolean path. **9-3 convention:**
the runner must register its arena under the scheduler's model name
(`ModelConfig.name` / `request_data.model`); an arena under any other key reads
as "no signal" and capacity admission will not apply.

**Deadlines.** `_expire_overdue_requests()` already scanned `running` every step,
so a request that survives many steps is re-checked per step. The expiry path
now also calls `_release_request_resources()`, so the row and its 9-2 slot are
freed and only that request errors (the loop keeps stepping).

**Abort/disconnect.** `abort_request()` releases request resources through a
duck-typed `executor.release_request(request_id)` hook (no `ExecutorBase` change;
9-3 implements it on the runner). Final/error results are expected to release
their own slot in the runner; the hook is for paths where the runner will never
see the request again.

**Loop liveness.** `AsyncHarmonySpeech.engine_step()` now returns
`bool(outputs or forwarded or engine.has_unfinished_requests())`. Without this a
progress-only step returned `False`, and `run_engine_loop()` would block in
`wait_for_new_requests()` until an unrelated request arrived — stalling the
decode. This is the one non-obvious engine-loop change.

**Progress metrics.** `get_ar_decode_stats()` reports per-AR-model
`active_rows`/slot occupancy (from 9-2 capacity) and cumulative
`tokens_total`/`steps_total`/`tokens_per_step` (fed by partial results; a
terminal result may optionally carry the final `progress`). Exposed on
`/health` (`ar_decode`) and `/metrics` (`hse_ar_decode_*`).

**Tests.** `tests/integration/test_ar_scheduling.py` (11 tests): scheduler
capacity composition + non-AR boolean preservation + no-capacity fallback;
survive-and-finish with exactly one client response; second request admitted
mid-decode; abort frees row+slot; deadline errors alone and the loop lives;
poisoned row errors alone; progress metrics; async progress-only step keeps the
loop stepping.

## Deviations (2026-11-07)

1. **Carry-over: `AsyncHarmonySpeech.get_whisper_lane_stats()` proxy + test.**
   9-2 recorded this Phase 10-1 gap (the serving layer holds the wrapper, so
   `/metrics` rendered zeroed Whisper-lane series in production). Added here
   because this task already touches the wrapper; the test lives in
   `tests/unit/inference_flow/test_whisper_lanes.py`.
2. **`get_ar_decode_stats()` proxy + a pure renderer module + `api_server.py`
   wiring.** The files table does not list a metrics module or the endpoint, but
   the checklist requires progress metrics. Following the 7-3/10-1 convention,
   the renderer is a separate torch-free module (`engine/ar_decode_metrics.py`)
   and `/metrics`/`/health` are wired for it. This is additive: the existing
   9-2 `hse_ar_cache_*` aggregate series are unchanged.
3. **`release_request(request_id)` is a duck-typed executor hook**, not a new
   `ExecutorBase` abstract method. That keeps every existing executor valid and
   confines 9-3's runner change to the runner; a missing hook is a no-op.
4. **The AR family set lives in `processing/scheduler.py`.** It is the runtime
   admission flag, separate from the shipped-config execution-class table (which
   9-6 moves); the set is the same two members, so the two cannot silently
   disagree without a deliberate edit.
5. **`docs/ar-cache.md` gained a short "Decode progress" subsection** for the
   new series. No CHANGELOG entry: 9-4 is internal plumbing with no
   user-visible capability until 9-3/9-6 land the runner and the config move.
6. **The shipped configs are untouched.** English/Turbo Chatterbox stay
   `max_batch_size: 1` (`loop`) until 9-6; with width 1 a single request still
   decodes across steps, which is the prerequisite 9-3 needs.

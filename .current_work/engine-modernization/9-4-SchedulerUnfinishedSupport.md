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

- [ ] Unfinished result variant + running-set semantics landed
- [ ] Capacity-based admission for AR models; boolean preserved elsewhere (regression green)
- [ ] Deadline + abort semantics verified for multi-step requests
- [ ] Progress metrics exposed
- [ ] `summary.md` Implementation Status updated

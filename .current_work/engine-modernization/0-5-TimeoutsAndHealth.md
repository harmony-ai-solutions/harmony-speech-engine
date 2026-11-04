# Phase 0-5: Timeouts & Health

> **Plan:** [Engine Modernization](summary.md) · **Phase:** 0 — Correctness & security foundation
> **Depends on:** 0-4 (error-result plumbing reused)
> **Codebase mapping consulted:** `.planning/codebase/CONCERNS.md`, `ARCHITECTURE.md`

## Objective

Replace the engine-step watchdog that **kills the entire background loop** after 120 s with per-request deadlines, so one slow or stuck request errors out while the engine keeps serving. Also fix the `check_health` bug that iterates dict *keys* instead of executors.

## Context (self-contained)

Current behavior (verified 2026-11-04):

- `harmonyspeech/engine/async_harmonyspeech.py:15` defines `ENGINE_ITERATION_TIMEOUT_S = 120`.
- `run_engine_loop()` (`:366-383`) wraps **the whole `engine_step()`** in `asyncio.wait_for(...)`. On `TimeoutError` it logs "This should never happen!", calls `self.set_errored(exc)` and **re-raises** — the background loop dies; every in-flight and future request hangs; the server is a zombie until restarted.
- `HarmonySpeechEngine.check_health()` (`harmonyspeech/engine/harmonyspeech_engine.py:697-699`) iterates `self.model_executors` (a dict keyed by model name) and calls `check_health()` on each **key string** — the health probe is broken today (re-read the exact lines; if it iterates `.values()` already, close this sub-item as done).

Why per-request: a single engine step legitimately runs multiple models and requests (OpenVoice clone = Whisper→TCE→Melo→TC hops); a global step timeout cannot distinguish "one stuck request" from "loop is dead", and any future slow-but-legal step (cold model load, Phase 6 remote step) would trip it.

## Implementation steps

### 1. Add a per-request deadline

- `harmonyspeech/common/request.py` (`EngineRequest`): add `deadline: float | None = None` (monotonic time) set at `add_request` from a configurable default.
- `harmonyspeech/common/config.py` or serving args: add `request_timeout_s` (engine-level default; suggested 120 s to preserve current worst-case latency for legitimate long syntheses; make it overridable per request later if needed — not in scope now).
- `AsyncHarmonySpeech.add_request` / `HarmonySpeechEngine.add_request`: stamp `request.deadline = time.monotonic() + request_timeout_s`.

### 2. Enforce deadlines in the engine loop

In the step machinery (`harmonyspeech_engine.py` `step()`/`step_async()` and/or `async_harmonyspeech.py` `engine_step()`), before/after executing a scheduled batch:

- Any request whose `deadline` has passed without finishing → produce a typed error output (`finish_reason="error"`, message "request exceeded timeout of Xs") via the same output path used by Phase 0-4, update scheduler status to finished, and drop it from `running`.
- Keep the per-model execution synchronous — we are not cancelling mid-forward (that would corrupt CUDA state); we are guaranteeing that once control returns to the engine, expired requests resolve as errors instead of being rescheduled forever.

### 3. Demote the loop watchdog to a safety net

`run_engine_loop()`:

- Keep `asyncio.wait_for(engine_step(), ENGINE_ITERATION_TIMEOUT_S)` **but** on `TimeoutError`: log critical, mark **all currently running requests** as errored (typed outputs, same plumbing), reset engine step state (clear `running` back to consistent state — re-read `scheduler.free_finished_requests()` interplay), and **continue the loop** instead of `raise`. Only re-raise if the same model errors repeatedly (e.g. 3 consecutive watchdog trips → `set_errored`, matching current semantics as a last resort).
- Document the chosen constants at the definition site.

### 4. Fix `check_health`

`harmonyspeech_engine.py:697-699`: iterate `self.model_executors.values()`; each executor's `check_health()` result (bool) aggregated — return overall health and per-model detail dict (this dict is extended in Phase 4-5 with VRAM numbers, so return a structured payload now).

## Files to create/modify

| File | Change |
|---|---|
| `harmonyspeech/common/request.py` | `deadline` field |
| `harmonyspeech/engine/harmonyspeech_engine.py` | deadline stamping + expiry-to-error path; `check_health` fix (`.values()`, structured payload) |
| `harmonyspeech/engine/async_harmonyspeech.py` | watchdog demoted to recoverable safety net; expired-request error outputs |
| `harmonyspeech/endpoints/args.py` (or config) | `request_timeout_s` setting |
| `tests/unit/error_handling/` (new) | `test_request_timeouts.py` |

## Tests

- Unit: request whose deadline is in the past → next `step()` produces `finish_reason="error"` output mentioning timeout; request removed from queues; loop state consistent.
- Unit: `check_health` returns per-model detail and does not raise (mock executors; previously would `AttributeError` on str).
- Integration (mocked slow model): `run_engine_loop` survives a step that exceeds `ENGINE_ITERATION_TIMEOUT_S` (patch constant to ~0.1 s): loop continues; in-flight requests get error outputs; a subsequent normal request succeeds.

## Checklist

- [ ] `EngineRequest.deadline` + default `request_timeout_s` implemented
- [ ] Expired requests resolve as typed errors; never rescheduled forever
- [ ] Watchdog no longer kills the loop on first trip (recoverable, with consecutive-failure escalation)
- [ ] `check_health` fixed and returns structured per-model payload
- [ ] Timeout unit/integration tests added and passing
- [ ] `CHANGELOG.md` entry
- [ ] `summary.md` Implementation Status updated

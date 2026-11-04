# Phase 5-1: Type-Checked Batches & Pairing Invariants

> **Plan:** [Engine Modernization](summary.md) · **Phase:** 5 — Honest batching (HSE only)
> **Depends on:** Phase 0 (0-3 especially), Phase 2 (artifacts)
> **Codebase mapping consulted:** `.planning/codebase/ARCHITECTURE.md` (scheduler/executor flow), research §1.5, §5.3(d)

## Objective

Make the *existing* batch flow provably safe and observable, as the foundation for real batched inference (5-2): request-type-homogeneous batches, verified input↔request alignment at every stage, and request-id-keyed result assembly — plus regression tests that run the engine at batch size > 1 end-to-end.

## Context (self-contained)

- The scheduler already admits up to `max_batch_size` requests per model per cycle (`scheduler.py:163-189` — `new_scheduled_for_models` allows adding requests for a model within one step; budget = `model_cfg.max_batch_size`, `:191-199`).
- 0-3 made handlers request-explicit and split mixed-type batches at dispatch; this task extends those invariants to **engine-level integration tests** and tightens the remaining seams:
  - `execute_model` dispatch (`model_runner_base.py:96-115`) — re-verify the split covers *every* handler, not just Chatterbox.
  - `prepare_inputs` entry (`inputs.py`) — alignment between prepared `inputs` and `requests_to_batch` after the split (prepare was possibly already called on the mixed list — re-read the call order; if prepare runs before dispatch, move the split before prepare).
- Async step path (`async_harmonyspeech.py:188-193`) awaits models sequentially — unchanged in this task (overlap is explicitly out of scope for Phase 5).

## Implementation steps

### 1. Hoist the type-split before input preparation

Re-read `execute_model` + `prepare_inputs` call order. Guarantee: one prepare call + one handler call per (model, request-type) partition, with `zip(requests, inputs)` passed to handlers. Add a shared invariant helper (used by all handlers):

```python
def assert_batch_aligned(handler_name: str, inputs: list, requests: list) -> None:
    if len(inputs) != len(requests):
        raise EngineBatchError(f"{handler_name}: {len(inputs)} inputs vs {len(requests)} requests")
```

(`EngineBatchError` → typed request errors, not crashes.)

### 2. Request-id keyed result assembly

Where outputs are collected (`outputs.append(_build_result(initial_request, ...))`), additionally verify uniqueness: no output may be built twice for the same `request_id` within a step (cheap set-check in `_build_result`). Catches future pairing regressions at runtime, not in review.

### 3. Engine-level batch integration tests

New `tests/integration/test_batching_invariants.py` (CPU models: KittenTTS + Silero + whisper-tiny config):

- N=4 concurrent identical requests to one model → all 4 outputs, correct ids, per-request outputs distinguishable when inputs differ (different texts).
- N=4 mixed-type requests to a Chatterbox-family executor (mocked model if CPU Chatterbox is too heavy — a stub executor registered under the Chatterbox model_type via test config) → split dispatch, all outputs.
- Reordering: submit A,B; artificially delay A's model (stub) → outputs still pair correctly (no index reuse).
- Scheduler interaction: forwarded (pipeline) request + fresh request to the same model in one cycle → no slot starvation assertion (documents current head-of-line behavior; fix out of scope).

## Files to create/modify

| File | Change |
|---|---|
| `harmonyspeech/task_handler/model_runner_base.py` | split-before-prepare; `assert_batch_aligned`; `_build_result` uniqueness check |
| `harmonyspeech/task_handler/inputs.py` | partition-aware prepare (if order requires) |
| `tests/integration/test_batching_invariants.py` (new) | the suite above |

## Tests

See step 3 — the suite *is* the deliverable. All existing suites must remain green.

## Checklist

- [ ] Type-split hoisted before prepare; zip-based pairing everywhere
- [ ] `EngineBatchError` + `_build_result` id-uniqueness guard
- [ ] Integration suite (identical, mixed, delayed/reordered) green
- [ ] `summary.md` Implementation Status updated

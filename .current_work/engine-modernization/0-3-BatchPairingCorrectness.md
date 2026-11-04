# Phase 0-3: Batch Pairing Correctness

> **Plan:** [Engine Modernization](summary.md) · **Phase:** 0 — Correctness & security foundation
> **Depends on:** nothing (parallel to 0-1/0-2)
> **Codebase mapping consulted:** `.planning/codebase/CONCERNS.md`, `ARCHITECTURE.md` (executor dispatch), `TESTING.md`

## Objective

Make per-request results provably paired with the right request and the right input data, at any batch size and any ordering. Today three defects make a batch of size ≥ 2 unsafe (and make Phase 5 impossible without rework):

1. **Closure late-binding**: inner worker functions read the *outer loop variable* `initial_request` instead of receiving the request as a parameter, so the audio bytes used can belong to whichever request the variable points to at call time.
2. **Mixed request-type batches**: a single executor batch can contain different request types; the handler is chosen by inspecting only the batch's request types (`model_runner_base.py:96-115`), so e.g. a `ChatterboxTTS` executor batch mixing `SpeechEmbeddingRequestInput` and `TextToSpeechRequestInput` dispatches the *whole* batch to one handler.
3. **Index-based invariants**: results are assembled positionally (`inputs[i]` ↔ `requests_to_batch[i]`) with no assertion that the two lists are aligned.

## Context (self-contained)

Dispatch: `model_runner_base.py:96-115` (`execute_model`) selects the `_execute_*` handler from the request types present in `requests_to_batch`; `inputs` come from the matching `prepare_*` function in `task_handler/inputs.py` (which uses `ThreadPoolExecutor.map`, preserving order — but nothing asserts it).

Closure bug pattern (repeats in every Chatterbox handler, verified 2026-11-04):

```python
# _execute_chatterbox_tts (model_runner_base.py:677-718)
def synthesize(input_params):
    ...
    elif hasattr(initial_request.request_data, "input_audio") and initial_request.request_data.input_audio:
        audio_bytes = base64.b64decode(initial_request.request_data.input_audio)  # ← outer variable!
        ...

for i, x in enumerate(inputs):
    initial_request = requests_to_batch[i]   # ← assigned AFTER synthesize is defined
    inference_result = synthesize(x)         # works only because call happens in same iteration
```

It "works" in the current sequential loop but is a landmine: any reordering, parallelization (`ThreadPoolExecutor`), or refactor that moves the call out of the same iteration pairs the wrong audio with the request. The research flags it as "blocks any parallelisation" (§1.6 #6).

## Implementation steps

### 1. Pass the request explicitly into every worker function

Refactor all handlers in `model_runner_base.py` (the full list of loops: `:163`, `:186`, `:234`, `:279`, `:327` (tone converter), `:397` (whisper), `:456`, `:666` (kitten), `:677`, `:720`, `:762` (chatterbox tts×3), `:806` (embedding), `:851` (vc)):

```python
# Before
def synthesize(input_params): ...          # reads initial_request from enclosing scope
for i, x in enumerate(inputs):
    initial_request = requests_to_batch[i]
    inference_result = synthesize(x)

# After
def synthesize(input_params, initial_request): ...
for i, x in enumerate(inputs):
    assert len(inputs) == len(requests_to_batch)
    initial_request = requests_to_batch[i]
    inference_result = synthesize(x, initial_request)
```

Where audio bytes are needed, take them from the **passed** request (or better: from `input_params`, which `prepare_*` already builds per request — check each `prepare_chatterbox_*` in `inputs.py:827-876` and prefer moving the audio bytes into `input_params` so the worker is request-agnostic).

### 2. Split mixed-type batches before handler dispatch

In `execute_model` (`model_runner_base.py:96-115`): group `requests_to_batch` by request-input type. If more than one type is present, partition both `requests_to_batch` and the already-prepared `inputs` consistently (prepare functions are typed per input type — re-read to confirm whether `prepare_inputs` already splits; if it prepares a superset, partition before preparing), then call the matching handler once per partition and concatenate outputs. Log at debug level when a split occurs.

### 3. Assert alignment invariants

- At handler entry: `assert len(inputs) == len(requests_to_batch), f"{handler}: {len(inputs)} inputs vs {len(requests_to_batch)} requests"`.
- Results are appended per iteration keyed by the *current* `initial_request` (already the case via `_build_result(initial_request, ...)`) — keep it, and after the loop optionally assert `{o.request_id for o in outputs} == {r.request_id for r in requests_to_batch}` where output objects carry request ids (verify attribute name in `common/outputs.py`).

### 4. Keep per-request error isolation

Existing per-item `try/except` → `_build_error_result(initial_request, e)` is correct — ensure the refactor does not accidentally hoist it around the whole loop.

## Files to create/modify

| File | Change |
|---|---|
| `harmonyspeech/task_handler/model_runner_base.py` | all handler loops refactored to explicit request passing; mixed-type split in `execute_model`; alignment asserts |
| `harmonyspeech/task_handler/inputs.py` | move per-request audio bytes into `input_params` where the worker currently reaches back into the request (chatterbox prepare functions `:827-876`) |
| `tests/unit/inference_flow/` (new) | `test_batch_pairing.py` |

## Tests

- Unit: construct a batch of two mocked requests with distinguishable payloads (e.g. different texts/audio markers) and a stubbed `self.model`; assert each output's request id maps to its own payload. Include a shuffled-order variant.
- Unit: mixed batch (one `SpeechEmbeddingRequestInput` + one `TextToSpeechRequestInput` on a Chatterbox executor) → both get their type-correct handler, both outputs present.
- Unit: `len(inputs) != len(requests_to_batch)` raises `AssertionError` (or converts to typed engine error if preferred — pick one and document).
- Regression: run existing `tests/e2e/tts` suite untouched (CPU) to prove no behavioral change.

## Checklist

- [x] All 11+ handler loops pass the request explicitly (no closure reads)
- [x] `execute_model` splits mixed-type batches and dispatches per partition
- [x] Alignment asserts in place at handler entry
- [x] Pairing unit tests (including reorder + mixed batch) added and passing
- [ ] Existing e2e suites still green
- [x] `summary.md` Implementation Status updated

## Implementation notes (2026-11-04)

- All 15 handler loops pass `initial_request` as an explicit parameter to the
  worker helper; no worker reads the enclosing loop variable any more.
- `execute_model` partitions `requests_to_batch` by `type(request.request_data)`
  before calling `prepare_inputs`, logs a debug line when a split happens, then
  dispatches each partition to its handler and concatenates the results. This
  fixes the mixed Chatterbox embed + TTS batch that used to drop TTS requests.
- Every handler starts with `_require_aligned(...)`, which raises
  `AssertionError(f"{handler}: N inputs vs M requests")`. `execute_model` also
  validates the prepare result and converts a mismatch into per-request typed
  error results, so the engine loop cannot be killed by a broken invariant.
  Documented choice: handlers assert; the engine path fails closed per request.
- Deviation: the `prepare_chatterbox_*` tuple contract was left unchanged
  (audio bytes are still read from the explicitly-passed request inside the
  worker). Moving them into `input_params` is a Phase 2 artifact-bus concern;
  doing it here would churn the tuple shapes consumed by existing tests without
  improving pairing, which is now guaranteed by explicit request passing.
- `tests/e2e/*` was **not** executed: those suites download real model weights.
  Collection is green (39 tests collected) and the new pairing tests use stubs;
  manual e2e verification remains outstanding for the owner.

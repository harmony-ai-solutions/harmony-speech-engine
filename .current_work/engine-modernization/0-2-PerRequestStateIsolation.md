# Phase 0-2: Per-Request State Isolation

> **Plan:** [Engine Modernization](summary.md) · **Phase:** 0 — Correctness & security foundation
> **Depends on:** 0-1 (serialization format may change conditionals handling)
> **Codebase mapping consulted:** `.planning/codebase/CONCERNS.md` (high finding: voice leakage across requests), `ARCHITECTURE.md`

## Objective

End cross-request voice leakage caused by **mutable state on process-wide model instances**. Today every Chatterbox execution writes speaker conditioning onto the shared model object and never resets it. A subsequent request that arrives *without* its own conditioning input silently inherits the previous request's voice. This is a correctness bug today, a privacy bug in multi-tenant use, and the **hard prerequisite** for Phase 4 (weight sharing) and Phase 5 (batching) — both would amplify it into cross-request contamination.

## Context (self-contained)

Mutation sites in `harmonyspeech/task_handler/model_runner_base.py` (verified 2026-11-04):

| Site | Handler | What is mutated |
|---|---|---|
| `:685` | `_execute_chatterbox_tts` | `self.model.conds = conditionals.to(self.device)` |
| `:690` | `_execute_chatterbox_tts` | `self.model.prepare_conditionals(audio_buf, ...)` (writes `self.model.conds` internally) |
| `:728`, `:733` | `_execute_chatterbox_turbo_tts` | same pattern |
| `:771`, `:775` | `_execute_chatterbox_multilingual_tts` | same pattern |
| `:826` / `:831` | `_execute_chatterbox_embedding` | `prepare_conditionals(...)` then reads `self.model.conds` |
| `:857-863` | `_execute_chatterbox_vc` | `self.model.ref_dict = ...` or `self.model.set_target_voice(...)` |

Leak scenario: request A (voice cloning, audio of speaker X) → `self.model.conds = X`. Request B (no `input_audio`, no `input_embedding` — e.g. malformed or a "default voice" call) → neither branch at `:683-690` fires → `generate()` runs **with X's conditionals still assigned**. Same for `ref_dict` in VC.

Constraint: the pinned upstream `chatterbox-tts==0.1.6` API reads conditioning from `self.model.conds` / `self.model.ref_dict` internally inside `generate()` — we cannot avoid the assignment without forking upstream. The fix is therefore **deterministic assignment + baseline restoration**, not removal.

## Implementation steps

### 1. Capture a baseline at model load

In the GPU/CPU model runner init path (`harmonyspeech/task_handler/gpu_model_runner.py` / `model_runner_base.py`, wherever `self.model` is assigned), after loading a Chatterbox-family model:

```python
import copy

# Baseline = whatever conditionals the model ships with (may be None until first prepare)
self._baseline_state = self._snapshot_chatterbox_state()

def _snapshot_chatterbox_state(self):
    state = {}
    if hasattr(self.model, "conds") and self.model.conds is not None:
        state["conds"] = copy.deepcopy(self.model.conds)  # tensors: deepcopy keeps device
    if hasattr(self.model, "ref_dict") and self.model.ref_dict is not None:
        state["ref_dict"] = copy.deepcopy(self.model.ref_dict)
    return state

def _restore_chatterbox_state(self):
    self.model.conds = self._baseline_state.get("conds")
    self.model.ref_dict = self._baseline_state.get("ref_dict")
```

Notes:

- `Conditionals` may hold device tensors; `copy.deepcopy` on tensors preserves device and storage — acceptable for a one-time snapshot (weights-sized, conditionals are small).
- If the model loads with `conds = None` (no default voice), the baseline is "no conditionals" — restoration then makes the leak impossible because the follow-up request fails loudly (step 3) instead of inheriting.

### 2. Wrap every Chatterbox handler execution in restore-on-exit

In `_execute_chatterbox_tts`, `_execute_chatterbox_turbo_tts`, `_execute_chatterbox_multilingual_tts`, `_execute_chatterbox_vc`, `_execute_chatterbox_embedding`: wrap the per-request inner call so that **every** exit path restores the baseline:

```python
try:
    ...  # existing synthesize/convert/compute body (assignment happens here)
finally:
    self._restore_chatterbox_state()
```

`try/finally`, not `except` — errors must not skip restoration.

### 3. Fail loudly on missing conditioning input

In each TTS/VC handler: if a request provides **neither** pre-computed conditionals/embedding **nor** `input_audio`/`target_audio`, raise `ValueError("voice cloning requires input_audio or a precomputed embedding/conditionals")` so the request resolves as a 400-style error (see `_build_error_result`). This converts today's silent inheritance into a typed rejection. Verify against product expectations for KittenTTS-style "default voice" calls — Chatterbox has no shipped default voice, so erroring is correct; if a future default is desired it must be an explicit config (`default_conditionals_path`), never the previous request's state.

### 4. Document the sharing invariant

Add a module-level comment block in `model_runner_base.py` near the handlers:

> Shared-weight invariant (Phase 4 dependency): per-request state lives on the *wrapper executor object* (`self.model.conds`, `self.model.ref_dict`) — never on shared backbone submodules (S3Gen/VE). Executors that share backbones must keep distinct wrapper instances or distinct per-request state slots.

## Files to create/modify

| File | Change |
|---|---|
| `harmonyspeech/task_handler/model_runner_base.py` | baseline snapshot/restore helpers; `try/finally` in the 5 Chatterbox handlers; missing-conditioning `ValueError` |
| `harmonyspeech/task_handler/gpu_model_runner.py` | baseline capture after model load (if init lives there; re-read to confirm) |
| `harmonyspeech/task_handler/cpu_executor.py` / worker path | same capture if Chatterbox runs on CPU configs |
| `tests/unit/inference_flow/` (new) | `test_chatterbox_state_isolation.py` |

## Tests

- Unit (mocked model or tiny CPU fixture): request A sets `conds`; assert after handler returns `self.model.conds` equals the baseline (identity/equality).
- Unit: request B without conditioning → error result, `finish_reason == "error"`, message mentions required input.
- Unit: VC path — `ref_dict` restored after `convert()`.
- If a GPU is available: two sequential engine requests with different reference audio produce different outputs (hash the waveforms); CPU mock acceptable for CI (tests are CPU-only today per `.planning/codebase/TESTING.md`).

## Checklist

- [ ] Baseline snapshot/restore implemented for all Chatterbox handlers (try/finally)
- [ ] Missing-conditioning requests fail with typed errors instead of inheriting state
- [ ] Sharing invariant documented in code
- [ ] Isolation unit tests added and passing
- [ ] `CHANGELOG.md` entry
- [ ] `summary.md` Implementation Status updated

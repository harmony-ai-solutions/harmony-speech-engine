# Phase 0-4: Startup Validation & No-Hang Guarantees

> **Plan:** [Engine Modernization](summary.md) · **Phase:** 0 — Correctness & security foundation
> **Depends on:** nothing (parallel to 0-1…0-3)
> **Codebase mapping consulted:** `.planning/codebase/CONCERNS.md` ("Requests Can Hang Forever"), `ARCHITECTURE.md` (scheduling/dispatch points)

## Objective

Guarantee that every admitted request either resolves or errors — never waits forever — and that a misconfigured engine **fails loudly at boot** instead of silently advertising pipelines it cannot run.

## Context (self-contained)

The hang mechanism (verified 2026-11-04):

1. A request enters with a *sentinel* model name (e.g. `"openvoice_v2"`, `"chatterbox"`). `check_reroute_request_to_model` (`harmonyspeech/engine/harmonyspeech_engine.py:337-361`) dispatches to a family reroute function that scans `self.model_configs` for a concrete `model_type` (e.g. `OpenVoiceV2ToneConverterEncoder`).
2. If the config is missing (user deleted a model from YAML), **no branch assigns `request.model`** — it stays at the sentinel string.
3. The scheduler (`harmonyspeech/processing/scheduler.py:27-30`, `:177-179`) computes `remaining_request_budget(model)`; the sentinel is not in `request_per_model_budget`, so the budget is 0 and the request is re-queued **forever**. The HTTP call hangs until the client disconnects.

Related weakness: model cards are built from `serving_*` group lists (e.g. `_TTS_MODEL_GROUPS` in `endpoints/openai/serving_text_to_speech.py:25-32`), which can advertise a family whose members are partially configured.

Phase 3 will replace the family reroute functions with `PipelineSpec` and makes this validation table-driven; this task installs the guarantee now, with the current structures, and defines the seams Phase 3 will reuse.

## Implementation steps

### 1. Startup toolchain validation

In `HarmonySpeechEngine.__init__` / `init_custom_executors()` (`harmonyspeech_engine.py:69-87`), after executors are built, validate every known family against the configured models:

- Encode the **required model types per family** in one place (a dict — this is the seed of Phase 3's `PipelineSpec`; keep it in a new module `harmonyspeech/processing/pipeline_registry.py` so Phase 3 can grow it):

```python
# harmonyspeech/processing/pipeline_registry.py
FAMILY_REQUIREMENTS: dict[str, list[str]] = {
    "harmonyspeech": ["HarmonySpeechEncoder", "HarmonySpeechSynthesizer", "HarmonySpeechVocoder"],
    "openvoice_v1": ["FasterWhisper", "OpenVoiceV1ToneConverterEncoder", "OpenVoiceV1Synthesizer", "OpenVoiceV1ToneConverter"],
    "openvoice_v2": ["FasterWhisper", "OpenVoiceV2ToneConverterEncoder", "MeloTTSSynthesizer", "OpenVoiceV2ToneConverter"],
    "voicefixer": ["VoiceFixerRestorer", "VoiceFixerVocoder"],
    "chatterbox": ["ChatterboxTTS"],
    "chatterbox_turbo": ["ChatterboxTurboTTS"],
    "chatterbox_multilingual": ["ChatterboxMultilingualTTS"],
    "chatterbox_vc": ["ChatterboxVC"],
}

def validate_families(model_configs: list[ModelConfig]) -> list[str]:
    """Returns list of family ids that are fully satisfied by the given configs."""
    available_types = {c.model_type for c in model_configs}
    return [f for f, reqs in FAMILY_REQUIREMENTS.items() if all(t in available_types for t in reqs)]
```

- Validation policy: log an explicit warning per incomplete family naming the family and the missing `model_type`s; expose the complete-family list on the engine (`self.available_families`). Do **not** hard-fail boot (operators may intentionally run subsets — e.g. CPU-only configs without Whisper), but…
- …the serving layer must only **advertise** complete families (step 3).

### 2. Fail fast on unresolvable routing

In `check_reroute_request_to_model` (or immediately after it in the request admission path, `harmonyspeech_engine.py` `add_request`): after rerouting, if `request.model` is still the sentinel (or not present in `self.model_executors`), build an **error output** for the request and finish it instead of scheduling:

```python
if request.model not in self.model_executors:
    error_result = RequestOutput(
        request_id=request_id,
        finish_reason="error",
        error=f"Model family '{request.requested_model}' is not fully configured on this engine "
              f"(missing one of: {FAMILY_REQUIREMENTS.get(request.requested_model, ['unknown'])})",
    )
    # route through the request tracker so the HTTP call resolves with an error, not a hang
```

Re-read how `RequestOutput` errors are surfaced (`common/outputs.py`, `async_harmonyspeech.py` `_request_tracker.process_request_output`) and follow that path exactly; the goal is the client receiving a 4xx/5xx body immediately.

### 3. Advertise only complete families

`endpoints/openai/serving_engine.py` model-card generation (`model_cards_from_config_groups`) and the per-serving `_MODEL_GROUPS` lists: filter groups against `engine.available_families`. A family missing a member must not appear in `/v1/models`.

### 4. Scheduler safety net

Optional but cheap: in `Scheduler._schedule_waiting_requests` (`scheduler.py:165-189`), if `budget.remaining_request_budget(model_name) <= 0` **and** `model_name` is not a key in `request_per_model_budget` at all (i.e. unknown model), route the request to `ignored_requests` with an error instead of `leftover_waiting_requests`. This is defense-in-depth for any future routing path that forgets to resolve.

## Files to create/modify

| File | Change |
|---|---|
| `harmonyspeech/processing/pipeline_registry.py` (new) | `FAMILY_REQUIREMENTS`, `validate_families()` |
| `harmonyspeech/engine/harmonyspeech_engine.py` | startup validation call + `available_families`; unresolvable-routing error path |
| `harmonyspeech/processing/scheduler.py` | unknown-model → `ignored_requests` safety net |
| `harmonyspeech/endpoints/openai/serving_engine.py` | advertise complete families only |
| `tests/unit/initialization/`, `tests/unit/error_handling/` | validation + no-hang tests |

## Tests

- Unit: config list missing `OpenVoiceV2ToneConverter` → `validate_families` excludes `openvoice_v2`; warning logged (caplog).
- Unit/integration: request to an incomplete family → request resolves with `finish_reason="error"` within one engine step; **no** infinite requeue (step the engine manually N times, assert queue drains).
- Unit: `/v1/models` card list omits incomplete families.
- Unit: scheduler safety net — request with never-configured model name lands in `ignored_requests`, not `waiting`.

## Checklist

- [x] `pipeline_registry.py` with family requirements + validator created
- [x] Startup validation logs missing members per incomplete family; `available_families` exposed
- [x] Unresolvable routing produces an immediate typed error (no hang) — verified by test
- [x] Model cards only advertise complete families
- [x] Scheduler unknown-model safety net in place
- [x] `docs/models.md` / `README.md` note the validation behavior
- [x] `summary.md` Implementation Status updated

## Implementation notes (2026-11-04)

- `harmonyspeech/processing/pipeline_registry.py` holds `FAMILY_REQUIREMENTS`,
  `validate_families()`, `missing_family_requirements()` and
  `is_family_available()`. Phase 3 grows this module into `PipelineSpec`.
- `HarmonySpeechEngine` logs one warning per incomplete family (naming the
  missing `model_type`s) and exposes `available_families`.
- Rejected routing: `add_request` builds a `finish_reason="error"` `RequestOutput`
  when `request.model` is not a configured executor and queues it as a
  *pre-step error*. Both `step()` and `step_async()` drain that queue through
  `_process_model_outputs`, so the request resolves within one engine step via
  the normal `_request_tracker.process_request_output` path (HTTP 400), without
  ever entering the scheduler.
- Model cards: `model_cards_from_config_groups` filters declared groups against
  `validate_families(configured_models)`; groups whose name is not declared in
  the registry keep the previous behaviour.
- Scheduler: a request whose model has no entry in the scheduling budget is
  appended to `ignored_requests` (propagated by `_schedule_default`) and becomes
  an error output in `_process_model_outputs` — defence in depth for future
  routing paths.
- Deviation: filtering is computed inside the static card builder from the model
  configs (equivalent to `engine.available_families`, but usable by the static
  `models_from_config` helpers in `api_server` and the e2e fixtures).

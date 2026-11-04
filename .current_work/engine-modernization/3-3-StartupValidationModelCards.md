# Phase 3-3: Startup Validation & Model Cards from Specs

> **Plan:** [Engine Modernization](summary.md) · **Phase:** 3 — Declarative PipelineSpec routing
> **Depends on:** 3-2
> **Codebase mapping consulted:** `.planning/codebase/ARCHITECTURE.md`, `CONCERNS.md` ("advertise an incomplete toolchain")

## Objective

Make specs the single source of truth for **what the deployment can do**: boot-time validation of every registered spec against the configured models (naming the pipeline *and* the missing step), and `/v1/models` cards generated from specs instead of the hand-maintained `_TTS_MODEL_GROUPS`-style lists in each serving module.

## Context (self-contained)

- 0-4 introduced `FAMILY_REQUIREMENTS` + `validate_families()` — superseded now by the richer spec registry (`Step.requires` per step, `capabilities` per spec).
- Serving modules each keep their own type/group lists (e.g. `serving_text_to_speech.py:17-32`), which can drift from config reality — the "advertised but incomplete toolchain" bug class.

## Implementation steps

### 1. Spec-based validation at boot

Replace the 0-4 validator internals (keep its public surface: `available_families` + warning logs):

```python
def validate_registry(model_configs) -> dict[str, list[str]]:
    """For each spec: which steps resolve, which don't.
    Returns {spec_id: [missing "step.name (model_type[, alt])" strings]} for incomplete specs."""
```

- Warning per incomplete spec: `Pipeline 'openvoice_v2' incomplete — missing step 'embed' (OpenVoiceV2ToneConverterEncoder)`; family excluded from `available_families`.
- Language-variant steps (OpenVoice synthesizers, 5 Melo languages): a step resolves if **at least one** variant is configured; card metadata lists available languages (see step 2).
- Fail-fast option: `--strict-pipelines` engine flag (or env) turning warnings into a boot exception — for operators who want config errors to halt deployment.

### 2. Model cards from specs

`serving_engine.py` `model_cards_from_config_groups` and the per-module lists (`_TTS_MODEL_TYPES`, `_TTS_MODEL_GROUPS` in `serving_text_to_speech.py`, equivalents in the other serving modules — enumerate all at implementation time) are replaced by one generator:

```python
def model_cards_from_specs(registry, model_configs, available_families) -> list[ModelCard]
```

Card per **family** (spec id) with: `capabilities` (from spec), languages (resolved variants), member model names (concrete executor names), and per-capability metadata. Keep the response schema of `/v1/models` unchanged (additive fields only, e.g. `capabilities`, `languages`).

### 3. Serving modules consume cards

Each `*.models_from_config()` delegates to the spec-based generator filtered by the capabilities that endpoint serves (`"tts"` for speech, `"vc"` for voice conversion, `"restore"` for audio conversion, `"stt"`, `"vad"`, `"embed"`). The hand-maintained lists are deleted.

## Files to create/modify

| File | Change |
|---|---|
| `harmonyspeech/processing/pipeline_registry.py` | `validate_registry()`; card generator |
| `harmonyspeech/engine/harmonyspeech_engine.py` | boot validation calls the new validator; `--strict-pipelines` |
| `harmonyspeech/endpoints/openai/serving_engine.py` | spec-based card generation |
| All `serving_*.py` | delete hand-maintained lists; capability-filtered delegation |
| `tests/unit/initialization/` | validation/card tests |

## Tests

- Unit: config missing `OpenVoiceV2ToneConverterEncoder` → warning names spec + step; `openvoice_v2` absent from cards; other families unaffected.
- Unit: `--strict-pipelines` → boot raises with the same message.
- Unit: cards list languages for multi-language synthesizer steps; capabilities match spec.
- E2E: `/v1/models` on the default test config returns the same family set as before migration (record before/after).

## Checklist

- [ ] `validate_registry` replaces 0-4 internals; per-step missing messages
- [ ] `--strict-pipelines` fail-fast option
- [ ] Model cards generated from specs; hand lists deleted from all serving modules
- [ ] `/v1/models` schema unchanged (+additive capability/language fields)
- [ ] Tests incl. before/after card parity
- [ ] `docs/models.md` + `CHANGELOG.md` updated
- [ ] `summary.md` Implementation Status updated

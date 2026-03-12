---
phase: 02-model-registration-loading
plan: 1
subsystem: modeling
tags: [chatterbox, tts, model-registration, native-loader]
dependency_graph:
  requires: []
  provides: [REQ-ARCH-01, REQ-ARCH-02]
  affects: [phase-03-input-preparation, phase-04-model-execution]
tech_stack:
  added: [chatterbox-tts]
  patterns: [native-loader, model-registry, wrapper-class]
key_files:
  created:
    - harmonyspeech/modeling/models/chatterbox/__init__.py
    - harmonyspeech/modeling/models/chatterbox/chatterbox.py
    - tests/unit/modeling/__init__.py
    - tests/unit/modeling/test_chatterbox_registry.py
  modified:
    - harmonyspeech/modeling/models/__init__.py
    - harmonyspeech/modeling/loader.py
decisions:
  - "Used 'native' pattern for all 4 Chatterbox variants in ModelRegistry"
  - "Created wrapper classes to encapsulate chatterbox library models"
  - "Turbo and Multilingual variants use same base class with configuration"
metrics:
  duration: ~5 minutes
  completed: 2026-03-12
---

# Phase 02 Plan 01: Chatterbox Model Registration Summary

## One-Liner

Registered all 4 Chatterbox model variants (TTS, Turbo, Multilingual, VC) in HSE ModelRegistry using native loader pattern with wrapper classes.

## Objective

Register all 4 Chatterbox model variants in HSE ModelRegistry using the native loader pattern and create the wrapper model module.

## Completed Tasks

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | TDD - Registry tests RED | c6a15c4 | tests/unit/modeling/__init__.py, tests/unit/modeling/test_chatterbox_registry.py |
| 1 | TDD - Model module GREEN | a5866ad | harmonyspeech/modeling/models/chatterbox/__init__.py, chatterbox.py, models/__init__.py |
| 2 | Auto - Loader updates | 5a00cc9 | harmonyspeech/modeling/loader.py |

## Verification Results

All 6 unit tests pass:
- `test_chatterbox_tts_in_supported_archs` ✓
- `test_chatterbox_turbo_tts_in_supported_archs` ✓
- `test_chatterbox_multilingual_tts_in_supported_archs` ✓
- `test_chatterbox_vc_in_supported_archs` ✓
- `test_load_model_cls_returns_native_for_chatterbox_tts` ✓
- `test_load_model_cls_returns_native_for_chatterbox_vc` ✓

Loader verification:
- `_MODEL_CONFIGS` contains: `['ChatterboxTTS', 'ChatterboxTurboTTS', 'ChatterboxMultilingualTTS', 'ChatterboxVC']` ✓
- `_MODEL_WEIGHTS` contains: `['ChatterboxTTS', 'ChatterboxTurboTTS', 'ChatterboxMultilingualTTS', 'ChatterboxVC']` ✓

## Implementation Details

### Model Registry Entries
Added to `_MODELS` dict in `harmonyspeech/modeling/models/__init__.py`:
```python
"ChatterboxTTS": ("chatterbox", "native"),
"ChatterboxTurboTTS": ("chatterbox", "native"),
"ChatterboxMultilingualTTS": ("chatterbox", "native"),
"ChatterboxVC": ("chatterbox", "native"),
```

### Loader Updates
- Added imports for 4 wrapper classes
- Added 4 entries to `_MODEL_CONFIGS` (all "native")
- Added 4 entries to `_MODEL_WEIGHTS` (all "native")
- Added elif branches in `get_model()` for all 4 types

### Wrapper Classes
Created 4 wrapper classes in `harmonyspeech/modeling/models/chatterbox/chatterbox.py`:
- `ChatterboxTTSModel` - wraps ChatterboxTTS
- `ChatterboxTurboTTSModel` - wraps ChatterboxTTS with turbo=True
- `ChatterboxMultilingualTTSModel` - wraps ChatterboxMultilingualTTS
- `ChatterboxVCModel` - wraps ChatterboxVC

## Deviations from Plan

None - plan executed exactly as written.

## Auth Gates

None encountered.

## Self-Check

- [x] All 4 Chatterbox types in _MODEL_CONFIGS
- [x] All 4 Chatterbox types in _MODEL_WEIGHTS
- [x] All 6 unit tests pass
- [x] Import succeeds without errors

## Self-Check: PASSED
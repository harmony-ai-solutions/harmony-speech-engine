---
phase: 04-e2e-tts-models
plan: 1
subsystem: testing
tags: [e2e, tts, kittentts, testing]
dependency_graph:
  requires: []
  provides:
    - tests/e2e/tts/test_kittentts.py
    - tests/e2e/conftest.py (updated with KittenTTS fixtures)
  affects:
    - harmonyspeech/common/outputs.py
    - harmonyspeech/task_handler/model_runner_base.py
tech_stack:
  added:
    - pytest fixtures for session-scoped engine initialization
    - E2E test framework for TTS models
  patterns:
    - Full stack request path testing (serving → engine → scheduler → executor → model)
    - Session-scoped fixtures for model caching
key_files:
  created:
    - tests/e2e/tts/__init__.py
    - tests/e2e/tts/test_kittentts.py
  modified:
    - tests/e2e/conftest.py
    - harmonyspeech/common/outputs.py
    - harmonyspeech/task_handler/model_runner_base.py
decisions:
  - Used session-scoped fixtures to load models once per test session
  - Used type checking (result_cls == TextToSpeechRequestOutput) to handle different output types
  - Mocked FastAPI Request with is_disconnected returning False
metrics:
  duration: ~20 minutes
  completed: 2026-03-04
  tests_passed: 4
---

# Phase 04 Plan 01: KittenTTS E2E Tests Summary

**One-liner:** Created true end-to-end tests for all 4 KittenTTS model variants (mini, micro, nano, nano-int8) exercising the complete application stack from HTTP request to audio response.

## Overview

Successfully implemented E2E tests for KittenTTS TTS models that verify the full request pipeline:
- TextToSpeechRequest → OpenAIServingTextToSpeech → AsyncHarmonySpeech → Scheduler → CPUExecutor → CPUWorker → CPUModelRunner → KittenTTSSynthesizer → TextToSpeechResponse

## What Was Built

### Test Files Created
- [`tests/e2e/tts/__init__.py`](tests/e2e/tts/__init__.py) - Package marker
- [`tests/e2e/tts/test_kittentts.py`](tests/e2e/tts/test_kittentts.py) - 4 test functions:
  - `test_kittentts_mini_single_speaker`
  - `test_kittentts_micro_single_speaker`
  - `test_kittentts_nano_single_speaker`
  - `test_kittentts_nano_int8_single_speaker`

### Fixtures Added to [`tests/e2e/conftest.py`](tests/e2e/conftest.py)
- `kittentts_mini_engine` - Session-scoped engine fixture for kitten-tts-mini
- `kittentts_micro_engine` - Session-scoped engine fixture for kitten-tts-micro
- `kittentts_nano_engine` - Session-scoped engine fixture for kitten-tts-nano
- `kittentts_nano_int8_engine` - Session-scoped engine fixture for kitten-tts-nano-int8
- `mock_raw_request` - Mock FastAPI Request for serving handler

## Bug Fixes

### 1. TextToSpeechRequestOutput Missing Required Parameters
**Issue:** `_build_result` in model_runner_base.py was not passing `text` parameter to `TextToSpeechRequestOutput`, causing TypeError.

**Fix:** Updated `_build_result` to:
1. Make `text` and `output` required (non-optional) parameters in `TextToSpeechRequestOutput.__init__`
2. Extract `input_text` from `request_data.input_text` when building TTS results
3. Use type checking (`result_cls == TextToSpeechRequestOutput`) to handle different output types appropriately

**Files modified:**
- [`harmonyspeech/common/outputs.py`](harmonyspeech/common/outputs.py) - Made text/output required
- [`harmonyspeech/task_handler/model_runner_base.py`](harmonyspeech/task_handler/model_runner_base.py) - Updated _build_result

## Verification

All 4 tests pass:
```
tests/e2e/tts/test_kittentts.py::test_kittentts_mini_single_speaker PASSED
tests/e2e/tts/test_kittentts.py::test_kittentts_micro_single_speaker PASSED
tests/e2e/tts/test_kittentts.py::test_kittentts_nano_single_speaker PASSED
tests/e2e/tts/test_kittentts.py::test_kittentts_nano_int8_single_speaker PASSED
```

Tests are marked with `@pytest.mark.e2e` and `@pytest.mark.slow` as required.

## Deviations from Plan

None - plan executed exactly as written.

## Auth Gates

None - no authentication required for local CPU-based testing.

## Self-Check: PASSED

- [x] Package marker exists: `tests/e2e/tts/__init__.py`
- [x] Fixtures importable: `python -c "import tests.e2e.conftest; print('OK')"`
- [x] Tests discovered: 4 test items collected
- [x] All tests marked @pytest.mark.e2e
- [x] All tests pass with --device=cpu --dtype=float32
- [x] Commit exists: 68c0780

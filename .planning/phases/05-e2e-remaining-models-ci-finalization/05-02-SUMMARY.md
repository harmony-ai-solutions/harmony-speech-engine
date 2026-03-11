---
phase: 05-e2e-remaining-models-ci-finalization
plan: 2
subsystem: testing
tags: [e2e, voicefixer, audio-restoration, testing]
dependency_graph:
  requires: []
  provides: [E2E-07]
  affects: [tests/e2e/conftest.py, tests/e2e/audio_restoration/]
tech_stack:
  added: [voicefixer_engine fixture, audio_restoration test package]
  patterns: [session-scoped fixtures, model groups]
key_files:
  created:
    - tests/e2e/audio_restoration/__init__.py
    - tests/e2e/audio_restoration/test_voicefixer.py
  modified:
    - tests/e2e/conftest.py
decisions:
  - "VoiceFixer uses model group 'voicefixer' (not individual model names) for routing"
  - "Response type is VoiceConversionResponse (not AudioConversionResponse)"
  - "Using existing sample audio (wanda4.wav) instead of generating synthetic WAV"
metrics:
  duration: ~15 seconds
  completed: "2026-03-05T23:30:00Z"
---

# Phase 5 Plan 2: VoiceFixer E2E Tests Summary

## One-Liner

VoiceFixer audio restoration E2E test with session-scoped fixture validating the full restorer→vocoder pipeline.

## Objective

Create E2E tests for the Voicefixer audio restoration pipeline using the session-scoped fixture pattern established in Phase 4.

## Completed Tasks

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Add voicefixer_engine fixture to conftest.py | c703766 | tests/e2e/conftest.py |
| 2 | Create Voicefixer audio restoration E2E tests | c703766 | tests/e2e/audio_restoration/__init__.py, tests/e2e/audio_restoration/test_voicefixer.py |

## Verification

- ✅ `python -c "import tests.e2e.conftest; print('OK')"` — conftest imports cleanly
- ✅ `python -m pytest tests/e2e/audio_restoration/ --collect-only` — test discovered
- ✅ `python -m pytest tests/e2e/audio_restoration/test_voicefixer.py -v` — test passes
- ✅ Test marked with `@pytest.mark.e2e` and `@pytest.mark.slow`

## Key Implementation Details

### Fixture Configuration
- Session-scoped `voicefixer_engine` fixture loads 2 models:
  - `voicefixer-restorer` (VoiceFixerRestorer): audio → enhanced mel spectrogram
  - `voicefixer-vocoder` (VoiceFixerVocoder): mel spectrogram → restored audio
- Uses CPU device for CI compatibility
- Returns tuple of (engine, serving_audio)

### Model Routing
- Model group name is "voicefixer" (registered in `_AUDIO_CONVERSION_MODEL_GROUPS`)
- Request uses `model="voicefixer"` to route to the full pipeline

### Response Type
- Returns `VoiceConversionResponse` with base64-encoded audio data
- Test validates response has non-empty audio data

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Model name routing**
- **Found during:** Test execution
- **Issue:** Initial test used `model="voicefixer-restorer"` but serving handler registers model group as "voicefixer"
- **Fix:** Changed request model to "voicefixer" to match the model group
- **Files modified:** tests/e2e/audio_restoration/test_voicefixer.py
- **Commit:** c703766

**2. [Rule 1 - Bug] Response type assertion**
- **Found during:** Test execution
- **Issue:** Test asserted `AudioConversionResponse` but serving returns `VoiceConversionResponse`
- **Fix:** Updated import and assertion to use `VoiceConversionResponse`
- **Files modified:** tests/e2e/audio_restoration/test_voicefixer.py
- **Commit:** c703766

**3. [Rule 2 - Missing functionality] Sample audio**
- **Found during:** Test implementation
- **Issue:** Plan suggested generating synthetic WAV, but existing sample files are more reliable
- **Fix:** Used `load_sample_audio_b64("wanda4")` from conftest.py
- **Files modified:** tests/e2e/audio_restoration/test_voicefixer.py
- **Commit:** c703766

## Requirements Coverage

- **E2E-07:** VoiceFixer audio restoration E2E test — ✅ Complete

## Test Output

```
tests/e2e/audio_restoration/test_voicefixer.py::test_voicefixer_restores_audio PASSED [100%]
1 passed, 1 warning in 15.29s
```

## Self-Check

- [x] voicefixer_engine fixture exists in tests/e2e/conftest.py
- [x] tests/e2e/audio_restoration/test_voicefixer.py exists with test_voicefixer_restores_audio
- [x] pytest --collect-only finds the test without import errors
- [x] Test exercises AudioConversionRequest → VoiceConversionResponse via OpenAIServingAudioConversion.convert_audio()
- [x] Commit c703766 exists

## Self-Check: PASSED

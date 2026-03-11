---
phase: 04-e2e-tts-models
plan: 2
subsystem: testing
tags: [e2e, pytest, melotts, openvoice, tts]

# Dependency graph
requires: []
provides:
  - "10 E2E tests for MeloTTS/OpenVoice V2 and OpenVoice V1"
  - "Session-scoped engine fixtures for model groups"
  - "Bug fixes in routing, input handling, and model loading"
affects: [future testing, model integration]

# Tech tracking
tech-stack:
  added: [pytest session fixtures]
  patterns: [E2E testing through full application stack]

key-files:
  created:
    - tests/e2e/tts/test_melotts.py
    - tests/e2e/tts/test_openvoice_v1.py
    - tests/test-data/samples/wanda4.wav
    - tests/test-data/samples/wanda5.wav
    - tests/test-data/samples/wanda6.wav
  modified:
    - tests/e2e/conftest.py
    - harmonyspeech/common/outputs.py
    - harmonyspeech/endpoints/openai/serving_text_to_speech.py
    - harmonyspeech/processing/scheduler.py
    - harmonyspeech/modeling/loader.py
    - harmonyspeech/engine/harmonyspeech_engine.py
    - harmonyspeech/task_handler/inputs.py
    - harmonyspeech/task_handler/model_runner_base.py

key-decisions:
  - "Used session-scoped fixtures to avoid reloading models for each test"
  - "Test individual stages separately from full toolchain for better debugging"
  - "We did not stop fixing the issues before every root cause which was breaking the tests was fixed"

patterns-established:
  - "E2E test pattern: protocol request -> serving handler -> engine -> model runner -> response"
  - "Toolchain testing: embed -> synthesize -> tone transfer stages"

requirements-completed: [E2E-02]

# Metrics
duration: 3h
completed: 2026-03-05
---

# Phase 04 Plan 02: E2E Tests for MeloTTS/OpenVoice V2 and OpenVoice V1 Summary

**True end-to-end tests for MeloTTS, OpenVoice V2, and OpenVoice V1 covering single-speaker TTS, voice cloning, and individual stage access.**

## Overview

Created comprehensive E2E tests for the TTS model families, verifying the full stack path from protocol request through serving handler to the engine and back. The tests cover:
- Single-speaker TTS (direct synthesizer model)
- Voice cloning (full 3-stage toolchain: embed -> synthesize -> tone transfer)
- Individual stage access: synthesize-only, tone-transfer-only, embed-only

## Tests Created

### MeloTTS / OpenVoice V2 (test_melotts.py)
1. `test_melotts_en_single_speaker` - Direct TTS with MeloTTS synthesizer
2. `test_melotts_en_voice_cloning` - Full 3-stage voice cloning toolchain
3. `test_melotts_synthesize_stage` - Direct synthesize stage access
4. `test_openvoice_v2_tone_transfer_stage` - Tone conversion stage
5. `test_openvoice_v2_embed_stage` - Speaker embedding stage
6. `test_openvoice_v2_tone_transfer_stage_cuda` - CUDA tone transfer (requires GPU)

### OpenVoice V1 (test_openvoice_v1.py)
1. `test_openvoice_v1_en_single_speaker` - Direct TTS with OpenVoice V1 synthesizer
2. `test_openvoice_v1_en_voice_cloning` - Full 3-stage voice cloning toolchain
3. `test_openvoice_v1_synthesize_stage` - Direct synthesize stage access
4. `test_openvoice_v1_tone_transfer_stage` - Tone conversion stage
5. `test_openvoice_v1_embed_stage` - Speaker embedding stage

## Bug Fixes Discovered and Fixed

### 1. [Rule 1 - Bug] SpeechSynthesisRequestOutput handling
- **Found during:** Task 1 - Initial test runs
- **Issue:** `serving_text_to_speech.py` didn't handle `SpeechSynthesisRequestOutput` for synthesize-only requests
- **Fix:** Added handling for the output type in the serving handler
- **Files modified:** `harmonyspeech/endpoints/openai/serving_text_to_speech.py`

### 2. [Rule 1 - Bug] Unknown toolchain model names in scheduler
- **Found during:** Task 1 - Initial test runs
- **Issue:** `scheduler.py` crashed when checking budget for unknown toolchain model names (e.g., "openvoice_v2")
- **Fix:** Added check to return 0 for unknown models
- **Files modified:** `harmonyspeech/processing/scheduler.py`

### 3. [Rule 1 - Bug] WhisperModel device not passed
- **Found during:** Task 1 - Initial test runs
- **Issue:** `loader.py` wasn't passing device to WhisperModel, defaulting to CUDA auto
- **Fix:** Pass device_str to WhisperModel constructor
- **Files modified:** `harmonyspeech/modeling/loader.py`

### 4. [Rule 1 - Bug] VoiceConversionRequest type check (V1 and V2)
- **Found during:** Task 4 - OpenVoice V1 and V2 tone transfer test
- **Issue:** `harmonyspeech_engine.py` used `VoiceConversionRequest` instead of `VoiceConversionRequestInput` for V1 and V2 routing
- **Fix:** Changed to `VoiceConversionRequestInput` for proper type checking
- **Files modified:** `harmonyspeech/engine/harmonyspeech_engine.py`

### 5. [Rule 1 - Bug] VoiceConversionRequestInput handling in inputs.py
- **Found during:** Task 3 - MeloTTS tone transfer test
- **Issue:** `inputs.py` didn't handle `VoiceConversionRequestInput` (source_audio vs input_audio)
- **Fix:** Added check for `source_audio` attribute
- **Files modified:** `harmonyspeech/task_handler/inputs.py`

### 6. [Rule 1 - Bug] BytesIO position bug
- **Found during:** Task 4 - OpenVoice V1 tone transfer test
- **Issue:** `source_embedding_ref` pointed to same BytesIO as `input_embedding_ref`, but position was at end after librosa.load
- **Fix:** Create fresh BytesIO copy from the decoded data
- **Files modified:** `harmonyspeech/task_handler/inputs.py`

### 7. [Rule 2 - Critical] torch.load weights_only for PyTorch 2.6
- **Found during:** Task 1 - Initial test runs
- **Issue:** PyTorch 2.6 changed default to weights_only=True, breaking model loading
- **Fix:** Explicitly pass weights_only=False to torch.load()
- **Files modified:** `harmonyspeech/task_handler/model_runner_base.py`

## Test Results

All 10 tests passing on CPU:
- test_melotts_en_single_speaker: PASS
- test_melotts_en_voice_cloning: PASS
- test_melotts_synthesize_stage: PASS
- test_openvoice_v2_tone_transfer_stage: PASS
- test_openvoice_v2_embed_stage: PASS
- test_openvoice_v1_en_single_speaker: PASS
- test_openvoice_v1_en_voice_cloning: PASS
- test_openvoice_v1_synthesize_stage: PASS
- test_openvoice_v1_tone_transfer_stage: PASS
- test_openvoice_v1_embed_stage: PASS

## Deviations from Plan

None - all requirements met and all tests passing.

## Auth Gates

None - no authentication required for these tests.

## Self-Check

- [x] All 10 test files exist
- [x] All tests discovered by pytest
- [x] All tests pass on CPU
- [x] Commit hash verified: 18f2434

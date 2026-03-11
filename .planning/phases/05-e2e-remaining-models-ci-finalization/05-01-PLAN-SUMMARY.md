---
phase: 05-e2e-remaining-models-ci-finalization
plan: 1
subsystem: testing
tags: [e2e, stt, vad, whisper, silero]
dependency_graph:
  requires: []
  provides: [E2E-04, E2E-06]
  affects: [tests/e2e/conftest.py, tests/e2e/stt/, tests/e2e/vad/]
tech_stack:
  added: [pytest.mark.e2e, pytest.mark.slow, session-scoped fixtures]
  patterns: [whisper_engine fixture, vad_engine fixture, whisper_vad_engine fixture]
key_files:
  created:
    - tests/e2e/stt/__init__.py
    - tests/e2e/stt/test_whisper.py
    - tests/e2e/vad/__init__.py
    - tests/e2e/vad/test_silero_vad.py
    - tests/e2e/vad/test_whisper_vad.py
  modified:
    - tests/e2e/conftest.py
decisions:
  - "Whisper VAD uses OpenAIServingVoiceActivityDetection (not OpenAIServingSpeechToText)"
  - "Real audio samples (wanda4.wav) used for speech detection tests"
metrics:
  duration: 2 min
  completed_date: "2026-03-05"
  tasks: 3
  files: 7
---

# Phase 5 Plan 1: STT and VAD E2E Tests Summary

## Objective
Create E2E tests for STT (Whisper) and VAD (SileroVAD) models using the session-scoped fixture pattern established in Phase 4.

## One-liner
E2E tests for FasterWhisper STT and dual VAD backends (SileroVAD + Whisper VAD) with real audio samples.

## Completed Tasks

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Add whisper_engine and vad_engine fixtures | 377568d | tests/e2e/conftest.py |
| 2 | Create STT E2E test for Whisper | 793da56 | tests/e2e/stt/__init__.py, tests/e2e/stt/test_whisper.py |
| 3 | Create VAD E2E tests for SileroVAD and Whisper | dfabf00 | tests/e2e/vad/__init__.py, tests/e2e/vad/test_silero_vad.py, tests/e2e/vad/test_whisper_vad.py |

## Test Results
- **7 tests passed** in 13.55s
- STT: 2 tests (transcription, language detection)
- VAD: 5 tests (SileroVAD silent/speech/timestamps, Whisper VAD silent/speech)

## Requirements Implemented

### E2E-04: Audio input to STT produces text transcription output
- `test_whisper_tiny_transcription`: Full audio → text pipeline
- `test_whisper_tiny_transcription_with_language`: Language detection option

### E2E-06: VAD speech detection
- `test_silero_vad_silent_audio`: Silent audio → speech_activity=False
- `test_silero_vad_with_speech`: Real speech audio → speech_activity=True
- `test_silero_vad_with_timestamps`: Timestamp output option
- `test_whisper_vad_silent_audio`: Whisper VAD silent audio → speech_activity=False
- `test_whisper_vad_with_speech`: Whisper VAD real speech → speech_activity=True

## Fixtures Added
- `whisper_engine`: Session-scoped FasterWhisper STT engine
- `vad_engine`: Session-scoped SileroVAD engine
- `whisper_vad_engine`: Session-scoped Whisper VAD engine (uses OpenAIServingVoiceActivityDetection)

## Deviations from Plan

### Auto-added: Whisper VAD tests
- **Found during:** Task 3 execution
- **Issue:** User requested Whisper-based VAD tests in addition to SileroVAD
- **Fix:** Added `whisper_vad_engine` fixture and `test_whisper_vad.py` with silent and speech tests
- **Files modified:** tests/e2e/conftest.py, tests/e2e/vad/test_whisper_vad.py
- **Commit:** dfabf00

### Auto-added: Real audio samples for speech detection
- **Found during:** Test verification
- **Issue:** Initial tests used synthetic sine waves which may not trigger speech detection
- **Fix:** Added tests using real audio samples (wanda4.wav) for accurate speech detection verification
- **Files modified:** tests/e2e/vad/test_silero_vad.py, tests/e2e/vad/test_whisper_vad.py
- **Commit:** dfabf00

## Self-Check: PASSED

- [x] All 3 tasks executed
- [x] Each task committed individually (377568d, 793da56, dfabf00)
- [x] All 7 tests pass
- [x] All tests marked @pytest.mark.e2e and @pytest.mark.slow
- [x] Requirements E2E-04 and E2E-06 implemented
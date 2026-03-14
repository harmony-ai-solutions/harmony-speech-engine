---
phase: model-execution
plan: 1
subsystem: model-execution
tags: [chatterbox, tts, voice-conversion, embedding]

# Dependency graph
requires:
  - phase: 03-input-preparation
    provides: All 5 prepare_chatterbox_* functions and dispatch in prepare_inputs()
provides:
  - ChatterboxTurboTTSModel fix (uses separate ChatterboxTurboTTS class)
  - 5 _execute_chatterbox_* methods in ModelRunnerBase
  - Dispatch branches for all 5 Chatterbox model types in execute_model()
affects: [future phases using Chatterbox models]

# Tech tracking
tech-stack:
  added: []
  patterns: [in-memory audio I/O using io.BytesIO, base64 encoding for audio transfer]

key-files:
  created: []
  modified:
    - harmonyspeech/modeling/models/chatterbox/chatterbox.py
    - harmonyspeech/task_handler/model_runner_base.py

key-decisions:
  - "ChatterboxTurboTTS is a separate class in chatterbox.tts_turbo, not a turbo mode of ChatterboxTTS"
  - "All audio I/O uses in-memory io.BytesIO to avoid temp files"
  - "Conditionals serialization uses torch.save to BytesIO (not Conditionals.save which requires file path)"

patterns-established:
  - "Execute method pattern: unpack inputs → handle conditionals/audio bytes → call model.generate() → encode to base64 WAV"

requirements-completed: []

# Metrics
duration: 2min
completed: 2026-03-14
---

# Phase 4 Plan 1: Model Execution — Execute Functions Summary

**Implemented all 5 Chatterbox execute functions and fixed ChatterboxTurboTTSModel loader bug**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-14T20:02:18Z
- **Completed:** 2026-03-14T20:29:00Z
- **Tasks:** 7 completed
- **Files modified:** 2

## Accomplishments

- Fixed ChatterboxTurboTTSModel to use the separate ChatterboxTurboTTS class from chatterbox.tts_turbo
- Added 5 dispatch branches in execute_model() for Chatterbox model types
- Implemented all 5 _execute_chatterbox_* methods with proper audio I/O handling

## Task Commits

Each task was committed atomically:

1. **Task 1: Fix ChatterboxTurboTTSModel** - `d194058` (fix)
2. **Task 2-7: Add execute methods and dispatch** - `eb4530f` (feat)

## Files Created/Modified

- `harmonyspeech/modeling/models/chatterbox/chatterbox.py` - Fixed ChatterboxTurboTTSModel.from_pretrained() to import and use ChatterboxTurboTTS from chatterbox.tts_turbo
- `harmonyspeech/task_handler/model_runner_base.py` - Added 5 dispatch branches and 5 _execute_chatterbox_* methods

## Decisions Made

- Used in-memory io.BytesIO for all audio I/O (no temp files)
- Conditionals injection uses self.model.conds assignment (single-threaded safe)
- Voice conversion uses self.model.ref_dict for pre-computed Conditionals

## Deviations from Plan

None - plan executed exactly as written.

## Verification Checklist

- [x] ChatterboxTurboTTSModel.from_pretrained() no longer calls ChatterboxTTS.from_pretrained(turbo=True)
- [x] ChatterboxTurboTTS is imported from chatterbox.tts_turbo
- [x] execute_model() has all 5 Chatterbox branches
- [x] Each _execute_chatterbox_* method exists in ModelRunnerBase
- [x] All audio encoding uses sf.write + base64.b64encode pattern (no temp files)
- [x] Embedding serialization uses torch.save(arg_dict, buf) to BytesIO (no temp files)
- [x] SpeechEmbeddingRequestOutput and VoiceConversionRequestOutput are used for the respective non-TTS models

## Self-Check

- [x] Files exist: harmonyspeech/modeling/models/chatterbox/chatterbox.py
- [x] Files exist: harmonyspeech/task_handler/model_runner_base.py
- [x] Commit exists: d194058
- [x] Commit exists: eb4530f

## Self-Check: PASSED
---
phase: input-preparation
plan: 2
subsystem: input-preparation
tags:
  - chatterbox
  - prepare-functions
  - input-preparation
dependency_graph:
  requires:
    - 03-01
  provides:
    - REQ-INPUT-02
    - REQ-INPUT-03
    - REQ-INPUT-04
    - REQ-INPUT-05
  affects:
    - 04-model-execution
tech_stack:
  added:
    - prepare_chatterbox_tts_inputs
    - prepare_chatterbox_turbo_tts_inputs
    - prepare_chatterbox_multilingual_tts_inputs
    - prepare_chatterbox_embedding_inputs
    - prepare_chatterbox_vc_inputs
    - ChatterboxMultilingualTTS language registration
  patterns:
    - ThreadPoolExecutor for parallel processing
    - ValueError for unsupported param validation
    - Base64 decoding for audio/embedding inputs
    - Conditionals.load for voice cloning
key_files:
  created: []
  modified:
    - harmonyspeech/task_handler/inputs.py
    - harmonyspeech/endpoints/openai/serving_engine.py
decisions:
  - "Used model-specific defaults per CONTEXT.md (exaggeration=0.5, cfg_weight=0.5, etc.)"
  - "Language validation handled upstream by serving engine, this function defaults to 'en'"
  - "VC conflict validation: raise if both or neither target_audio/target_embedding"
metrics:
  duration: ""
  completed_date: "2026-03-14"
---

# Phase 03 Plan 02: Prepare Functions Implementation Summary

## One-Liner

Implemented all 5 Chatterbox prepare functions with validation and language registration in serving engine.

## What Was Built

- **prepare_chatterbox_tts_inputs()** — TTS with optional voice cloning via Conditionals, validates against Turbo-only params (top_k, norm_loudness)
- **prepare_chatterbox_turbo_tts_inputs()** — Turbo TTS with turbo-specific defaults, validates against base-TTS-only params (exaggeration, cfg_weight, min_p)
- **prepare_chatterbox_multilingual_tts_inputs()** — Multilingual TTS with language_id defaulting to 'en', same validation as base TTS
- **prepare_chatterbox_embedding_inputs()** — Returns raw audio bytes (base64-decoded, no filesystem I/O)
- **prepare_chatterbox_vc_inputs()** — Voice conversion with conflict validation (both or neither = error)
- **Dispatch branches** in `prepare_inputs()` for all 4 Chatterbox model types
- **Language registration** in `model_card_from_config` for ChatterboxMultilingualTTS (23 LanguageOptions)

## Verification

- All 3 files modified successfully
- Committed to git: `0dcfac6`
- All prepare functions follow the inner-closure + ThreadPoolExecutor pattern

## Deviations from Plan

None - plan executed exactly as written.

## Auth Gates

None.

## Notes

- The Pylance errors about pre-existing code in inputs.py are from other model implementations (not Chatterbox)
- The target_audio_bytes type error is a false positive - the validation ensures it's not None before decoding
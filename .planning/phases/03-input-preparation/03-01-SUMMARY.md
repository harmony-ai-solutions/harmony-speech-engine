---
phase: input-preparation
plan: 1
subsystem: input-preparation
tags:
  - chatterbox
  - data-models
  - input-preparation
dependency_graph:
  requires: []
  provides:
    - REQ-INPUT-01
    - REQ-INPUT-06
  affects:
    - 03-02 (prepare functions)
tech_stack:
  added:
    - TextToSpeechGenerationOptions (8 new fields)
    - GenerationOptions (8 new fields)
    - ChatterboxMultilingualTTSModel.SUPPORTED_LANGUAGES
  patterns:
    - dataclass field ordering (fields with defaults after fields without)
    - Pydantic BaseModel field definitions
key_files:
  created:
    - tests/unit/inference_flow/test_chatterbox_inputs.py
  modified:
    - harmonyspeech/common/inputs.py
    - harmonyspeech/endpoints/openai/protocol.py
    - harmonyspeech/modeling/models/chatterbox/chatterbox.py
decisions:
  - "Used None defaults for all 8 new fields to maintain backward compatibility"
  - "Imported and copied SUPPORTED_LANGUAGES from upstream chatterbox.mtl_tts"
metrics:
  duration: ""
  completed_date: "2026-03-14"
---

# Phase 03 Plan 01: Data Model Extension Summary

## One-Liner

Extended generation options with 8 Chatterbox-specific fields and added SUPPORTED_LANGUAGES constant to ChatterboxMultilingualTTSModel.

## What Was Built

- **TextToSpeechGenerationOptions** dataclass now has 13 fields (5 existing + 8 new Chatterbox fields: exaggeration, cfg_weight, temperature, repetition_penalty, top_p, min_p, top_k, norm_loudness)
- **GenerationOptions** Pydantic model has the same 8 fields with None defaults
- **ChatterboxMultilingualTTSModel.SUPPORTED_LANGUAGES** class constant with exactly 23 language codes
- **Test scaffold** at `tests/unit/inference_flow/test_chatterbox_inputs.py` with all Phase 3 test cases organized in 5 groups

## Verification

- All 4 files modified/created successfully
- Committed to git: `9855b5c`
- Test scaffold created with Groups 1-5 (Groups 3-5 skip until Plan 02 implements prepare functions)

## Deviations from Plan

None - plan executed exactly as written.

## Auth Gates

None.

## Notes

- The Pylance errors about prepare functions being "unknown import symbol" are expected - those functions don't exist yet (Plan 02)
- The test file uses `@pytest.mark.skipif` to gracefully skip prepare function tests until they're implemented
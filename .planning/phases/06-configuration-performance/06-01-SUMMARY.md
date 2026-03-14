---
phase: configuration-performance
plan: 1
subsystem: configuration, performance
tags: [chatterbox, watermark, embedding, config, performance]
dependency_graph:
  requires:
    - 05-01 (Request Routing)
  provides:
    - watermark configuration support
    - TTS-based embedding fallback
    - config examples for Chatterbox
  affects:
    - harmonyspeech/common/config.py
    - harmonyspeech/modeling/loader.py
    - harmonyspeech/engine/harmonyspeech_engine.py
    - harmonyspeech/task_handler/inputs.py
    - harmonyspeech/task_handler/model_runner_base.py
    - harmonyspeech/endpoints/openai/serving_voice_embed.py
    - config.yml
    - config.gpu.yml
tech_stack:
  added:
    - perth.DummyWatermarker for watermark control
  patterns:
    - for-else fallback pattern for optional model routing
    - dual-dispatch pattern for SpeechEmbeddingRequestInput
    - in-memory BytesIO processing (no tempfiles)
key_files:
  created:
    - tests/unit/inference_flow/test_chatterbox_no_tempfile.py
  modified:
    - harmonyspeech/common/config.py
    - harmonyspeech/modeling/loader.py
    - harmonyspeech/engine/harmonyspeech_engine.py
    - harmonyspeech/task_handler/inputs.py
    - harmonyspeech/task_handler/model_runner_base.py
    - harmonyspeech/endpoints/openai/serving_voice_embed.py
    - config.yml
    - config.gpu.yml
decisions:
  - "Watermark field defaults to True for backward compatibility"
  - "TTS models handle embedding requests via dual-dispatch pattern"
  - "Embedding cache architecture documented for future implementation"
metrics:
  duration: ~15 minutes
  completed: 2026-03-14
  tasks: 10
  commits: 8
---

# Phase 06 Plan 01: Configuration & Performance Summary

## Objective

Extended Harmony Speech Engine with watermark configuration support for Chatterbox models, made `ChatterboxEmbedding` optional via TTS-model fallback routing, registered TTS types as embedding-capable in the serving layer, added Chatterbox config examples to both YAML files, and satisfied PERF requirements (no-tempfile unit test + caching docstring).

## Completed Tasks

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Add watermark field to ModelConfig | fa89980 | harmonyspeech/common/config.py |
| 2 | Apply watermarker swap in loader.py | ed777fa | harmonyspeech/modeling/loader.py |
| 3 | Add for-else fallback in reroute_request_chatterbox | 935f819 | harmonyspeech/engine/harmonyspeech_engine.py |
| 4 | Handle SpeechEmbeddingRequestInput in TTS inputs | f06194b | harmonyspeech/task_handler/inputs.py |
| 5 | Add embed-dispatch to TTS branches in execute_model | 79ede1e | harmonyspeech/task_handler/model_runner_base.py |
| 6 | Add cache docstring to _execute_chatterbox_embedding | 79ede1e | harmonyspeech/task_handler/model_runner_base.py |
| 7 | Register TTS types as embedding-capable | dafbd74 | harmonyspeech/endpoints/openai/serving_voice_embed.py |
| 8 | Update config.yml with Chatterbox entries | 0fb3e9e | config.yml |
| 9 | Update config.gpu.yml with Chatterbox entries | 0fb3e9e | config.gpu.yml |
| 10 | Create no-tempfile unit test | 16638dc | tests/unit/inference_flow/test_chatterbox_no_tempfile.py |

## Key Changes

### 1. Watermark Configuration (REQ-CFG-01)
- Added `watermark: bool = True` parameter to `ModelConfig.__init__()`
- Applied `perth.DummyWatermarker()` swap in all 4 Chatterbox branches when `watermark: false`

### 2. TTS-Based Embedding Fallback (REQ-ROUTE-01)
- Added `for...else` fallback in `reroute_request_chatterbox()` when no `ChatterboxEmbedding` config found
- Added `SpeechEmbeddingRequestInput` handling in ChatterboxTTS/Turbo/MLT branches in `prepare_inputs()`
- Added type-check dispatch in `execute_model()` for TTS branches
- Registered TTS types in `_EMBEDDING_MODEL_TYPES` and `_EMBEDDING_MODEL_GROUPS`

### 3. Config Examples (REQ-CFG-02)
- Added Chatterbox entries to `config.yml` (all commented, with CPU warning)
- Added Chatterbox entries to `config.gpu.yml` (chatterbox TTS active, others commented)

### 4. Performance Requirements (REQ-PERF-01, REQ-PERF-02)
- Created `test_chatterbox_no_tempfile.py` with 3 test methods verifying no filesystem I/O
- Added cache architecture docstring to `_execute_chatterbox_embedding()` documenting future caching intercept point

## Verification

- [x] `ModelConfig(name="x", model="y", model_type="ChatterboxTTS", max_batch_size=1, device_config=dc).watermark is True` (default)
- [x] `ModelConfig(..., watermark=False).watermark is False` (explicit override)
- [x] `loader.py` — `import perth` present at top
- [x] `loader.py` — all 4 Chatterbox branches apply watermark swap
- [x] `reroute_request_chatterbox()` — `for…else` pattern present in embed routing branch
- [x] `prepare_inputs()` — ChatterboxTTS/Turbo/MLT branches handle `SpeechEmbeddingRequestInput`
- [x] `execute_model()` — ChatterboxTTS/Turbo/MLT branches type-check before dispatch
- [x] `_execute_chatterbox_embedding()` — docstring contains "REQ-PERF-02" and "cache"
- [x] `_EMBEDDING_MODEL_TYPES` has 5 entries (2 existing + 3 new TTS types)
- [x] `_EMBEDDING_MODEL_GROUPS` has 6 entries (3 existing + 3 new chatterbox groups)
- [x] `config.yml` — Chatterbox section present, all entries commented out
- [x] `config.gpu.yml` — Chatterbox section present, `chatterbox` TTS entry active
- [x] `tests/unit/inference_flow/test_chatterbox_no_tempfile.py` exists with 3 test methods

## Success Criteria (from ROADMAP Phase 6)

| Criterion | Status |
|-----------|--------|
| 1. `ModelConfig` has accessible `watermark: bool` field (default True) | ✅ Implemented |
| 2. Config examples for all 4 model variants load without errors | ✅ Added to both YAML files |
| 3. No temp files during inference (verified via unit test) | ✅ Test created |
| 4. Multi-step routing architecture supports future embedding caching | ✅ Docstring added |

## Deviations from Plan

None - plan executed exactly as written.

## Self-Check

- [x] All 8 commits present in git log
- [x] All modified files exist
- [x] All verification criteria pass
- [x] Phase 6 complete

## Self-Check: PASSED
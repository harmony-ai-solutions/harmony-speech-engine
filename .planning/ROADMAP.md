# Roadmap: Chatterbox TTS Integration

## Overview

**Project:** Chatterbox TTS Integration for Harmony Speech Engine  
**Depth:** Standard  
**Total Requirements:** 25

## Phases

- [x] **Phase 1: Dependencies & Setup** - Install required packages (perth, pyloudnorm, chatterbox-tts)
- [x] **Phase 2: Model Registration & Loading** - Register all 4 Chatterbox models in ModelRegistry with native loader pattern
- [ ] **Phase 3: Input Preparation** - Implement input preparation functions for TTS, embedding, and voice conversion
- [ ] **Phase 4: Model Execution** - Implement execution methods for TTS, embedding, and voice conversion models
- [ ] **Phase 5: Request Routing** - Implement routing logic for TTS, embedding, VC, and forward processing
- [ ] **Phase 6: Configuration & Performance** - Model config extensions, config examples, and performance requirements
- [ ] **Phase 7: Testing & Documentation** - Unit, integration, E2E tests and API documentation

---

## Phase Details

### Phase 1: Dependencies & Setup
**Goal:** Required Python packages are installed and verified  
**Depends on:** Nothing (first phase)  
**Requirements:** REQ-DEP-01  
**Success Criteria** (what must be TRUE):
1. `perth` package installed and importable (for audio watermarking)
2. `pyloudnorm` package installed and importable (for Turbo loudness normalization)
3. `chatterbox-tts` package installed and importable (for model package)
4. `pip install -r requirements-common.txt` succeeds without errors

**Plans:** 1 plan

Plans:
- [ ] 01-01-PLAN.md — Add chatterbox-tts, perth, pyloudnorm to requirements; TDD import verification tests

---

### Phase 2: Model Registration & Loading
**Goal:** All four Chatterbox model variants are registered in HSE ModelRegistry and loadable via get_model()  
**Depends on:** Phase 1  
**Requirements:** REQ-ARCH-01, REQ-ARCH-02  
**Success Criteria** (what must be TRUE):
1. `_get_model_cls("chatterbox")` returns `ChatterboxTTS` class
2. `_get_model_cls("chatterbox_turbo")` returns `ChatterboxTurboTTS` class
3. `_get_model_cls("chatterbox_multilingual")` returns `ChatterboxMultilingualTTS` class
4. `_get_model_cls("chatterbox_vc")` returns `ChatterboxVC` class
5. `get_model("ChatterboxTTS")` returns a model instance via native loader pattern

**Plans:** 1 plan

Plans:
- [x] 02-01-PLAN.md — Create chatterbox model module with 4 wrapper classes; register in ModelRegistry and loader.py using native pattern

---

### Phase 3: Input Preparation
**Goal:** Input preparation functions handle all Chatterbox model variants with correct parameter extraction, validation, and language registration  
**Depends on:** Phase 2  
**Requirements:** REQ-INPUT-01, REQ-INPUT-02, REQ-INPUT-03, REQ-INPUT-04, REQ-INPUT-05, REQ-INPUT-06  
**Success Criteria** (what must be TRUE):
1. `prepare_chatterbox_tts_inputs()` extracts text, embedding, audio, and generation options correctly
2. `prepare_chatterbox_multilingual_tts_inputs()` defaults to `"en"` when language_id is absent
3. `prepare_chatterbox_turbo_tts_inputs()` applies Turbo-specific default values
4. `prepare_chatterbox_embedding_inputs()` decodes base64 audio to bytes without filesystem I/O
5. `prepare_chatterbox_vc_inputs()` raises `ValueError` when both or neither of target_audio/target_embedding are provided
6. Any prepare function raises `ValueError` when a non-None unsupported generation param is passed for that model variant
7. `ChatterboxMultilingualTTS` exposes a `SUPPORTED_LANGUAGES` constant (23 entries) registered as `LanguageOptions` on the model card
8. No temp files created — all processing uses in-memory BytesIO

**Plans:** TBD

---

### Phase 4: Model Execution
**Goal:** All Chatterbox model variants execute and return properly formatted outputs  
**Depends on:** Phase 3  
**Requirements:** REQ-EXEC-01, REQ-EXEC-02, REQ-EXEC-03  
**Success Criteria** (what must be TRUE):
1. `_execute_chatterbox_tts()` generates audio and returns base64 WAV in `TextToSpeechRequestOutput`
2. `_execute_chatterbox_turbo_tts()` applies Turbo-specific params (norm_loudness, top_k, etc.)
3. `_execute_chatterbox_multilingual_tts()` forwards language_id to model
4. `_execute_chatterbox_embedding()` computes Conditionals from audio and returns base64 serialized embedding
5. `_execute_chatterbox_vc()` performs voice conversion and returns output audio

**Plans:** TBD

---

### Phase 5: Request Routing
**Goal:** Requests are correctly routed to appropriate models based on type and parameters  
**Depends on:** Phase 4  
**Requirements:** REQ-ROUTE-01, REQ-ROUTE-02, REQ-ROUTE-03, REQ-ROUTE-04  
**Success Criteria** (what must be TRUE):
1. TTS requests without voice cloning route directly to TTS model
2. TTS requests with pre-computed embedding route directly to TTS model
3. TTS requests with input_audio route to Embedding model first, then forward to TTS (multi-step)
4. Embedding requests route to Chatterbox Embedding model
5. VoiceConversion requests route to ChatterboxVC model
6. Forward processing correctly transfers embedding from embed step to synthesize step

**Plans:** TBD

---

### Phase 6: Configuration & Performance
**Goal:** Model configuration support complete and architecture meets performance requirements  
**Depends on:** Phase 5  
**Requirements:** REQ-CFG-01, REQ-CFG-02, REQ-PERF-01, REQ-PERF-02  
**Success Criteria** (what must be TRUE):
1. `ModelConfig` has accessible `watermark: bool` field (default True)
2. Config examples for all 4 model variants load without errors
3. No temp files created during inference (verified via integration test)
4. Multi-step routing architecture supports future embedding caching (serializable Conditionals)

**Plans:** TBD

---

### Phase 7: Testing & Documentation
**Goal:** Comprehensive test coverage and API documentation complete  
**Depends on:** Phase 6  
**Requirements:** REQ-TEST-01, REQ-TEST-02, REQ-TEST-03, REQ-TEST-04, REQ-DOC-01  
**Success Criteria** (what must be TRUE):
1. Unit tests pass: `pytest tests/unit/inference_flow/test_chatterbox_*.py -v`
2. Input validation unit tests pass: all `ValueError` branches covered for all model variants and conflict cases (REQ-TEST-04)
3. Integration tests pass: `pytest tests/integration/test_chatterbox_flow.py -v`
4. E2E tests pass or skip gracefully: `pytest tests/e2e/test_chatterbox_e2e.py -v`
5. API documentation accessible via `/docs` endpoint
6. New generation options fields documented in OpenAPI spec

**Plans:** TBD

---

## Coverage Map

| Requirement | Phase | Status |
|-------------|-------|--------|
| REQ-DEP-01 | Phase 1 | Pending |
| REQ-ARCH-01 | Phase 2 | Pending |
| REQ-ARCH-02 | Phase 2 | Pending |
| REQ-INPUT-01 | Phase 3 | Pending |
| REQ-INPUT-02 | Phase 3 | Pending |
| REQ-INPUT-03 | Phase 3 | Pending |
| REQ-INPUT-04 | Phase 3 | Pending |
| REQ-INPUT-05 | Phase 3 | Pending |
| REQ-INPUT-06 | Phase 3 | Pending |
| REQ-EXEC-01 | Phase 4 | Pending |
| REQ-EXEC-02 | Phase 4 | Pending |
| REQ-EXEC-03 | Phase 4 | Pending |
| REQ-ROUTE-01 | Phase 5 | Pending |
| REQ-ROUTE-02 | Phase 5 | Pending |
| REQ-ROUTE-03 | Phase 5 | Pending |
| REQ-ROUTE-04 | Phase 5 | Pending |
| REQ-CFG-01 | Phase 6 | Pending |
| REQ-CFG-02 | Phase 6 | Pending |
| REQ-PERF-01 | Phase 6 | Pending |
| REQ-PERF-02 | Phase 6 | Pending |
| REQ-TEST-01 | Phase 7 | Pending |
| REQ-TEST-02 | Phase 7 | Pending |
| REQ-TEST-03 | Phase 7 | Pending |
| REQ-TEST-04 | Phase 7 | Pending |
| REQ-DOC-01 | Phase 7 | Pending |

---

## Progress Table

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Dependencies & Setup | 0/1 | Not started | - |
| 2. Model Registration & Loading | 0/1 | Not started | - |
| 3. Input Preparation | 0/1 | Not started | - |
| 4. Model Execution | 0/1 | Not started | - |
| 5. Request Routing | 0/1 | Not started | - |
| 6. Configuration & Performance | 0/1 | Not started | - |
| 7. Testing & Documentation | 0/1 | Not started | - |

---

*Last updated: 2026-03-13*
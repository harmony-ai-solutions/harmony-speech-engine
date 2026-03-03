# Roadmap: Harmony Speech Engine Testing Framework

**Generated:** 2026-02-28  
**Depth:** Standard (5 phases)  
**Total v1 Requirements:** 24

---

## Phases

- [x] **Phase 1: Test Framework Foundation** - Core pytest setup, fixtures, directory structure, basic CI
 (completed 2026-03-03)
- [x] **Phase 2: Unit Testing Core Components** - Unit tests for config, engine, model loaders
 (completed 2026-03-03)
- [x] **Phase 3: Integration Testing** - API endpoint and CLI integration tests
 (completed 2026-03-03)
- [ ] **Phase 4: E2E Testing - TTS Models** - End-to-end tests for TTS pipelines
- [ ] **Phase 5: E2E Testing - Remaining Models & CI Finalization** - STT, VC, VAD, Voicefixer + coverage

---

## Phase Details

### Phase 1: Test Framework Foundation

**Goal:** Establish core pytest infrastructure with fixtures, directory structure, and basic CI configuration

**Depends on:** Nothing (first phase)

**Requirements:** TFW-01, TFW-02, TFW-03, TFW-04, CI-01, CI-02, DOC-01

**Success Criteria** (what must be TRUE):
1. `pytest --version` runs successfully and shows pytest is installed
2. Test discovery finds all tests in `tests/unit/`, `tests/integration/`, `tests/e2e/` directories
3. Running `pytest --collect-only` shows all test files and functions are discovered
4. Shared fixtures in `conftest.py` are available to all tests (config fixture, tmp_path fixture)
5. `pytest` runs in CPU-only mode (enforced by device fixtures or markers)
6. GitHub Actions workflow runs pytest on push/pull_request events
7. Test documentation exists in `docs/testing.md` explaining how to run tests

**Plans:** 2/2 plans complete

Plans:
- [ ] 01-01-PLAN.md - pytest directory structure, conftest.py fixtures, --device/--dtype CLI hooks
- [ ] 01-02-PLAN.md - GitHub Actions CI workflow and docs/testing.md developer guide

---

### Phase 2: Unit Testing Core Components

**Goal:** Create fast, isolated unit tests for core engine components

**Depends on:** Phase 1

**Requirements:** UNIT-01, UNIT-02, UNIT-03, UNIT-04, UNIT-05

**Success Criteria** (what must be TRUE):
1. Unit tests for `harmonyspeech/common/config.py` pass - config parsing handles valid/invalid inputs
2. Unit tests for `harmonyspeech/engine/` pass - engine initialization works correctly
3. Unit tests for `harmonyspeech/modeling/loader.py` pass - model loading logic is validated
4. Mock objects can be used in tests via `pytest-mock` fixture
5. Async test functions can be executed with proper event loop handling via `pytest-asyncio`

**Plans:** 2/2 plans complete

Plans:
- [x] 02-01-PLAN.md - Config unit tests (DeviceConfig, ModelConfig, EngineConfig, dtype resolution)
- [ ] 02-02-PLAN.md - Engine initialization and model loader unit tests

---

### Phase 3: Integration Testing

**Goal:** Validate component interactions and API endpoint functionality

**Depends on:** Phase 2

**Requirements:** INT-01, INT-02, INT-03

**Success Criteria** (what must be TRUE):
1. Integration tests for OpenAI-compatible API endpoints pass - endpoints respond correctly
2. Integration tests for CLI commands pass - CLI arguments parse correctly
3. Integration tests verify request/response cycle - data flows correctly through the system

**Plans:** 2/2 plans executed

Plans:
- [x] 03-01-PLAN.md - API endpoint integration tests (conftest mock engine + test_api_endpoints.py)
- [x] 03-02-PLAN.md - CLI argument parser integration tests (test_cli.py)

---

### Phase 4: E2E Testing - TTS Models

**Goal:** Validate complete TTS inference pipelines end-to-end

**Depends on:** Phase 3

**Requirements:** E2E-01, E2E-02, E2E-03

**Success Criteria** (what must be TRUE):
1. E2E tests for KittenTTS pass - text input produces audio output
2. E2E tests for MeloTTS pass - text input produces audio output
3. E2E tests for HarmonySpeech pass - text input produces audio output
4. All E2E tests run in CPU mode without GPU dependencies

**Plans:** TBD

---

### Phase 5: E2E Testing - Remaining Models & CI Finalization

**Goal:** Complete E2E coverage for all model types and finalize CI with coverage reporting

**Depends on:** Phase 4

**Requirements:** E2E-04, E2E-05, E2E-06, E2E-07, CI-03, DOC-02

**Success Criteria** (what must be TRUE):
1. E2E tests for STT (Whisper) pass - audio input produces text output
2. E2E tests for Voice Conversion (OpenVoice) pass - source audio converts to target voice
3. E2E tests for VAD pass - voice activity is detected in audio streams
4. E2E tests for Audio Restoration (Voicefixer) pass - audio quality is improved
5. Coverage reporting is generated via pytest-cov
6. Test documentation explains how to interpret test results and coverage reports

**Plans:** TBD

---

## Progress Table

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Test Framework Foundation | 2/2 | Complete   | 2026-03-03 |
| 2. Unit Testing Core Components | 2/2 | Complete   | 2026-03-03 |
| 3. Integration Testing | 2/2 | Complete   | 2026-03-03 |
| 4. E2E Testing - TTS Models | 0/1 | Not started | - |
| 5. E2E Testing - Remaining Models & CI Finalization | 0/1 | Not started | - |

---

## Coverage Summary

| Requirement Category | Count | Phase |
|---------------------|-------|-------|
| Test Framework (TFW) | 4 | Phase 1 |
| Unit Tests (UNIT) | 5 | Phase 2 |
| Integration Tests (INT) | 3 | Phase 3 |
| E2E TTS (E2E-01 to E2E-03) | 3 | Phase 4 |
| E2E Remaining (E2E-04 to E2E-07) | 4 | Phase 5 |
| CI/CD (CI) | 3 | Phase 1, Phase 5 |
| Documentation (DOC) | 2 | Phase 1, Phase 5 |
| **Total** | **24** | |

---

*Last updated: 2026-02-28*

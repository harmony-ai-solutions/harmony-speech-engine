# Requirements: Harmony Speech Engine Testing Framework

## v1 Requirements

### Test Framework (TFW)

- [x] **TFW-01**: Project uses pytest as the testing framework
- [x] **TFW-02**: pytest configuration added to pyproject.toml with appropriate options
- [x] **TFW-03**: Test directory structure created (unit/, integration/, e2e/)
- [x] **TFW-04**: conftest.py created with shared fixtures for test setup/teardown

### Unit Tests (UNIT)

- [ ] **UNIT-01**: Unit tests created for config parsing (harmonyspeech/common/config.py)
- [ ] **UNIT-02**: Unit tests created for engine initialization (harmonyspeech/engine/)
- [ ] **UNIT-03**: Unit tests created for model loader (harmonyspeech/modeling/loader.py)
- [ ] **UNIT-04**: Mocking support configured using pytest-mock
- [ ] **UNIT-05**: Async testing support configured using pytest-asyncio

### Integration Tests (INT)

- [ ] **INT-01**: Integration tests created for OpenAI-compatible API endpoints
- [ ] **INT-02**: Integration tests created for CLI commands
- [ ] **INT-03**: Integration tests verify request/response cycle correctly

### End-to-End Tests (E2E)

- [ ] **E2E-01**: End-to-end tests created for KittenTTS model
- [ ] **E2E-02**: End-to-end tests created for MeloTTS model
- [ ] **E2E-03**: End-to-end tests created for HarmonySpeech model
- [ ] **E2E-04**: End-to-end tests created for STT (Whisper) model
- [ ] **E2E-05**: End-to-end tests created for Voice Conversion (OpenVoice)
- [ ] **E2E-06**: End-to-end tests created for VAD
- [ ] **E2E-07**: End-to-end tests created for Audio Restoration (Voicefixer)

### CI/CD (CI)

- [ ] **CI-01**: GitHub Actions workflow created for automated test execution
- [ ] **CI-02**: CI configured to run tests in CPU-only mode
- [ ] **CI-03**: Coverage reporting configured using pytest-cov

### Documentation (DOC)

- [ ] **DOC-01**: Test documentation created explaining how to run tests
- [ ] **DOC-02**: Test documentation explains how to interpret results

## v2 Requirements (Deferred)

- [ ] **COV-01**: Coverage thresholds enforced (80% target for core modules)
- [ ] **PERF-01**: Performance benchmarks added for model inference
- [ ] **PARA-01**: Parallel test execution configured using pytest-xdist

## Out of Scope

- **GPU-based tests in CI** — CI environment is CPU-only; GPU testing done manually
- **Load/stress testing** — defer to future phase with dedicated tools
- **Security penetration testing** — defer to security-focused phase
- **100% coverage requirement** — impractical for v1; target 80% for core modules

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| TFW-01 | 1 | Complete |
| TFW-02 | 1 | Complete |
| TFW-03 | 1 | Complete |
| TFW-04 | 1 | Complete |
| CI-01 | 1 | Pending |
| CI-02 | 1 | Pending |
| DOC-01 | 1 | Pending |
| UNIT-01 | 2 | Pending |
| UNIT-02 | 2 | Pending |
| UNIT-03 | 2 | Pending |
| UNIT-04 | 2 | Pending |
| UNIT-05 | 2 | Pending |
| INT-01 | 3 | Pending |
| INT-02 | 3 | Pending |
| INT-03 | 3 | Pending |
| E2E-01 | 4 | Pending |
| E2E-02 | 4 | Pending |
| E2E-03 | 4 | Pending |
| E2E-04 | 5 | Pending |
| E2E-05 | 5 | Pending |
| E2E-06 | 5 | Pending |
| E2E-07 | 5 | Pending |
| CI-03 | 5 | Pending |
| DOC-02 | 5 | Pending |

---
*Last updated: 2026-02-28 after requirements definition*
# Project Research Summary

**Project:** Harmony Speech Engine
**Domain:** Testing Framework for ML/AI Models (pytest-based)
**Researched:** 2026-02-28
**Confidence:** HIGH

## Executive Summary

This project aims to build a robust, pytest-based testing framework for the Harmony Speech Engine, an ML/AI speech model system. Expert approaches in this domain emphasize comprehensive test coverage across unit, integration, and end-to-end (E2E) levels, utilizing extensive fixture management and mocking to isolate logic from expensive model inference and external dependencies. A key differentiator is enabling CPU-only CI execution, ensuring accessibility and rapid feedback without the overhead of GPU infrastructure.

The recommended approach centers on `pytest` and its ecosystem (`pytest-asyncio`, `pytest-mock`, `pytest-cov`), establishing a structured test hierarchy (`unit/`, `integration/`, `e2e/`). By leveraging shared fixtures in `conftest.py` for test data and configuration, the framework can efficiently test individual components, API endpoints, and complete inference pipelines (TTS, STT, Voice Conversion, VAD, Audio Restoration).

The primary risks involve implicit GPU dependencies failing in CI, flaky tests due to non-deterministic model outputs, and over-mocking that masks true integration failures. Mitigating these requires explicit device overrides (enforcing CPU in CI), tolerance-based assertions for floating-point comparisons, and maintaining a clear boundary between mocked unit tests and genuine E2E execution using representative audio fixtures.

## Key Findings

### Recommended Stack

The testing framework relies on the industry-standard `pytest` ecosystem, utilizing already installed core dependencies to minimize adoption friction. The stack prioritizes deterministic, fast, and scalable testing capabilities.

**Core technologies:**
- `pytest` (9.0.2): Core testing framework — Industry standard, enables fixtures, parametrization, and rich plugin ecosystem.
- `pytest-asyncio` (1.3.0): Async test support — Required for testing async engine methods and proper event loop management.
- `pytest-mock` (3.15.1): Mocking utilities — Provides a cleaner fixture API over `unittest.mock` for isolating external dependencies.
- `pytest-cov` (^6.0): Coverage reporting — Essential for tracking test completeness across core modules.

### Expected Features

The feature landscape balances essential framework capabilities with ML-specific enhancements, prioritizing robust model validation and developer experience.

**Must have (table stakes):**
- pytest Framework Setup — Core infrastructure.
- Test Fixtures (`conftest.py`) — Shared data and model configurations.
- Unit Tests (Config, Engine) — Validating core logic.
- Integration Tests — API endpoints and CLI interactions.
- E2E Tests for TTS Models — Full inference pipeline validation.

**Should have (competitive):**
- CPU-Only CI Execution — Fast, accessible automated testing without GPU requirements.
- Model-by-Model Test Structure — Targeted, modular testing for individual models.
- Coverage Reporting with Thresholds — Automated quality gates.
- Async Testing Support — Full coverage of async interfaces.

**Defer (v2+):**
- Load/Stress Testing Framework — Complex infrastructure needs.
- Security Penetration Testing — Specialized focus.
- Golden Output Comparison — Strict regression detection.

### Architecture Approach

The testing architecture follows a layered design, separating test execution, organization, fixtures, and data to ensure maintainability and scalability.

**Major components:**
1. **pytest Runner & Plugins** — Manages discovery, execution, async support, mocking, and coverage reporting.
2. **Test Organization (`unit/`, `integration/`, `e2e/`)** — Segregates tests by scope and execution requirements.
3. **Fixture & Mock Layer (`conftest.py`)** — Provides reusable setup, mock objects, and data injection across all tests.
4. **Test Data Layer (`fixtures/`)** — Stores static assets like sample audio tensors, reference files, and model configs.

### Critical Pitfalls

1. **GPU-Dependent Tests in CPU-Only CI** — Avoid by explicitly forcing `torch.device("cpu")` in fixtures and using device-specific markers.
2. **Flaky Tests from Non-Deterministic Outputs** — Avoid by utilizing approximate equality assertions (`pytest.approx`) and setting random seeds.
3. **Missing Test Fixtures for Model Artifacts** — Avoid by creating robust fixtures for downloading test models or including small sample data locally.
4. **Over-Mocking Core Model Logic** — Avoid by maintaining clear boundaries (mock I/O, not inference) and ensuring true E2E pipeline tests exist.
5. **No Test Isolation (Shared State)** — Avoid by strictly scoping fixtures, cleaning up global state, and utilizing `tmp_path` for file operations.

## Implications for Roadmap

Based on research dependencies and critical path analysis, the following phase structure is suggested:

### Phase 1: Test Framework Setup & Foundation
**Rationale:** Core infrastructure must be established before tests can be written. Fixtures and CPU-only enforcement are foundational requirements.
**Delivers:** pytest configuration, `conftest.py` with base fixtures, directory structure, mock data.
**Addresses:** pytest Framework, Test Fixtures, CI/CD Configuration basics.
**Avoids:** Missing Test Fixtures, No Test Isolation, GPU-Dependent Tests in CI.

### Phase 2: Unit Testing Core Components
**Rationale:** Fast, isolated tests for the engine and configuration catch obvious issues early without requiring complex model setup.
**Delivers:** Unit tests for `common/config.py`, `engine/`, `args_tools.py`, async coverage.
**Uses:** `pytest`, `pytest-mock`, `pytest-asyncio`.
**Implements:** `unit/` test suite.

### Phase 3: Integration & API Testing
**Rationale:** Validates component interactions and the OpenAI-compatible REST endpoints before tackling heavy model inference.
**Delivers:** Integration tests for API endpoints and CLI commands.
**Uses:** HTTP mocking, shared fixtures.
**Implements:** `integration/` test suite.

### Phase 4: E2E Testing for TTS Pipelines
**Rationale:** TTS is the primary use case. Testing these pipelines end-to-end ensures the core value proposition functions correctly.
**Delivers:** E2E tests for KittenTTS, MeloTTS, and HarmonySpeech.
**Uses:** Real model inference (CPU), audio fixtures.
**Implements:** `e2e/test_tts/` test suite.

### Phase 5: Comprehensive E2E & CI Finalization
**Rationale:** Expands coverage to all remaining model types and enforces quality gates via coverage reporting in automated CI pipelines.
**Delivers:** E2E tests for STT (Whisper), Voice Conversion, VAD, Audio Restoration; full GitHub Actions integration with pytest-cov.
**Uses:** `pytest-cov`, GitHub Actions.
**Implements:** Remaining `e2e/` suites, CI workflow.

### Phase Ordering Rationale
- **Dependency Flow:** Configuration -> Fixtures -> Unit Logic -> Integration -> Full E2E -> CI Validation.
- **Risk Mitigation:** Foundational phases establish CPU enforcement and mock data handling early, preventing the most common architectural pitfalls.
- **Value Delivery:** Core functionality and API interactions are validated before expanding to edge-case model pipelines.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 4 (E2E TTS):** Understanding the specific loading and initialization quirks of KittenTTS and MeloTTS for local CPU testing.
- **Phase 5 (Comprehensive E2E):** Voicefixer and STT dependencies may require specialized lightweight mock models or fixtures.

Phases with standard patterns (skip research-phase):
- **Phase 1 (Foundation):** Standard pytest boilerplate.
- **Phase 2 (Unit Testing):** Standard Python mocking and unit testing patterns.
- **Phase 3 (Integration):** Standard FastAPI/HTTP endpoint testing.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Based on verified environment dependencies and pytest documentation. |
| Features | HIGH | Clear mapping of MVP requirements vs. future enhancements. |
| Architecture | HIGH | Established patterns for pytest organization in ML projects. |
| Pitfalls | MEDIUM | ML testing is inherently complex; some flakiness may still emerge during implementation. |

**Overall confidence:** HIGH

### Gaps to Address

- **Audio File Management:** Determining the exact strategy for managing audio fixtures (committing small files vs. generating them programmatically) needs validation during Phase 1.
- **Model Download Mocking:** Need to determine if small "dummy" models should be uploaded to HF, or if model loading should be mocked entirely during E2E CPU tests to avoid large downloads in CI.

## Sources

### Primary (HIGH confidence)
- Project Context (`.planning/research/STACK.md`, `FEATURES.md`, `ARCHITECTURE.md`, `PITFALLS.md`) — Aggregated findings and technical directives.
- Official Pytest Documentation — Capabilities, fixtures, and marker implementations.

### Secondary (MEDIUM confidence)
- Industry Experience — Common ML testing patterns, CPU vs. GPU behavior differences, and tolerance-based testing methodologies.

---
*Research completed: 2026-02-28*
*Ready for roadmap: yes*

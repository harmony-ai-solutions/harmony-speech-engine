# Architecture Research: Testing Framework for ML/AI Models

**Domain:** Testing Framework Architecture
**Researched:** 2026-02-28
**Confidence:** HIGH

## Standard Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Test Execution Layer                          │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────┐ │
│  │   pytest    │  │  pytest-    │  │   pytest-   │  │  pytest-  │ │
│  │   Runner    │  │   asyncio   │  │    mock     │  │   cov     │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └─────┬─────┘ │
│         │                │                │               │        │
├─────────┴────────────────┴────────────────┴───────────────┴────────┤
│                        Test Organization Layer                       │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────┐ │
│  │    unit/    │  │ integration/│  │    e2e/     │  │  conftest │ │
│  │  test_*.py  │  │  test_*.py  │  │  test_*.py  │  │   .py     │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └─────┬─────┘ │
│         │                │                │               │        │
├─────────┴────────────────┴────────────────┴───────────────┴────────┤
│                        Fixture & Mock Layer                          │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────┐ │
│  │   Sample    │  │   Model     │  │   Audio     │  │   Mock    │ │
│  │   Configs   │  │   Fixtures  │  │   Fixtures  │  │  Objects  │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └───────────┘ │
├─────────────────────────────────────────────────────────────────────┤
│                        Test Data Layer                               │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │
│  │   fixtures/ │  │  __init__.py│  │  pyproject  │                  │
│  │ audio_samples│ │             │  │   .toml     │                  │
│  └─────────────┘  └─────────────┘  └─────────────┘                  │
└─────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Typical Implementation |
|-----------|----------------|------------------------|
| **pytest Runner** | Discovers and executes tests; manages test collection, execution, and reporting | pytest 9.x — scans `tests/` for `test_*.py`, executes, reports results |
| **pytest-asyncio** | Handles async test functions; provides event loop management | `@pytest.mark.asyncio` decorator; auto-detects async tests |
| **pytest-mock** | Provides `mocker` fixture for mocking; cleaner API than unittest.mock | `mocker.patch()`, `mocker.Mock()` |
| **pytest-cov** | Measures and reports code coverage | `--cov=harmonyspeech` flag; generates coverage reports |
| **conftest.py** | Shared fixtures and hooks; defines reusable test data and setup | `@pytest.fixture` decorators; `pytest_configure()` hooks |
| **unit/** | Tests individual components in isolation | Mock external dependencies; test single class/function |
| **integration/** | Tests component interactions and API endpoints | Test request/response; test CLI commands |
| **e2e/** | Tests complete inference pipelines end-to-end | Full model execution; input-to-output validation |
| **fixtures/** | Static test data (audio samples, config files) | Sample audio tensors, model configs, reference outputs |
| **Mock Objects** | Replace external dependencies (HF downloads, GPU, network) | `unittest.mock.Mock`, `mocker.patch` |

## Recommended Project Structure

```
harmony-speech-engine/
├── tests/                              # Test root directory
│   ├── __init__.py                     # Makes tests a package
│   ├── conftest.py                     # Shared fixtures and pytest hooks
│   ├── pyproject.toml                  # pytest configuration
│   ├── unit/                           # Unit tests
│   │   ├── __init__.py
│   │   ├── test_config.py              # Tests for common/config.py
│   │   ├── test_engine.py              # Tests for engine/ module
│   │   ├── test_args_tools.py          # Tests for args_tools.py
│   │   └── test_processing/            # Processing module tests
│   │       ├── __init__.py
│   │       └── test_scheduler.py
│   ├── integration/                    # Integration tests
│   │   ├── __init__.py
│   │   ├── test_api_endpoints.py       # OpenAI-compatible endpoint tests
│   │   └── test_cli.py                 # CLI command tests
│   ├── e2e/                            # End-to-end tests
│   │   ├── __init__.py
│   │   ├── test_tts/                   # TTS model tests
│   │   │   ├── __init__.py
│   │   │   ├── test_kittentts.py
│   │   │   ├── test_melotts.py
│   │   │   └── test_harmonyspeech.py
│   │   ├── test_stt_whisper.py         # STT model tests
│   │   ├── test_voice_conversion.py    # OpenVoice tests
│   │   ├── test_vad.py                 # VAD tests
│   │   └── test_audio_restoration.py   # Voicefixer tests
│   └── fixtures/                       # Static test data
│       ├── __init__.py
│       ├── audio_samples/              # Sample audio files
│       │   ├── sample_16khz.wav        # 1-second test audio
│       │   └── sample_48khz.wav        # High-res test audio
│       └── configs/                    # Sample config files
│           ├── cpu_config.yaml
│           └── tts_config.yaml
├── .github/workflows/
│   └── test.yml                        # CI test execution
└── pyproject.toml                      # pytest configuration (root)
```

### Structure Rationale

- **`tests/` at root:** Follows Python convention; pytest auto-discovers without configuration
- **`unit/integration/e2e/`:** Separates test types by scope; enables selective execution with `pytest -m unit`
- **`conftest.py` at root:** Provides shared fixtures to all tests; pytest loads automatically
- **`e2e/model-specific folders:`** Enables targeted testing; run `pytest tests/e2e/test_tts/` for TTS only
- **`fixtures/` for static data:** Keeps test data separate from test code; enables reuse across test types

## Architectural Patterns

### Pattern 1: Fixture-Based Test Setup

**What:** Use pytest fixtures for shared test setup and teardown.

**When to use:** Any test requiring common setup (model configs, sample data, engine instances).

**Trade-offs:**
- ✅ Reduces duplication; single source of truth
- ✅ Manages lifecycle (setup/teardown) automatically
- ✅ Supports dependency injection between fixtures
- ❌ Can lead to overly complex fixture graphs if not careful

**Example:**
```python
# conftest.py
import pytest
import torch

@pytest.fixture
def sample_audio_tensor():
    """Generate sample audio tensor for testing."""
    return torch.randn(1, 16000)  # 1 second at 16kHz

@pytest.fixture
def model_config_cpu():
    """Create test model config for CPU."""
    from harmonyspeech.common.config import ModelConfig, DeviceConfig
    return ModelConfig(
        name="test-tts",
        model="test/model",
        model_type="tts",
        max_batch_size=1,
        device_config=DeviceConfig("cpu")
    )

# test_file.py
def test_config_parsing(model_config_cpu):
    """Test config uses fixture automatically."""
    assert model_config_cpu.device_config.device_type == "cpu"
```

### Pattern 2: Mock-Based Isolation

**What:** Replace external dependencies with mock objects to isolate unit tests.

**When to use:** Unit tests that would otherwise require network, GPU, or file system access.

**Trade-offs:**
- ✅ Fast execution; no external dependencies
- ✅ Deterministic results; no network flakiness
- ✅ Tests logic, not integration
- ❌ Doesn't test actual integration; mocks may diverge from real behavior

**Example:**
```python
# test_engine.py
from unittest.mock import Mock, patch
import pytest

class TestHarmonySpeechEngine:
    @patch('harmonyspeech.engine.harmonyspeech_engine.CPUExecutor')
    def test_init_with_cpu(self, mock_executor):
        """Test engine initialization with CPU executor."""
        mock_executor.return_value = Mock()
        
        from harmonyspeech.engine.harmonyspeech_engine import HarmonySpeechEngine
        engine = HarmonySpeechEngine(
            model_configs=[],
            log_stats=False
        )
        mock_executor.assert_called()
```

### Pattern 3: Test Markers for Selective Execution

**What:** Use pytest markers to categorize tests; run subsets with `pytest -m marker`.

**When to use:** Large test suites where not all tests need to run every time.

**Trade-offs:**
- ✅ Faster development feedback; run unit tests only
- ✅ Selective CI execution; skip slow tests on PRs
- ❌ Requires discipline to mark tests correctly
- ❌ Can lead to missed test coverage if markers are wrong

**Example:**
```python
# conftest.py
def pytest_configure(config):
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "e2e: End-to-end tests")
    config.addinivalue_line("markers", "slow: Slow-running tests")

# test_file.py
@pytest.mark.unit
def test_config_validation():
    """Fast unit test."""
    pass

@pytest.mark.slow
@pytest.mark.e2e
def test_full_tts_pipeline():
    """Slow E2E test."""
    pass

# Run only unit tests
# pytest -m unit

# Run all except slow
# pytest -m "not slow"
```

### Pattern 4: Parametrized Tests for Edge Cases

**What:** Use `@pytest.mark.parametrize` to run the same test with multiple inputs.

**When to use:** Testing the same logic with different inputs (languages, configs, edge cases).

**Trade-offs:**
- ✅ Catches edge cases efficiently; single test function
- ✅ Clear intent; each parameter combination is explicit
- ❌ Can become unwieldy with many parameters
- ❌ Harder to debug which parameter failed

**Example:**
```python
@pytest.mark.parametrize("language,expected_lang_code", [
    ("en", "english"),
    ("es", "spanish"),
    ("fr", "french"),
    ("zh", "chinese"),
])
def test_language_detection(language, expected_lang_code):
    """Test language detection for multiple languages."""
    from harmonyspeech.common.utils import detect_language
    result = detect_language(language)
    assert result == expected_lang_code
```

### Pattern 5: Async Test Support

**What:** Use pytest-asyncio to test async engine methods.

**When to use:** Testing async inference methods, async model loading.

**Trade-offs:**
- ✅ Tests actual async code paths
- ✅ Proper event loop management
- ❌ Requires understanding of asyncio nuances
- ❌ Can have event loop conflicts if not configured correctly

**Example:**
```python
# conftest.py - configure asyncio mode
pytest_plugins = ('pytest_asyncio',)

# test_async.py
import pytest

@pytest.mark.asyncio
async def test_async_inference():
    """Test async inference methods."""
    from harmonyspeech.engine.async_harmonyspeech import AsyncEngine
    
    engine = AsyncEngine(device="cpu")
    result = await engine.generate("Hello world")
    assert result is not None
```

## Data Flow

### Test Execution Flow

```
1. pytest Discovery
   └─> Scans tests/ directory for test_*.py files
       └─> Collects test functions and classes

2. Fixture Resolution
   └─> For each test, resolve fixture dependencies
       └─> Execute fixture functions (setup)
           └─> Inject fixtures into test function

3. Test Execution
   └─> Run test function with injected fixtures
       └─> If async: manage event loop
       └─> If marked: check marker filters

4. Assertion & Reporting
   └─> Execute assertions
       └─> Capture failures
       └─> Generate test report

5. Teardown
   └─> Execute fixture teardown (yield cleanup)
       └─> Release resources
```

### Fixture Dependency Flow

```
conftest.py (root)
    │
    ├── @pytest.fixture sample_audio_tensor()
    │       └── Injected into: e2e tests
    │
    ├── @pytest.fixture model_config_cpu()
    │       └── Injected into: unit & e2e tests
    │
    ├── @pytest.fixture mock_executor()
    │       └── Injected into: unit tests
    │
    └── @pytest.fixture running_server()
            └── Injected into: integration tests
```

### Test Type Data Flow

```
Unit Tests:
  test_*.py ──uses──> conftest.py fixtures ──uses──> Mock objects
       │                                              │
       └────────────── No external I/O ──────────────┘

Integration Tests:
  test_api.py ──uses──> conftest.py fixtures ──uses──> HTTP mocks
       │                                              │
       └────────── Test API endpoints ────────────────┘

E2E Tests:
  test_tts.py ──uses──> conftest.py fixtures ──uses──> fixtures/audio/
       │                                              │
       └────────── Full model inference ──────────────┘
```

## Build Order & Dependencies

### Component Dependency Graph

```
Phase 1: Foundation
├── pytest configuration (pyproject.toml)
├── conftest.py with basic fixtures
└── Test directory structure
    │
    ├── tests/__init__.py
    ├── tests/conftest.py
    ├── tests/unit/
    ├── tests/integration/
    ├── tests/e2e/
    └── tests/fixtures/

Phase 2: Unit Tests
├── tests/unit/test_config.py
├── tests/unit/test_engine.py
└── tests/unit/test_args_tools.py
    │
    └──requires: conftest.py fixtures, mock objects

Phase 3: Integration Tests
├── tests/integration/test_api_endpoints.py
└── tests/integration/test_cli.py
    │
    └──requires: Unit tests passing, conftest.py

Phase 4: E2E Tests (TTS Priority)
├── tests/e2e/test_tts/test_kittentts.py
├── tests/e2e/test_tts/test_melotts.py
└── tests/e2e/test_tts/test_harmonyspeech.py
    │
    └──requires: Integration tests passing, fixtures/audio/

Phase 5: Additional E2E Tests
├── tests/e2e/test_stt_whisper.py
├── tests/e2e/test_voice_conversion.py
├── tests/e2e/test_vad.py
└── tests/e2e/test_audio_restoration.py

Phase 6: CI/CD Integration
├── .github/workflows/test.yml
└── pytest-cov configuration
    │
    └──requires: All test types implemented
```

### Build Order Rationale

1. **Foundation first:** Cannot write tests without directory structure and pytest configuration
2. **Unit before integration:** Unit tests are faster and catch obvious issues early
3. **Integration before E2E:** Integration tests validate component interactions before full pipeline tests
4. **TTS first:** TTS is primary use case; validate core functionality first
5. **CI last:** CI requires tests to exist before configuring automated execution

### Critical Path

```
conftest.py fixtures
        │
        ▼
Unit tests (config, engine)
        │
        ▼
Integration tests (API)
        │
        ▼
E2E tests (TTS models)
        │
        ▼
CI/CD configuration
```

## Scalability Considerations

| Concern | At 10 tests | At 100 tests | At 500 tests |
|---------|-------------|--------------|--------------|
| **Execution Time** | < 1 second | < 30 seconds | < 5 minutes |
| **Strategy** | Run all | Run all | Use markers: `pytest -m unit` |
| **Parallelization** | Not needed | Consider pytest-xdist | Required: `pytest -n auto` |
| **Coverage** | Manual review | pytest-cov | Enforce thresholds in CI |
| **Fixtures** | Simple conftest | Organized by test type | Fixture modules per component |

## Anti-Patterns to Avoid

### Anti-Pattern 1: Monolithic Fixtures

**What:** Creating one giant fixture that does everything.

**Why bad:** Hard to maintain, creates hidden dependencies, difficult to debug.

**Instead:** Create focused, single-responsibility fixtures that compose together.

### Anti-Pattern 2: Testing Implementation Details

**What:** Testing internal function calls, private methods, or specific code paths.

**Why bad:** Brittle tests that break on refactoring; tests should verify behavior, not implementation.

**Instead:** Test public interfaces and expected outputs.

### Anti-Pattern 3: No Test Isolation

**What:** Tests that depend on execution order or share state.

**Why bad:** Flaky tests that pass sometimes; difficult to debug failures.

**Instead:** Use fixtures with proper setup/teardown; use `pytest-randomly` to catch order dependencies.

### Anti-Pattern 4: Over-Mocking

**What:** Mocking everything, including the code under test.

**Why bad:** Tests don't actually verify anything; mocks can diverge from real behavior.

**Instead:** Mock only external dependencies; test real code paths.

## Sources

- **Context7**: pytest, pytest-asyncio, pytest-mock — verified current versions and architecture
- **Official Documentation**: https://docs.pytest.org/ — confirmed pytest 9.x patterns
- **Project Context**: `.planning/codebase/TESTING.md` — existing testing recommendations
- **STACK.md**: Technology stack recommendations (pytest 9.0.2, pytest-asyncio 1.3.0, pytest-mock 3.15.1)
- **FEATURES.md**: Feature landscape and dependencies

---

*Architecture research for: Testing Framework for ML/AI Models*
*Researched: 2026-02-28*
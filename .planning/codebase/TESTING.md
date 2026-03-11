# Testing Patterns

_Last updated: 2026-03-11_

## Overview

The Harmony Speech Engine has a comprehensive three-tier test structure (unit, integration, E2E) with full CI/CD integration. Tests use pytest with async support, mocking, and coverage reporting.

## Test Framework

**Runner:** pytest
- Version: Latest compatible with Python 3.12+
- Configuration in `pyproject.toml`:
  ```toml
  [tool.pytest.ini_options]
  testpaths = ["tests"]
  asyncio_mode = "auto"
  addopts = [
      "--strict-markers",
      "-ra",
      "--cov=harmonyspeech",
      "--cov-report=term-missing",
      "--cov-report=xml",
  ]
  ```

**Key Dependencies:**
- `pytest` - Test runner
- `pytest-asyncio` - Async test support (mode: auto)
- `pytest-mock` - Mocking utilities
- `pytest-cov` - Coverage reporting
- `torch` - For device/dtype fixtures

## Test Directory Structure

```
tests/
├── conftest.py              # Root: CLI hooks, shared fixtures
├── test-data/               # Binary test assets (audio files, configs)
├── unit/
│   ├── conftest.py          # Unit fixtures (mock_model_loader, mock_hf_downloader)
│   ├── __init__.py
│   ├── initialization/
│   │   ├── test_config.py
│   │   └── test_engine.py
│   ├── inference_flow/
│   │   └── test_loader.py
│   └── error_handling/
│       └── test_config_errors.py
├── integration/
│   ├── conftest.py          # Integration fixtures (mock_engine_app, test_app)
│   ├── __init__.py
│   ├── test_api_endpoints.py
│   └── test_cli.py
└── e2e/
    ├── conftest.py          # E2E fixtures (engine fixtures, models_cache_dir)
    ├── __init__.py
    ├── stt/
    │   └── test_whisper.py
    ├── tts/
    │   ├── test_kittentts.py
    │   ├── test_melotts.py
    │   ├── test_openvoice_v1.py
    │   └── test_harmonyspeech.py
    ├── vad/
    │   ├── test_silero_vad.py
    │   └── test_whisper_vad.py
    └── audio_restoration/
        └── test_voicefixer.py
```

## Test Tiers

| Tier | Speed | Mocking | Use Case |
|------|-------|---------|----------|
| `unit` | Fast (<1s) | Full mocking | Logic in isolation — config parsing, data transformations |
| `integration` | Medium (1-30s) | Partial (no real models) | API endpoints, CLI, component wiring |
| `e2e` | Slow (minutes) | None (real models) | Full inference pipelines with actual model output |

**Test Markers:**
- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.e2e` - End-to-end tests (auto-applied to `tests/e2e/`)
- `@pytest.mark.slow` - Tests taking >30 seconds

## Running Tests

### Basic Commands

```bash
# Run all tests (CPU mode, default)
pytest

# Run specific tier
pytest tests/unit/
pytest tests/integration/
pytest tests/e2e/

# Run by marker
pytest -m unit              # Only unit tests
pytest -m "not e2e"         # Skip E2E tests
pytest -m e2e               # Only E2E tests
pytest -m "not slow"        # Skip slow tests
```

### Device and dtype Options

The test suite supports hardware-specific testing via CLI flags:

```bash
# Override device (defaults to cpu)
pytest --device=cuda        # Requires CUDA availability

# Override dtype (defaults to float32)
pytest --dtype=float16
```

| Flag | Default | Values | Description |
|------|---------|--------|-------------|
| `--device` | `cpu` | `cpu`, `cuda` | Target device for tensor operations |
| `--dtype` | `float32` | `float32`, `float16` | Default tensor dtype |

> **Strict Validation:** If `--device=cuda` is passed on a machine without CUDA, the test suite exits immediately with an error. This prevents silent failures.

### Run Specific Model E2E Tests

```bash
pytest -k "whisper" tests/e2e/ --device=cpu --dtype=float32
pytest -k "kittentts" tests/e2e/ --device=cpu --dtype=float32
pytest -k "voicefixer" tests/e2e/ --device=cpu --dtype=float32
pytest -k "vad" tests/e2e/ --device=cpu --dtype=float32
```

## Root Fixtures (tests/conftest.py)

These fixtures are available to all tests:

| Fixture | Scope | Description |
|---------|-------|-------------|
| `device` | session | Target device string from `--device` flag |
| `dtype` | session | Target dtype string from `--dtype` flag |
| `tests_root` | session | `Path` to the `tests/` directory |
| `test_data_dir` | session | `Path` to `tests/test-data/` |
| `sample_config` | function | Minimal valid model config dict |

**CLI Hooks:**
- `pytest_addoption` - Registers `--device` and `--dtype` options
- `pytest_configure` - Validates CUDA availability early

## Unit Test Fixtures (tests/unit/conftest.py)

| Fixture | Description |
|---------|-------------|
| `mock_model_loader` | Returns MagicMock for `harmonyspeech.modeling.loader.get_model_class` |
| `mock_hf_downloader` | Patches HuggingFace Hub download to prevent network calls |

## Integration Test Fixtures (tests/integration/conftest.py)

| Fixture | Scope | Description |
|---------|-------|-------------|
| `test_app` | module | FastAPI TestClient for API endpoint tests |
| `mock_engine_app` | module | TestClient with all serving module globals mocked |

The `mock_engine_app` fixture patches all 7 serving module globals:
- `openai_serving_tts`, `openai_serving_stt`, `openai_serving_vc`
- `openai_serving_embedding`, `openai_serving_vad`, `openai_serving_ac`
- `engine`, `engine_args`

## E2E Test Fixtures (tests/e2e/conftest.py)

### Engine Fixtures

| Fixture | Scope | Description |
|---------|-------|-------------|
| `kittentts_mini_engine` | session | KittenTTS mini variant engine |
| `kittentts_micro_engine` | session | KittenTTS micro variant engine |
| `kittentts_nano_engine` | session | KittenTTS nano variant engine |
| `kittentts_nano_int8_engine` | session | KittenTTS nano int8 variant engine |
| `melotts_en_engine` | session | MeloTTS / OpenVoice V2 engine |
| `openvoice_v1_en_engine` | session | OpenVoice V1 engine |
| `harmonyspeech_engine` | session | HarmonySpeech v1 engine |
| `whisper_engine` | session | FasterWhisper STT engine |
| `vad_engine` | session | SileroVAD engine |
| `whisper_vad_engine` | session | Whisper-based VAD engine |
| `voicefixer_engine` | session | VoiceFixer audio restoration engine |

### Helper Fixtures

| Fixture | Scope | Description |
|---------|-------|-------------|
| `models_cache_dir` | session | Temp directory for model weight caching |
| `mock_raw_request` | session | Mock FastAPI Request for serving handler calls |

### Test Data Helpers

| Function | Description |
|----------|-------------|
| `make_silent_wav_b64()` | Generate minimal silent WAV as base64 |
| `load_sample_audio_b64()` | Load WAV from `tests/test-data/samples/` |

### Automatic E2E Marker

The `pytest_collection_modifyitems` hook automatically marks all tests in `tests/e2e/` with `@pytest.mark.e2e`.

## Test Structure Patterns

### Unit Test Example

From `tests/unit/initialization/test_config.py`:

```python
"""Unit tests for harmonyspeech/common/config.py — DeviceConfig, ModelConfig, EngineConfig."""
import pytest
import torch
from harmonyspeech.common.config import DeviceConfig, ModelConfig, EngineConfig


def test_device_config_explicit_cpu():
    cfg = DeviceConfig("cpu")
    assert cfg.device_type == "cpu"
    assert cfg.device == torch.device("cpu")


def test_device_config_auto_falls_back_to_cpu(monkeypatch):
    """When CUDA unavailable and is_cpu() returns True, device_type is 'cpu'."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("harmonyspeech.common.config.is_cpu", lambda: True)
    cfg = DeviceConfig("auto")
    assert cfg.device_type == "cpu"


def test_model_config_invalid_dtype_raises(cpu_device):
    with pytest.raises(ValueError, match="Unknown dtype"):
        ModelConfig(
            name="t", model="m", model_type="MeloTTSSynthesizer",
            max_batch_size=1, device_config=cpu_device, dtype="invalid_dtype",
        )
```

### Integration Test Example

From `tests/integration/test_api_endpoints.py`:

```python
"""Integration tests for OpenAI-compatible API endpoints."""
import pytest


@pytest.fixture(scope="module")
def client(mock_engine_app):
    """Provides the mocked TestClient for API endpoint tests."""
    return mock_engine_app


def test_health(client):
    """Test GET /health returns 200 with health status JSON."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "text_to_speech" in data
    assert "speech_to_text" in data


def test_tts_request_cycle(client):
    """Test POST /v1/audio/speech with valid body returns 200 with base64 audio."""
    response = client.post(
        "/v1/audio/speech",
        json={"model": "test-tts-model", "input": "Hello world"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "audio" in data or "data" in data
```

## Mocking Patterns

### What to Mock (Unit Tests)
- External API calls (HuggingFace model downloads)
- GPU/torch operations (for CPU-only testing)
- File I/O operations
- Network requests

### What NOT to Mock (E2E Tests)
- Core business logic that needs validation
- Configuration parsing
- Model loading (real model tests)
- Actual inference output

### Using monkeypatch

```python
def test_device_config_auto_no_device_raises(monkeypatch):
    """When neither CUDA nor CPU available, RuntimeError is raised."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("harmonyspeech.common.config.is_cpu", lambda: False)
    with pytest.raises(RuntimeError):
        DeviceConfig("auto")
```

## Coverage

**Requirements:** Not currently enforced, but coverage is collected.

**Coverage Configuration in `pyproject.toml`:**
```toml
[tool.coverage.run]
omit = [
    "harmonyspeech/modeling/models/*",
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
]
```

**View Coverage:**
```bash
# Terminal output (default with --cov)
pytest --cov=harmonyspeech --cov-report=term-missing

# HTML report
pytest --cov=harmonyspeech --cov-report=html
open htmlcov/index.html   # macOS
xdg-open htmlcov/index.html  # Linux
```

**Coverage Exclusions:**
- `harmonyspeech/modeling/models/*` - Third-party model code
- Deprecation warnings and user warnings are filtered

## CI Pipeline

From `.github/workflows/test.yml`:

### Jobs

1. **test** - Unit & Integration Tests
   - Runs: `pytest --device=cpu --dtype=float32 -v --tb=short -m "not e2e"`
   - Uploads `coverage.xml` artifact

2. **lint** - Code Quality
   - Runs: `black --check --diff harmonyspeech/ tests/`
   - Runs: `flake8 harmonyspeech/ tests/`

3. **e2e-kittentts** - KittenTTS E2E Tests
4. **e2e-melotts-openvoice-v2** - MeloTTS and OpenVoice V2 E2E Tests
5. **e2e-openvoice-v1** - OpenVoice V1 E2E Tests
6. **e2e-harmonyspeech** - HarmonySpeech E2E Tests
7. **e2e-whisper** - Whisper STT E2E Tests
8. **e2e-vad** - SileroVAD E2E Tests
9. **e2e-voicefixer** - VoiceFixer E2E Tests

All E2E jobs:
- Require the `test` job to pass first (`needs: [test]`)
- Run with `--device=cpu --dtype=float32`
- Use `--no-cov` for long-running tests to reduce overhead

## Adding New Tests

### Unit Tests
1. Add `tests/unit/{category}/test_{module}.py`
2. Import the module under test
3. Mock dependencies with `mock_model_loader` or `unittest.mock.patch`
4. Write test functions with descriptive names

### Integration Tests
1. Add `tests/integration/test_{feature}.py`
2. Use the `test_app` or `mock_engine_app` fixture
3. Test API endpoints, CLI commands, or component wiring

### E2E Tests
1. Add `tests/e2e/{task}/test_{model}.py`
2. Use `models_cache_dir` fixture to avoid redundant model downloads
3. Use appropriate engine fixture from `tests/e2e/conftest.py`

### File Naming
- Always prefix with `test_` for pytest discovery
- Example: `test_config.py`, `test_api_endpoints.py`, `test_whisper.py`

---

_Last updated: 2026-03-11_
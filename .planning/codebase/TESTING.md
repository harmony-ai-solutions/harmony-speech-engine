# Testing Patterns

**Analysis Date:** 2026-02-28

## Test Framework

**Status:** No tests currently exist in the codebase.

**Recommended Framework:**
- `pytest` - Standard Python testing framework
- Version: Latest compatible with Python 3.12+

**Configuration:**
- No `pytest.ini`, `setup.cfg`, or `pyproject.toml` test configuration found
- Should be added to `pyproject.toml` under `[tool.pytest.ini_options]`

**Run Commands (recommended):**
```bash
pytest                    # Run all tests
pytest --watch            # Watch mode (with pytest-watch)
pytest --cov=harmonyspeech # Coverage
pytest -v                 # Verbose output
```

## Test File Organization

**Location:**
- No `tests/` directory with test files exists
- Should follow standard Python convention: `tests/` at project root or co-located with modules

**Recommended Structure:**
```
harmony-speech-engine/
├── tests/
│   ├── __init__.py
│   ├── conftest.py           # Shared fixtures
│   ├── unit/
│   │   ├── test_config.py
│   │   ├── test_engine.py
│   │   └── test_models/
│   ├── integration/
│   │   └── test_endpoints.py
│   └── fixtures/
│       └── audio_samples/
```

**Naming:**
- `test_*.py` for test modules
- `*_test.py` alternative pattern (less common in Python)

## Test Structure

**Recommended Pattern:**
```python
import pytest
from harmonyspeech.common.config import ModelConfig, DeviceConfig

class TestModelConfig:
    """Tests for ModelConfig class."""
    
    def test_default_values(self):
        """Test that default values are set correctly."""
        config = ModelConfig(
            name="test",
            model="test/model",
            model_type="tts",
            max_batch_size=1,
            device_config=DeviceConfig("cpu")
        )
        assert config.name == "test"
        assert config.dtype == "bfloat16"
    
    def test_auto_device_detection(self):
        """Test automatic device type detection."""
        config = DeviceConfig("auto")
        assert config.device_type in ["cuda", "cpu"]
```

**Patterns:**
- Use `pytest` fixtures in `conftest.py` for shared setup
- Use `unittest.mock` or `pytest-mock` for mocking
- Parametrized tests with `@pytest.mark.parametrize` for multiple inputs

## Mocking

**Recommended Framework:** `pytest-mock` or `unittest.mock`

**Patterns:**
```python
from unittest.mock import Mock, patch, MagicMock
import pytest

class TestHarmonySpeechEngine:
    @patch('harmonyspeech.engine.harmonyspeech_engine.CPUExecutor')
    def test_init_with_cpu(self, mock_executor):
        """Test engine initialization with CPU executor."""
        from harmonyspeech.engine.harmonyspeech_engine import HarmonySpeechEngine
        mock_executor.return_value = Mock()
        
        engine = HarmonySpeechEngine(
            model_configs=[],
            log_stats=False
        )
        mock_executor.assert_called()
```

**What to Mock:**
- External API calls (HuggingFace model downloads)
- GPU/torch operations (for CPU-only testing)
- File I/O operations
- Network requests

**What NOT to Mock:**
- Core business logic that needs validation
- Configuration parsing (test with real configs)
- Model loading (integration tests)

## Fixtures and Factories

**Test Data:**
```python
# conftest.py
import pytest
import torch
import numpy as np

@pytest.fixture
def sample_audio_tensor():
    """Generate sample audio tensor for testing."""
    return torch.randn(1, 16000)  # 1 second of audio at 16kHz

@pytest.fixture
def model_config_cpu():
    """Create a test model config for CPU."""
    from harmonyspeech.common.config import ModelConfig, DeviceConfig
    return ModelConfig(
        name="test-tts",
        model="test/model",
        model_type="tts",
        max_batch_size=1,
        device_config=DeviceConfig("cpu")
    )
```

**Location:**
- `tests/conftest.py` for shared fixtures
- `tests/fixtures/` for static test data (audio files, config samples)

## Coverage

**Requirements:** Not currently enforced

**Recommended Target:**
- Minimum 80% coverage for core modules
- 100% coverage for `common/` utilities
- Focus on `engine/`, `processing/`, and `modeling/` modules

**View Coverage:**
```bash
pytest --cov=harmonyspeech --cov-report=html --cov-report=term
```

## Test Types

**Unit Tests:**
- Test individual classes and functions in isolation
- Focus on `common/`, `engine/args_tools.py`, `processing/scheduler.py`
- Mock external dependencies

**Integration Tests:**
- Test full request/response cycle
- Test CLI commands
- Test model loading (may require GPU or large downloads)
- Location: `tests/integration/`

**E2E Tests:**
- Not currently implemented
- Could use `pytest` with subprocess to test CLI
- Could test against running server with `requests` library

## Async Testing

**Note:** The codebase uses both sync and async patterns

**Sync Testing:**
- Standard pytest (default)

**Async Testing (if needed):**
```python
import pytest
import asyncio

@pytest.mark.asyncio
async def test_async_inference():
    """Test async inference methods."""
    # Test async code
    result = await some_async_function()
    assert result is not None
```

Requires `pytest-asyncio` package.

## Known Testing Gaps

**Critical Missing Tests:**
1. No tests for `HarmonySpeechEngine` class
2. No tests for model loading/initialization
3. No tests for request/response handling
4. No tests for CLI argument parsing
5. No tests for configuration validation

**Priority Areas for Test Coverage:**
1. `harmonyspeech/engine/harmonyspeech_engine.py` - Core engine
2. `harmonyspeech/common/config.py` - Configuration parsing
3. `harmonyspeech/processing/scheduler.py` - Request scheduling
4. `harmonyspeech/endpoints/openai/` - API endpoints

---

*Testing analysis: 2026-02-28*

# Coding Conventions

_Last updated: 2026-03-11_

## Overview

This document outlines the coding standards and conventions used in the Harmony Speech Engine codebase. All code must pass black formatting and flake8 linting before being committed.

## Python Version and Environment

**Required:** Python 3.12+
- Enforced in `pyproject.toml`: `requires-python = ">=3.12"`
- All type hints should use modern Python syntax (PEP 585 generics, `Self` for methods)

## Code Formatting

**Formatter:** black
- Configuration in `pyproject.toml`:
  ```toml
  [tool.black]
  line-length = 100
  target-version = ["py312"]
  ```
- Run formatting: `black harmonyspeech/ tests/`
- Check formatting without changes: `black --check --diff harmonyspeech/ tests/`

**Key Settings:**
- Line length: 100 characters
- No trailing commas in single-element tuples
- Use implicit string concatenation where appropriate

## Linting

**Linter:** flake8
- Configuration in `pyproject.toml`:
  ```toml
  [tool.flake8]
  max-line-length = 100
  extend-ignore = ["E203", "W503"]
  exclude = [".git", "__pycache__", "models/", ".planning/"]
  ```
- Run linting: `flake8 harmonyspeech/ tests/`
- Key ignores:
  - E203: whitespace before ':' (conflicts with black)
  - W503: line break before binary operator (conflicts with black)

## Naming Patterns

### Files
- **Modules:** snake_case: `harmonyspeech_engine.py`, `cpu_executor.py`, `args_tools.py`
- **Test files:** `test_*.py` prefix (pytest discovery)
- **Package directories:** Use `__init__.py` for package exports

### Functions
- **Public functions:** snake_case: `init_custom_executors()`, `get_loading_progress_bar()`, `unwrap()`
- **Private functions:** Prefix with underscore: `_log_formatter()`, `_internal_helper()`
- **Async functions:** Prefix with `a` or use async def: `async def generate()`

### Variables
- **Standard variables:** snake_case: `model_configs`, `log_stats`, `local_logging_interval_sec`
- **Constants:** UPPER_SNAKE_CASE: `HARMONYSPEECH_LOG_LEVEL`, `DEFAULT_TIMEOUT_SEC`

### Classes and Types
- **Classes:** PascalCase: `HarmonySpeechEngine`, `ModelConfig`, `DeviceConfig`, `EngineRequest`
- **Dataclasses:** Used for configuration objects: `@dataclass` decorator
- **Type aliases:** Use TypeAlias: `Tensor = torch.Tensor`

## Type Hints

**Usage:** Extensive throughout the codebase
- All function parameters and return values must have type hints
- Use `Optional` for nullable types: `Optional[str]`
- Use `Union` for multiple types: `Union[str, None]`
- Use `List`, `Dict`, `Tuple` from typing (or PEP 585 generics: `list[str]`)
- Example from `harmonyspeech/common/config.py`:
  ```python
  from typing import Optional, Union, List
  
  def __init__(
      self,
      name: str,
      model: str,
      model_type: str,
      max_batch_size: int,
      device_config: DeviceConfig,
      language: Optional[str] = None,
      voices: Optional[List[str]] = None,
  ) -> None:
  ```

## Import Organization

**Standard Order:**
1. Standard library: `import os`, `import time`, `from typing import ...`
2. Third-party packages: `from loguru import logger`, `import torch`, `import yaml`
3. Local modules: `from harmonyspeech.common.config import ...`, `from harmonyspeech.engine...`

**Path Aliases:**
- No path aliases configured
- Use full package paths: `harmonyspeech.common`, `harmonyspeech.engine`

**Avoid circular imports:**
- Import at module level when possible
- Use deferred imports inside functions when circular dependency exists

## Code Style

### Indentation
- 4 spaces (no tabs)
- Python 3.12+ enforces this

### Docstrings
- Use Google-style docstrings for classes and public methods
- Example from `harmonyspeech/engine/harmonyspeech_engine.py`:
  ```python
  class HarmonySpeechEngine:
      """
      An Inference Engine for AI Speech that receives requests and generates outputs.
      """
  ```

### Inline Comments
- Minimal inline comments; code should be self-explanatory
- Use TODO comments sparingly: `# TODO(issue): description`

## Error Handling

**Patterns:**
- Use exceptions with descriptive messages:
  ```python
  raise RuntimeError("No supported device detected.")
  raise ValueError(f"Unknown dtype: {dtype}")
  ```
- Use `loguru` for runtime errors and info:
  ```python
  from loguru import logger
  logger.error(f"Failed to load model: {e}")
  ```
- Avoid bare `except:` — catch specific exceptions

**Logging:**
- Framework: `loguru` (imported as `from loguru import logger`)
- Environment variable for log level: `HARMONYSPEECH_LOG_LEVEL` (default: "INFO")
- Example from `harmonyspeech/common/logger.py`:
  ```python
  from loguru import logger
  LOG_LEVEL = os.getenv("HARMONYSPEECH_LOG_LEVEL", "INFO").upper()
  ```

## Async/Await Patterns

**Async Functions:**
- Use `async def` for async functions
- Use `await` for all async calls
- Use `asyncio` for async utilities

**Testing Async:**
- Use `pytest-asyncio` with `asyncio_mode = "auto"` in `pyproject.toml`
- Mark async tests with `@pytest.mark.asyncio` (or let auto mode handle it)

## Function Design

### Size
- Functions should be focused and single-purpose
- Large classes (e.g., `HarmonySpeechEngine`) contain multiple methods but each method should be focused

### Parameters
- Type hints on all parameters
- Default values provided where appropriate
- Example: `def __init__(self, model_configs: Optional[List[ModelConfig]], log_stats: bool):`

### Return Values
- Type hints on return values
- Use `Optional` for nullable returns

## Module Design

### Exports
- Centralized in `__init__.py` files using `__all__`:
  ```python
  __all__ = [
      "__commit__",
      "__short_commit__",
      "__version__",
      "HarmonySpeechEngine"
  ]
  ```

### Barrel Files
- Use `__init__.py` for package-level exports
- Re-export in `harmonyspeech/common/__init__.py`, `harmonyspeech/engine/__init__.py`

## Configuration Patterns

### Environment Variables
- Use `os.getenv()` with defaults:
  ```python
  _LOCAL_LOGGING_INTERVAL_SEC = int(os.environ.get("HARMONYSPEECH_LOCAL_LOGGING_INTERVAL_SEC", "5"))
  ```
- Pattern: `os.environ.get("HARMONYSPEECH_<FEATURE>", "default_value")`

### YAML Configuration
- Uses `yaml` library for config file parsing
- Config files: `config.yml`, `config.gpu.yml`

### Dataclasses
- Use `@dataclass` for configuration objects
- Example from `harmonyspeech/common/config.py`:
  ```python
  @dataclass
  class ModelConfig:
      name: str
      model: str
      model_type: str
      max_batch_size: int
      device_config: DeviceConfig
  ```

## CLI Patterns

**Argument Parsing:**
- Uses `argparse` for CLI
- Subcommands via `subparsers`
- Example from `harmonyspeech/endpoints/cli.py`:
  ```python
  parser = argparse.ArgumentParser(description="Harmony Speech Engine CLI")
  subparsers = parser.add_subparsers()
  serve_parser = subparsers.add_parser("run", help="...")
  ```

## Testing Conventions

Tests follow the same formatting and linting rules as production code.

### Test File Structure
- Location: `tests/unit/`, `tests/integration/`, `tests/e2e/`
- Naming: `test_*.py` for test modules
- Use `pytest` with fixtures from `conftest.py`

### Test Patterns
- Use `pytest` fixtures for shared setup
- Use `unittest.mock` or `pytest-mock` for mocking
- Use `@pytest.mark.parametrize` for multiple inputs

---

_Last updated: 2026-03-11_
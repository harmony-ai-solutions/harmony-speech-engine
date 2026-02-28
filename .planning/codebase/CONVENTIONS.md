# Coding Conventions

**Analysis Date:** 2026-02-28

## Naming Patterns

**Files:**
- snake_case: `harmonyspeech_engine.py`, `cpu_executor.py`, `args_tools.py`
- Modules use `__init__.py` for package exports

**Functions:**
- snake_case: `init_custom_executors()`, `get_loading_progress_bar()`, `unwrap()`
- Private functions prefixed with underscore: `_log_formatter()`

**Variables:**
- snake_case: `model_configs`, `log_stats`, `local_logging_interval_sec`
- Constants: UPPER_SNAKE_CASE: `HARMONYSPEECH_LOG_LEVEL`

**Types/Classes:**
- PascalCase: `HarmonySpeechEngine`, `ModelConfig`, `DeviceConfig`, `EngineRequest`
- Dataclasses used for configuration objects

## Code Style

**Formatting:**
- No explicit formatter configuration found (no `.prettierrc`, `pyproject.toml` formatter section)
- Uses standard Python indentation (4 spaces)

**Linting:**
- No explicit linter configuration found (no `.pylintrc`, `pyproject.toml` linting section)
- Python 3.12+ required (enforced in `pyproject.toml`)

**Type Hints:**
- Extensive use of type hints throughout codebase
- Examples from [`harmonyspeech/common/config.py`](harmonyspeech/common/config.py:1):
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
  ```

## Import Organization

**Order (observed pattern):**
1. Standard library: `import os`, `import time`, `from typing import ...`
2. Third-party packages: `from loguru import logger`, `import torch`, `import yaml`
3. Local modules: `from harmonyspeech.common.config import ...`, `from harmonyspeech.engine...`

**Path Aliases:**
- No path aliases configured (uses relative imports within package)
- Full package paths: `harmonyspeech.common`, `harmonyspeech.engine`, etc.

## Error Handling

**Patterns:**
- Uses exceptions with descriptive messages: `raise RuntimeError("No supported device detected.")`
- Logging via `loguru` for runtime errors and info
- No try/except patterns observed in core files

**Logging:**
- Framework: `loguru` (imported as `from loguru import logger`)
- Console output: `rich` library for formatted console output
- Environment variable for log level: `HARMONYSPEECH_LOG_LEVEL` (default: "INFO")
- Example from [`harmonyspeech/common/logger.py`](harmonyspeech/common/logger.py:1):
  ```python
  from loguru import logger
  LOG_LEVEL = os.getenv("HARMONYSPEECH_LOG_LEVEL", "INFO").upper()
  ```

## Comments

**When to Comment:**
- Docstrings on classes and public methods
- Example from [`harmonyspeech/engine/harmonyspeech_engine.py`](harmonyspeech/engine/harmonyspeech_engine.py:21):
  ```python
  class HarmonySpeechEngine:
      """
      An Inference Engine for AI Speech that receives requests and generates outputs.
      """
  ```

**Inline Comments:**
- Minimal inline comments; code is generally self-explanatory
- TODO comments present (e.g., in engine initialization)

## Function Design

**Size:**
- Functions tend to be focused and single-purpose
- Large classes (e.g., `HarmonySpeechEngine` is 647 lines) contain multiple methods

**Parameters:**
- Type hints on all parameters
- Default values provided where appropriate
- Example: `def __init__(self, model_configs: Optional[List[ModelConfig]], log_stats: bool):`

**Return Values:**
- Type hints on return values
- Uses `Optional` for nullable returns

## Module Design

**Exports:**
- Centralized in `__init__.py` files
- Example from [`harmonyspeech/__init__.py`](harmonyspeech/__init__.py:1):
  ```python
  __all__ = [
      "__commit__",
      "__short_commit__",
      "__version__",
      "HarmonySpeechEngine"
  ]
  ```

**Barrel Files:**
- Uses `__init__.py` for package-level exports
- Re-exports in `harmonyspeech/common/__init__.py`, `harmonyspeech/engine/__init__.py`, etc.

## Configuration Patterns

**Environment Variables:**
- Uses `os.getenv()` with defaults
- Pattern: `os.environ.get("HARMONYSPEECH_<FEATURE>", "default_value")`
- Example from [`harmonyspeech/engine/harmonyspeech_engine.py`](harmonyspeech/engine/harmonyspeech_engine.py:18):
  ```python
  _LOCAL_LOGGING_INTERVAL_SEC = int(os.environ.get("HARMONYSPEECH_LOCAL_LOGGING_INTERVAL_SEC", "5"))
  ```

**YAML Configuration:**
- Uses `yaml` library for config file parsing
- Config files: `config.yml`, `config.gpu.yml`

## CLI Patterns

**Argument Parsing:**
- Uses `argparse` for CLI
- Subcommands via `subparsers`
- Example from [`harmonyspeech/endpoints/cli.py`](harmonyspeech/endpoints/cli.py:1):
  ```python
  parser = argparse.ArgumentParser(description="Harmony Speech Engine CLI")
  subparsers = parser.add_subparsers()
  serve_parser = subparsers.add_parser("run", help="...")
  ```

---

*Convention analysis: 2026-02-28*

# Coding Conventions

**Analysis Date:** 2026-02-28

## Naming Patterns

**Files:**
- Snake case for all Python files: `harmonyspeech/common/logger.py`, `harmonyspeech/endpoints/openai/api_server.py`.
- Package names are snake case: `harmonyspeech`, `modeling`, `task_handler`.

**Functions:**
- Snake case for functions and methods: `setup_logger()`, `create_speech()`, `model_dump()`.
- Protected/Internal methods use a single leading underscore: `_log_formatter()`, `_force_log()`.

**Variables:**
- Snake case for local variables and parameters: `status_code`, `engine_args`, `request`.
- UPPER_SNAKE_CASE for constants: `LOG_LEVEL`, `SERVICE_NAME`, `UVICORN_LOG_CONFIG`.
- Global variables (if any) follow snake case: `engine`, `openai_serving_tts`.

**Types:**
- PascalCase for classes: `EngineConfig`, `AsyncHarmonySpeech`, `OpenAIServingTextToSpeech`.
- PascalCase for Pydantic models: `ErrorResponse`, `TextToSpeechRequest`, `AudioDataResponse`.

## Code Style

**Formatting:**
- PEP8 compliant style observed throughout the codebase.
- Indentation: 4 spaces.
- Line Length: Not strictly enforced by config but generally around 80-100 characters.

**Linting:**
- Not detected (no `.flake8`, `.pylintrc`, or `ruff.toml` in root).
- `pyproject.toml` exists but does not specify linting tools.

## Import Organization

**Order:**
1. Standard library imports: `import asyncio`, `import os`.
2. Third-party library imports: `import fastapi`, `from loguru import logger`.
3. Local/Internal imports: `from harmonyspeech.common.config import EngineConfig`.
- Multi-line imports use parentheses: `from fastapi.responses import (HTMLResponse, JSONResponse, Response, StreamingResponse)`.

**Path Aliases:**
- Not detected. Absolute imports within the `harmonyspeech` package are preferred.

## Error Handling

**Patterns:**
- Extensive use of `try/except` blocks in API endpoints: `harmonyspeech/endpoints/openai/api_server.py`.
- Return of specialized `ErrorResponse` pydantic models for API errors.
- Internal errors are often caught and logged via `loguru` before returning a 500 status code.

## Logging

**Framework:**
- `loguru` is the primary logging framework: `harmonyspeech/common/logger.py`.
- `rich` is used for console formatting and progress bars: `harmonyspeech/common/logger.py`.

**Patterns:**
- Custom `_log_formatter` in `harmonyspeech/common/logger.py` handles color-coded output based on log levels.
- `logger.log_once()` utility provided for deduplicating repeated log messages.
- Uvicorn logs are intercepted and routed through `loguru`.

## Comments

**When to Comment:**
- Descriptive comments for non-obvious logic.
- TODOs for pending features: `TODO: Add other Endpoint serving classes here`.
- References to external documentation/APIs: `# Based on: https://platform.openai.com/docs/api-reference/audio/createSpeech`.

**JSDoc/TSDoc:**
- Triple-quote docstrings for classes and functions.
- Generally a simplified descriptive style, occasionally including "Inspiration taken from..." or "Based on...".

## Function Design

**Size:**
- Functions are generally concise and focused on a single responsibility.
- Large setup functions like `run_server` are used for bootstrapping the application.

**Parameters:**
- Extensive use of type hints for parameters.
- FastAPI dependency injection patterns for endpoint parameters: `x_api_key: Optional[str] = Header(None)`.

**Return Values:**
- Extensive use of type hints for return values.
- API endpoints return `JSONResponse` or `StreamingResponse`.

## Module Design

**Exports:**
- Modules expose functionality via classes and functions.
- `__init__.py` files are present in most directories to establish packages.

**Barrel Files:**
- `__init__.py` often exports version info or key classes from submodules.

---

*Convention analysis: 2026-02-28*

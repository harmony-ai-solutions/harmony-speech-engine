<!-- refreshed: 2026-11-02 -->
# Coding Conventions

**Analysis Date:** 2026-11-02

## Language and Runtime

**Required:** Python 3.12+
- Declared in `pyproject.toml`: `requires-python = ">=3.12"`
- Use PEP 604 unions (`str | None`) and built-in generics (`list[str]`, `dict[str, int]`) in new code.
- Legacy files still use `typing.Optional` / `typing.List` / `typing.Dict` (for example `harmonyspeech/modeling/models/__init__.py`). Match the surrounding file; prefer the modern form for new modules.

## Code Style

**Formatter and linter: ruff** (single tool, configured in `pyproject.toml`).

```toml
[tool.ruff]
line-length = 120
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "N", "UP", "B", "C4", "SIM"]
ignore = ["E501"]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
skip-magic-trailing-comma = true
line-ending = "auto"
```

Commands (same as CI, `.github/workflows/test.yml`):

```bash
ruff format harmonyspeech/ tests/
ruff format --check --diff harmonyspeech/ tests/
ruff check harmonyspeech/ tests/
```

**Rules of thumb:**
- Line length target is 120; `E501` is ignored, so long strings/URLs may exceed it, but keep code readable.
- Double quotes everywhere (`quote-style = "double"`).
- 4-space indentation, no tabs.
- Imports are sorted by ruff's `I` rules: standard library, third-party, then first-party `harmonyspeech`.
- Do not add magic trailing commas if it changes formatting unexpectedly (`skip-magic-trailing-comma = true`).

**Legacy tooling:** `requirements-common.txt` still lists `black>=24.0.0` and `flake8>=7.0.0`, and `docs/testing.md` still describes a black/flake8 lint job. CI no longer runs them — `ruff` is authoritative. Do not introduce black/flake8-specific `# noqa` codes or formatting.

## Naming Patterns

### Files and Directories
- Python modules: `snake_case` — `harmonyspeech_engine.py`, `cpu_executor.py`, `args_tools.py`.
- Packages: `snake_case` — `harmonyspeech/task_handler/`, `harmonyspeech/endpoints/openai/`, `harmonyspeech/modeling/models/chatterbox/`.
- Test modules: `test_*.py`.
- Configuration files: `config.yml`, `config.gpu.yml`; dependency sets: `requirements-<variant>.txt`.
- Markdown docs: `lowercase-with-hyphens.md` under `docs/` (`docs/testing.md`, `docs/routing.md`).

### Functions and Methods
- Public functions/methods: `snake_case` — `prepare_chatterbox_tts_inputs()`, `reroute_request_chatterbox()`, `check_forward_processing()`.
- Private helpers: leading underscore — `_execute_chatterbox_tts()`, `_build_error_result()`, `_build_result()`, `_log_formatter()`, `_patched_torch_load()`.
- Model integration naming is systematic:
  - input preparation: `prepare_<model_family>_inputs()` in `harmonyspeech/task_handler/inputs.py`
  - execution: `_execute_<model_family>()` in `harmonyspeech/task_handler/model_runner_base.py`
  - routing: `reroute_request_<model_family>()` in `harmonyspeech/engine/harmonyspeech_engine.py`
- Test helpers: leading underscore — `_make_tts_request()`, `_make_engine()`, `_make_model_config()`.
- Async functions use `async def`; do not use an `a` prefix.

### Variables and Constants
- Variables: `snake_case` — `model_configs`, `log_stats`, `local_logging_interval_sec`.
- Module constants: `UPPER_SNAKE_CASE` — `_CHATTERBOX_TURBO_REPO_ID`, `KITTENTTS_VOICES`, `TEXT_INPUT`, `REFERENCE_AUDIO`.
- Local dispatch tables use `UPPER_SNAKE_CASE` inside the method when scoped there, e.g. `_CHATTERBOX_TTS_TYPE_MAP` in `harmonyspeech/engine/harmonyspeech_engine.py:294`.
- Model-type registry keys are PascalCase strings matching class names: `"ChatterboxTTS"`, `"ChatterboxTurboTTS"`, `"ChatterboxMultilingualTTS"`, `"ChatterboxEmbedding"`, `"ChatterboxVC"` (`harmonyspeech/modeling/models/__init__.py`).
- User-facing sentinel model IDs are lowercase with underscores: `chatterbox`, `chatterbox_turbo`, `chatterbox_multilingual`, `chatterbox_vc` (mapped in `harmonyspeech/engine/harmonyspeech_engine.py`).

### Classes and Types
- Classes: `PascalCase` — `HarmonySpeechEngine`, `ModelConfig`, `ModelRunnerBase`.
- Internal request/option data: `@dataclass` in `harmonyspeech/common/inputs.py` (e.g. `TextToSpeechGenerationOptions`).
- API request/response schemas: Pydantic `BaseModel` in `harmonyspeech/endpoints/openai/protocol.py`.
- Upstream model wrapper classes: `<Name>Model` with a `from_pretrained` classmethod — `ChatterboxTTSModel`, `ChatterboxTurboTTSModel`, `ChatterboxMultilingualTTSModel`, `ChatterboxVCModel` in `harmonyspeech/modeling/models/chatterbox/chatterbox.py`.

## Type Hints

- New public functions and methods should annotate parameters and returns; use `X | None` for optional values.
- Internal dataclasses in `harmonyspeech/common/inputs.py` annotate every field, e.g.:
  ```python
  @dataclass
  class TextToSpeechGenerationOptions:
      seed: int | None
      style: int | None
      exaggeration: float | None = None
      cfg_weight: float | None = None
      top_k: int | None = None
      norm_loudness: bool | None = None
  ```
- Pydantic fields use `Field(default, description="...")` for anything exposed through the OpenAI-compatible API (`harmonyspeech/endpoints/openai/protocol.py`).
- Some older helpers omit return annotations (e.g. `prepare_*_inputs()` in `harmonyspeech/task_handler/inputs.py` returns an untyped list). When editing such a function, add annotations only if it does not create churn.
- Do not add `from __future__ import annotations` — Python 3.12 is required.

## Import Organization

Order (enforced by ruff `I`):
1. Standard library — `import io`, `import base64`, `from pathlib import Path`
2. Third-party — `import torch`, `import numpy as np`, `from loguru import logger`, `from chatterbox.tts import Conditionals`
3. First-party — `from harmonyspeech.common.config import ModelConfig`
4. Test-only imports of fixtures/helpers — `from tests.e2e.conftest import load_sample_audio_b64`

**Path aliases:** none. Always use full package paths (`harmonyspeech.common`, `harmonyspeech.task_handler`).

**Circular imports:** avoid module-level cycles by deferring imports inside functions when needed. Example: `tests/integration/test_chatterbox_flow.py` imports serving handlers inside fixtures; `tests/e2e/conftest.py` imports engine classes inside the `chatterbox_engine` fixture.

## Error Handling

**Validation errors use `ValueError` with a specific message** that names the offending option or field. This is relied on by tests via `pytest.raises(ValueError, match="top_k")`:

```python
if opts.top_k is not None:
    raise ValueError("top_k is not supported by ChatterboxTTS")
if getattr(request, "input_audio", None) is not None and getattr(request, "input_embedding", None) is not None:
    raise ValueError("Provide either input_audio or input_embedding, not both.")
```

(`harmonyspeech/task_handler/inputs.py:652-882`)

**Never let a per-request inference failure kill the engine loop.** In `harmonyspeech/task_handler/model_runner_base.py`:
- `execute_model()` wraps `prepare_inputs()` in `try/except Exception` and returns `[self._build_error_result(req, e) for req in requests_to_batch]`.
- Each `_execute_*` method wraps its per-item inference call in `try/except Exception` and builds an error result with `_build_error_result()`.
- `_build_error_result()` logs the full traceback with `logger.opt(exception=exception).error(...)` and returns `RequestOutput(finish_reason="error", error=str(exception))`.
- `HarmonySpeechEngine.check_forward_processing()` (`harmonyspeech/engine/harmonyspeech_engine.py:363`) checks `finish_reason == "error"` and returns `RequestStatus.FINISHED_STOPPED` without forwarding.

**Upstream exceptions:** convert third-party validation `AssertionError` into `ValueError` so the serving layer returns a 400 instead of crashing:

```python
except AssertionError as e:
    raise ValueError(str(e)) from e
```

(`harmonyspeech/task_handler/model_runner_base.py:823-830`)

**Serving layer:** catch `ValueError` and return `ErrorResponse` via `self.create_error_response(...)`; see `harmonyspeech/endpoints/openai/serving_text_to_speech.py:63-94`.

**Do not use bare `except:`.** Catch `Exception` only at the per-request boundary described above; elsewhere catch specific exceptions.

## Logging

**Framework:** `loguru`, imported as `from loguru import logger`.

- Module setup and rich console integration: `harmonyspeech/common/logger.py`.
- Log level: `HARMONYSPEECH_LOG_LEVEL`, default `"INFO"` (`harmonyspeech/common/logger.py:14`).
- Include the stack trace when logging a caught exception:
  ```python
  logger.opt(exception=e).error(f"Inference error for request {request_id}: {e}")
  ```
- Use `logger.warning()` for degraded/fallback paths, `logger.info()` for lifecycle events.
- Do not use `print()` in library/server code; CLI help output is handled by argparse.
- `log_once()` and `logger.log_once` exist for repeated warnings (`harmonyspeech/common/logger.py:76-79`).

## Comments and Docstrings

**Docstrings:**
- Every module starts with a one-to-three-line docstring describing responsibility, for example `harmonyspeech/modeling/models/chatterbox/chatterbox.py` and `tests/e2e/tts/test_chatterbox_turbo.py`.
- Classes and non-trivial public methods use short summary docstrings. Model wrappers use Google-style `Args:` / `Returns:` sections (`ChatterboxTTSModel.from_pretrained`).
- Test docstrings state the invariant or regression they protect, not just the function name. Newer Chatterbox tests cite requirement IDs and upstream error text:
  ```python
  """Turbo embedding with audio < 5s returns 400 ErrorResponse; engine survives to serve next request.

  Regression test for:
    AssertionError: "Audio prompt must be longer than 5 seconds!"
  """
  ```
  (`tests/e2e/tts/test_chatterbox_turbo.py:88-96`)

**Inline comments:** use them to explain *why*, especially for third-party workarounds. Accepted example: the module-level monkey-patch blocks in `harmonyspeech/modeling/models/chatterbox/chatterbox.py:22-76` each carry a block comment describing the upstream bug and the fallback behavior.

**Task markers:** `FIXME` is used for known correctness gaps (`harmonyspeech/task_handler/model_runner_base.py:163` — `# FIXME: This is not properly batched`). Do not add new `TODO` without an owner/issue context.

## Function Design

- Keep per-model execution methods focused on one model type and follow the established shape:
  1. define a nested `synthesize` / `compute_embedding` / `convert` closure,
  2. iterate `for i, x in enumerate(inputs)`,
  3. pair with `requests_to_batch[i]`,
  4. wrap the call in `try/except Exception` and append either `_build_result()` or `_build_error_result()`.
- Batched input preparation uses a nested `prepare(request)` closure plus `ThreadPoolExecutor().map(...)` (`harmonyspeech/task_handler/inputs.py:646-882`). Preserve ordering: `executor.map` returns results in input order.
- Prefer explicit tuple returns with a documented tuple layout in the docstring (e.g. `(input_text, language_id, conditionals, ...)`).
- Avoid optional mutable defaults; use `None` and apply model defaults inside the prepare function (Chatterbox pattern at `harmonyspeech/task_handler/inputs.py:686-694`).

## Module Design

- Package exports live in `__init__.py` with `__all__`:
  ```python
  from .chatterbox import (
      ChatterboxMultilingualTTSModel,
      ChatterboxTTSModel,
      ChatterboxTurboTTSModel,
      ChatterboxVCModel,
  )

  __all__ = ["ChatterboxTTSModel", "ChatterboxTurboTTSModel", "ChatterboxMultilingualTTSModel", "ChatterboxVCModel"]
  ```
  (`harmonyspeech/modeling/models/chatterbox/__init__.py`)
- Re-export public engine types from `harmonyspeech/engine/__init__.py` and `harmonyspeech/common/__init__.py`.
- No barrel file for every subpackage; only add an `__init__.py` re-export when callers outside the package need it.

## Model Integration Conventions

Adding or changing a model family touches these locations in this order:

1. **Registry:** add the architecture string and module/class to `_MODELS` in `harmonyspeech/modeling/models/__init__.py` (use `"native"` for models implemented directly by HSE).
2. **Wrapper:** add `<Name>Model` with `from_pretrained(cls, device: str | torch.device | None = None, **kwargs)` in `harmonyspeech/modeling/models/<family>/`.
3. **Input preparation:** add `prepare_<family>_inputs()` and dispatch it from `prepare_inputs()` in `harmonyspeech/task_handler/inputs.py`.
4. **Execution:** add `_execute_<family>()` and dispatch it from `execute_model()` in `harmonyspeech/task_handler/model_runner_base.py`.
5. **Routing:** add/extend `reroute_request_<family>()` and `check_reroute_request_to_model()` in `harmonyspeech/engine/harmonyspeech_engine.py`.
6. **Serving:** register models via `models_from_config()` in the relevant `harmonyspeech/endpoints/openai/serving_*.py`.
7. **Tests:** add unit tests (routing, prepare functions, registry), an integration flow test, and an E2E test with a fixture in `tests/e2e/conftest.py`.

**Chatterbox specifics to preserve:**
- Chatterbox request handling must be filesystem-free. Use `io.BytesIO` for audio and `torch.save` into a `BytesIO` buffer for `Conditionals`; never `tempfile` or `open()` (`harmonyspeech/task_handler/model_runner_base.py:677-873`; guard test `tests/unit/inference_flow/test_chatterbox_no_tempfile.py`).
- `Conditionals` are base64-encoded and serialized in-memory. Embedding output is a base64 `Conditionals` payload (`_execute_chatterbox_embedding`).
- Each Chatterbox variant accepts a distinct generation-option subset; reject non-`None` unsupported fields with `ValueError` instead of silently ignoring them. Variant defaults live in the prepare functions, not in the API schema.
- Supported multilingual language codes are exposed as `ChatterboxMultilingualTTSModel.SUPPORTED_LANGUAGES` and must contain 23 entries (`tests/unit/inference_flow/test_chatterbox_inputs.py`).

## Async/Await Patterns

- `async def` for async paths; `await` every coroutine.
- `asyncio_mode = "auto"` in `pyproject.toml`, so `@pytest.mark.asyncio` is optional for tests. Existing integration tests still declare it; either style is accepted.
- Sync E2E tests invoke async serving methods with `asyncio.run(...)`, one call per test (`tests/e2e/tts/test_chatterbox.py`).
- Do not run blocking inference on the event loop; the engine uses background loops (`harmonyspeech/engine/async_harmonyspeech.py`).

## Configuration Patterns

- Environment variables use the `HARMONYSPEECH_` prefix with a default:
  ```python
  LOG_LEVEL = os.getenv("HARMONYSPEECH_LOG_LEVEL", "INFO").upper()
  _LOCAL_LOGGING_INTERVAL_SEC = int(os.environ.get("HARMONYSPEECH_LOCAL_LOGGING_INTERVAL_SEC", "5"))
  ```
- YAML configs: `config.yml` (CPU) and `config.gpu.yml`; parsed with `yaml` and mapped into `EngineConfig`/`ModelConfig` dataclasses (`harmonyspeech/common/config.py`).
- Do not read or commit `.env` files. Local `.env`, `.env-amd`, and `.env.example` exist at the repo root, but code consumes process environment variables, not the files.
- Model downloads honour `HF_HOME` / `HUGGINGFACE_HUB_CACHE`; E2E fixtures rely on this (`tests/e2e/conftest.py:479-496`).

## CLI Patterns

- Argument parsing lives in `harmonyspeech/endpoints/openai/args.py`; build parsers with `make_arg_parser()` so tests can reuse them (`tests/integration/test_cli.py`).
- Use `argparse` subparsers for commands (`harmonyspeech/endpoints/cli.py`).
- Console entry point is declared in `pyproject.toml`: `harmonyspeech = "harmonyspeech.endpoints.cli:main"`.
- Keep CLI defaults in the parser, not in call sites, so `parser.parse_args([])` remains a valid test case.

## Testing Conventions

- Tests follow the same ruff formatting/linting rules as production code.
- Location by tier and category: `tests/unit/<category>/`, `tests/integration/`, `tests/e2e/<task>/`.
- Test names describe behavior: `test_reroute_tts_with_input_audio_to_embed_step`, `test_chatterbox_turbo_embedding_short_audio_returns_error`.
- Prefer flat test functions with fixtures. Class grouping is allowed for a coherent behavior set (`class TestChatterboxNoTempfile` in `tests/unit/inference_flow/test_chatterbox_no_tempfile.py`).
- Mark unit tests with `@pytest.mark.unit`; E2E files declare `pytestmark = [pytest.mark.e2e, pytest.mark.slow]`. E2E directories are also auto-marked by `tests/e2e/conftest.py`.
- Mock only boundaries (HuggingFace download, model objects, serving globals, file I/O); never load real models in unit or integration tests.
- See `.planning/codebase/TESTING.md` for fixtures, mocking patterns, and CI commands.

---

*Convention analysis: 2026-11-02*

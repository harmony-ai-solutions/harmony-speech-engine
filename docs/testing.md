# Testing Guide

This document explains how to run tests for the Harmony Speech Engine, what the test structure looks like, and how to interpret results.

---

## Prerequisites

Install test dependencies (included in `requirements-common.txt`):

```bash
pip install -r requirements-cpu.txt
pip install -r requirements-common.txt
```

---

## Running Tests

### Run all tests (CPU mode, default)

```bash
pytest
```

This runs all tests under `tests/` in CPU-only mode. No GPU is required.

### Run a specific tier

```bash
# Unit tests only (fast, fully mocked)
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# End-to-end tests only (slow — downloads real models)
pytest tests/e2e/
```

### Run by marker

```bash
# Only unit tests (via marker)
pytest -m unit

# Skip slow tests
pytest -m "not slow"

# Only e2e tests
pytest -m e2e
```

### Device and dtype options

The test suite defaults to CPU and float32. Use these flags to override:

| Flag | Default | Values | Description |
|------|---------|--------|-------------|
| `--device` | `cpu` | `cpu`, `cuda` | Target device for tensor operations |
| `--dtype` | `float32` | `float32`, `float16` | Default tensor dtype |

```bash
# Override device (will fail on CPU-only machines if set to cuda)
pytest --device=cuda

# Override dtype
pytest --dtype=float16
```

> **Strict validation:** If `--device=cuda` is passed on a machine without CUDA, the test suite
> will exit immediately with an error. This is by design to prevent silent failures.

---

## Test Structure

```
tests/
├── conftest.py          # Root: shared fixtures, CLI hooks (device, dtype, sample_config)
├── test-data/           # Binary test assets (mock audio files, sample YAML configs)
├── unit/
│   ├── conftest.py      # Unit-specific fixtures (mock_model_loader, mock_hf_downloader)
│   └── test_*.py        # Fast, fully mocked unit tests
├── integration/
│   ├── conftest.py      # Integration fixtures (test_app FastAPI TestClient)
│   └── test_*.py        # Component interaction tests
└── e2e/
    ├── conftest.py      # E2E fixtures (models_cache_dir, e2e marker auto-application)
    └── test_*.py        # Full pipeline tests with real models
```

### Test tiers

| Tier | Speed | Mocking | When to use |
|------|-------|---------|-------------|
| `unit` | Fast (<1s/test) | Full mocking | Logic in isolation — config parsing, data transformations, input validation |
| `integration` | Medium (1–30s) | Partial (no real models) | Component wiring — API endpoints, scheduler, executor dispatch |
| `e2e` | Slow (minutes) | None (real models) | Full inference pipelines — verifies actual model output |

---

## Shared Fixtures

These fixtures are available to all tests via `tests/conftest.py`:

| Fixture | Scope | Description |
|---------|-------|-------------|
| `device` | session | Target device string from `--device` flag |
| `dtype` | session | Target dtype string from `--dtype` flag |
| `tests_root` | session | `Path` to the `tests/` directory |
| `test_data_dir` | session | `Path` to `tests/test-data/` |
| `sample_config` | function | Minimal valid model config dict for testing config parsing |

---

## CI Integration

Tests run automatically via GitHub Actions on every push and pull request:

- **`test` job:** Installs CPU dependencies, runs `pytest --device=cpu`
- **`lint` job:** Runs `black --check` and `flake8` (runs in parallel with tests)
- Pip packages are cached between runs to reduce CI time.

See `.github/workflows/test.yml` for the full workflow definition.

---

## Adding New Tests

- **Unit tests:** Add `tests/unit/test_{module}.py` — import the module under test, mock its dependencies with `mock_model_loader` or `unittest.mock.patch`.
- **Integration tests:** Add `tests/integration/test_{feature}.py` — use the `test_app` fixture.
- **E2E tests:** Add `tests/e2e/test_{model}.py` — use `models_cache_dir` to avoid redundant downloads.

File naming: always prefix with `test_` for pytest discovery.

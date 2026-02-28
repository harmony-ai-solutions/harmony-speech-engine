# Testing Patterns

**Analysis Date:** 2026-02-28

## Test Framework

**Runner:**
- `pytest` (>=7.0.0 planned, as per `memory-bank/techContext.md`)
- Config: `pytest.ini` or `pyproject.toml` (currently minimal configuration)

**Assertion Library:**
- Standard Python `assert` statements used with `pytest`.

**Run Commands:**
```bash
pytest tests/unit/              # Run unit tests
pytest tests/integration/       # Run integration tests
pytest tests/performance/ --benchmark # Run performance benchmarks
pytest tests/ --cov=harmonyspeech # Run tests with coverage
```

## Test File Organization

**Location:**
- Dedicated `tests/` directory at the repository root.
- Planned subdirectories for different test types: `tests/unit/`, `tests/integration/`, `tests/performance/`.

**Naming:**
- Files: `test_*.py` or `*_test.py` (consistent with standard pytest conventions).
- Classes: `Test*`.
- Functions: `test_*`.

**Structure:**
```
[project-root]/
└── tests/
    ├── unit/
    ├── integration/
    ├── performance/
    └── __pycache__/ (contains remnants of `test_example.py`)
```

## Test Structure

**Suite Organization:**
(Planned/Remnants)
```python
# tests/test_example.py (referenced in __pycache__)
def test_example():
    assert True
```

**Patterns:**
- `pytest` fixtures for setup/teardown.
- `@pytest.mark.asyncio` for testing async functions (given extensive use of `async/await` in `harmonyspeech/`).
- `parametrize` for data-driven testing.

## Mocking

**Framework:**
- `unittest.mock` or `pytest-mock` (implied by Python standard practices for `pytest`).

**Patterns:**
- Mocking external AI models and API calls is essential given the engine's focus on inference.
- Use of `AsyncMock` for mocking asynchronous methods like `engine.generate()`.

**What to Mock:**
- External service calls (HuggingFace, etc.).
- Resource-intensive AI model loads in unit tests.
- File system operations.

**What NOT to Mock:**
- Internal utility functions in unit tests.
- Pydantic models for data validation.

## Fixtures and Factories

**Test Data:**
(Planned)
- Audio samples for TTS/STT testing.
- Sample configuration YAML files.
- Mock speaker embeddings in base64.

**Location:**
- Planned `tests/fixtures/` or `tests/data/`.

## Coverage

**Requirements:**
- Tracked via `pytest-cov`.
- Target: High coverage for `harmonyspeech/common/` and `harmonyspeech/endpoints/`.

**View Coverage:**
```bash
pytest tests/ --cov=harmonyspeech --cov-report=html
```

## Test Types

**Unit Tests:**
- Focus on individual components like `harmonyspeech/common/utils.py`, `harmonyspeech/common/logger.py`.

**Integration Tests:**
- Focus on full request-response cycles in `harmonyspeech/endpoints/openai/api_server.py`.
- Validation of `AsyncHarmonySpeech` engine initialization and model loading.

**E2E Tests:**
- Interaction with `vllm` or `ray` clusters (if used).
- Full system health checks via `/health` endpoint.

## Common Patterns

**Async Testing:**
```python
@pytest.mark.asyncio
async def test_async_function():
    result = await my_async_function()
    assert result == expected
```

**Error Testing:**
```python
def test_error_case():
    with pytest.raises(ValueError):
        function_that_raises()
```

---

*Testing analysis: 2026-02-28*

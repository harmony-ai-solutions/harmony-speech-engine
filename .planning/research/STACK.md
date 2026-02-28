# Stack Research

**Domain:** Python Testing Framework
**Researched:** 2026-02-28
**Confidence:** HIGH

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| pytest | 9.0.2 | Core testing framework | Industry standard for Python testing. Already installed in project. Supports fixtures, parametrization, and extensive plugin ecosystem. |
| pytest-asyncio | 1.3.0 | Async test support | Required for testing async engine components. Already installed. Provides automatic async test detection and event loop management. |
| pytest-mock | 3.15.1 | Mocking utilities | Provides `mocker` fixture for cleaner mocking API. Already installed. Integrates seamlessly with pytest. |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pytest-cov | ^6.0 | Coverage reporting | Always - for measuring test coverage across harmonyspeech modules |
| pytest-xdist | ^3.6 | Parallel test execution | When test suite grows - enables `-n auto` for faster CI |
| pytest-timeout | ^2.3 | Test timeout enforcement | Always - prevents hanging tests in CI, use `pytest --timeout=60` |
| pytest-randomly | ^3.16 | Random test ordering | During development - catches order-dependent test failures |
| responses | ^0.25 | HTTP mocking | When testing API endpoints that make external HTTP calls |
| respx | ^0.22 | HTTPX mocking | If using httpx client in tests (future-proofing) |
| freezegun | ^1.5 | Time mocking | When testing time-dependent logic (schedulers, caches) |
| hypothesis | ^6.112 | Property-based testing | For generating edge-case test data automatically |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| GitHub Actions | CI/CD pipeline | Already exists in `.github/workflows/` - add pytest step |
| coverage.py | Coverage analysis | Integrated via pytest-cov, configured in `pyproject.toml` |
| pre-commit | Code quality checks | Add pytest to pre-commit run for staged files |

## Installation

```bash
# Core testing dependencies (already installed)
conda run -n hse pip install pytest==9.0.2 pytest-asyncio==1.3.0 pytest-mock==3.15.1

# Additional recommended dependencies
conda run -n hse pip install pytest-cov pytest-xdist pytest-timeout pytest-randomly

# Optional: advanced testing
conda run -n hse pip install responses freezegun hypothesis
```

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| pytest | unittest | Only if migrating legacy code; pytest is strictly superior |
| pytest-mock | unittest.mock | pytest-mock provides cleaner fixture API |
| pytest-cov | coverage CLI | pytest-cov integrates coverage into test runs seamlessly |
| pytest-xdist | tox | tox is for matrix testing across environments, not parallel execution |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| nose2 | In maintenance mode since 2020, pytest is the de facto standard | pytest |
| testify | Deprecated, less community support than pytest | pytest + pytest-mock |
| pytest-benchmark | Not needed for initial test suite, add later if performance tests required | Skip for MVP |
| allure-pytest | Over-engineered for initial needs, adds CI complexity | pytest-cov is sufficient |
| doctest | Good for documentation but not for comprehensive testing | pytest unit tests |

## Stack Patterns by Variant

**If building initial test infrastructure:**
- Use pytest + pytest-mock + pytest-asyncio (already installed)
- Add pytest-cov for coverage visibility
- Add pytest-timeout to prevent CI hangs

**If CI execution time becomes problematic:**
- Add pytest-xdist for parallel execution
- Use `pytest -n auto` to auto-detect CPU cores

**If testing async model inference:**
- Use pytest-asyncio (already installed)
- Configure `asyncio_mode = "auto"` in pytest.ini
- Use `pytest.mark.asyncio` decorator on async tests

## Version Compatibility

| Package | Compatible With | Notes |
|---------|-----------------|-------|
| pytest 9.0.2 | Python 3.12+ | Current installed version |
| pytest-asyncio 1.3.0 | pytest 8.x, Python 3.9+ | Current installed version |
| pytest-mock 3.15.1 | pytest 8.x, Python 3.8+ | Current installed version |
| pytest-cov ^6.0 | pytest 8.x, coverage 7.x | Add to requirements |
| pytest-xdist ^3.6 | pytest 8.x, Python 3.8+ | Add to requirements |
| pytest-timeout ^2.3 | pytest 8.x, Python 3.8+ | Add to requirements |

## Sources

- **Context7**: pytest, pytest-asyncio, pytest-mock — verified current versions and capabilities
- **Official Documentation**: https://docs.pytest.org/ — confirmed pytest 9.x as current stable
- **Project Context**: `.planning/codebase/TESTING.md` — existing testing recommendations
- **Environment Check**: Verified pytest 9.0.2, pytest-asyncio 1.3.0, pytest-mock 3.15.1 already installed in conda hse environment

---

*Stack research for: Python Testing Framework*
*Researched: 2026-02-28*
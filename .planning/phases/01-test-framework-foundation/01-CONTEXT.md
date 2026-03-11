# Phase 1: Test Framework Foundation - Context

**Gathered:** 2026-03-03
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase delivers the core testing infrastructure for the Harmony Speech Engine. It establishes the `pytest` configuration, directory structure, shared fixtures, and the initial GitHub Actions CI workflow. It ensures that subsequent phases can add unit, integration, and E2E tests to a stable and well-organized environment.

</domain>

<decisions>
## Implementation Decisions

### Test Organization & Data
- **Structure**: Utilize a three-tier directory structure: `tests/unit/`, `tests/integration/`, and `tests/e2e/`.
- **Fixtures**: Use distributed `conftest.py` files. Each sub-directory will have its own `conftest.py` for specialized fixtures, while shared fixtures reside in `tests/conftest.py`.
- **Assets**: Create a root-level `tests-data/` folder to store binary assets like mock audio files and sample configuration files. This keeps the main `tests/` directory focused on code.
- **Mocking Strategy**: 
    - Unit tests must mock model loading and external dependencies (like HuggingFace) to remain fast and offline-capable.
    - E2E tests (Phases 4 & 5) are allowed to download and use real models from HuggingFace.

### CI Workflow (GitHub Actions)
- **Runtime**: Only test on Python 3.12 for now to keep the initial CI fast and simple.
- **Triggers**: Enable CI runs for all pushes to any branch and for all Pull Requests.
- **Performance**: Implement caching for both `pip` dependencies and HuggingFace models to optimize run times.
- **Code Quality**: Run linting (Black, Flake8) as a parallel job in the CI workflow for fast feedback.

### Hardware Testing Strategy
- **Default Device**: Tests will use `cpu` by default.
- **Override Mechanism**: Support custom CLI flags for `pytest` to allow manual overrides:
    - `--device`: (e.g., `pytest --device cuda`)
    - `--dtype`: (e.g., `pytest --dtype float16`)
- **Strict Validation**: If a developer requests a device that is not available (e.g., `cuda` on a CPU-only machine), the test suite must fail strictly rather than falling back silently.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `pyproject.toml`: Already contains an exclusion for `tests*`, which will be updated to configure `pytest`.
- `.github/workflows/`: Contains Docker release workflows; the new testing workflow will follow a similar YAML structure.

### Established Patterns
- The project follows a modular structure (`harmonyspeech/common`, `harmonyspeech/engine`, etc.), which the `tests/unit/` structure should mirror for consistency.
- Standardized model configuration via YAML will be used to create mock configurations for testing.

### Integration Points
- `pytest` CLI hooks will be used to implement the `--device` and `--dtype` flags.
- GitHub Actions environment will be configured to handle the caching of models and dependencies.

</code_context>

<specifics>
## Specific Ideas
- No specific requirements — open to standard approaches using `pytest-asyncio` for the FastAPI/Engine async components.
</specifics>

<deferred>
## Deferred Ideas
- **Performance Testing**: A `tests/performance/` directory was discussed but deferred to a future phase once core functionality is tested.
- **Cross-version Matrix**: Testing on Python 3.8-3.11 is deferred to future stability passes.
</deferred>

---

*Phase: 01-test-framework-foundation*
*Context gathered: 2026-03-03*

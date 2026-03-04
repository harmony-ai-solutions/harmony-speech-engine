# Phase 2: Unit Testing Core Components - Context

**Gathered:** 2026-03-03
**Status:** Ready for planning

<domain>
## Phase Boundary

Create fast, isolated unit tests for core engine components, including config parsing, engine initialization, and model loading logic. Deliverables include unit test files in `tests/unit/` that provide comprehensive coverage while maintaining high execution speed through minimal mocking.

</domain>

<decisions>
## Implementation Decisions

### Mock Strategy
- **Minimal Mocking**: Prefer real logic for config parsing and validation. Mock only external dependencies like GPU hardware, model downloads (HuggingFace Hub), and heavy executors.
- **Inline Mocks**: Define mocks directly within test files using `pytest-mock` (the `mocker` fixture) for clarity and locality.
- **Executor Mocking**: Mock `CPUExecutor` and `GPUExecutor` to verify engine-to-executor coordination without running actual inference.
- **Async Handling**: Use `pytest-asyncio` in auto mode. Mark async tests with `@pytest.mark.asyncio`.

### Test Coverage Scope
- **Comprehensive Coverage**: Target all public methods, edge cases, and error handling paths in `config.py`, `engine/`, and `loader.py`.
- **Config Validation**: Test invalid YAML syntax, missing required fields, and invalid field values (e.g., bad device types, negative batch sizes).
- **Engine Propagation**: Verify that configuration (batch size, dtype, device) flows correctly from the engine to the mocked executors by calling `infer/tts` methods with simple inputs.
- **Model Loader**: Test model path resolution, caching logic, and fallback behaviors with mocked HF API.

### Test Organization
- **By Test Type**: Group tests by behavior rather than strictly matching the `harmonyspeech/` file structure.
  - `tests/unit/test_initialization.py`: Covers engine and executor setup.
  - `tests/unit/test_inference_flow.py`: Covers config propagation and parameter handling.
  - `tests/unit/test_error_handling.py`: Covers config validation and loading failures.

### Test Data
- **Minimal Data**: Use string and dictionary inputs directly in code where possible to keep tests fast and self-contained. Avoid creating physical test data files unless strictly necessary for file I/O testing.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `tests/unit/conftest.py`: Contains `mock_model_loader` and `mock_hf_downloader` fixtures that can be leveraged or referenced.
- `pyproject.toml`: Already configured with `[tool.pytest.ini_options]` from Phase 1.

### Established Patterns
- **Three-Tier Testing**: The project uses `tests/unit`, `tests/integration`, and `tests/e2e` separation.
- **CLI Hooks**: `tests/conftest.py` has hooks for `--device` and `--dtype` which should be respected in unit tests if they interact with global state (though unit tests should ideally be isolated).

### Integration Points
- `harmonyspeech/common/config.py`: Primary entry point for configuration logic.
- `harmonyspeech/engine/harmonyspeech_engine.py`: Coordination layer between API and executors.
- `harmonyspeech/modeling/loader.py`: Handles model acquisition and instantiation.

</code_context>

<specifics>
## Specific Ideas
- "Engine: real initialization logic, Mock: CPUExecutor/GPUExecutor — test that engine correctly selects executor based on device"
- "Engine's infer/tts method — call it with dummy input and verify executor receives correct batch_size and dtype"
- "Direct unit tests for config and loader — test config validation errors and model path resolution directly"

</specifics>

<deferred>
## Deferred Ideas
- None — discussion stayed within phase scope.

</deferred>

---

*Phase: 02-unit-testing-core-components*
*Context gathered: 2026-03-03*

# Pitfalls Research

**Domain:** pytest-based testing framework for ML/AI speech models
**Researched:** 2026-02-28
**Confidence:** MEDIUM

## Critical Pitfalls

### Pitfall 1: GPU-Dependent Tests in CPU-Only CI

**What goes wrong:**
Tests pass on developer machines with GPU but fail in CI because they implicitly depend on CUDA-specific behavior (torch device placement, GPU memory availability, CUDA kernels). The test suite appears to work locally but is unusable in CI.

**Why it happens:**
Developers test on their GPU-enabled machines and don't realize their tests have implicit GPU dependencies. PyTorch's default device selection often works transparently, masking the dependency until CI runs on CPU-only runners.

**How to avoid:**
- Explicitly force CPU device in all test fixtures using `torch.device("cpu")`
- Add CI-specific markers that verify tests run on CPU: `@pytest.mark.cpu_only`
- Create a `conftest.py` fixture that enforces CPU mode globally for tests
- Run a subset of tests in CI with explicit device override to catch issues early

**Warning signs:**
- Tests use `torch.cuda.is_available()` checks that skip on CPU
- Tests pass locally but fail in GitHub Actions with "RuntimeError: CUDA..."
- Test fixtures don't specify device, relying on default behavior

**Phase to address:**
Phase 1: Test Framework Setup — Include CPU enforcement in base fixtures

---

### Pitfall 2: Flaky Tests from Non-Deterministic Model Outputs

**What goes wrong:**
ML models produce slightly different outputs on each run due to floating-point precision, random initialization, or batch processing order. Tests that assert exact equality fail intermittently, causing CI instability and erosion of trust in the test suite.

**Why it happens:**
Developers test with simple assertions like `assert output == expected_output` without accounting for numerical variance. This is especially problematic with TTS models where small audio differences are imperceptible but cause test failures.

**How to avoid:**
- Use approximate equality assertions: `pytest.approx()` or `np.allclose()` with appropriate tolerances
- For audio: use structural similarity metrics (SSIM) or waveform similarity thresholds
- For text: test semantic equivalence rather than exact match
- Set random seeds where possible: `torch.manual_seed()`, `np.random.seed()`
- Allow configurable tolerance per model type

**Warning signs:**
- Tests fail intermittently with small numerical differences
- Tests pass on one machine, fail on another (floating-point variance)
- No tolerance parameters in test assertions

**Phase to address:**
Phase 2: Unit Tests — Implement proper tolerance-based assertions from the start

---

### Pitfall 3: Missing Test Fixtures for Model Artifacts

**What goes wrong:**
Tests fail in CI because they depend on model weights, config files, or audio samples that exist on developer machines but aren't available in the CI environment. Tests can't run without manual setup.

**Why it happens:**
Developers create tests that depend on downloaded model files without considering CI environment. Tests work on their machine because they've already run the model download, but CI has a clean environment.

**How to avoid:**
- Create fixtures that handle model downloading or use small test models
- Include sample audio files in the repository under `tests/fixtures/`
- Use `@pytest.fixture(scope="session")` for expensive downloads to avoid repeated fetches
- Mock model loading entirely for unit tests: use `unittest.mock` to patch `torch.load`
- Document required fixtures in a README within tests/

**Warning signs:**
- Tests fail with "FileNotFoundError" or "Model not found" in CI
- Tests require manual model download before running
- No `tests/fixtures/` directory exists

**Phase to address:**
Phase 1: Test Framework Setup — Create fixture infrastructure for test data

---

### Pitfall 4: Over-Mocking Core Model Logic

**What goes wrong:**
Tests mock too much of the model pipeline, testing only the wrapper code but not the actual model behavior. This gives false confidence — tests pass but the model integration is broken.

**Why it happens:**
Developers mock to avoid slow model inference in tests, but end up testing nothing meaningful. The mock becomes a "trusted" component that never gets validated against real behavior.

**How to avoid:**
- Define clear boundaries: mock I/O, network, file system — NOT model inference
- Create integration tests that run the full model pipeline (even if slow)
- Use `@pytest.mark.slow` to separate fast unit tests from slow integration tests
- Validate mocks against real implementations periodically
- Test at least one model end-to-end to catch integration issues

**Warning signs:**
- Most tests use `@patch` or `Mock` for the core model classes
- No tests actually run model inference
- Test coverage is high but confidence in model behavior is low

**Phase to address:**
Phase 3: End-to-End Tests — Ensure at least one model pipeline is tested without mocking

---

### Pitfall 5: Not Testing Model Loading Failures

**What goes wrong:**
Tests only cover the happy path. When model loading fails (corrupted weights, missing config, incompatible version), the error handling is untested and users get unhelpful error messages.

**Why it happens:**
Developers test what should work, not what could go wrong. Model loading seems stable during development, so error handling is overlooked.

**How to avoid:**
- Create tests for error conditions: missing model files, corrupted weights, invalid configs
- Test that appropriate exceptions are raised with helpful messages
- Test fallback behavior when primary model fails
- Test version compatibility checks

**Warning signs:**
- No tests for exception handling in model loading code
- Error messages aren't validated
- No tests for recovery from partial failures

**Phase to address:**
Phase 2: Unit Tests — Include error handling tests alongside happy path

---

### Pitfall 6: Ignoring Audio/Text Preprocessing Pipeline

**What goes wrong:**
Tests only verify the core model inference, not the full pipeline including text normalization, phoneme conversion, audio post-processing. Bugs in preprocessing go undetected.

**Why it happens:**
It's easier to test the model in isolation, but real bugs often occur in the preprocessing steps that prepare input for the model.

**How to avoid:**
- Test text normalization: numbers, abbreviations, special characters
- Test phoneme conversion for various languages
- Test audio post-processing: normalization, format conversion
- Create integration tests that verify the full pipeline from text input to audio output

**Warning signs:**
- No tests for `text/` modules in model packages
- Tests use pre-processed inputs rather than raw user input
- Pipeline tests only verify final output exists, not correctness

**Phase to address:**
Phase 3: End-to-End Tests — Test full preprocessing pipelines

---

### Pitfall 7: Test Data Not Representative of Real Usage

**What goes wrong:**
Tests use trivial inputs ("hello world") that don't reflect real usage patterns. Edge cases like long text, special characters, mixed languages aren't tested, leading to production bugs.

**Why it happens:**
Developers use simple test cases for convenience. Complex real-world inputs require more setup and expected output validation.

**How to avoid:**
- Create a test fixture with diverse, representative inputs
- Include edge cases: very long text, emoji, mixed scripts, special characters
- Use real-world text samples from public datasets
- Parameterize tests with multiple input variations using `@pytest.mark.parametrize`

**Warning signs:**
- All tests use "test" or "hello" as input
- No tests for text longer than typical sentence
- No tests for non-English text

**Phase to address:**
Phase 2: Unit Tests — Create representative test data fixtures

---

### Pitfall 8: No Test Isolation (Shared State Between Tests)

**What goes wrong:**
Tests affect each other through shared state (global model cache, singleton patterns, file system). One test's failure causes cascading failures in unrelated tests.

**Why it happens:**
ML models are expensive to load, so developers use global caches or singletons. These work fine in production but cause test isolation issues.

**How to avoid:**
- Use pytest's `function` scope for fixtures that load models
- Clean up global state in `@pytest.fixture(autouse=True)` teardown
- Use `tmp_path` for any file system operations
- Avoid class-level or module-level model caching in tests

**Warning signs:**
- Test order affects results (run tests in different order to verify)
- Tests use class-level fixtures with `@pytest.fixture(scope="class")`
- Global variables for model instances

**Phase to address:**
Phase 1: Test Framework Setup — Enforce test isolation in conftest.py

---

### Pitfall 9: Missing Async Test Support

**What goes wrong:**
The codebase uses async inference, but tests are written synchronously. Async bugs (missing awaits, race conditions) aren't caught, and tests don't reflect real usage patterns.

**Why it happens:**
Writing async tests requires `pytest-asyncio` and understanding of async fixtures. Developers take the simpler synchronous path.

**How to avoid:**
- Install `pytest-asyncio` and configure it in `pyproject.toml`
- Use `@pytest.mark.asyncio` for async test functions
- Test concurrent requests to find race conditions
- Test timeout behavior for async operations

**Warning signs:**
- No tests with `async def` test functions
- Async engine methods aren't tested
- No tests for concurrent request handling

**Phase to address:**
Phase 2: Unit Tests — Include async test coverage for engine components

---

### Pitfall 10: No CPU vs GPU Behavioral Difference Testing

**What goes wrong:**
The same model produces different outputs on CPU vs GPU (different precision, different kernel implementations). Tests pass on CPU but fail in production where GPU is used.

**Why it happens:**
Floating-point operations have different precision on different devices. Tests only run on CPU in CI, so GPU-specific issues aren't caught.

**How to avoid:**
- Document known CPU/GPU differences for each model
- Create tests that verify output is within acceptable tolerance across devices
- If possible, run a subset of tests on both CPU and GPU in CI
- Add device-specific test markers: `@pytest.mark.gpu`, `@pytest.mark.cpu`

**Warning signs:**
- No device-specific test markers
- Tests don't account for device-dependent precision
- No documentation of known CPU/GPU differences

**Phase to address:**
Phase 4: Additional Model Tests — Include device-specific testing when GPU available

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Skip slow tests in CI | Faster CI runs | Integration bugs undetected | Only for unit tests, never for e2e |
| Use hardcoded test data | Simple to implement | Doesn't catch preprocessing bugs | Only for initial scaffolding |
| Mock all model inference | Fast tests | False confidence, integration untested | Only for API/interface tests |
| Skip error handling tests | Fewer tests to write | Poor user experience when errors occur | Never — error paths are critical |
| No test documentation | Faster initial development | Others can't understand test intent | Only for obvious test names |

---

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| HuggingFace Hub | Not handling download failures gracefully | Mock downloads or use small test models |
| PyTorch | Implicit device dependencies | Explicitly use CPU device in tests |
| Audio libraries (torchaudio) | Not handling different sample rates | Normalize all audio to single sample rate in fixtures |
| Model config files | Not handling missing/invalid configs | Test with both valid and invalid configs |
| File system | Tests pollute working directory | Use `tmp_path` fixture for all file operations |

---

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Loading models in every test | Tests take 10+ minutes | Use session-scoped fixtures with lazy loading | At 50+ tests |
| No test parallelization | CI takes 30+ minutes | Use `pytest-xdist` for parallel execution | At 100+ tests |
| Large audio fixtures in git | Repo size balloons | Use small audio samples, compress or reference external | When repo exceeds 100MB |
| No test timeout | Tests hang indefinitely | Add `@pytest.mark.timeout(seconds)` | With flaky model loading |

---

## Security Mistakes

| Mistake | Risk | Prevention |
|---------|------|------------|
| Including model weights in tests | License violations, large repo | Use small test models or mock weights |
| Testing against real external APIs | Data leakage, rate limiting | Mock all external API calls |
| Storing test credentials in repo | Credential exposure | Use environment variables, never commit secrets |
| Not testing input validation | Injection attacks | Test with malicious inputs (empty, very long, special chars) |

---

## "Looks Done But Isn't" Checklist

- [ ] **Test fixtures:** Often missing device specification — verify `torch.device("cpu")` is used
- [ ] **Model tests:** Often only test happy path — verify error handling tests exist
- [ ] **Async tests:** Often missing for async engine methods — verify `pytest-asyncio` is configured
- [ ] **Integration tests:** Often just API tests, not full model pipeline — verify e2e tests run actual inference
- [ ] **Test isolation:** Often overlooked with shared state — verify tests pass in random order
- [ ] **Tolerance:** Often using exact equality — verify numerical tolerance is used for ML outputs

---

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| GPU-dependent tests in CI | MEDIUM | Add CPU device enforcement, re-run CI |
| Flaky tests from non-determinism | LOW | Add tolerance to assertions, set seeds |
| Missing test fixtures | MEDIUM | Create fixtures directory, add sample data |
| Over-mocking | HIGH | Rewrite tests to use real inference, add integration tests |
| No error handling tests | MEDIUM | Add negative tests, verify error messages |

---

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| GPU-dependent tests in CPU CI | Phase 1: Test Framework Setup | Run tests in CI, verify they pass on CPU |
| Flaky tests from non-determinism | Phase 2: Unit Tests | Run tests multiple times, verify stability |
| Missing test fixtures | Phase 1: Test Framework Setup | Verify all tests can run without manual setup |
| Over-mocking core logic | Phase 3: End-to-End Tests | Run e2e tests, verify actual model inference occurs |
| Not testing model loading failures | Phase 2: Unit Tests | Verify error handling tests exist |
| Ignoring preprocessing pipeline | Phase 3: End-to-End Tests | Verify full pipeline tests exist |
| Test data not representative | Phase 2: Unit Tests | Review test inputs, verify diversity |
| No test isolation | Phase 1: Test Framework Setup | Run tests in random order, verify no dependencies |
| Missing async test support | Phase 2: Unit Tests | Verify async tests exist for async methods |
| No CPU vs GPU difference testing | Phase 4: Additional Models | Document differences, add device-specific tests |

---

## Sources

- Pytest documentation on fixtures and markers
- PyTorch testing best practices
- .planning/codebase/TESTING.md (existing codebase recommendations)
- Common ML testing patterns from industry experience
- pytest-asyncio documentation for async testing

---

*Pitfalls research for: pytest-based testing framework for ML/AI speech models*
*Researched: 2026-02-28*

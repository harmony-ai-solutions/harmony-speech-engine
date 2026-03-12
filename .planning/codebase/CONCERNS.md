# Codebase Concerns

**Analysis Date:** 2026-03-11

## Overview

This document captures technical debt, known issues, test coverage gaps, and risks in the Harmony Speech Engine codebase. The engine has grown significantly with new model support (KittenTTS, ChatterBox) and a testing framework, but several areas remain untested or carry technical debt from rapid development.

---

## Recent Updates (March 2026)

### KittenTTS Status
**Resolved:** KittenTTS was introduced without tests (commit `3f63eb3`). All 4 e2e tests (mini, micro, nano, nano-int8) have since been validated and pass on CPU.

### ChatterBox Removed
**Resolved:** The stale `harmonyspeech/modeling/models/chatterbox/` directory was removed as it was incomplete and untested.

---

## Test Coverage & Gaps

The test suite consists of 80 tests with 80% overall coverage.

### What IS Tested

| Tier | Count | Scope |
|------|-------|-------|
| **Unit** | 28 | Config parsing, dtype resolution, engine init, model loader registry |
| **Integration** | 17 | CLI arg parsing, all 7 HTTP endpoints via mocked serving layer |
| **E2E** | 35 | Full model pipelines for KittenTTS, MeloTTS, OpenVoice V1/V2, HarmonySpeech, FasterWhisper, SileroVAD, VoiceFixer |

### Critical Gaps (Untested Areas)

| Gap | Detail |
|-----|--------|
| **GPU Executor Path** | `gpu_model_runner.py` and `gpu_worker.py` are 0% covered (358 lines) |
| **Integration Mocks** | Integration tier uses full mocks; real engine/scheduler interaction not tested at this level |
| **Utility Functions** | `common/utils.py` is only 50% covered |
| **Error Handling** | Exception handling branches in `serving_*.py` and `async_harmonyspeech.py` are largely untested |
| **Streaming Output** | Feature not implemented; "Stream output is not yet supported" TODO in TTS endpoints |

---

## Test Infrastructure Issues

### Missing Commit ID Module
**Problem:** `RuntimeWarning: Failed to read commit hash` from `harmonyspeech/__init__.py:3`.
**Impact:** The `harmonyspeech.commit_id` module is expected but not generated or present, leading to "unknown" version reporting in logs.

---

## Existing Tech Debt (Previously Documented)

### Missing Batched Inference

- **Issue:** Most model execution functions iterate over requests and perform inference one by one, missing out on GPU batching performance gains.
- **Files:** `harmonyspeech/task_handler/model_runner_base.py`
- **Impact:** Significantly reduced throughput and higher latency for concurrent requests.
- **Fix approach:** Refactor `_execute_*` methods to use batched versions of model inference calls.

### Hardcoded Routing Logic

- **Issue:** Complex routing for multi-step processing (e.g., OpenVoice, Harmonyspeech) is hardcoded using string-based `model_type` comparisons.
- **Files:** `harmonyspeech/engine/harmonyspeech_engine.py`
- **Impact:** High maintenance overhead when adding or modifying models/workflows.
- **Fix approach:** Implement a more flexible, configuration-driven routing or pipeline management system.

### Unoptimized MeloTTS Integration

- **Issue:** Integration includes "super complicated, imperformant and underoptimized code" (per code comments), including redundant model allocations.
- **Files:** `harmonyspeech/modeling/models/melo/inputs.py`, `harmonyspeech/modeling/models/melo/melo.py`
- **Impact:** Unnecessary memory usage and higher latency for MeloTTS-based generation.
- **Fix approach:** Rewrite the text normalization and Bert-based embedding logic to be more efficient.

### Generic Error Handling

- **Issue:** Widespread use of `ValueError` and generic exceptions instead of specialized error classes.
- **Files:** `harmonyspeech/endpoints/openai/serving_*`, `harmonyspeech/engine/async_harmonyspeech.py`
- **Impact:** Difficult to distinguish between different types of failures.
- **Fix approach:** Define a comprehensive set of domain-specific exception classes.

---

## Security Considerations

### API Key Fail-Open Behavior

- **Issue:** API key checks are disabled if `ENDPOINT_URL` is not set or empty, potentially exposing the service without authentication.
- **Files:** `auth/apikeys.py:108`
- **Risk:** Unauthorized access if environment variable is misconfigured.
- **Current mitigation:** Relies on correct environment configuration.
- **Recommendations:** Implement fail-closed behavior or mandatory configuration check.

### Non-Distributed API Key Cache

- **Issue:** `_api_key_cache` is a simple in-memory dictionary.
- **Files:** `auth/apikeys.py:31`
- **Risk:** In distributed Ray environment, rate limits and API key validity may not be synchronized across workers.
- **Recommendations:** Use distributed cache (Redis) or centralized auth service.

---

## Performance Bottlenecks

### Potential GIL Bottleneck

- **Problem:** The engine uses `ThreadPoolExecutor` to run executors in parallel.
- **Files:** `harmonyspeech/engine/harmonyspeech_engine.py:592`
- **Cause:** If executor work doesn't release GIL (heavy Python-based pre/post-processing), parallelism limited.
- **Improvement path:** Verify GIL release patterns; consider `ProcessPoolExecutor` or Ray tasks.

### Redundant Text Normalization Allocations

- **Problem:** Bert is allocated for each normalized part of text during MeloTTS inference.
- **Files:** `harmonyspeech/modeling/models/melo/inputs.py:30`
- **Cause:** Unoptimized pre-processing pipeline.
- **Improvement path:** Cache or reuse Bert model instance for request/worker lifecycle.

---

## Fragile Areas

### Multi-Step Request Forwarding

- **Files:** `harmonyspeech/engine/harmonyspeech_engine.py:314` (`check_forward_processing`)
- **Why fragile:** Re-adding requests to scheduler for multi-step processing creates complex state management.
- **Safe modification:** Carefully test changes to `step()` loop or `Scheduler` state transitions.

### String-Based Model Type Identification

- **Files:** `harmonyspeech/task_handler/model_runner_base.py:62`, `harmonyspeech/engine/harmonyspeech_engine.py`
- **Why fragile:** Magic strings for `model_type` create risk of typos or routing breaks during refactoring.
- **Safe modification:** Use central enumeration or constant-based mapping for model types.

---

## Scaling Limits

### Single-Instance Scheduler

- **Current capacity:** Single engine instance manages all requests.
- **Limit:** Centralized scheduler may bottleneck at high request volumes or across many Ray workers.
- **Scaling path:** Distribute scheduling or use more efficient queueing mechanisms.

---

## Missing Critical Features

### Streaming Support

- **Problem:** "Stream output is not yet supported" in TTS endpoints.
- **Files:** `harmonyspeech/endpoints/openai/serving_text_to_speech.py:68`
- **Blocks:** Real-time applications requiring low time-to-first-byte.

### Silero VAD Routing

- **Problem:** "TODO: add switch for silero here using request.input_vad_mode"
- **Files:** `harmonyspeech/engine/harmonyspeech_engine.py:165`
- **Blocks:** Users from choosing alternative VAD engines via API parameters.

---

## Known Bugs (from TODO/FIXME Comments)

### FIXME Comments Found (47 total)

| Location | Issue |
|----------|-------|
| `harmonyspeech/task_handler/model_runner_base.py:122,144,189,200,236,280,344,392` | "FIXME: This is not properly batched" |
| `harmonyspeech/task_handler/inputs.py:333` | "TODO: This expects Whisper VAD input, needs rework to be compatible with Silero VAD" |
| `harmonyspeech/engine/harmonyspeech_engine.py:165,236` | "TODO: add switch for silero here using request.input_vad_mode" |
| `harmonyspeech/engine/harmonyspeech_engine.py:188,259` | "FIXME: This code needs refactor" |
| `harmonyspeech/modeling/models/melo/inputs.py:29-31` | "FIXME: This is a bunch of super complicated, imperformant and underoptimized code" |
| Multiple serving files | "TODO: Basic checks for [request type] request" |
| Multiple serving files | "TODO: Use an aphrodite-specific Validation Error" |

### Critical FIXME Items

1. **MeloTTS inputs.py (lines 29-31):** Code explicitly states "needs to be completely rewritten when there is time" - indicates known poor quality code
2. **Model batching (8 locations):** Performance impact - not batched properly
3. **Silero VAD routing (2 locations):** Feature not implemented, blocks user choice
4. **VAD compatibility (1 location):** Whisper VAD format expected, Silero incompatible

---

## Dependency Risks

### Pinned Dependencies with Comments

- **ctranslate2 == 4.4.0:** Pinned with comment "Downgrade ctranslate, see: https://github.com/SYSTRAN/faster-whisper/issues/1086" - indicates known compatibility issue
- **onnxruntime:** No version pin - could have breaking changes
- **torch >= 2.7.1:** Required but no upper bound - could break on torch 3.x

### Chatterbox TTS Dependency Pinning

- **Package:** `chatterbox-tts==0.1.6`
- **Problem:** The package declares hard version pins in its pip metadata (`torch==2.6.0`, `torchaudio==2.6.0`, `transformers==4.46.3`, `numpy<1.26.0`, `safetensors==0.5.3`, `diffusers==0.29.0`) that conflict with the versions required by the rest of HSE. A normal `pip install chatterbox-tts` would downgrade torch, transformers, and numpy, breaking the engine.
- **Current mitigation:** Installed via `--no-deps` with its transitive dependencies listed explicitly in `requirements-common.txt`. Verified working on torch 2.10, transformers 5.0, numpy 2.4, safetensors 0.7, diffusers 0.37.
- **Risk:** Every new `chatterbox-tts` release must be manually evaluated before bumping the pin:
  1. Check the new release's `requires_dist` for any newly required packages or changed pins.
  2. Run `pip install chatterbox-tts==<new_version> --no-deps` in a test environment.
  3. Verify all four classes import: `ChatterboxTTS`, `ChatterboxVC`, `ChatterboxMultilingualTTS`, `ChatterboxTurboTTS`.
  4. Run `pytest tests/unit/initialization/test_chatterbox_imports.py -v`.
  5. Update the version pin and transitive deps in `requirements-common.txt` if needed.
- **Upgrade owner:** Whoever bumps the pin is responsible for completing steps 1–5 above.
- **Long-term fix:** If Resemble AI relaxes their version pins in a future release, the `--no-deps` workaround can be removed and the package installed normally.

### Complex Dependency Tree

- Multiple phonemizers: `gruut[de,es,fr]`, `misaki[en]`, `espeakng_loader`, `g2p_en`, `g2pkk`
- Multiple NLP tools: `spacy`, `nltk`, `mecab-python3`, `unidic-lite`, `pykakasi`, `eunjeon`, `fugashi`, `cn2an`, `pypinyin`, `jieba`
- **Risk:** Dependency conflicts likely when adding new features; testing overhead for multilingual support

These items will break in upcoming dependency versions:

| Warning | File | Breaking Version |
|---------|------|-----------------|
| `audioop` deprecated | `pydub` (third-party) | Python 3.13 |
| `aifc`/`sunau` deprecated | `audioread` (third-party) | Python 3.13 |
| `scipy.ndimage.morphology` | `harmonyspeech/modeling/models/harmonyspeech/common.py:7` | SciPy 2.0 |
| `kakasi.setMode`/`getConverter` | `harmonyspeech/modeling/models/melo/text/japanese.py:548-551` | kakasi v3.0 |
| `torch stft return_complex=False` | MeloTTS voice cloning path | Future PyTorch |

### No Lock File

- No `requirements.lock` or `poetry.lock` - builds not reproducible
- Recommendation: Add pip-tools or poetry for reproducible builds

---

## Breaking Change Risks

### API Server Evolution

- **Files:** `harmonyspeech/endpoints/openai/serving_*.py`, `harmonyspeech/endpoints/openai/api_server.py`
- **Risk:** Many TODO comments for validation errors and endpoint expansion - API may change
- **Current:** OpenAI-compatible API, but not guaranteed stable

### Model Type Registry

- **Files:** `harmonyspeech/modeling/models/*/__init__.py`
- **Risk:** No central registry - adding new models requires manual wiring in multiple places
- **Impact:** Easy to miss adding a model to configuration, causing runtime errors

---

## Test Infrastructure Observations

### Test Markers Defined

From `pyproject.toml`:
- `@pytest.mark.unit` - Fast, mocked
- `@pytest.mark.integration` - Component interaction
- `@pytest.mark.e2e` - Real models, slow
- `@pytest.mark.slow` - Tests >30 seconds

### E2E Test Fixtures

Session-scoped fixtures in `tests/e2e/conftest.py`:
- `kittentts_mini_engine`, `kittentts_micro_engine`, `kittentts_nano_engine`, `kittentts_nano_int8_engine`
- `melotts_en_engine`
- `openvoice_v1_en_engine`
- `harmonyspeech_engine`
- `whisper_engine`
- `vad_engine` (Silero)
- `whisper_vad_engine`
- `voicefixer_engine`

### Integration Test State

- Uses mocked serving classes - not testing real component interaction
- Coverage limited to HTTP layer, not engine internals

---

## Recommendations

### Priority 1 (Critical)

1. **Run and validate KittenTTS e2e tests** - Currently marked untested
2. **Add ChatterBox tests or remove** - Unknown implementation status
3. **Add multi-step pipeline tests** - VAD → Embedding → TTS workflow untested

### Priority 2 (High Impact)

4. **Fix Silero VAD routing** - TODO in engine prevents user choice
5. **Implement streaming support** - Missing critical feature
6. **Address batching FIXMEs** - Performance impact on 8+ code paths

### Priority 3 (Maintenance)

7. **Add pip lock file** - Reproducible builds
8. **Refactor error handling** - Domain-specific exceptions
9. **Centralize model type registry** - Reduce magic strings

---

*Concerns audit: 2026-03-11*
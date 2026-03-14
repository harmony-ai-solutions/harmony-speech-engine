# Phase 7: Testing & Documentation — CONTEXT

## Phase Summary

**Goal:** Comprehensive test coverage across all Chatterbox model variants and routes, plus OpenAPI field documentation for new generation options.

**Requirements:** REQ-TEST-01, REQ-TEST-02, REQ-TEST-03, REQ-TEST-04, REQ-DOC-01

**Depends on:** Phase 6 (watermark config, optional ChatterboxEmbedding, config examples all complete ✅)

---

## What Prior Phases Left Ready

The following tests and infrastructure are **already implemented** — do **not** recreate or modify:

| File | Created By | What it tests |
|------|-----------|--------------|
| [`tests/unit/initialization/test_chatterbox_imports.py`](tests/unit/initialization/test_chatterbox_imports.py) | Phase 1 | `chatterbox-tts`, `perth`, `pyloudnorm` importable |
| [`tests/unit/modeling/test_chatterbox_registry.py`](tests/unit/modeling/test_chatterbox_registry.py) | Phase 2 | All 4 model types in `ModelRegistry`, `load_model_cls` returns `"native"` |
| [`tests/unit/inference_flow/test_chatterbox_inputs.py`](tests/unit/inference_flow/test_chatterbox_inputs.py) | Phase 3 | All 5 `prepare_*` functions, `ValueError` branches, language defaults |
| [`tests/unit/inference_flow/test_chatterbox_no_tempfile.py`](tests/unit/inference_flow/test_chatterbox_no_tempfile.py) | Phase 6 | `builtins.open` + `tempfile.*` never called; `io.BytesIO` used |

Phase 7 adds **3 new test files** and **1 documentation update**:
1. `tests/unit/inference_flow/test_chatterbox_routing.py` — routing logic unit tests (REQ-TEST-01, REQ-TEST-02)
2. `tests/integration/test_chatterbox_flow.py` — engine-level integration tests (REQ-TEST-03)
3. `tests/e2e/tts/test_chatterbox_e2e.py` — full-stack E2E tests with real model, CUDA skip (REQ-TEST-04)
4. [`harmonyspeech/endpoints/openai/protocol.py`](harmonyspeech/endpoints/openai/protocol.py) — `Field(description=...)` on 8 Chatterbox params (REQ-DOC-01)

---

## Gray Areas Resolved

### GA-1: Integration Test Mock Boundary → **Option B (engine-level mocking)**

Integration tests mock at the **model runner** level, not the serving layer. This means:
- A real `HarmonySpeechEngine` (or `AsyncHarmonySpeech`) instance is created with real `ModelConfig` objects
- The executor's `execute_model()` method is mocked to return fake outputs
- This exercises `reroute_request_chatterbox()` and `check_forward_processing()` in `harmonyspeech_engine.py` — the most critical Chatterbox-specific logic that is **not** tested by any prior phase tests

The existing [`tests/integration/conftest.py`](tests/integration/conftest.py) `mock_engine_app` fixture (which mocks at the serving layer) is **unchanged** — it covers general API endpoint shapes. Chatterbox integration tests are self-contained in `test_chatterbox_flow.py`.

### GA-2: E2E CUDA Skip → **`pytest.mark.skipif` + CUDA availability + `cuda` marker**

Chatterbox requires CUDA for practical use (noted in Phase 6 config). Decision:

1. **Add `cuda` marker** to [`pyproject.toml`](pyproject.toml) `[tool.pytest.ini_options]` markers list:
   ```
   "cuda: marks tests that require CUDA/GPU hardware"
   ```
2. Every Chatterbox E2E test gets both `@pytest.mark.e2e` + `@pytest.mark.slow` + `@pytest.mark.cuda`
3. Each test also carries `@pytest.mark.skipif(not torch.cuda.is_available(), reason="Chatterbox requires CUDA")`

**Note:** GPU is available on the dev system, so tests will run without skipping. The `skipif` guard ensures CI environments without GPUs don't fail.

### GA-3: OpenAPI Documentation → **Option A (Field descriptions in `GenerationOptions` only)**

Add `Field(None, description="...")` to the 8 Chatterbox-specific params in `GenerationOptions` in [`protocol.py`](harmonyspeech/endpoints/openai/protocol.py). This makes them visible and described in `/docs` Swagger UI and ReDoc automatically — no separate doc file update needed.

### GA-4: Conftest Strategy → **E2E first, integration self-contained**

The user confirmed: integration tests cover general API shapes (already done); for Chatterbox, E2E tests with a real model are the meaningful coverage. Chatterbox integration fixtures are **self-contained** inside `test_chatterbox_flow.py` — no shared conftest changes for integration.

---

## Implementation Details

### 1. `pyproject.toml` — Add `cuda` marker

**File:** [`pyproject.toml`](pyproject.toml)

Add to `[tool.pytest.ini_options]` `markers` list:

```toml
markers = [
    "unit: marks tests as unit tests (fast, fully mocked)",
    "integration: marks tests as integration tests (component interaction, partially mocked)",
    "e2e: marks tests as end-to-end tests (real models, slow, CPU-only)",
    "slow: marks tests that take > 30 seconds to run",
    "cuda: marks tests that require CUDA/GPU hardware",   # ← ADD THIS
]
```

---

### 2. `tests/unit/inference_flow/test_chatterbox_routing.py` (new)

**Purpose:** Unit-test the routing logic in `harmonyspeech_engine.py` for all Chatterbox model route cases.

These tests use `unittest.mock.patch` and `MagicMock` — no real models loaded.

**Test cases to cover:**

| Test name | What it asserts |
|-----------|----------------|
| `test_reroute_tts_no_cloning` | TTS request without audio → routed to `ChatterboxTTS` executor directly |
| `test_reroute_tts_with_precomputed_embedding` | TTS request with `input_embedding` → routed to `ChatterboxTTS` executor directly |
| `test_reroute_tts_with_input_audio_to_embed_step` | TTS request with `input_audio` + no embedding → routed to `ChatterboxEmbedding` executor first |
| `test_reroute_tts_embedding_fallback_to_tts` | TTS with `input_audio`, no `ChatterboxEmbedding` in config → falls back to `ChatterboxTTS` executor |
| `test_reroute_embed_request` | `SpeechEmbeddingRequestInput` → routed to `ChatterboxEmbedding` executor |
| `test_reroute_embed_fallback_to_tts` | `SpeechEmbeddingRequestInput`, no `ChatterboxEmbedding` in config → falls back to `ChatterboxTTS` |
| `test_reroute_vc_request` | `VoiceConversionRequestInput` → routed to `ChatterboxVC` executor |
| `test_reroute_turbo_tts` | TTS on `chatterbox_turbo` model → routed to `ChatterboxTurboTTS` executor |
| `test_reroute_multilingual_tts` | TTS on `chatterbox_multilingual` model → routed to `ChatterboxMultilingualTTS` executor |
| `test_forward_processing_transfers_embedding` | After embed step completes, `check_forward_processing()` sets `input_embedding` on TTS re-submission |

**Fixture pattern:**

```python
def _make_engine_with_configs(model_configs):
    """Create a HarmonySpeechEngine with the given ModelConfig list (no real models loaded)."""
    engine_config = MagicMock()
    engine_config.model_configs = model_configs
    engine = HarmonySpeechEngine.__new__(HarmonySpeechEngine)
    engine.model_configs = model_configs
    return engine
```

**Key imports:**
```python
from harmonyspeech.engine.harmonyspeech_engine import HarmonySpeechEngine
from harmonyspeech.common.config import ModelConfig, DeviceConfig
from harmonyspeech.common.request import RequestInput
from harmonyspeech.common.inputs import (
    TextToSpeechRequestInput, SpeechEmbeddingRequestInput, VoiceConversionRequestInput
)
```

---

### 3. `tests/integration/test_chatterbox_flow.py` (new)

**Purpose:** Integration test the full request lifecycle through `AsyncHarmonySpeech` for all Chatterbox model routes, with model runner mocked to avoid actual GPU inference.

**Approach:** Create a real engine with real `ModelConfig` objects, but patch `model_runner_base.ModelRunnerBase.execute_model` to return fake outputs. This tests:
- `reroute_request_chatterbox()` path selection
- `check_forward_processing()` embedding→TTS transfer
- Complete request lifecycle: `add_request` → scheduler → executor dispatch → output

**Self-contained fixture** (inside `test_chatterbox_flow.py`):

```python
@pytest.fixture(scope="module")
def chatterbox_engine_mocked():
    """Engine with ChatterboxTTS model config and mocked model runner."""
    from harmonyspeech.common.config import EngineConfig, ModelConfig, DeviceConfig
    from harmonyspeech.engine.args_tools import AsyncEngineArgs
    from harmonyspeech.engine.async_harmonyspeech import AsyncHarmonySpeech

    model_configs = [
        ModelConfig(
            name="chatterbox",
            model="resemble-ai/chatterbox",
            model_type="ChatterboxTTS",
            max_batch_size=1,
            dtype="float32",
            device_config=DeviceConfig(device="cpu"),
        ),
    ]
    engine_config = EngineConfig(model_configs=model_configs)
    engine_args = AsyncEngineArgs(disable_log_stats=True, disable_log_requests=True)

    with patch("harmonyspeech.modeling.loader.get_model") as mock_get_model, \
         patch("harmonyspeech.task_handler.model_runner_base.ModelRunnerBase.execute_model") as mock_execute:
        # Mock model loader to return a fake model
        mock_get_model.return_value = MagicMock()
        # Mock execute_model to return fake TTS output
        mock_execute.return_value = [TextToSpeechRequestOutput(audio=b"fake_audio_wav", sample_rate=24000)]

        engine = AsyncHarmonySpeech.from_engine_args_and_config(
            engine_args, engine_config, start_engine_loop=True
        )
        serving_tts = OpenAIServingTextToSpeech(
            engine,
            OpenAIServingTextToSpeech.models_from_config(engine_config.model_configs)
        )
        serving_embed = OpenAIServingVoiceEmbedding(
            engine,
            OpenAIServingVoiceEmbedding.models_from_config(engine_config.model_configs)
        )
        serving_vc = OpenAIServingVoiceConversion(
            engine,
            OpenAIServingVoiceConversion.models_from_config(engine_config.model_configs)
        )
        yield engine, serving_tts, serving_embed, serving_vc
```

**Test cases:**

| Test name | Route tested |
|-----------|-------------|
| `test_chatterbox_tts_direct` | TTS without audio/embedding → direct to `ChatterboxTTS` |
| `test_chatterbox_tts_with_embedding` | TTS with `input_embedding` → direct to `ChatterboxTTS` (no embed step) |
| `test_chatterbox_tts_voice_cloning_multistep` | TTS with `input_audio` → embed step then TTS step (multi-step routing) |
| `test_chatterbox_embed_standalone` | Standalone `/v1/embeddings` request → `ChatterboxEmbedding` executor |
| `test_chatterbox_embed_fallback` | `/v1/embeddings` when no `ChatterboxEmbedding` in config → falls back to `ChatterboxTTS` |
| `test_chatterbox_vc_request` | Voice conversion request → `ChatterboxVC` executor |

**Note on mock outputs:** The integration tests' `execute_model` mock must return output types that match what the serving layer expects:
- TTS → `TextToSpeechRequestOutput`
- Embed → `SpeechEmbeddingRequestOutput`
- VC → `VoiceConversionRequestOutput`

---

### 4. `tests/e2e/tts/test_chatterbox_e2e.py` (new)

**Purpose:** Full-stack E2E tests with the **real Chatterbox model** loaded from HuggingFace. Tests every model route end-to-end on real audio.

**Path note:** The ROADMAP references `tests/e2e/test_chatterbox_e2e.py` but the project convention puts TTS e2e tests in `tests/e2e/tts/`. Use `tests/e2e/tts/test_chatterbox_e2e.py` — the `pytest` command in the ROADMAP should be updated to reflect this path.

**Fixture in `tests/e2e/conftest.py`** (add alongside existing fixtures):

```python
@pytest.fixture(scope="session")
def chatterbox_engine(models_cache_dir, device):
    """Session-scoped engine fixture for Chatterbox TTS E2E tests.

    Loads 2 models:
    - chatterbox (ChatterboxTTS) — TTS + voice cloning
    - chatterbox-vc (ChatterboxVC) — voice conversion

    Requires CUDA. Skipped automatically if not available.
    """
    import torch
    if not torch.cuda.is_available():
        pytest.skip("Chatterbox E2E requires CUDA")

    model_configs = [
        ModelConfig(
            name="chatterbox",
            model="resemble-ai/chatterbox",
            model_type="ChatterboxTTS",
            max_batch_size=1,
            dtype="float32",
            device_config=DeviceConfig(device="cuda"),
        ),
        ModelConfig(
            name="chatterbox-vc",
            model="resemble-ai/chatterbox",
            model_type="ChatterboxVC",
            max_batch_size=1,
            dtype="float32",
            device_config=DeviceConfig(device="cuda"),
        ),
    ]
    engine_config = EngineConfig(model_configs=model_configs)
    engine_args = AsyncEngineArgs(disable_log_stats=True, disable_log_requests=True)
    engine = AsyncHarmonySpeech.from_engine_args_and_config(
        engine_args, engine_config, start_engine_loop=True
    )
    serving_tts = OpenAIServingTextToSpeech(
        engine,
        OpenAIServingTextToSpeech.models_from_config(engine_config.model_configs)
    )
    serving_embed = OpenAIServingVoiceEmbedding(
        engine,
        OpenAIServingVoiceEmbedding.models_from_config(engine_config.model_configs)
    )
    serving_vc = OpenAIServingVoiceConversion(
        engine,
        OpenAIServingVoiceConversion.models_from_config(engine_config.model_configs)
    )
    return (engine, serving_tts, serving_embed, serving_vc)
```

**Test cases in `test_chatterbox_e2e.py`:**

All tests marked with `@pytest.mark.e2e`, `@pytest.mark.slow`, `@pytest.mark.cuda`, and `@pytest.mark.skipif(not torch.cuda.is_available(), reason="Chatterbox requires CUDA")`.

| Test name | Route | What it verifies |
|-----------|-------|-----------------|
| `test_chatterbox_tts_no_cloning` | TTS direct | Text → base64 WAV returned; non-empty |
| `test_chatterbox_tts_with_precomputed_embedding` | TTS + embedding | Pre-embed audio → use embedding in TTS; non-empty WAV |
| `test_chatterbox_tts_voice_cloning` | Multi-step TTS+embed | `input_audio` provided → embed step + TTS step; non-empty WAV |
| `test_chatterbox_standalone_embedding` | Embed only | Audio → base64 Conditionals embedding; non-empty |
| `test_chatterbox_vc_with_target_audio` | VC + target audio | Source + target audio → converted audio; non-empty WAV |
| `test_chatterbox_vc_with_target_embedding` | VC + pre-computed | Source audio + embedding → converted audio; non-empty WAV |

**Reference audio:** Use `load_sample_audio_b64("wanda4")` — already available in `tests/test-data/samples/`.

**TEXT_INPUT constant:**
```python
TEXT_INPUT = "Hello, world. This is a test of the Chatterbox voice cloning system."
REFERENCE_AUDIO = load_sample_audio_b64("wanda4")
```

---

### 5. `protocol.py` — Chatterbox Field Descriptions (REQ-DOC-01)

**File:** [`harmonyspeech/endpoints/openai/protocol.py`](harmonyspeech/endpoints/openai/protocol.py)

In the `GenerationOptions` Pydantic model, add `Field(None, description="...")` to the 8 Chatterbox-specific fields:

```python
from pydantic import BaseModel, Field

class GenerationOptions(BaseModel):
    # ... existing fields ...

    # Chatterbox TTS generation parameters
    exaggeration: Optional[float] = Field(
        None,
        description="(Chatterbox TTS/Multilingual) Emotion exaggeration factor. "
                    "Higher values amplify emotional expressiveness. Default: 0.5"
    )
    cfg_weight: Optional[float] = Field(
        None,
        description="(Chatterbox TTS/Multilingual) Classifier-free guidance weight. "
                    "Controls how closely the model follows the voice prompt. Default: 0.5"
    )
    temperature: Optional[float] = Field(
        None,
        description="(Chatterbox TTS/Turbo/Multilingual) Sampling temperature. "
                    "Higher values increase randomness. Default: 0.8"
    )
    repetition_penalty: Optional[float] = Field(
        None,
        description="(Chatterbox TTS/Turbo/Multilingual) Repetition penalty applied during sampling. "
                    "Values > 1.0 reduce repetition. Default: 1.2"
    )
    top_p: Optional[float] = Field(
        None,
        description="(Chatterbox TTS/Turbo/Multilingual) Top-p (nucleus) sampling probability. "
                    "Default TTS: 1.0, Turbo: 0.95"
    )
    min_p: Optional[float] = Field(
        None,
        description="(Chatterbox TTS/Multilingual only) Minimum probability threshold for sampling. "
                    "Default: 0.05. Not supported by Turbo variant."
    )
    top_k: Optional[int] = Field(
        None,
        description="(Chatterbox Turbo only) Top-k sampling — number of candidates kept. "
                    "Default: 1000. Not supported by standard TTS or Multilingual variants."
    )
    norm_loudness: Optional[bool] = Field(
        None,
        description="(Chatterbox Turbo only) Apply pyloudnorm loudness normalization to output. "
                    "Default: True. Not supported by standard TTS or Multilingual variants."
    )
```

**Verification:** After this change, `GET /docs` will show all 8 Chatterbox fields with descriptions in the `GenerationOptions` schema.

---

## Test Coverage Map (Phase 7 completion)

| Requirement | Test Location | Coverage |
|-------------|--------------|---------|
| REQ-TEST-01 (unit tests pass) | `tests/unit/inference_flow/test_chatterbox_*.py` | All existing + new routing tests |
| REQ-TEST-02 (ValueError branches covered) | `tests/unit/inference_flow/test_chatterbox_inputs.py` | Already covers all branches from Phase 3 |
| REQ-TEST-03 (integration tests pass) | `tests/integration/test_chatterbox_flow.py` | 6 route cases, engine-level mocking |
| REQ-TEST-04 (E2E tests pass or skip) | `tests/e2e/tts/test_chatterbox_e2e.py` | 6 full-stack tests, CUDA-skippable |
| REQ-DOC-01 (API docs accessible + fields documented) | `protocol.py` `GenerationOptions` | 8 Chatterbox fields with `Field(description=...)` |

---

## Files to Create

| File | Purpose |
|------|---------|
| `tests/unit/inference_flow/test_chatterbox_routing.py` | Unit tests for `reroute_request_chatterbox()` + `check_forward_processing()` |
| `tests/integration/test_chatterbox_flow.py` | Engine-level integration tests (all 6 routes) |
| `tests/e2e/tts/test_chatterbox_e2e.py` | Full E2E tests with real Chatterbox model on CUDA |

## Files to Modify

| File | Change |
|------|--------|
| [`pyproject.toml`](pyproject.toml) | Add `"cuda: marks tests that require CUDA/GPU hardware"` to markers list |
| [`harmonyspeech/endpoints/openai/protocol.py`](harmonyspeech/endpoints/openai/protocol.py) | Add `Field(description=...)` to 8 Chatterbox fields in `GenerationOptions` |
| [`tests/e2e/conftest.py`](tests/e2e/conftest.py) | Add `chatterbox_engine` session-scoped fixture |

---

## Architecture: Test Layers for Chatterbox

```
┌─────────────────────────────────────────────────────────────┐
│  E2E Layer  tests/e2e/tts/test_chatterbox_e2e.py           │
│  Real model (CUDA) + real audio data                        │
│  @pytest.mark.cuda + skipif(not cuda.is_available())        │
│  Tests: TTS direct, TTS+embed, voice cloning, VC            │
└──────────────────────────┬──────────────────────────────────┘
                           │ exercises full stack
┌──────────────────────────▼──────────────────────────────────┐
│  Integration  tests/integration/test_chatterbox_flow.py     │
│  Real engine + real ModelConfigs + mocked executor          │
│  Tests: routing logic, multi-step forwarding, all routes    │
└──────────────────────────┬──────────────────────────────────┘
                           │ exercises engine routing
┌──────────────────────────▼──────────────────────────────────┐
│  Unit  tests/unit/inference_flow/test_chatterbox_routing.py │
│  Mocked engine, direct function calls on reroute_*/check_*  │
│  Tests: all 10 routing branch variants                      │
└─────────────────────────────────────────────────────────────┘

Already covered by prior phases:
  test_chatterbox_registry.py       (Phase 2 — registry)
  test_chatterbox_inputs.py         (Phase 3 — input prep + ValueError)
  test_chatterbox_no_tempfile.py    (Phase 6 — PERF-01 BytesIO)
  test_chatterbox_imports.py        (Phase 1 — imports)
```

---

## ROADMAP Path Correction

The ROADMAP references `pytest tests/e2e/test_chatterbox_e2e.py -v` but the project convention places TTS E2E tests under `tests/e2e/tts/`. The correct path is:

```bash
pytest tests/e2e/tts/test_chatterbox_e2e.py -v --device=cuda
```

The unit and integration paths are unchanged:
```bash
pytest tests/unit/inference_flow/test_chatterbox_*.py -v
pytest tests/integration/test_chatterbox_flow.py -v
```

---

## Open Questions from Prior Phases

| ID | Question | Resolution |
|----|----------|-----------|
| — | None carried forward | Phase 7 introduces no new unresolved questions |

---

*Phase 7 CONTEXT generated: 2026-03-14*

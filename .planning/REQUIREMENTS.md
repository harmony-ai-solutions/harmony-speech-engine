# Requirements: Chatterbox TTS Integration for HSE

## REQ-ARCH-01: Model Registration

**Description:** All four Chatterbox model variants must be registered in the HSE ModelRegistry and be loadable through the standard model loading pipeline.

**Details:**
- Register `ChatterboxTTS` in `_MODELS` dict with path `("chatterbox", "ChatterboxTTS")`
- Register `ChatterboxTurboTTS` in `_MODELS` dict with path `("chatterbox", "ChatterboxTurboTTS")`
- Register `ChatterboxMultilingualTTS` in `_MODELS` dict with path `("chatterbox", "ChatterboxMultilingualTTS")`
- Register `ChatterboxVC` in `_MODELS` dict with path `("chatterbox", "ChatterboxVC")`
- All four must be marked as "native" loader pattern (like KittenTTS)

**Verification:** Unit test that `_get_model_cls()` returns correct class for each type.

---

## REQ-ARCH-02: Model Loading

**Description:** All four Chatterbox model types must be loadable via `get_model()` using the native loader pattern.

**Details:**
- Add entries to `_MODEL_CONFIGS` dict: `"ChatterboxTTS": "native"`, etc.
- Add entries to `_MODEL_WEIGHTS` dict: `"ChatterboxTTS": "native"`, etc.
- Add loader branches in `get_model()` that call `ChatterboxTTS.from_pretrained()` (or equivalent)
- Support device placement via `torch.device`

**Verification:** Mock test that model objects are returned from `get_model()` with correct config.

---

## REQ-INPUT-01: Extended Generation Options

**Description:** Extend `TextToSpeechGenerationOptions` with optional Chatterbox-specific parameters.

**Details:**
- Add field `exaggeration: Optional[float] = None` (range 0.0-1.0, default 0.5)
- Add field `cfg_weight: Optional[float] = None` (range 0.0-1.0, default 0.5)
- Add field `temperature: Optional[float] = None` (default 0.8)
- Add field `repetition_penalty: Optional[float] = None` (default 1.2 for TTS, 2.0 for MTL)
- Add field `top_p: Optional[float] = None` (default 1.0 for TTS, 0.95 for Turbo)
- Add field `min_p: Optional[float] = None` (default 0.05 for TTS, 0.00 for Turbo)
- Add field `top_k: Optional[int] = None` (default 1000, Turbo only)
- Add field `norm_loudness: Optional[bool] = None` (default True, Turbo only)

**Verification:** Unit test that extended fields are accessible on the dataclass.

---

## REQ-INPUT-02: TTS Input Preparation

**Description:** Implement input preparation functions for Chatterbox TTS models.

**Details:**
- Implement `prepare_chatterbox_tts_inputs()` that handles:
  - `input_text` extraction
  - `input_embedding` deserialization (if provided, for pre-computed Conditionals)
  - `input_audio` handling for multi-step voice cloning (route to embedder)
  - Generation options extraction (exaggeration, cfg_weight, temperature, etc.)
- Implement `prepare_chatterbox_multilingual_tts_inputs()` with additional `language_id` validation
- Implement `prepare_chatterbox_turbo_tts_inputs()` with Turbo-specific defaults
- No temp file creation — use in-memory BytesIO pattern

**Verification:** Unit tests with mocked requests return correct parameter tuples.

---

## REQ-INPUT-03: Embedding Input Preparation

**Description:** Implement input preparation for the dedicated embedding endpoint.

**Details:**
- Implement `prepare_chatterbox_embedding_inputs()` that:
  - Decodes `input_audio` from base64
  - Returns raw audio bytes for Conditionals computation
- Must work with in-memory data (no filesystem)

**Verification:** Unit tests verify base64 decode and byte handling.

---

## REQ-INPUT-04: Voice Conversion Input Preparation

**Description:** Implement input preparation for ChatterboxVC.

**Details:**
- Implement `prepare_chatterbox_vc_inputs()` that handles:
  - `source_audio` decoding
  - `target_audio` or `target_embedding` handling
  - Returns source audio bytes + target Conditionals (or bytes to compute)

**Verification:** Unit tests verify correct parameter extraction.

---

## REQ-EXEC-01: TTS Model Execution

**Description:** Implement model execution methods for Chatterbox TTS variants.

**Details:**
- Implement `_execute_chatterbox_tts()` that:
  - Accepts text + optional pre-computed Conditionals
  - Calls `model.generate()` with appropriate params
  - Encodes output audio as base64 WAV
  - Returns `TextToSpeechRequestOutput`
- Implement `_execute_chatterbox_turbo_tts()` with Turbo-specific params (norm_loudness, etc.)
- Implement `_execute_chatterbox_multilingual_tts()` with language_id forwarding

**Verification:** Mock model returns known audio; output is base64 WAV of expected length.

---

## REQ-EXEC-02: Embedding Model Execution

**Description:** Implement model execution for embedding generation.

**Details:**
- Implement `_execute_chatterbox_embedding()` that:
  - Accepts raw audio bytes
  - Calls `prepare_conditionals_from_audio()` wrapper (no temp files)
  - Serializes `Conditionals` via `Conditionals.save()` → BytesIO → base64
  - Returns `SpeechEmbeddingRequestOutput`

**Verification:** Mock test verifies Conditionals serialization and base64 encoding.

---

## REQ-EXEC-03: Voice Conversion Execution

**Description:** Implement model execution for ChatterboxVC.

**Details:**
- Implement `_execute_chatterbox_vc()` that:
  - Tokenizes source audio with S3Gen
  - Uses target Conditionals (or computes from target audio)
  - Decodes with S3Gen
  - Returns `VoiceConversionRequestOutput`

**Verification:** Mock test verifies correct output type.

---

## REQ-ROUTE-01: TTS Routing

**Description:** Implement request routing for Chatterbox TTS models.

**Details:**
- Add `reroute_request_chatterbox()` in `harmonyspeech_engine.py`
- Handle single-speaker TTS: route directly to TTS model
- Handle voice cloning with pre-computed embedding: route directly to TTS model
- Handle voice cloning with input_audio: route to Embedding model first
- Register `"chatterbox"` and `"chatterbox_turbo"` and `"chatterbox_multilingual"` as model groups

**Verification:** Unit tests verify correct routing for each request type.

---

## REQ-ROUTE-02: Embedding Routing

**Description:** Implement request routing for embedding endpoint.

**Details:**
- Route `SpeechEmbeddingRequestInput` with `input_audio` → Chatterbox Embedding
- Add Chatterbox to `_EMBEDDING_MODEL_TYPES` and `_EMBEDDING_MODEL_GROUPS` in `serving_voice_embed.py`

**Verification:** Integration test verifies request flows to embedding model.

---

## REQ-ROUTE-03: Voice Conversion Routing

**Description:** Implement request routing for ChatterboxVC.

**Details:**
- Route `VoiceConversionRequestInput` → `ChatterboxVC` model
- Add ChatterboxVC to VC endpoint registration

**Verification:** Integration test verifies VC request flows to ChatterboxVC.

---

## REQ-ROUTE-04: Forward Processing

**Description:** Implement forward processing for multi-step voice cloning.

**Details:**
- In `check_forward_processing()`:
  - If result is `SpeechEmbeddingRequestOutput` and requested_model is Chatterbox group → forward to TTS
  - Clear `input_audio`, set `input_embedding` from result
- Single-step completion for all other cases

**Verification:** Integration test verifies full voice cloning pipeline (embed → forward → synthesize).

---

## REQ-CFG-01: Model Config Extension

**Description:** Support watermark configuration in ModelConfig.

**Details:**
- Add `watermark: bool = True` field to `ModelConfig`
- Read in loader and pass to model generation
- Default to True (perth watermarking enabled)

**Verification:** Unit test verifies config field is accessible.

---

## REQ-CFG-02: Config Examples

**Description:** Add example model configurations to config.yml.

**Details:**
- Add `chatterbox-tts` config entry
- Add `chatterbox-turbo-tts` config entry
- Add `chatterbox-multilingual-tts` config entry
- Add `chatterbox-vc` config entry
- Include `voices` list where applicable
- Include `watermark` field demonstration

**Verification:** Config loads without errors in HSE initialization.

---

## REQ-DEP-01: Dependencies

**Description:** Add required dependencies to requirements-common.txt.

**Details:**
- Add `perth` for watermarking
- Add `pyloudnorm` for Turbo loudness normalization
- Add `chatterbox-tts` for model package

**Verification:** `pip install -r requirements-common.txt` succeeds without errors.

---

## REQ-TEST-01: Unit Tests

**Description:** Implement unit tests for new components.

**Details:**
- `test_chatterbox_loader.py` — ModelRegistry and loader tests
- `test_chatterbox_inputs.py` — Input preparation tests
- `test_chatterbox_routing.py` — Routing logic tests

**Verification:** `pytest tests/unit/inference_flow/test_chatterbox_*.py -v` passes.

---

## REQ-TEST-02: Integration Tests

**Description:** Implement integration tests for request flows.

**Details:**
- `test_chatterbox_flow.py` — Full request → output flow with mocked models
- Test voice cloning pipeline (embed → forward → synthesize)
- Test single-speaker TTS
- Test voice conversion

**Verification:** `pytest tests/integration/test_chatterbox_flow.py -v` passes.

---

## REQ-TEST-03: E2E Tests

**Description:** Implement end-to-end tests with real model downloads.

**Details:**
- `test_chatterbox_e2e.py` — Actual model inference
- Mark with `@pytest.mark.e2e` and `@pytest.mark.slow`
- Skip if model not available or GPU required but unavailable

**Verification:** `pytest tests/e2e/test_chatterbox_e2e.py -v` passes (or skips gracefully).

---

## REQ-DOC-01: API Documentation

**Description:** Update API documentation for new model endpoints.

**Details:**
- Document Chatterbox model group in OpenAPI spec
- Document new generation options fields
- Document supported languages for multilingual model
- Add example cURL requests

**Verification:** API docs accessible via `/docs` endpoint.

---

## REQ-PERF-01: No Temp File I/O

**Description:** Ensure no temporary files are created during inference.

**Details:**
- All audio processing uses in-memory BytesIO
- Embedding computation uses `prepare_conditionals_from_audio()` wrapper
- No `NamedTemporaryFile` usage for Chatterbox

**Verification:** Code review + integration test monitoring file handle count.

---

## REQ-PERF-02: Embedding Cache Potential

**Description:** Architecture supports future embedding caching.

**Details:**
- Multi-step routing ensures embeddings are computed as separate step
- Embedding output is serializable (Conditionals → base64)
- Future cache layer can intercept between embed and synthesize steps

**Verification:** Architecture review confirms cacheability.

---

## Open Questions

| ID | Question | Status |
|---|---|---|
| OQ-01 | Should multilingual model languages be exposed in model card voices list? | Open — depends on API design preference |
| OQ-02 | How to handle Turbo's ignored params (exaggeration, cfg_weight, min_p) — warn or silent ignore? | Open — need consistency with other model error handling |
| OQ-03 | Should we implement eager loading of Chatterbox models on startup, or lazy load? | Open — follow existing HSE pattern (lazy) |

---

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| REQ-DEP-01 | Phase 1: Dependencies & Setup | Pending |
| REQ-ARCH-01 | Phase 2: Model Registration & Loading | Pending |
| REQ-ARCH-02 | Phase 2: Model Registration & Loading | Pending |
| REQ-INPUT-01 | Phase 3: Input Preparation | Pending |
| REQ-INPUT-02 | Phase 3: Input Preparation | Pending |
| REQ-INPUT-03 | Phase 3: Input Preparation | Pending |
| REQ-INPUT-04 | Phase 3: Input Preparation | Pending |
| REQ-EXEC-01 | Phase 4: Model Execution | Pending |
| REQ-EXEC-02 | Phase 4: Model Execution | Pending |
| REQ-EXEC-03 | Phase 4: Model Execution | Pending |
| REQ-ROUTE-01 | Phase 5: Request Routing | Pending |
| REQ-ROUTE-02 | Phase 5: Request Routing | Pending |
| REQ-ROUTE-03 | Phase 5: Request Routing | Pending |
| REQ-ROUTE-04 | Phase 5: Request Routing | Pending |
| REQ-CFG-01 | Phase 6: Configuration & Performance | Pending |
| REQ-CFG-02 | Phase 6: Configuration & Performance | Pending |
| REQ-PERF-01 | Phase 6: Configuration & Performance | Pending |
| REQ-PERF-02 | Phase 6: Configuration & Performance | Pending |
| REQ-TEST-01 | Phase 7: Testing & Documentation | Pending |
| REQ-TEST-02 | Phase 7: Testing & Documentation | Pending |
| REQ-TEST-03 | Phase 7: Testing & Documentation | Pending |
| REQ-DOC-01 | Phase 7: Testing & Documentation | Pending |

*Last updated: 2026-03-12*
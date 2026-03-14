# Phase 6: Configuration & Performance — CONTEXT

## Phase Summary

**Goal:** Extend `ModelConfig` with a `watermark` field, add Chatterbox config examples to both YAML files, make `ChatterboxEmbedding` optional via TTS-model fallback routing, and satisfy both performance requirements (no-tempfile unit test + caching docstring).

**Requirements:** REQ-CFG-01, REQ-CFG-02, REQ-PERF-01, REQ-PERF-02

**Depends on:** Phase 5 (all routing logic already implemented and working ✅)

---

## What Phase 5 Left Ready

The following is already implemented and working (do **not** touch):

- `reroute_request_chatterbox()` in [`harmonyspeech/engine/harmonyspeech_engine.py`](harmonyspeech/engine/harmonyspeech_engine.py) — searches `model_configs` for `cfg.model_type == "ChatterboxEmbedding"` exclusively
- All 3 Chatterbox TTS variants + `ChatterboxEmbedding` + `ChatterboxVC` registered in serving layer
- `_execute_chatterbox_embedding()` in `model_runner_base.py` — uses in-memory `BytesIO` (no temp files)
- `check_forward_processing()` Chatterbox block — transfers embedding → TTS via `input_embedding`

Phase 6 extends 5 subsystems:
1. `ModelConfig` — add `watermark` field
2. `loader.py` — apply watermark swap based on config
3. `harmonyspeech_engine.py` — make `ChatterboxEmbedding` optional (fallback to TTS)
4. `task_handler/inputs.py` + `model_runner_base.py` — dual-dispatch for TTS models handling embed requests
5. `serving_voice_embed.py` — register all 3 TTS types as embedding-capable
6. `config.yml` + `config.gpu.yml` — add Chatterbox example entries

---

## Gray Areas Resolved

### GA-1: Watermark propagation path

**Finding:** [`ChatterboxTTS.generate()`](C:/Users/sge20/miniconda3/envs/hse/Lib/site-packages/chatterbox/tts.py) has **no `watermark` parameter**. All 4 Chatterbox model variants always call `self.watermarker.apply_watermark(wav, sample_rate=self.sr)` unconditionally as the last line of `generate()`. The `perth` library provides `DummyWatermarker` — which simply rounds the audio array — as the documented opt-out mechanism.

**Decision:** After loading a Chatterbox model in `loader.py`, swap `model.watermarker = perth.DummyWatermarker()` when `model_config.watermark is False`. All 4 model variants (`ChatterboxTTS`, `ChatterboxTurboTTS`, `ChatterboxMultilingualTTS`, `ChatterboxVC`) need this swap applied.

```python
# In loader.py — after each ChatterboxXxx.from_pretrained():
import perth
if not getattr(model_config, 'watermark', True):
    model.watermarker = perth.DummyWatermarker()
```

### GA-2: ModelConfig.watermark threading

**Decision:** Add `watermark: bool = True` as an optional kwarg to [`ModelConfig.__init__()`](harmonyspeech/common/config.py:44). Since `EngineConfig.load_config_from_yaml()` does `ModelConfig(device_config=..., **model_cfg)`, any `watermark:` field in YAML is automatically passed through to `ModelConfig`. The full propagation chain is:

```
config.yml: watermark: false
    → EngineConfig.load_config_from_yaml()
    → ModelConfig(watermark=False)   ← stored as self.watermark
    → loader.py get_model() reads model_config.watermark
    → model.watermarker = perth.DummyWatermarker()
    → model.generate() → self.watermarker.apply_watermark() → noop (rounds audio)
```

No executor changes needed — watermark is fully handled at model-load time.

### GA-3: Config model: field + ChatterboxEmbedding as optional

**`model:` field value:** Use `"resemble-ai/chatterbox"` for all Chatterbox variants — consistent with KittenTTS precedent (`"KittenML/kitten-tts-mini-0.8"`) of using the real source HF repo for clarity even when the native loader doesn't use it for downloads.

**Config placement:**
- `config.yml` (CPU): Chatterbox entries as **commented-out** with a `# NOTE: Chatterbox TTS requires CUDA for practical use` comment
- `config.gpu.yml` (GPU/CUDA): Chatterbox entries **active** with `device: cuda`
- `ChatterboxEmbedding` is a **commented-out optional** entry in both files

**ChatterboxEmbedding optional fallback:** The routing in `reroute_request_chatterbox()` is enhanced to fall back to the TTS model executor when no `ChatterboxEmbedding` config entry is present. This lets users run with a single `ChatterboxTTS` entry and still serve embedding requests.

### GA-4: PERF requirement deliverables

| Requirement | Phase 6 Deliverable | Phase 7 Deliverable |
|-------------|---------------------|---------------------|
| PERF-01 (no temp files) | Unit test: mock patching verifies `tempfile` + `open()` not called | Integration test: file handle count monitoring |
| PERF-02 (caching architecture) | Code comment in `_execute_chatterbox_embedding()` documenting cache extension point | Nothing additional |

---

## Implementation Details

### 1. ModelConfig Extension (REQ-CFG-01)

**File:** [`harmonyspeech/common/config.py`](harmonyspeech/common/config.py)

Add `watermark: bool = True` to `ModelConfig.__init__()`:

```python
def __init__(
    self,
    name: str,
    model: str,
    model_type: str,
    max_batch_size: int,
    device_config: DeviceConfig,
    language: Optional[str] = None,
    voices: Optional[List[str]] = None,
    trust_remote_code: Optional[bool] = False,
    download_dir: Optional[str] = None,
    load_format: Optional[str] = "auto",
    dtype: Optional[Union[str, torch.dtype]] = "bfloat16",
    seed: Optional[int] = 0,
    revision: Optional[str] = None,
    code_revision: Optional[str] = None,
    enforce_eager: bool = True,
    watermark: bool = True,          # ← ADD THIS
) -> None:
    ...
    self.watermark = watermark       # ← ADD THIS
```

**Verification:** `assert model_config.watermark is True` / `is False` depending on YAML value.

---

### 2. Loader Watermark Swap (REQ-CFG-01)

**File:** [`harmonyspeech/modeling/loader.py`](harmonyspeech/modeling/loader.py)

Add `import perth` at the top. After each native Chatterbox load branch, apply the watermarker swap:

```python
# In the Chatterbox native block (after each from_pretrained call):

if model_config.model_type == "ChatterboxTTS":
    device = str(device_config.device)
    model = ChatterboxTTSModel.from_pretrained(device=device)
    if not getattr(model_config, 'watermark', True):
        model.watermarker = perth.DummyWatermarker()
    return model

elif model_config.model_type == "ChatterboxTurboTTS":
    device = str(device_config.device)
    model = ChatterboxTurboTTSModel.from_pretrained(device=device)
    if not getattr(model_config, 'watermark', True):
        model.watermarker = perth.DummyWatermarker()
    return model

elif model_config.model_type == "ChatterboxMultilingualTTS":
    device = str(device_config.device)
    model = ChatterboxMultilingualTTSModel.from_pretrained(device=device)
    if not getattr(model_config, 'watermark', True):
        model.watermarker = perth.DummyWatermarker()
    return model

elif model_config.model_type == "ChatterboxVC":
    device = str(device_config.device)
    model = ChatterboxVCModel.from_pretrained(device=device)
    if not getattr(model_config, 'watermark', True):
        model.watermarker = perth.DummyWatermarker()
    return model
```

**Note:** `perth` is already a transitive dependency (installed via `requirements-common.txt`) — no new requirements entry needed.

---

### 3. Optional ChatterboxEmbedding — Routing Fallback

**File:** [`harmonyspeech/engine/harmonyspeech_engine.py`](harmonyspeech/engine/harmonyspeech_engine.py)

Modify `reroute_request_chatterbox()` to fall back to the TTS model when no `ChatterboxEmbedding` executor is configured:

```python
def reroute_request_chatterbox(self, request: RequestInput):
    _CHATTERBOX_TTS_TYPE_MAP = {
        "chatterbox": "ChatterboxTTS",
        "chatterbox_turbo": "ChatterboxTurboTTS",
        "chatterbox_multilingual": "ChatterboxMultilingualTTS",
    }

    if (isinstance(request, SpeechEmbeddingRequestInput) or
        (
            isinstance(request, TextToSpeechRequestInput) and
            request.input_audio is not None and
            request.input_embedding is None
        )
    ):
        # Prefer dedicated ChatterboxEmbedding executor if configured
        for cfg in self.model_configs:
            if cfg.model_type == "ChatterboxEmbedding":
                request.model = cfg.name
                break
        else:
            # Fallback: route embedding request to the TTS model executor
            # (ChatterboxTTS also has prepare_conditionals — no separate model needed)
            tts_type = _CHATTERBOX_TTS_TYPE_MAP.get(request.requested_model, "ChatterboxTTS")
            for cfg in self.model_configs:
                if cfg.model_type == tts_type:
                    request.model = cfg.name
                    break

    elif (
        isinstance(request, TextToSpeechRequestInput) and
        (request.input_audio is None)
    ):
        tts_type = _CHATTERBOX_TTS_TYPE_MAP.get(request.requested_model)
        if tts_type:
            for cfg in self.model_configs:
                if cfg.model_type == tts_type:
                    request.model = cfg.name
                    break
```

**⚠️ Python `for…else` note:** The `else` clause on a `for` loop executes when the loop completes **without a `break`** — i.e., when no `ChatterboxEmbedding` config entry was found. This is the correct Python idiom for "try to find X, fall back to Y".

---

### 4. Dual-Dispatch: TTS Models Handle Embed Requests

When no `ChatterboxEmbedding` is configured, embedding requests are routed to the TTS executor. Two dispatch points need updating:

#### 4a. `task_handler/inputs.py` — `prepare_inputs()` dispatch

**File:** [`harmonyspeech/task_handler/inputs.py`](harmonyspeech/task_handler/inputs.py)

Add `SpeechEmbeddingRequestInput` handling to the TTS model branches. Currently around line 212 the `ChatterboxEmbedding` branch exists. Add parallel handling to the TTS branches:

```python
elif model_config.model_type in ["ChatterboxTTS", "ChatterboxTurboTTS", "ChatterboxMultilingualTTS"]:
    inputs = []
    for r in requests_to_batch:
        if isinstance(r.request_data, SpeechEmbeddingRequestInput):
            # Embedding request routed to TTS model (no dedicated ChatterboxEmbedding)
            inputs.extend(prepare_chatterbox_embedding_inputs([r]))
        elif isinstance(r.request_data, TextToSpeechRequestInput):
            if model_config.model_type == "ChatterboxTTS":
                inputs.extend(prepare_chatterbox_tts_inputs([r]))
            elif model_config.model_type == "ChatterboxTurboTTS":
                inputs.extend(prepare_chatterbox_turbo_tts_inputs([r]))
            elif model_config.model_type == "ChatterboxMultilingualTTS":
                inputs.extend(prepare_chatterbox_multilingual_tts_inputs([r]))
        else:
            raise ValueError(
                f"ChatterboxTTS prepare_inputs: request ID {r.request_id} is not "
                f"TextToSpeechRequestInput or SpeechEmbeddingRequestInput"
            )
    return inputs
```

**Current code (before this phase):** The existing `ChatterboxTTS/Turbo/MLT` branches in `prepare_inputs()` only handle `TextToSpeechRequestInput` — adding the `SpeechEmbeddingRequestInput` case enables fallback embedding.

#### 4b. `model_runner_base.py` — `execute_model()` dispatch

**File:** [`harmonyspeech/task_handler/model_runner_base.py`](harmonyspeech/task_handler/model_runner_base.py)

Add `SpeechEmbeddingRequestInput` handling inside the existing TTS dispatch branches:

```python
elif model_type == "ChatterboxTTS":
    # Check if this is an embedding request routed to the TTS model
    if requests_to_batch and isinstance(requests_to_batch[0].request_data, SpeechEmbeddingRequestInput):
        outputs = self._execute_chatterbox_embedding(inputs, requests_to_batch)
    else:
        outputs = self._execute_chatterbox_tts(inputs, requests_to_batch)

elif model_type == "ChatterboxTurboTTS":
    if requests_to_batch and isinstance(requests_to_batch[0].request_data, SpeechEmbeddingRequestInput):
        outputs = self._execute_chatterbox_embedding(inputs, requests_to_batch)
    else:
        outputs = self._execute_chatterbox_turbo_tts(inputs, requests_to_batch)

elif model_type == "ChatterboxMultilingualTTS":
    if requests_to_batch and isinstance(requests_to_batch[0].request_data, SpeechEmbeddingRequestInput):
        outputs = self._execute_chatterbox_embedding(inputs, requests_to_batch)
    else:
        outputs = self._execute_chatterbox_multilingual_tts(inputs, requests_to_batch)
```

**Note on batching:** All Chatterbox models use `max_batch_size: 1` — so `requests_to_batch[0]` is safe for dispatch type checking. If a batch ever contains mixed request types (embed + TTS), that is a scheduler invariant violation, not a model runner concern.

---

### 5. Embedding Serving Layer — Register TTS Types

**File:** [`harmonyspeech/endpoints/openai/serving_voice_embed.py`](harmonyspeech/endpoints/openai/serving_voice_embed.py)

Add the 3 TTS model types to `_EMBEDDING_MODEL_TYPES` and their groups to `_EMBEDDING_MODEL_GROUPS`:

```python
_EMBEDDING_MODEL_TYPES = [
    "HarmonySpeechEncoder",
    "ChatterboxEmbedding",
    "ChatterboxTTS",              # ADD — handles embed when no ChatterboxEmbedding configured
    "ChatterboxTurboTTS",         # ADD
    "ChatterboxMultilingualTTS",  # ADD
]

_EMBEDDING_MODEL_GROUPS = {
    "harmonyspeech": ["HarmonySpeechEncoder"],
    "openvoice_v1": ["FasterWhisper", "OpenVoiceV1ToneConverterEncoder"],
    "openvoice_v2": ["FasterWhisper", "OpenVoiceV2ToneConverterEncoder"],
    "chatterbox": ["ChatterboxEmbedding", "ChatterboxTTS"],              # ADD ChatterboxTTS fallback
    "chatterbox_turbo": ["ChatterboxTurboTTS"],                          # ADD
    "chatterbox_multilingual": ["ChatterboxMultilingualTTS"],            # ADD
}
```

**Note on `"chatterbox"` group:** When `ChatterboxEmbedding` is present in config, it appears in the group first — the serving engine uses this list to check which models to activate. The `ChatterboxTTS` fallback entry ensures the group is non-empty even when no dedicated embedding model is configured.

---

### 6. Config Examples (REQ-CFG-02)

#### `config.yml` (CPU — commented out with warning)

Add after the KittenTTS section:

```yaml
  # ===========================================================================
  # Chatterbox TTS - High-quality voice cloning TTS (requires CUDA for practical use)
  # ===========================================================================
  # NOTE: Chatterbox models are ~1-2GB and run very slowly on CPU.
  #       For real-time use, enable in config.gpu.yml instead.
  #
  # Chatterbox TTS (standard) - voice cloning + 8 languages
  # - name: "chatterbox"
  #   model: "resemble-ai/chatterbox"
  #   model_type: "ChatterboxTTS"
  #   watermark: true          # Set to false to disable perth audio watermarking
  #   max_batch_size: 1
  #   dtype: "float32"
  #   device_config:
  #     device: "cpu"

  # Chatterbox Turbo TTS - faster synthesis with loudness normalization
  # - name: "chatterbox_turbo"
  #   model: "resemble-ai/chatterbox"
  #   model_type: "ChatterboxTurboTTS"
  #   watermark: true
  #   max_batch_size: 1
  #   dtype: "float32"
  #   device_config:
  #     device: "cpu"

  # Chatterbox Multilingual TTS - 23-language support with voice cloning
  # - name: "chatterbox_multilingual"
  #   model: "resemble-ai/chatterbox"
  #   model_type: "ChatterboxMultilingualTTS"
  #   watermark: true
  #   max_batch_size: 1
  #   dtype: "float32"
  #   device_config:
  #     device: "cpu"

  # Chatterbox Voice Conversion - convert speech from one voice to another
  # - name: "chatterbox_vc"
  #   model: "resemble-ai/chatterbox"
  #   model_type: "ChatterboxVC"
  #   watermark: true
  #   max_batch_size: 1
  #   dtype: "float32"
  #   device_config:
  #     device: "cpu"

  # Chatterbox Embedding (OPTIONAL) - dedicated executor for /v1/embeddings endpoint
  # If omitted, embedding requests are handled by the ChatterboxTTS executor above.
  # Add this only if you need concurrent TTS + high-throughput standalone embeddings.
  # - name: "chatterbox_embedding"
  #   model: "resemble-ai/chatterbox"
  #   model_type: "ChatterboxEmbedding"
  #   max_batch_size: 1
  #   dtype: "float32"
  #   device_config:
  #     device: "cpu"
```

#### `config.gpu.yml` (CUDA — active)

Add after the KittenTTS section:

```yaml
  # ===========================================================================
  # Chatterbox TTS - High-quality voice cloning TTS
  # ===========================================================================

  # Chatterbox TTS (standard) - voice cloning + 8 languages
  - name: "chatterbox"
    model: "resemble-ai/chatterbox"
    model_type: "ChatterboxTTS"
    watermark: true          # Set to false to disable perth audio watermarking
    max_batch_size: 1
    dtype: "float32"
    device_config:
      device: "cuda"

  # Chatterbox Turbo TTS - faster synthesis with loudness normalization
  # - name: "chatterbox_turbo"
  #   model: "resemble-ai/chatterbox"
  #   model_type: "ChatterboxTurboTTS"
  #   watermark: true
  #   max_batch_size: 1
  #   dtype: "float32"
  #   device_config:
  #     device: "cuda"

  # Chatterbox Multilingual TTS - 23-language support with voice cloning
  # - name: "chatterbox_multilingual"
  #   model: "resemble-ai/chatterbox"
  #   model_type: "ChatterboxMultilingualTTS"
  #   watermark: true
  #   max_batch_size: 1
  #   dtype: "float32"
  #   device_config:
  #     device: "cuda"

  # Chatterbox Voice Conversion - convert speech from one voice to another
  # - name: "chatterbox_vc"
  #   model: "resemble-ai/chatterbox"
  #   model_type: "ChatterboxVC"
  #   watermark: true
  #   max_batch_size: 1
  #   dtype: "float32"
  #   device_config:
  #     device: "cuda"

  # Chatterbox Embedding (OPTIONAL) - dedicated executor for /v1/embeddings endpoint
  # If omitted, embedding requests are handled by the ChatterboxTTS executor above.
  # - name: "chatterbox_embedding"
  #   model: "resemble-ai/chatterbox"
  #   model_type: "ChatterboxEmbedding"
  #   max_batch_size: 1
  #   dtype: "float32"
  #   device_config:
  #     device: "cuda"
```

---

### 7. PERF-01: No-Tempfile Unit Test

**File:** `tests/unit/inference_flow/test_chatterbox_no_tempfile.py` (new)

```python
"""Unit test: verify Chatterbox execution path does not create temporary files."""
import io
import base64
from unittest.mock import patch, MagicMock
import pytest


def test_chatterbox_tts_no_tempfile(mock_model_runner_chatterbox):
    """_execute_chatterbox_tts() must not call tempfile or open() for file I/O."""
    with patch("builtins.open") as mock_open, \
         patch("tempfile.NamedTemporaryFile") as mock_tmp, \
         patch("tempfile.mkstemp") as mock_mkstemp:
        # Execute a TTS request (using mocked model)
        mock_model_runner_chatterbox._execute_chatterbox_tts(
            inputs=[("Hello world", None, 0.5, 0.5, 0.8, 1.2, 1.0, 0.05)],
            requests_to_batch=[...]
        )
        mock_open.assert_not_called()
        mock_tmp.assert_not_called()
        mock_mkstemp.assert_not_called()
```

**Note:** Full test implementation details (fixtures, mock setup) are left to the plan executor. The key assertion is that `builtins.open`, `tempfile.NamedTemporaryFile`, and `tempfile.mkstemp` are **never called** during Chatterbox execute methods.

---

### 8. PERF-02: Cache Extension Point (Documentation)

**File:** [`harmonyspeech/task_handler/model_runner_base.py`](harmonyspeech/task_handler/model_runner_base.py)

Add a docstring comment to `_execute_chatterbox_embedding()`:

```python
def _execute_chatterbox_embedding(self, inputs, requests_to_batch):
    """Execute ChatterboxEmbedding to compute voice Conditionals from audio.

    Architecture note (REQ-PERF-02):
    The output is a base64-encoded serialized Conditionals object (via torch.save →
    BytesIO → base64). This design supports a future embedding cache layer:

        Cache key:   hash(audio_bytes)  or  user-provided voice_id
        Cache value: base64 Conditionals string (returned by this method)
        Cache hit:   skip this executor entirely; pass Conditionals directly to TTS

    The cache intercept point is in check_forward_processing() in harmonyspeech_engine.py,
    between the embed step completion and the TTS re-submission. No code changes required
    to this method to enable caching — only the engine's forward-processing block needs
    a cache lookup before re-submitting the forwarding_request.
    """
    ...  # existing implementation unchanged
```

---

## Files to Modify

| File | Change |
|------|--------|
| [`harmonyspeech/common/config.py`](harmonyspeech/common/config.py) | Add `watermark: bool = True` to `ModelConfig.__init__()` |
| [`harmonyspeech/modeling/loader.py`](harmonyspeech/modeling/loader.py) | Add `import perth`; apply `model.watermarker = perth.DummyWatermarker()` in all 4 Chatterbox native branches |
| [`harmonyspeech/engine/harmonyspeech_engine.py`](harmonyspeech/engine/harmonyspeech_engine.py) | Add `for…else` fallback in `reroute_request_chatterbox()` for embedding → TTS model |
| [`harmonyspeech/task_handler/inputs.py`](harmonyspeech/task_handler/inputs.py) | Add `SpeechEmbeddingRequestInput` handling to `ChatterboxTTS/Turbo/MLT` branches in `prepare_inputs()` |
| [`harmonyspeech/task_handler/model_runner_base.py`](harmonyspeech/task_handler/model_runner_base.py) | Add embed-request dispatch to TTS branches in `execute_model()`; add cache docstring to `_execute_chatterbox_embedding()` |
| [`harmonyspeech/endpoints/openai/serving_voice_embed.py`](harmonyspeech/endpoints/openai/serving_voice_embed.py) | Add `ChatterboxTTS/Turbo/MLT` to `_EMBEDDING_MODEL_TYPES` and `_EMBEDDING_MODEL_GROUPS` |
| [`config.yml`](config.yml) | Add Chatterbox entries (commented out, with CUDA warning) |
| [`config.gpu.yml`](config.gpu.yml) | Add Chatterbox entries (active, `chatterbox` TTS enabled; others commented) |

## Files to Create

| File | Purpose |
|------|---------|
| `tests/unit/inference_flow/test_chatterbox_no_tempfile.py` | PERF-01 unit test |

---

## Success Criteria Mapping (from ROADMAP Phase 6)

| Criterion | How it's satisfied |
|-----------|-------------------|
| 1. `ModelConfig` has accessible `watermark: bool` field (default True) | `watermark: bool = True` added to `ModelConfig.__init__()` |
| 2. Config examples for all 4 model variants load without errors | Entries added to `config.yml` (commented) and `config.gpu.yml` (active) |
| 3. No temp files during inference (verified via integration test) | `test_chatterbox_no_tempfile.py` unit test + architecture uses BytesIO throughout |
| 4. Multi-step routing architecture supports future embedding caching (serializable Conditionals) | Docstring in `_execute_chatterbox_embedding()` documents the cache intercept point |

---

## Key Architecture Diagram: Embedding Fallback Flow

```
User config.yml:
  - name: "chatterbox"        ← only this entry (no ChatterboxEmbedding)
    model_type: ChatterboxTTS

API: POST /v1/embeddings  { "model": "chatterbox", "input_audio": "..." }
    │
    ▼
serving_voice_embed.py
  _EMBEDDING_MODEL_TYPES: [..., "ChatterboxTTS"]      ← Phase 6 added
  _EMBEDDING_MODEL_GROUPS["chatterbox"]: ["ChatterboxEmbedding", "ChatterboxTTS"]  ← Phase 6
    │
    ▼
HarmonySpeechEngine.add_request()
    │
    ▼
check_reroute_request_to_model()
  requested_model == "chatterbox" → reroute_request_chatterbox()
    │
    ├── for cfg in model_configs: cfg.model_type == "ChatterboxEmbedding" → NOT FOUND
    └── else: fallback → cfg.model_type == "ChatterboxTTS" → request.model = "chatterbox"
    │
    ▼
Scheduler → ChatterboxTTS executor
    │
    ▼
model_runner_base.execute_model()
  model_type == "ChatterboxTTS"
  requests_to_batch[0] is SpeechEmbeddingRequestInput
    → _execute_chatterbox_embedding(inputs, requests_to_batch)  ← dual-dispatch
    │
    ▼
SpeechEmbeddingRequestOutput (base64 Conditionals) returned to caller
```

---

## Watermark Config Diagram

```
config.yml:
  - name: "chatterbox"
    model_type: ChatterboxTTS
    watermark: false        ← user opts out

EngineConfig.load_config_from_yaml()
  → ModelConfig(watermark=False)

loader.py get_model()
  ChatterboxTTSModel.from_pretrained(device=device)
  if not model_config.watermark:
    model.watermarker = perth.DummyWatermarker()

model.generate(text, ...)
  ... inference ...
  watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)
  #   ↑ DummyWatermarker.apply_watermark() just rounds the float array → noop
  return torch.from_numpy(watermarked_wav).unsqueeze(0)
```

---

## Open Questions from Prior Phases

| ID | Question | Resolution |
|----|----------|-----------|
| OQ-03 | Eager vs. lazy loading of Chatterbox? | Follow existing HSE pattern (lazy — loaded on first request) |

---

*Phase 6 CONTEXT generated: 2026-03-14*

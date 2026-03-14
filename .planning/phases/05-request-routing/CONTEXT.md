# Phase 5: Request Routing — CONTEXT

## Phase Summary

**Goal:** Implement routing logic so TTS, Embedding, and VoiceConversion requests for Chatterbox model variants are dispatched to the correct model executor — including the multi-step `input_audio → embed → synthesize` chain.

**Requirements:** REQ-ROUTE-01, REQ-ROUTE-02, REQ-ROUTE-03, REQ-ROUTE-04

**Depends on:** Phase 4 (all 5 `_execute_chatterbox_*` methods and dispatch branches in `execute_model()` are already implemented)

---

## What Phase 4 Left Ready

The following is already implemented and working (do **not** touch):

- All 5 execute functions in [`model_runner_base.py`](harmonyspeech/task_handler/model_runner_base.py):
  - `_execute_chatterbox_tts()`
  - `_execute_chatterbox_turbo_tts()`
  - `_execute_chatterbox_multilingual_tts()`
  - `_execute_chatterbox_embedding()`
  - `_execute_chatterbox_vc()`
- Dispatch branches in `execute_model()` for `ChatterboxTTS`, `ChatterboxTurboTTS`, `ChatterboxMultilingualTTS`, `ChatterboxEmbedding`, `ChatterboxVC`
- All 5 `prepare_chatterbox_*` input functions in [`task_handler/inputs.py`](harmonyspeech/task_handler/inputs.py)

Phase 5 wires up **three layers** to connect API requests to these executors:
1. Engine rerouting (`harmonyspeech_engine.py`)
2. Engine forward processing (`harmonyspeech_engine.py`)
3. Serving layer model registration (`serving_text_to_speech.py`, `serving_voice_embed.py`, `serving_voice_conversion.py`)

---

## Architecture Overview: Routing Flow

```
API Request (TextToSpeechRequest / EmbedSpeakerRequest / VoiceConversionRequest)
    │
    ▼
serving_*.py  →  engine.generate(request_data=...) where requested_model = "chatterbox*"
    │
    ▼
HarmonySpeechEngine.add_request()
    │
    ▼
check_reroute_request_to_model()          ← [Phase 5: add Chatterbox branches]
    │
    ├── request.model = cfg.name (ChatterboxEmbedding)   ← if input_audio present (multi-step)
    └── request.model = cfg.name (ChatterboxTTS/Turbo/Multilingual)  ← direct path
    │
    ▼
Scheduler → Executor → execute_model()    ← [Phase 4: already done]
    │
    ▼
check_forward_processing()                ← [Phase 5: add Chatterbox block]
    │
    ├── SpeechEmbeddingRequestOutput received?
    │     └── input_audio=None, input_embedding=result, re-submit  (→ loops back to synthesize step)
    └── TextToSpeechRequestOutput received?
          └── FINISHED_STOPPED (return to API caller)
```

---

## Key Architecture: `check_reroute_request_to_model()` Pattern

The function at [`harmonyspeech_engine.py:300`](harmonyspeech/engine/harmonyspeech_engine.py:300) dispatches by `request.requested_model`, a string that equals the user's YAML config `name:` field.

**Existing pattern (reference — do not modify):**
```python
def check_reroute_request_to_model(self, request: RequestInput):
    if request.requested_model == "harmonyspeech":
        self.reroute_request_harmonyspeech(request)
    if request.requested_model == "openvoice_v1":
        self.reroute_request_openvoice_v1(request)
    ...
```

**Phase 5 additions (add AFTER existing `if` blocks):**
```python
    # Chatterbox TTS (standard)
    if request.requested_model == "chatterbox":
        self.reroute_request_chatterbox(request)
    # Chatterbox Turbo TTS
    if request.requested_model == "chatterbox_turbo":
        self.reroute_request_chatterbox(request)
    # Chatterbox Multilingual TTS
    if request.requested_model == "chatterbox_multilingual":
        self.reroute_request_chatterbox(request)
    # Chatterbox Voice Conversion
    if request.requested_model == "chatterbox_vc":
        self.reroute_request_chatterbox_vc(request)
```

All three TTS variants share the same `reroute_request_chatterbox()` function because the routing logic (does it have `input_audio`?) is the same. The target `model_type` differs per variant but is looked up from configs.

---

## `reroute_request_chatterbox()` Implementation

**Routing rules for TTS variants (`requested_model` ∈ `["chatterbox", "chatterbox_turbo", "chatterbox_multilingual"]`):**

| Condition | Route to model_type |
|-----------|---------------------|
| `TextToSpeechRequestInput` + `input_audio is not None` + `input_embedding is None` | `ChatterboxEmbedding` (Step 1 of multi-step) |
| `TextToSpeechRequestInput` + `input_audio is None` + `input_embedding is not None` | `ChatterboxTTS` / `ChatterboxTurboTTS` / `ChatterboxMultilingualTTS` (Step 2) |
| `TextToSpeechRequestInput` + `input_audio is None` + `input_embedding is None` | `ChatterboxTTS` / `ChatterboxTurboTTS` / `ChatterboxMultilingualTTS` (direct, no voice cloning) |
| `SpeechEmbeddingRequestInput` | `ChatterboxEmbedding` (standalone embed request) |

**model_type lookup table:**
| `requested_model` | TTS model_type to route to |
|-------------------|---------------------------|
| `"chatterbox"` | `"ChatterboxTTS"` |
| `"chatterbox_turbo"` | `"ChatterboxTurboTTS"` |
| `"chatterbox_multilingual"` | `"ChatterboxMultilingualTTS"` |

**Implementation:**
```python
def reroute_request_chatterbox(self, request: RequestInput):
    # Map sentinel name → TTS model_type
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
        # Route to embedding model (Step 1 of multi-step, or standalone embed)
        for cfg in self.model_configs:
            if cfg.model_type == "ChatterboxEmbedding":
                request.model = cfg.name
                break
    elif (
        isinstance(request, TextToSpeechRequestInput) and
        (request.input_audio is None)  # direct path or Step 2 after embedding
    ):
        # Route to the appropriate TTS variant
        tts_type = _CHATTERBOX_TTS_TYPE_MAP.get(request.requested_model)
        if tts_type:
            for cfg in self.model_configs:
                if cfg.model_type == tts_type:
                    request.model = cfg.name
                    break
```

---

## `reroute_request_chatterbox_vc()` Implementation

**Routing rules for VC (`requested_model == "chatterbox_vc"`):**

| Condition | Route to model_type |
|-----------|---------------------|
| `VoiceConversionRequestInput` | `ChatterboxVC` |

```python
def reroute_request_chatterbox_vc(self, request: RequestInput):
    if isinstance(request, VoiceConversionRequestInput):
        for cfg in self.model_configs:
            if cfg.model_type == "ChatterboxVC":
                request.model = cfg.name
                break
```

---

## `check_forward_processing()` Extension

The function at [`harmonyspeech_engine.py:314`](harmonyspeech/engine/harmonyspeech_engine.py:314) handles what happens after a model returns a result. Phase 5 adds a **new separate block** for Chatterbox (after the existing HarmonySpeech/OpenVoice blocks).

**Add this block AFTER the existing `if isinstance(input_data, TextToSpeechRequestInput) and requested_model in ["harmonyspeech", "openvoice_v1", "openvoice_v2"]` block:**

```python
# Multi-step TTS Processing (Chatterbox — embed then synthesize)
if isinstance(input_data, TextToSpeechRequestInput) and requested_model in [
    "chatterbox",
    "chatterbox_turbo",
    "chatterbox_multilingual"
]:
    forwarding_request = input_data

    if isinstance(result.result_data, SpeechEmbeddingRequestOutput):
        # Embedding step finished — forward to TTS synthesize step
        # Mark as forwarded in scheduler
        new_status = RequestStatus.FINISHED_FORWARDED
        self.scheduler.update_request_status(result.request_id, new_status)
        # Embedding result becomes input_embedding; clear input_audio
        forwarding_request.input_audio = None
        forwarding_request.input_embedding = result.result_data.output
        self.add_request(result.request_id, forwarding_request)
```

**Why no additional branches (transcription, vocoder)?**
- Chatterbox does NOT use VAD/transcription as a preprocessing step (unlike OpenVoice)
- Chatterbox does NOT have a separate vocoder model (unlike HarmonySpeech)
- The chain is exactly: `[input_audio → ChatterboxEmbedding → input_embedding → ChatterboxTTS*]`

---

## Serving Layer Registrations

### `serving_text_to_speech.py`

Add to `_TTS_MODEL_TYPES` and `_TTS_MODEL_GROUPS`:

```python
_TTS_MODEL_TYPES = [
    "OpenVoiceV1Synthesizer",
    "MeloTTSSynthesizer",
    "KittenTTSSynthesizer",
    "ChatterboxTTS",             # ADD
    "ChatterboxTurboTTS",        # ADD
    "ChatterboxMultilingualTTS", # ADD
]

_TTS_MODEL_GROUPS = {
    "harmonyspeech": ["HarmonySpeechSynthesizer", "HarmonySpeechVocoder"],
    "openvoice_v1": ["OpenVoiceV1Synthesizer", "OpenVoiceV1ToneConverter"],
    "openvoice_v2": ["MeloTTSSynthesizer", "OpenVoiceV2ToneConverter"],
    "chatterbox": ["ChatterboxTTS"],              # ADD
    "chatterbox_turbo": ["ChatterboxTurboTTS"],   # ADD
    "chatterbox_multilingual": ["ChatterboxMultilingualTTS"],  # ADD
}
```

Each group has exactly **one member** because Chatterbox TTS variants are self-contained (no secondary tone converter needed). The group key (`"chatterbox"`, `"chatterbox_turbo"`, `"chatterbox_multilingual"`) matches the `requested_model` sentinel used in the engine rerouting logic.

### `serving_voice_embed.py`

Add to `_EMBEDDING_MODEL_TYPES` and `_EMBEDDING_MODEL_GROUPS`:

```python
_EMBEDDING_MODEL_TYPES = [
    "HarmonySpeechEncoder",
    "ChatterboxEmbedding",     # ADD
]

_EMBEDDING_MODEL_GROUPS = {
    "harmonyspeech": ["HarmonySpeechEncoder"],
    "openvoice_v1": ["FasterWhisper", "OpenVoiceV1ToneConverterEncoder"],
    "openvoice_v2": ["FasterWhisper", "OpenVoiceV2ToneConverterEncoder"],
    "chatterbox": ["ChatterboxEmbedding"],  # ADD
}
```

**Why expose `ChatterboxEmbedding` via API?** Enables pre-computation + caching:
1. User calls `/embed` → gets base64 `Conditionals`
2. User stores it client-side
3. User passes `input_embedding` to every future TTS request → skips embedding step, direct TTS

This supports Phase 6's "future embedding caching" goal and directly implements REQ-ROUTE-02.

### `serving_voice_conversion.py`

Add to `_VOICE_CONVERSION_MODEL_TYPES` and `_VOICE_CONVERSION_MODEL_GROUPS`:

```python
_VOICE_CONVERSION_MODEL_TYPES = [
    "OpenVoiceV1ToneConverter",
    "OpenVoiceV2ToneConverter",
    "ChatterboxVC",            # ADD
]

_VOICE_CONVERSION_MODEL_GROUPS = {
    "openvoice_v1": ["OpenVoiceV1ToneConverter"],
    "openvoice_v2": ["OpenVoiceV2ToneConverter"],
    "chatterbox_vc": ["ChatterboxVC"],  # ADD
}
```

---

## Success Criteria Mapping (from ROADMAP)

| Criterion | How it's satisfied |
|-----------|-------------------|
| 1. TTS without voice cloning routes directly to TTS model | `reroute_request_chatterbox()` — `input_audio is None` + `input_embedding is None` → ChatterboxTTS* |
| 2. TTS with pre-computed embedding routes directly to TTS model | `reroute_request_chatterbox()` — `input_audio is None` + `input_embedding is not None` → ChatterboxTTS* |
| 3. TTS with `input_audio` routes to Embedding first, then forward to TTS | `reroute_request_chatterbox()` routes to `ChatterboxEmbedding`; `check_forward_processing()` then forwards with `input_embedding` set |
| 4. Embedding requests route to ChatterboxEmbedding model | `reroute_request_chatterbox()` — `SpeechEmbeddingRequestInput` → `ChatterboxEmbedding` |
| 5. VoiceConversion requests route to ChatterboxVC model | `reroute_request_chatterbox_vc()` → `ChatterboxVC` |
| 6. Forward processing transfers embedding from embed step to synthesize step | `check_forward_processing()` new Chatterbox block — `input_audio=None`, `input_embedding=result`, re-submit |

---

## Files to Modify

| File | Change |
|------|--------|
| [`harmonyspeech/engine/harmonyspeech_engine.py`](harmonyspeech/engine/harmonyspeech_engine.py) | Add `reroute_request_chatterbox()`, `reroute_request_chatterbox_vc()`; extend `check_reroute_request_to_model()`; add Chatterbox block to `check_forward_processing()` |
| [`harmonyspeech/endpoints/openai/serving_text_to_speech.py`](harmonyspeech/endpoints/openai/serving_text_to_speech.py) | Add 3 Chatterbox types to `_TTS_MODEL_TYPES`; add 3 groups to `_TTS_MODEL_GROUPS` |
| [`harmonyspeech/endpoints/openai/serving_voice_embed.py`](harmonyspeech/endpoints/openai/serving_voice_embed.py) | Add `ChatterboxEmbedding` to `_EMBEDDING_MODEL_TYPES`; add `"chatterbox"` group to `_EMBEDDING_MODEL_GROUPS` |
| [`harmonyspeech/endpoints/openai/serving_voice_conversion.py`](harmonyspeech/endpoints/openai/serving_voice_conversion.py) | Add `ChatterboxVC` to `_VOICE_CONVERSION_MODEL_TYPES`; add `"chatterbox_vc"` group to `_VOICE_CONVERSION_MODEL_GROUPS` |

---

## Imports Needed in `harmonyspeech_engine.py`

The file already imports:
```python
from harmonyspeech.common.inputs import *
from harmonyspeech.common.outputs import *
from harmonyspeech.common.request import EngineRequest, ExecutorResult, RequestStatus
```

No new imports are needed — all required types (`SpeechEmbeddingRequestInput`, `TextToSpeechRequestInput`, `VoiceConversionRequestInput`, `SpeechEmbeddingRequestOutput`) are already available via `*` imports.

---

## Config Example (what users write in YAML)

For the routing to work, users must name their Chatterbox model configs using the sentinel names:

```yaml
# config.yml example — Chatterbox TTS (standard)
models:
  - name: chatterbox               # MUST match sentinel
    model_type: ChatterboxTTS
    device: cuda
    max_batch_size: 1

  - name: chatterbox_embedding     # optional — for standalone /embed
    model_type: ChatterboxEmbedding
    device: cuda
    max_batch_size: 1

# Alternatively, for Turbo:
  - name: chatterbox_turbo
    model_type: ChatterboxTurboTTS
    device: cuda
    max_batch_size: 1

# Voice Conversion:
  - name: chatterbox_vc
    model_type: ChatterboxVC
    device: cuda
    max_batch_size: 1
```

⚠️ **Note:** `ChatterboxEmbedding` is loaded as a separate model executor from `ChatterboxTTS`. They share the same underlying Chatterbox library model internally, but from HSE's perspective, each config entry gets its own executor and model instance.

---

## Resolved Gray Areas

### Gray Area 1: Group Detection Mechanism

**Decision: Name sentinels** — `"chatterbox"`, `"chatterbox_turbo"`, `"chatterbox_multilingual"`, `"chatterbox_vc"` — consistent with existing pattern (`"harmonyspeech"`, `"openvoice_v1"`, etc.).

### Gray Area 2: Forward Chain Isolation

**Decision: New separate Chatterbox block** in `check_forward_processing()` — no coupling to HarmonySpeech/OpenVoice blocks. Chatterbox has a simpler 1-hop chain (embed → synthesize) vs OpenVoice's 3-hop chain (transcribe → embed → synthesize → tone_convert).

### Gray Area 3: ChatterboxEmbedding API Exposure

**Decision: Expose via `/embed` endpoint** — enables pre-computation + caching workflow. Group key: `"chatterbox"` in `_EMBEDDING_MODEL_GROUPS`.

### Gray Area 4: TTS Serving Registration

**Decision: One group per variant** — `"chatterbox"`, `"chatterbox_turbo"`, `"chatterbox_multilingual"` as independent groups. Each group has a single member because Chatterbox TTS variants are self-contained.

---

## Success Criteria (from ROADMAP)

1. TTS requests without voice cloning route directly to TTS model ✓
2. TTS requests with pre-computed embedding route directly to TTS model ✓
3. TTS requests with `input_audio` route to Embedding model first, then forward to TTS (multi-step) ✓
4. Embedding requests route to Chatterbox Embedding model ✓
5. VoiceConversion requests route to ChatterboxVC model ✓
6. Forward processing correctly transfers embedding from embed step to synthesize step ✓

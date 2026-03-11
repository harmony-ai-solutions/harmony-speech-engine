# Project: Chatterbox TTS Integration for Harmony Speech Engine

## Summary

Integrate the Chatterbox TTS model family (by ResembleAI) into the Harmony Speech Engine (HSE), following the established architecture, routing system, and test-driven development conventions. The integration covers all four model variants, multilingual support (23 languages), voice cloning via the multi-step embedding pipeline, standalone voice conversion, and production optimizations for embedding persistence.

## Context

**Repository:** `harmony-ai-solutions/harmony-speech-engine`  
**Reference Codebase:** `.current_work/chatterbox-tts/` (official ResembleAI Chatterbox repo)  
**HuggingFace Repos:**
- `ResembleAI/chatterbox` — standard TTS + multilingual TTS + VC checkpoints
- `ResembleAI/chatterbox-turbo` — turbo TTS variant

**HSE Architecture Pattern:**
- Models registered in `harmonyspeech/modeling/models/__init__.py` (`_MODELS` dict)
- Loader entries in `harmonyspeech/modeling/loader.py` (`_MODEL_CONFIGS`, `_MODEL_WEIGHTS`)
- Input preparation in `harmonyspeech/task_handler/inputs.py` (`prepare_inputs()`)
- Execution dispatch in `harmonyspeech/task_handler/model_runner_base.py` (`execute_model()`)
- Routing logic in `harmonyspeech/engine/harmonyspeech_engine.py` (`reroute_request_*`, `check_forward_processing`)
- Endpoint model group registration in `harmonyspeech/endpoints/openai/serving_*.py`

## Model Variants to Integrate

| Model Type Name | Class | HF Repo | Capabilities |
|---|---|---|---|
| `ChatterboxTTS` | `ChatterboxTTS` | `ResembleAI/chatterbox` | Single-speaker TTS (EN), voice cloning |
| `ChatterboxTurboTTS` | `ChatterboxTurboTTS` | `ResembleAI/chatterbox-turbo` | Same as above, 2-step CFM, GPT2-medium tokenizer |
| `ChatterboxMultilingualTTS` | `ChatterboxMultilingualTTS` | `ResembleAI/chatterbox` | 23-language TTS, voice cloning |
| `ChatterboxVC` | `ChatterboxVC` | `ResembleAI/chatterbox` | Voice conversion only (S3Gen-based) |

## Chatterbox Architecture (Key Components)

**ChatterboxTTS / ChatterboxTurboTTS / ChatterboxMultilingualTTS:**
- **T3** — Transformer-based text-to-speech-token model (with `T3Config`)
- **S3Gen** — Speech token decoder (Flow Matching based; Turbo uses `meanflow=True`)
- **VoiceEncoder** — Computes speaker embedding (`ve_embed`) from reference audio
- **Tokenizer** — `EnTokenizer` (standard), `AutoTokenizer` (Turbo), `MTLTokenizer` (multilingual)
- **Conditionals** — Dataclass holding T3Cond + S3Gen ref dict; serializable via `Conditionals.save()` / `Conditionals.load()`

**ChatterboxVC:**
- Only needs **S3Gen** — tokenizes source audio, decodes with target voice reference dict
- No T3 or VoiceEncoder needed

## Key Architectural Decision: Multi-Step Routing for Voice Cloning

Unlike OpenVoice (which uses NamedTemporaryFile for inline embedding), Chatterbox voice cloning follows the HarmonySpeech V1 pattern:

```
TTS Request (voice cloning, input_audio provided)
    │
    ├─[Reroute]─► Embedding Model (ChatterboxTTS in embed mode)
    │                    │
    │                    ▼
    │              SpeechEmbeddingRequestOutput
    │              (Conditionals serialized → base64)
    │                    │
    ├─[Forward]──► Synthesis Model (ChatterboxTTS)
                          │
                          ▼
                    TextToSpeechRequestOutput
```

**Rationale:**
1. Future caching potential — if many requests use same input audio
2. No Filesystem I/O — pure in-memory audio handling using `io.BytesIO` with `librosa.load()`

## Embedding / Conditionals Strategy

**For the dedicated embedding endpoint (`/embed-speaker`):**
- Use `prepare_conditionals_from_audio()` wrapper method that accepts raw bytes (no temp files)
- Serialize `Conditionals` via `Conditionals.save()` → `BytesIO` → base64 → return as `SpeechEmbeddingRequestOutput`

**For TTS voice cloning pipeline (multi-step routing):**
- If `input_audio` provided and `input_embedding` is None → route to embedding model first
- Embedding model computes Conditionals → returns serialized embedding
- Forward request to synthesis model with `input_embedding` set, `input_audio` cleared
- Synthesis model deserializes Conditionals → generates audio

**For ChatterboxVC:**
- `ref_dict` (S3Gen embedding) provided via `target_audio` or pre-computed `target_embedding`
- Routed through `VoiceConversionRequestInput`

## Generation Parameters

Extend `TextToSpeechGenerationOptions` with optional Chatterbox fields (other models ignore unused fields):

| Field | Type | Default | Applicable To |
|---|---|---|---|
| `exaggeration` | float (0-1) | 0.5 | TTS, Multilingual |
| `cfg_weight` | float (0-1) | 0.5 | TTS, Multilingual |
| `temperature` | float > 0 | 0.8 | All |
| `repetition_penalty` | float | 1.2 (TTS), 2.0 (MTL) | All |
| `top_p` | float | 1.0 (TTS), 0.95 (Turbo) | TTS, Turbo |
| `min_p` | float | 0.05 (TTS), 0.00 (Turbo) | TTS, Turbo |
| `top_k` | int | 1000 | Turbo only |
| `norm_loudness` | bool | True | Turbo only |

**Validation:** Model runners validate unsupported param combinations and warn/ignore as appropriate.

## Watermarking

Configurable per-model via `ModelConfig`. Add a `watermark: bool = True` field to enable/disable perth watermarking.

## Dependencies to Add

| Package | Purpose | Required By |
|---|---|---|
| `perth` | Implicit audio watermarking | All Chatterbox models |
| `pyloudnorm` | Loudness normalization | ChatterboxTurboTTS |
| `chatterbox-tts` | Model package | All Chatterbox models |

## Testing Strategy

- **Unit tests**: Mock model loading; test input preparation, routing logic, ModelRegistry registration
- **Integration tests**: Mock model inference; test full request flow through engine → runner → output
- **E2E tests**: Actual model download and inference (slow, GPU optional, CPU fallback)

## Key Decisions

| Decision | Rationale | Outcome |
|---|---|---|
| Wrap Chatterbox classes as "native" loader pattern | Matches KittenTTS precedent; Chatterbox's `from_pretrained()` handles all weight loading | Pending |
| Conditionals serialized as base64 torch tensors | Matches existing HSE embedding pattern | Pending |
| Extend `TextToSpeechGenerationOptions` with optional Chatterbox fields | Single request type; models ignore irrelevant fields | Pending |
| Multi-step routing for voice cloning (no temp files) | HarmonySpeech V1 pattern; no I/O; enables future caching | Pending |
| ChatterboxVC through VoiceConversionRequestInput | Consistent with existing pattern | Pending |
| perth + pyloudnorm + chatterbox-tts in requirements-common.txt | All lightweight dependencies | Pending |

---
*Last updated: 2026-03-12 after initialization and planning review*
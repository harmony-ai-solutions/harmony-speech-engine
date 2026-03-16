# Phase 4-1: Speed Factor and Peak Normalization Wiring

## Objective

Wire up the `speed` field from `GenerationOptions` and the `normalize_audio` field as post-synthesis audio processing steps applied in the serving layer. These steps apply **after** single-call TTS synthesis (and after stitching in the chunked path). Both are model-agnostic.

## Prerequisites

- Phase 1-1 complete: `audio_utils.py` has `apply_speed_factor()` and `normalize_peak()`
- Phase 2-1 complete: `GenerationOptions.speed` has updated docstring; `normalize_audio` replaces `norm_loudness`
- Phase 3-1 complete: `_create_text_to_speech_single()` and `_create_text_to_speech_chunked()` exist

## Files to Modify

- `harmonyspeech/endpoints/openai/serving_text_to_speech.py`
- `harmonyspeech/task_handler/inputs.py` (for `norm_loudness` → `normalize_audio` rename in Turbo path)

## Detailed Implementation Steps

### Step 1: Define which model types support native speed

Based on a search of `task_handler/inputs.py`, the following model types **natively consume `speed`** inside their input preparation and model call:

| Model type | Speed support |
|---|---|
| `HarmonySpeechSynthesizer` | ✅ native (speed_function) |
| `OpenVoiceV1Synthesizer` | ✅ native (speed_modifier) |
| `MeloTTSSynthesizer` | ✅ native (speed_modifier) |
| `KittenTTSSynthesizer` | ✅ native (speed_modifier) |
| `ChatterboxTTS` | ❌ — no native speed |
| `ChatterboxTurboTTS` | ❌ — no native speed |
| `ChatterboxMultilingualTTS` | ❌ — no native speed |
| `ChatterboxVC` | ❌ — no native speed |
| `VoiceFixerRestorer` / `VoiceFixerVocoder` | ❌ |
| `FasterWhisper`, `SileroVAD` | ❌ (N/A) |

**Add a constant** to [`harmonyspeech/common/audio_utils.py`](../../harmonyspeech/common/audio_utils.py):

```python
# Set of model_type strings that handle speed natively in their input preparation.
# When the resolved model type is in this set, post-synthesis time-stretching is
# skipped to avoid double-applying speed.
NATIVE_SPEED_MODEL_TYPES: frozenset[str] = frozenset({
    "HarmonySpeechSynthesizer",
    "OpenVoiceV1Synthesizer",
    "MeloTTSSynthesizer",
    "KittenTTSSynthesizer",
})
```

### Step 2: Wire `apply_speed_factor()` in `_create_text_to_speech_single()`

After the engine returns audio and it is decoded to a numpy array (inside `_create_text_to_speech_single()`), apply post-synthesis speed adjustment **only if the resolved model type is NOT in `NATIVE_SPEED_MODEL_TYPES`** and `generation_options.speed` is set and not 1.0.

The serving layer has access to the engine config and can resolve the model type from the request. Use a helper to look up the model type:

```python
# --- Post-synthesis: speed adjustment (only for models without native speed) ---
gen_opts = request.generation_options
if gen_opts and gen_opts.speed is not None and gen_opts.speed != 1.0:
    from harmonyspeech.common.audio_utils import apply_speed_factor, NATIVE_SPEED_MODEL_TYPES
    resolved_model_type = self._get_model_type(request.model)  # see step below
    if resolved_model_type not in NATIVE_SPEED_MODEL_TYPES:
        audio_array = apply_speed_factor(audio_array, sample_rate, gen_opts.speed)
        logger.debug(
            f"Post-synthesis speed {gen_opts.speed}x applied to {resolved_model_type} output"
        )
    else:
        logger.debug(
            f"Speed {gen_opts.speed}x handled natively by {resolved_model_type}, skipping post-synthesis time-stretch"
        )
```

**Add `_get_model_type()` helper to the serving class:**

```python
def _get_model_type(self, model_name: str) -> str | None:
    """Resolve the model_type string for the given model name from engine config.

    Returns None if the model is not found in the engine's model configs.
    """
    for model_config in self.engine.engine_config.model_configs:
        if model_config.name == model_name:
            return model_config.model_type
    return None
```

**Important:** The speed adjustment must be applied **before** normalization.

### Step 2: Wire `normalize_peak()` in `_create_text_to_speech_single()`

After speed adjustment (or directly after decode if no speed adjustment), apply peak normalization if `normalize_audio` is `True` (default).

```python
# --- Post-synthesis: peak normalization ---
do_normalize = (
    gen_opts.normalize_audio if gen_opts and gen_opts.normalize_audio is not None else True
)
if do_normalize:
    from harmonyspeech.common.audio_utils import normalize_peak
    audio_array = normalize_peak(audio_array, threshold=0.99, target=0.95)
```

**Note:** In the chunked path (`_create_text_to_speech_chunked()`), peak normalization is applied **once on the final stitched audio** (already implemented in Phase 3-1), not on individual chunks. Individual chunks in `_create_text_to_speech_single()` should **skip normalization** when called from the chunked path to avoid double-normalization.

To handle this cleanly, add an optional `_skip_postprocess: bool = False` parameter to `_create_text_to_speech_single()`:

```python
async def _create_text_to_speech_single(
    self,
    request: TextToSpeechRequest,
    raw_request,
    _skip_postprocess: bool = False,
) -> TextToSpeechResponse | ErrorResponse:
    ...
    # Post-synthesis processing block
    if not _skip_postprocess:
        # Apply speed factor
        if gen_opts and gen_opts.speed is not None and gen_opts.speed != 1.0:
            audio_array = apply_speed_factor(audio_array, sample_rate, gen_opts.speed)
        # Apply peak normalization
        do_normalize = (gen_opts.normalize_audio if gen_opts and gen_opts.normalize_audio is not None else True)
        if do_normalize:
            audio_array = normalize_peak(audio_array)
```

In `_create_text_to_speech_chunked()`, call single with `_skip_postprocess=True` per chunk, then apply speed + normalization once on the stitched result:

```python
# In _create_text_to_speech_chunked(), call single-path with skip:
chunk_response = await self._create_text_to_speech_single(
    chunk_request, raw_request, _skip_postprocess=True
)

# After stitching, apply speed + normalization:
if gen_opts and gen_opts.speed is not None and gen_opts.speed != 1.0:
    stitched = apply_speed_factor(stitched, detected_sample_rate, gen_opts.speed)
if do_normalize:
    stitched = normalize_peak(stitched)
```

### Step 3: Handle `normalize_audio` rename in `task_handler/inputs.py`

The `prepare_chatterbox_turbo_tts_inputs()` function in [`harmonyspeech/task_handler/inputs.py`](../../harmonyspeech/task_handler/inputs.py) currently passes `norm_loudness` from `GenerationOptions` to the ChatterboxTurbo model call. Since we are renaming `norm_loudness` to `normalize_audio` in the protocol, update the mapping:

**Find the code that reads `opts.norm_loudness`** and change it to `opts.normalize_audio`:

Before:
```python
norm_loudness = opts.norm_loudness if opts is not None and opts.norm_loudness is not None else True
```

After:
```python
norm_loudness = opts.normalize_audio if opts is not None and opts.normalize_audio is not None else True
```

This preserves the Turbo model's behavior (it still receives `norm_loudness` as a positional/keyword arg to the model call) but maps it from the renamed protocol field.

**Also update the validation** that previously rejected `norm_loudness` in non-Turbo model paths. In `prepare_chatterbox_tts_inputs()` and `prepare_chatterbox_multilingual_tts_inputs()`, the validation block currently says:
```python
if opts.norm_loudness is not None:
    raise ValueError("norm_loudness is not supported by ChatterboxTTS")
```
This validation should be **removed** entirely since `normalize_audio` is now handled purely in the serving layer and never passed to any model runner. No model runner needs to validate it.

### Step 4: Update `TextToSpeechGenerationOptions` dataclass in `common/inputs.py`

The [`TextToSpeechGenerationOptions`](../../harmonyspeech/common/inputs.py:17) dataclass mirrors `GenerationOptions` for internal use. Apply the same rename:

Before:
```python
norm_loudness: bool | None = None
```

After:
```python
normalize_audio: bool | None = None
```

Confirm all usages of `norm_loudness` in `inputs.py` are updated accordingly.

---

## Summary of Post-Processing Order

The order in which post-processing is applied (both in single and chunked paths) is:

1. **Model inference** (engine call)
2. **Decode** audio from base64 WAV → numpy float32
3. *(Chunked path only)* **Stitch** audio segments with crossfade
4. **Speed factor** (`apply_speed_factor`) — changes duration
5. **Peak normalization** (`normalize_peak`) — applied last to prevent clipping

---

## Progress Checklist

- [ ] Add `apply_speed_factor()` call in `_create_text_to_speech_single()` with `_skip_postprocess` guard
- [ ] Add `normalize_peak()` call in `_create_text_to_speech_single()` with `_skip_postprocess` guard
- [ ] Update `_create_text_to_speech_chunked()` to pass `_skip_postprocess=True` and apply speed+normalize on stitched result
- [ ] Update `prepare_chatterbox_turbo_tts_inputs()` in `task_handler/inputs.py`: `norm_loudness` → `normalize_audio`
- [ ] Remove `norm_loudness` validation from `prepare_chatterbox_tts_inputs()` and `prepare_chatterbox_multilingual_tts_inputs()`
- [ ] Update `TextToSpeechGenerationOptions` dataclass in `common/inputs.py`: `norm_loudness` → `normalize_audio`
- [ ] Search entire codebase for remaining `norm_loudness` references and update them all
- [ ] Run existing e2e tests to confirm Chatterbox Turbo behavior is unchanged

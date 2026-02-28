# Phase 2: Model Registry Registration

## Objective

Register `KittenTTSSynthesizer` in the HSE model registry so the loader and model runner can resolve it by type name.

## File to Modify

### `harmonyspeech/modeling/models/__init__.py`

**Current `_MODELS` dict** (excerpt):
```python
_MODELS = {
    ...
    "VoiceFixerVocoder": ("voicefixer", "VoiceFixerVocoder"),
    "FasterWhisper": ("faster-wisper", "native"),
    "SileroVAD": ("silero-vad", "native"),
}
```

**Add the following entry:**
```python
"KittenTTSSynthesizer": ("kittentts", "native"),
```

The `"native"` class marker tells `_get_model_cls()` in the loader that this model is not loaded via the standard PyTorch `nn.Module` path — it uses its own initialization logic (same pattern as `FasterWhisper` and `SileroVAD`).

### Full diff to apply:

In [`harmonyspeech/modeling/models/__init__.py`](harmonyspeech/modeling/models/__init__.py), find the `_MODELS` dictionary and add the new entry **before the closing brace**:

```python
    # KittenTTS
    "KittenTTSSynthesizer": ("kittentts", "native"),
```

## Progress Checklist

- [x] Add `"KittenTTSSynthesizer": ("kittentts", "native")` to `_MODELS` in `harmonyspeech/modeling/models/__init__.py`

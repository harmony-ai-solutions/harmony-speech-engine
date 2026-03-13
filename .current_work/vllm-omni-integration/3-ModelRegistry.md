# Phase 3: Model Registry

## Objective

Register `"VllmOmniTTS"` as a recognised model type in HSE's `ModelRegistry`. This is needed for two reasons:
1. `ModelRegistry.get_supported_archs()` is used in several places (health checks, model listing) — the type should appear there
2. The registry lookup pattern is used throughout the codebase; keeping it consistent makes the system discoverable

## Important Note

`VllmOmniExecutor` does **not** call `loader.get_model()` or `ModelRegistry.load_model_cls()` — it handles loading entirely inside itself. The `"native"` class name in the registry is a marker that says "this model type does not use the standard PyTorch model loader path". This is the same pattern used by `FasterWhisper`, `SileroVAD`, and `KittenTTSSynthesizer`.

## File to Modify

**`harmonyspeech/modeling/models/__init__.py`**

## Current State

The `_MODELS` dict currently ends with:
```python
# KittenTTS
"KittenTTSSynthesizer": ("kittentts", "native"),
```

## Implementation

### Step 1: Add VllmOmniTTS to `_MODELS`

Add the following entry at the end of the `_MODELS` dict, after `KittenTTSSynthesizer`:

```python
# vllm-omni TTS models (Qwen3-TTS, CosyVoice3, Fish Speech)
# Note: Loading is handled entirely by VllmOmniExecutor, not by loader.get_model()
"VllmOmniTTS": ("vllm_omni", "native"),
```

The `"vllm_omni"` module name is never actually imported by the registry (because `model_cls_name == "native"` returns the string `"native"` early before any import). This entry purely serves as:
- A discoverable entry in `get_supported_archs()` output
- An explicit documentation anchor in the registry

## Complete Modified `_MODELS` Dict (relevant portion)

```python
_MODELS = {
    # ... existing entries ...

    # KittenTTS
    "KittenTTSSynthesizer": ("kittentts", "native"),

    # vllm-omni TTS models (Qwen3-TTS, CosyVoice3, Fish Speech)
    # Note: Loading is handled entirely by VllmOmniExecutor, not by loader.get_model()
    "VllmOmniTTS": ("vllm_omni", "native"),
}
```

## Verification

After this change:
```python
from harmonyspeech.modeling.models import ModelRegistry
assert "VllmOmniTTS" in ModelRegistry.get_supported_archs()
assert ModelRegistry.load_model_cls("VllmOmniTTS") == "native"
```

## Progress Checklist

- [ ] Add `"VllmOmniTTS": ("vllm_omni", "native")` to `_MODELS` in `harmonyspeech/modeling/models/__init__.py`
- [ ] Add explanatory comment above the entry

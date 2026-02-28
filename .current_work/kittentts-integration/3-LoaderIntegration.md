# Phase 3: Loader Integration

## Objective

Add `KittenTTSSynthesizer` to the `_MODEL_CONFIGS` and `_MODEL_WEIGHTS` dictionaries in the loader, and add the native model bailout in `get_model()` to instantiate `KittenTTSSynthesizer` directly.

## File to Modify

### `harmonyspeech/modeling/loader.py`

KittenTTS follows the **native model pattern** (same as `FasterWhisper` and `SileroVAD`). The model downloads its own files from HuggingFace during initialization, so config and weights are marked `"native"`.

---

#### Step 1: Add to `_MODEL_CONFIGS`

Find the `_MODEL_CONFIGS` dict and add:

```python
    # KittenTTS
    "KittenTTSSynthesizer": {
        "default": "native"
    },
```

#### Step 2: Add to `_MODEL_WEIGHTS`

Find the `_MODEL_WEIGHTS` dict and add:

```python
    # KittenTTS
    "KittenTTSSynthesizer": {
        "default": "native"
    },
```

#### Step 3: Add top-level import in `loader.py`

At the top of `loader.py`, alongside the existing native model imports:

```python
from faster_whisper import WhisperModel
from silero_vad import load_silero_vad
```

Add the KittenTTS import:

```python
from harmonyspeech.modeling.models.kittentts.kittentts import KittenTTSSynthesizer
```

#### Step 4: Add native bailout in `get_model()`

In the `get_model()` function, find the block that handles native models:

```python
            # Bailout for native models
            if model_class == "native" and hf_config == "native":
                if model_config.model_type == "FasterWhisper":
                    model = WhisperModel(model_config.model)
                    return model
                elif model_config.model_type == "SileroVAD":
                    ...
                    return model
```

Add a new `elif` **before the final `else` or after `SileroVAD`**:

```python
                elif model_config.model_type == "KittenTTSSynthesizer":
                    model = KittenTTSSynthesizer(model_name_or_path=model_config.model)
                    return model
```

The `model_config.model` field will contain the HuggingFace repo ID as specified in `config.yml` (e.g. `"KittenML/kitten-tts-mini-0.8"`).

---

## Notes

- No `get_model_config()` call is needed for KittenTTS — the model fetches its own `config.json` internally.
- No `get_model_weights()` call is needed — weights are downloaded as ONNX and voices NPZ files during `KittenTTSSynthesizer.__init__()`.
- The `model_config.language` field is not used by KittenTTS (English only for now), but can be added to the bailout in future if multilingual support is added.

## Progress Checklist

- [ ] Add `"KittenTTSSynthesizer"` entry to `_MODEL_CONFIGS` in [`harmonyspeech/modeling/loader.py`](harmonyspeech/modeling/loader.py)
- [ ] Add `"KittenTTSSynthesizer"` entry to `_MODEL_WEIGHTS` in [`harmonyspeech/modeling/loader.py`](harmonyspeech/modeling/loader.py)
- [ ] Add `elif model_config.model_type == "KittenTTSSynthesizer"` bailout in `get_model()` in [`harmonyspeech/modeling/loader.py`](harmonyspeech/modeling/loader.py)

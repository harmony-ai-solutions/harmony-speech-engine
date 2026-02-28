# Phase 1: KittenTTS Model Module

## Objective

Copy the KittenTTS source code from `.current_work/KittenTTS/kittentts/` into `harmonyspeech/modeling/models/kittentts/` and create a `KittenTTSSynthesizer` wrapper class that HSE's model runner can use.

## Files to Create

### `harmonyspeech/modeling/models/kittentts/__init__.py`

```python
from .kittentts import KittenTTSSynthesizer

__all__ = [
    "KittenTTSSynthesizer",
]
```

---

### `harmonyspeech/modeling/models/kittentts/onnx_model.py`

Copy verbatim from `.current_work/KittenTTS/kittentts/onnx_model.py` with one import change:

- Change `from .preprocess import TextPreprocessor` to remain as-is (relative import within the new package).

The full file contents are unchanged from the KittenTTS source. Imports used:
- `misaki`, `numpy`, `phonemizer`, `soundfile`, `onnxruntime`

---

### `harmonyspeech/modeling/models/kittentts/preprocess.py`

Copy verbatim from `.current_work/KittenTTS/kittentts/preprocess.py` with no changes needed.

---

### `harmonyspeech/modeling/models/kittentts/onnx_model.py`

Copy verbatim from `.current_work/KittenTTS/kittentts/onnx_model.py`. Only change the internal relative import:

- Change `from .preprocess import TextPreprocessor` → `from harmonyspeech.modeling.models.kittentts.preprocess import TextPreprocessor`

---

### `harmonyspeech/modeling/models/kittentts/kittentts.py`

This is the main integration wrapper. It wraps `KittenTTS_1_Onnx` and exposes a clean interface for HSE's model runner.

```python
"""KittenTTS integration for Harmony Speech Engine.

KittenTTS is an ultra-lightweight ONNX-based TTS model by KittenML.
Models are downloaded from HuggingFace Hub and run via ONNX Runtime.
Sample rate: 24000 Hz, English only.
"""
import json
from typing import Optional

import numpy as np
from huggingface_hub import hf_hub_download

from harmonyspeech.modeling.models.kittentts.onnx_model import KittenTTS_1_Onnx


# Available KittenTTS HuggingFace model repositories
KITTENTTS_MODEL_REPOS = {
    "kitten-tts-mini": "KittenML/kitten-tts-mini-0.8",
    "kitten-tts-micro": "KittenML/kitten-tts-micro-0.8",
    "kitten-tts-nano": "KittenML/kitten-tts-nano-0.8-fp32",
    "kitten-tts-nano-int8": "KittenML/kitten-tts-nano-0.8-int8",
}

KITTENTTS_SAMPLE_RATE = 24000


class KittenTTSSynthesizer:
    """
    Wrapper for KittenTTS ONNX model for use in Harmony Speech Engine.

    Supports all KittenTTS model variants:
    - kitten-tts-mini  (80MB, 80M params)
    - kitten-tts-micro (41MB, 40M params)
    - kitten-tts-nano  (56MB, 15M params, fp32)
    - kitten-tts-nano-int8 (25MB, 15M params, int8)

    Available voices: Bella, Jasper, Luna, Bruno, Rosie, Hugo, Kiki, Leo
    """

    def __init__(self, model_name_or_path: str, cache_dir: Optional[str] = None):
        """
        Initialize KittenTTSSynthesizer by downloading model from HuggingFace.

        Args:
            model_name_or_path: HuggingFace repo ID (e.g. "KittenML/kitten-tts-mini-0.8")
                                 or short name (e.g. "kitten-tts-mini")
            cache_dir: Optional directory to cache downloaded model files
        """
        # Resolve repo ID
        if "/" not in model_name_or_path and model_name_or_path in KITTENTTS_MODEL_REPOS:
            repo_id = KITTENTTS_MODEL_REPOS[model_name_or_path]
        else:
            repo_id = model_name_or_path

        self.repo_id = repo_id
        self.sample_rate = KITTENTTS_SAMPLE_RATE
        self._model = self._load_model(repo_id, cache_dir)

    def _load_model(self, repo_id: str, cache_dir: Optional[str]) -> KittenTTS_1_Onnx:
        """Download and initialize KittenTTS_1_Onnx from HuggingFace."""
        # Download config
        config_path = hf_hub_download(
            repo_id=repo_id,
            filename="config.json",
            cache_dir=cache_dir
        )
        with open(config_path, 'r') as f:
            config = json.load(f)

        if config.get("type") not in ["ONNX1", "ONNX2"]:
            raise ValueError(
                f"Unsupported KittenTTS model type '{config.get('type')}'. "
                f"Expected 'ONNX1' or 'ONNX2'."
            )

        # Download model ONNX file
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=config["model_file"],
            cache_dir=cache_dir
        )

        # Download voices NPZ file
        voices_path = hf_hub_download(
            repo_id=repo_id,
            filename=config["voices"],
            cache_dir=cache_dir
        )

        return KittenTTS_1_Onnx(
            model_path=model_path,
            voices_path=voices_path,
            speed_priors=config.get("speed_priors", {}),
            voice_aliases=config.get("voice_aliases", {})
        )

    def generate(self, text: str, voice: str = "Jasper", speed: float = 1.0, clean_text: bool = True) -> np.ndarray:
        """
        Generate speech audio from text.

        Args:
            text: Input text to synthesize
            voice: Voice name (friendly: 'Jasper', 'Bella', etc.) or internal voice ID
            speed: Speech speed multiplier (1.0 = normal)
            clean_text: If True, preprocess text (normalize numbers, etc.)

        Returns:
            numpy float32 array of audio samples at 24000 Hz
        """
        return self._model.generate(text, voice=voice, speed=speed, clean_text=clean_text)

    @property
    def available_voices(self):
        """Return list of friendly voice names."""
        return self._model.all_voice_names

    def load_weights(self, checkpoint, hf_config=None):
        """No-op: KittenTTS loads its own weights during __init__."""
        pass
```

## Progress Checklist

- [x] Create `harmonyspeech/modeling/models/kittentts/` directory
- [x] Create `harmonyspeech/modeling/models/kittentts/__init__.py`
- [x] Copy `onnx_model.py` from `.current_work/KittenTTS/kittentts/onnx_model.py`
- [x] Copy `preprocess.py` from `.current_work/KittenTTS/kittentts/preprocess.py`
- [x] Create `harmonyspeech/modeling/models/kittentts/kittentts.py` with `KittenTTSSynthesizer`

# Phase 2-3: VllmOmniExecutor — Prompt Building for Qwen3-TTS

## Objective

Implement the tokenizer management and prompt building methods in `VllmOmniExecutor`:
- `_get_tokenizer()` — lazy-load and cache the model's `AutoTokenizer`
- `_get_talker_config()` — lazy-load and cache the `Qwen3TTSConfig.talker_config`
- `_estimate_prompt_len(additional_information)` — estimate placeholder token count for Qwen3-TTS
- `_build_prompt(request_data)` — convert `TextToSpeechRequestInput` to vllm-omni prompt dict

## Background: Why Qwen3-TTS Needs Special Prompt Building

Unlike other HSE models that receive plain text or audio, vllm-omni Qwen3-TTS requires a structured prompt dict with a **placeholder token array** and an **`additional_information`** payload:

```python
prompt = {
    "prompt_token_ids": [0] * estimated_length,   # length matters; values don't
    "additional_information": {
        "task_type": ["CustomVoice"],
        "text": ["Hello world"],
        "language": ["English"],
        "speaker": ["Vivian"],
        "instruct": [""],
        "max_new_tokens": [2048],
    }
}
```

The `estimated_length` must match what the model's talker will produce after embedding preprocessing (the AR talker replaces all token values via `preprocess`, but the length is used to allocate the KV cache). The `Qwen3TTSTalkerForConditionalGeneration.estimate_prompt_len_from_additional_information()` method computes this from the tokenizer and config.

### Qwen3-TTS Task Types

| HSE Input | vllm-omni `task_type` | Required fields |
|---|---|---|
| `voice_id` set, no `input_audio` | `"CustomVoice"` | `text`, `language`, `speaker`, `instruct` |
| `input_audio` set (reference audio) | `"Base"` with `x_vector_only_mode=True` | `text`, `language`, `ref_audio`, `x_vector_only_mode` |
| Neither | `"CustomVoice"` with default speaker | `text`, `language`, `speaker="Vivian"`, `instruct` |

> **Note**: `"VoiceDesign"` task type (instruct-based voice description) is **not yet mapped** because HSE's `TextToSpeechRequestInput` has no dedicated `instructions` field. This can be added later by extending the request format.

### Language Mapping (HSE → vllm-omni)

HSE uses short language codes; vllm-omni uses full English names:

```python
_HSE_TO_VLLM_OMNI_LANGUAGE = {
    "EN": "English",
    "ZH": "Chinese",
    "JA": "Japanese",
    "KR": "Korean",
    "DE": "German",
    "FR": "French",
    "ES": "Spanish",
    "IT": "Italian",
    "PT": "Portuguese",
    "RU": "Russian",
}
```

If `language_id` is not in the map (e.g., already a full name like `"English"`) or is `None`, default to `"Auto"`.

### Reference Audio Format (Base task)

When `input_audio` is provided (base64-encoded WAV/MP3/etc.), it must be converted to `[[wav_samples_list, sample_rate]]` for vllm-omni:

```python
# input_audio is base64 string from HSE request
audio_bytes = base64.b64decode(request_data.input_audio)
wav_np, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)  # sr=None preserves original
# vllm-omni expects: [[list_of_float_samples, int_sample_rate]]
additional_information["ref_audio"] = [[wav_np.tolist(), int(sr)]]
additional_information["x_vector_only_mode"] = [True]
```

`x_vector_only_mode=True` means: use the reference audio for speaker embedding only, without requiring a transcript (`ref_text`). This simplifies the API since HSE requests don't carry a `ref_text` field.

## File to Modify

**`harmonyspeech/executor/vllm_omni_executor.py`** — replace the three stubs

## Implementation

### Add language map constant (module level, before the class)

```python
# Language code mapping: HSE short codes → vllm-omni full names
_HSE_TO_VLLM_OMNI_LANGUAGE = {
    "EN": "English",
    "ZH": "Chinese",
    "JA": "Japanese",
    "KR": "Korean",
    "DE": "German",
    "FR": "French",
    "ES": "Spanish",
    "IT": "Italian",
    "PT": "Portuguese",
    "RU": "Russian",
}
```

### Replace `_get_tokenizer()` stub

```python
def _get_tokenizer(self):
    """Lazily load and cache the HuggingFace AutoTokenizer for this model.

    The tokenizer is needed by _estimate_prompt_len() to measure text token
    lengths for the Qwen3-TTS talker stage prompt placeholder calculation.

    Returns:
        Loaded AutoTokenizer instance (cached after first call).
    """
    if self._tokenizer is None:
        from transformers import AutoTokenizer  # noqa: PLC0415
        logger.info(
            f"[VllmOmniExecutor] Loading tokenizer for {self.model_config.model}..."
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.model,
            trust_remote_code=True,
            padding_side="left",
        )
        logger.info("[VllmOmniExecutor] Tokenizer loaded.")
    return self._tokenizer
```

### Replace `_get_talker_config()` stub

```python
def _get_talker_config(self):
    """Lazily load and cache the Qwen3-TTS talker configuration.

    The talker config provides codec_language_id and spk_is_dialect parameters
    needed by estimate_prompt_len_from_additional_information().

    Returns:
        talker_config object (Qwen3TTSTalkerConfig), or None if not available
        (e.g., for non-Qwen3-TTS models like CosyVoice3 or Fish Speech).
    """
    if self._talker_config is None:
        try:
            from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import (  # noqa: PLC0415
                Qwen3TTSConfig,
            )
            cfg = Qwen3TTSConfig.from_pretrained(
                self.model_config.model, trust_remote_code=True
            )
            self._talker_config = getattr(cfg, "talker_config", None)
        except Exception as e:
            logger.debug(
                f"[VllmOmniExecutor] Could not load Qwen3TTSConfig "
                f"(model may not be Qwen3-TTS): {e}"
            )
            self._talker_config = False  # False sentinel: don't try again
    # Return None if load failed (False sentinel → None for callers)
    return self._talker_config if self._talker_config is not False else None
```

### Replace `_estimate_prompt_len()` stub

```python
def _estimate_prompt_len(self, additional_information: dict) -> int:
    """Estimate the number of placeholder tokens for the Qwen3-TTS talker stage.

    The AR Talker replaces all input embeddings via preprocess, so placeholder
    token values are irrelevant — only the LENGTH matters for KV cache allocation.

    Falls back to 2048 if estimation fails (e.g., for non-Qwen3-TTS models,
    or if the tokenizer fails to load).

    Args:
        additional_information: The additional_information dict to be sent to Omni.

    Returns:
        Estimated integer token count.
    """
    try:
        from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_talker import (  # noqa: PLC0415
            Qwen3TTSTalkerForConditionalGeneration,
        )
        tok = self._get_tokenizer()
        talker_config = self._get_talker_config()
        task_type = (additional_information.get("task_type") or ["CustomVoice"])[0]

        return Qwen3TTSTalkerForConditionalGeneration.estimate_prompt_len_from_additional_information(
            additional_information=additional_information,
            task_type=task_type,
            tokenize_prompt=lambda t: tok(t, padding=False)["input_ids"],
            codec_language_id=getattr(talker_config, "codec_language_id", None) if talker_config else None,
            spk_is_dialect=getattr(talker_config, "spk_is_dialect", None) if talker_config else None,
        )
    except Exception as e:
        logger.warning(
            f"[VllmOmniExecutor] Prompt length estimation failed: {e}. "
            "Using fallback length 2048."
        )
        return 2048
```

### Replace `_build_prompt()` stub

```python
def _build_prompt(self, request_data) -> dict:
    """Convert a TextToSpeechRequestInput to a vllm-omni prompt dict.

    Builds the additional_information payload and estimates the prompt token
    placeholder length required by the Qwen3-TTS talker stage.

    Task type selection logic:
        1. If request_data.input_audio is set → "Base" task (voice cloning)
           using x_vector_only_mode=True (no ref_text required)
        2. Elif request_data.voice_id is set → "CustomVoice" with named speaker
        3. Else → "CustomVoice" with default speaker "Vivian"

    Args:
        request_data: TextToSpeechRequestInput instance.

    Returns:
        Dict with "prompt_token_ids" (placeholder list) and "additional_information".
    """
    import librosa  # noqa: PLC0415

    additional_information: dict = {}

    # --- Text ---
    additional_information["text"] = [request_data.input_text]

    # --- Language ---
    lang_code = getattr(request_data, "language_id", None)
    vllm_language = _HSE_TO_VLLM_OMNI_LANGUAGE.get(lang_code, lang_code) if lang_code else "Auto"
    additional_information["language"] = [vllm_language]

    # --- Task type and voice ---
    if getattr(request_data, "input_audio", None):
        # Base task: voice cloning from reference audio
        additional_information["task_type"] = ["Base"]
        audio_bytes = base64.b64decode(request_data.input_audio)
        wav_np, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)
        additional_information["ref_audio"] = [[wav_np.tolist(), int(sr)]]
        additional_information["x_vector_only_mode"] = [True]

    elif getattr(request_data, "voice_id", None):
        # CustomVoice task: use named speaker
        additional_information["task_type"] = ["CustomVoice"]
        additional_information["speaker"] = [request_data.voice_id]
        additional_information["instruct"] = [""]

    else:
        # CustomVoice task: use default speaker
        additional_information["task_type"] = ["CustomVoice"]
        additional_information["speaker"] = ["Vivian"]
        additional_information["instruct"] = [""]

    # --- Generation parameters ---
    additional_information["max_new_tokens"] = [2048]

    # --- Estimate prompt length ---
    estimated_len = self._estimate_prompt_len(additional_information)
    logger.debug(
        f"[VllmOmniExecutor] Prompt: task={additional_information['task_type'][0]}, "
        f"lang={vllm_language}, len={estimated_len}"
    )

    return {
        "prompt_token_ids": [0] * estimated_len,
        "additional_information": additional_information,
    }
```

## Notes

- **`librosa`** is already in `requirements-common.txt` — no new dependency.
- `_get_talker_config()` uses a `False` sentinel (not `None`) to distinguish "not loaded yet" from "load attempted but unavailable". This prevents repeated failed loading attempts on non-Qwen3-TTS models.
- For CosyVoice3 and Fish Speech, `_estimate_prompt_len` will fall back to 2048 (the Qwen3-TTS-specific function won't be found). A future phase can add model-specific estimators.
- `sr=None` in `librosa.load()` preserves the original sample rate of the reference audio — vllm-omni handles resampling internally.

## Progress Checklist

- [ ] Add `_HSE_TO_VLLM_OMNI_LANGUAGE` dict at module level (before class)
- [ ] Replace `_get_tokenizer()` stub with lazy-loading implementation
- [ ] Replace `_get_talker_config()` stub with lazy-loading implementation (with `False` sentinel)
- [ ] Replace `_estimate_prompt_len()` stub with full implementation
- [ ] Replace `_build_prompt()` stub with full implementation
- [ ] Verify Base task: `input_audio` is decoded and converted to `[[samples_list, sr]]`
- [ ] Verify CustomVoice task: `voice_id` maps to `speaker` field
- [ ] Verify default CustomVoice: no input_audio, no voice_id → speaker "Vivian"
- [ ] Verify language mapping: `"EN"` → `"English"`, `None` → `"Auto"`

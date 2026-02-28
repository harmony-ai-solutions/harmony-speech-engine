# Phase 5: Endpoint Registration + Model Runner Execution Handler

## Objective

1. Register `KittenTTSSynthesizer` in the TTS endpoint's model type list so it appears in the model catalog and is routable.
2. Add the `_execute_kittentts_synthesizer()` method and its dispatch in [`harmonyspeech/task_handler/model_runner_base.py`](harmonyspeech/task_handler/model_runner_base.py).

## File to Modify: `harmonyspeech/endpoints/openai/serving_text_to_speech.py`

At the top of this file, two lists control which model types the TTS endpoint knows about:

```python
_TTS_MODEL_TYPES = [
    "OpenVoiceV1Synthesizer",
    "MeloTTSSynthesizer"
]
_TTS_MODEL_GROUPS = {
    "harmonyspeech": ["HarmonySpeechSynthesizer", "HarmonySpeechVocoder"],
    "openvoice_v1": ["OpenVoiceV1Synthesizer", "OpenVoiceV1ToneConverter"],
    "openvoice_v2": ["MeloTTSSynthesizer", "OpenVoiceV2ToneConverter"]
}
```

**Add `"KittenTTSSynthesizer"` to `_TTS_MODEL_TYPES`:**

```python
_TTS_MODEL_TYPES = [
    "OpenVoiceV1Synthesizer",
    "MeloTTSSynthesizer",
    "KittenTTSSynthesizer"
]
```

**No change needed to `_TTS_MODEL_GROUPS`** — KittenTTS is a single-step synthesizer. When a request arrives with `model: "kitten-tts-mini"` (the config name), HSE routes it directly to the matching executor. No multi-step pipeline is needed.

### Why this is required

`model_cards_from_config_groups()` in [`serving_engine.py`](harmonyspeech/endpoints/openai/serving_engine.py) builds the list of available models from `_TTS_MODEL_TYPES` and `_TTS_MODEL_GROUPS`. Without adding `"KittenTTSSynthesizer"` to `_TTS_MODEL_TYPES`, a KittenTTS model configured in `config.yml` will **not** appear in the TTS model catalog, and the endpoint will respond with "model not found" for all requests.

### No engine routing changes needed

In [`harmonyspeech_engine.py`](harmonyspeech/engine/harmonyspeech_engine.py), `check_reroute_request_to_model()` only handles the named group workflows (`harmonyspeech`, `openvoice_v1`, `openvoice_v2`, `voicefixer`). For KittenTTS, the request arrives with `model` already set to the config name (e.g. `"kitten-tts-mini"`), so it routes directly to the correct executor without any additional routing logic.

### API usage note

The `_check_model()` in [`serving_engine.py`](harmonyspeech/endpoints/openai/serving_engine.py) validates that `mode` must be `"single_speaker_tts"` or `"voice_cloning"`. KittenTTS users must set `mode: "single_speaker_tts"` in their requests (no changes needed in the validation logic).

---

## File to Modify: `harmonyspeech/task_handler/model_runner_base.py`

#### Step 1: Add dispatch case in `execute_model()`

In the `execute_model()` method, find the `elif model_type == "SileroVAD":` block and add a new `elif` before the final `else: raise NotImplementedError`:

```python
        elif model_type == "KittenTTSSynthesizer":
            outputs = self._execute_kittentts_synthesizer(inputs, requests_to_batch)
```

#### Step 2: Add `_execute_kittentts_synthesizer()` method

Add the following method to the `ModelRunnerBase` class, after `_execute_silero_vad()`:

```python
    def _execute_kittentts_synthesizer(self, inputs, requests_to_batch):
        """Execute KittenTTSSynthesizer to generate speech audio from text."""

        def synthesize_text(input_params):
            input_text, voice, speed_modifier = input_params

            # Run KittenTTS generation
            # Returns numpy float32 array at 24000 Hz
            audio_array = self.model.generate(
                text=input_text,
                voice=voice,
                speed=speed_modifier,
                clean_text=True
            )

            # Flatten to 1D if needed (KittenTTS may return 2D)
            if audio_array.ndim > 1:
                audio_array = audio_array.flatten()

            # Encode to WAV in memory and return as base64 string
            with io.BytesIO() as wav_buffer:
                sf.write(wav_buffer, audio_array, self.model.sample_rate, format='WAV')
                wav_bytes = wav_buffer.getvalue()

            encoded_audio = base64.b64encode(wav_bytes).decode('utf-8')
            return encoded_audio

        outputs = []
        for i, x in enumerate(inputs):
            initial_request = requests_to_batch[i]
            inference_result = synthesize_text(x)
            result = self._build_result(initial_request, inference_result, TextToSpeechRequestOutput)
            outputs.append(result)
        return outputs
```

## Notes

- `self.model` will be a `KittenTTSSynthesizer` instance (set by `_load_model()` which calls `get_model()`).
- `sf` (`soundfile`) and `io` are already imported in `model_runner_base.py`.
- `base64` is already imported in `model_runner_base.py`.
- The output format is `TextToSpeechRequestOutput` (same as `MeloTTSSynthesizer` and `OpenVoiceV1Synthesizer`).
- KittenTTS always outputs at 24,000 Hz. The sample rate is accessed via `self.model.sample_rate`.
- `audio_array.flatten()` handles the case where KittenTTS returns a `[1, N]` shaped array from the ONNX model (the `outputs[0][..., :-5000]` in `generate_single_chunk` can produce a 2D array).

## Progress Checklist

- [ ] Add `elif model_type == "KittenTTSSynthesizer"` dispatch in `execute_model()` in [`harmonyspeech/task_handler/model_runner_base.py`](harmonyspeech/task_handler/model_runner_base.py)
- [ ] Add `_execute_kittentts_synthesizer()` method to `ModelRunnerBase` in [`harmonyspeech/task_handler/model_runner_base.py`](harmonyspeech/task_handler/model_runner_base.py)

# Phase 4: Input Preparation Handler

## Objective

Add input preparation logic for `KittenTTSSynthesizer` to [`harmonyspeech/task_handler/inputs.py`](harmonyspeech/task_handler/inputs.py). KittenTTS takes raw text, a voice name, and a speed modifier — no BERT preprocessing or config loading needed.

## File to Modify

### `harmonyspeech/task_handler/inputs.py`

#### Step 1: Add new case to `prepare_inputs()`

In the `prepare_inputs()` function, find the final `elif` block (for `SileroVAD`) and add a new `elif` after it, before the final `else: raise NotImplementedError`:

```python
    elif model_config.model_type == "KittenTTSSynthesizer":
        for r in requests_to_batch:
            if (
                isinstance(r.request_data, TextToSpeechRequestInput) or
                isinstance(r.request_data, SynthesisRequestInput)
            ):
                inputs.append(r.request_data)
            else:
                raise ValueError(
                    f"request ID {r.request_id} is not of type TextToSpeechRequestInput or "
                    f"SynthesisRequestInput"
                )
        return prepare_kittentts_synthesizer_inputs(inputs)
```

#### Step 2: Add `prepare_kittentts_synthesizer_inputs()` function

Add the following new function at the end of the file (after `prepare_silero_vad_inputs`):

```python
def prepare_kittentts_synthesizer_inputs(requests_to_batch: List[Union[
    TextToSpeechRequestInput,
    SynthesisRequestInput
]]):
    """
    Prepare inputs for KittenTTSSynthesizer model.
    Extracts text, voice name, and speed from the request.
    No preprocessing or BERT tokenization needed — KittenTTS handles this internally.
    """
    def prepare(request):
        input_text = request.input_text

        # Voice selection: use voice_id from request or fall back to default
        voice = request.voice_id if request.voice_id else "Jasper"

        # Speed modifier
        speed_modifier = 1.0
        if request.generation_options:
            if request.generation_options.speed:
                speed_modifier = float(request.generation_options.speed)

        return input_text, voice, speed_modifier

    with ThreadPoolExecutor() as executor:
        inputs = list(executor.map(prepare, requests_to_batch))

    return inputs
```

## Notes

- Voice names should match one of the 8 available KittenTTS voices: `Bella`, `Jasper`, `Luna`, `Bruno`, `Rosie`, `Hugo`, `Kiki`, `Leo`.
- The `voice_id` field in the API request carries the voice name (same convention as existing models using `voice_id`).
- `clean_text` is not exposed as a request parameter for now — it defaults to `True` inside `KittenTTSSynthesizer.generate()`.
- No `language_id` handling needed (English only in current release).

## Progress Checklist

- [ ] Add `elif model_config.model_type == "KittenTTSSynthesizer"` case to `prepare_inputs()` in [`harmonyspeech/task_handler/inputs.py`](harmonyspeech/task_handler/inputs.py)
- [ ] Add `prepare_kittentts_synthesizer_inputs()` function to [`harmonyspeech/task_handler/inputs.py`](harmonyspeech/task_handler/inputs.py)

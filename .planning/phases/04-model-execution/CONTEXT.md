# Phase 4: Model Execution — CONTEXT

## Phase Summary

**Goal:** Implement all 5 Chatterbox execute functions in `model_runner_base.py` and register them in the `execute_model()` dispatch.

**Requirements:** REQ-EXEC-01, REQ-EXEC-02, REQ-EXEC-03

**Depends on:** Phase 3 (all 5 prepare functions and dispatch branches in `inputs.py` are already implemented)

---

## What Phase 3 Left Ready

The following is already implemented and working (do **not** touch):

- [`prepare_chatterbox_tts_inputs()`](harmonyspeech/task_handler/inputs.py:664) — returns `(text, conditionals_or_None, exaggeration, cfg_weight, temperature, repetition_penalty, top_p, min_p)`
- [`prepare_chatterbox_turbo_tts_inputs()`](harmonyspeech/task_handler/inputs.py:715) — returns `(text, conditionals_or_None, temperature, repetition_penalty, top_p, top_k, norm_loudness)`
- [`prepare_chatterbox_multilingual_tts_inputs()`](harmonyspeech/task_handler/inputs.py:766) — returns `(text, language_id, conditionals_or_None, exaggeration, cfg_weight, temperature, repetition_penalty, top_p, min_p)`
- [`prepare_chatterbox_embedding_inputs()`](harmonyspeech/task_handler/inputs.py:823) — returns `[audio_bytes, ...]` (raw base64-decoded bytes, no filesystem I/O)
- [`prepare_chatterbox_vc_inputs()`](harmonyspeech/task_handler/inputs.py:839) — returns `(source_bytes, target_conditionals_or_None, target_audio_bytes_or_None)`
- Dispatch branches in `prepare_inputs()` for `ChatterboxTTS`, `ChatterboxTurboTTS`, `ChatterboxMultilingualTTS`, `ChatterboxVC`, `ChatterboxEmbedding`

The `execute_model()` dispatch in [`model_runner_base.py`](harmonyspeech/task_handler/model_runner_base.py:49) currently **does not** have branches for any Chatterbox model type. It ends with `else: raise NotImplementedError(...)`.

---

## Key Architecture: ModelRunnerBase Pattern

All execute functions follow this pattern (see [`_execute_kittentts_synthesizer()`](harmonyspeech/task_handler/model_runner_base.py:570) as canonical reference):

```python
def _execute_chatterbox_tts(self, inputs, requests_to_batch):
    def synthesize(input_params):
        text, conditionals, exaggeration, ... = input_params
        # ... call self.model.*
        return base64_encoded_wav  # or base64_encoded_embedding

    outputs = []
    for i, x in enumerate(inputs):
        initial_request = requests_to_batch[i]
        inference_result = synthesize(x)
        result = self._build_result(initial_request, inference_result, TextToSpeechRequestOutput)
        outputs.append(result)
    return outputs
```

Key facts:
- `self.model` is the loaded model instance (returned by `get_model()`)
- `self.device` is the target device (e.g., `"cuda"`, `"cpu"`)
- `self._build_result(initial_request, inference_result, ResultClass)` constructs the `ExecutorResult`
- For `TextToSpeechRequestOutput`, `_build_result` will use `initial_request.request_data.input_text` as the text field automatically
- **No batching** — iterate over inputs one by one (all existing models do this; see FIXME comments in codebase)

---

## Output Classes (from `harmonyspeech/common/outputs.py`)

| Execute Function | Output Class |
|---|---|
| `_execute_chatterbox_tts` | `TextToSpeechRequestOutput` |
| `_execute_chatterbox_turbo_tts` | `TextToSpeechRequestOutput` |
| `_execute_chatterbox_multilingual_tts` | `TextToSpeechRequestOutput` |
| `_execute_chatterbox_embedding` | `SpeechEmbeddingRequestOutput` |
| `_execute_chatterbox_vc` | `VoiceConversionRequestOutput` |

`TextToSpeechRequestOutput` constructor: `(request_id, text, output, finish_reason, metrics)`
`SpeechEmbeddingRequestOutput` constructor: `(request_id, output, finish_reason, metrics)`
`VoiceConversionRequestOutput` constructor: `(request_id, output, finish_reason, metrics)`

All `output` fields must be **base64-encoded strings** (WAV audio or serialized Conditionals).

---

## Model Classes and Their Actual Python Types

This is a critical discovery from scouting:

| Model Type string | Loaded model Python class | Source module |
|---|---|---|
| `"ChatterboxTTS"` | `chatterbox.tts.ChatterboxTTS` | `chatterbox.tts` |
| `"ChatterboxTurboTTS"` | `chatterbox.tts_turbo.ChatterboxTurboTTS` | `chatterbox.tts_turbo` |
| `"ChatterboxMultilingualTTS"` | `chatterbox.mtl_tts.ChatterboxMultilingualTTS` | `chatterbox.mtl_tts` |
| `"ChatterboxVC"` | `chatterbox.vc.ChatterboxVC` | `chatterbox.vc` |
| `"ChatterboxEmbedding"` | same as `ChatterboxTTS` (shares model) | `chatterbox.tts` |

⚠️ **Critical bug to fix in this phase:** [`ChatterboxTurboTTSModel.from_pretrained()`](harmonyspeech/modeling/models/chatterbox/chatterbox.py:52) currently calls `ChatterboxTTS.from_pretrained(turbo=True)` which is **wrong** — `ChatterboxTurboTTS` is a **separate class** in `chatterbox.tts_turbo`. The wrapper must be updated to use `ChatterboxTurboTTS.from_pretrained(device=device)` from `chatterbox.tts_turbo`.

---

## Chatterbox Library API Reference

### ChatterboxTTS (`chatterbox.tts`)

```python
# Generate audio from text
model.generate(
    text,
    repetition_penalty=1.2,
    min_p=0.05,
    top_p=1.0,
    audio_prompt_path=None,     # file path — NOT used; use model.conds instead
    exaggeration=0.5,
    cfg_weight=0.5,
    temperature=0.8,
)
# Returns: torch.Tensor shape [1, samples] at model.sr (S3GEN_SR = 24000 Hz)

model.prepare_conditionals(wav_fpath, exaggeration=0.5)
# Sets model.conds (Conditionals object)
# wav_fpath: accepts BytesIO (librosa.load accepts file-like objects)

model.conds  # Conditionals object (can be set directly)
model.sr     # sample rate (24000)
```

### ChatterboxTurboTTS (`chatterbox.tts_turbo`)

```python
model.generate(
    text,
    repetition_penalty=1.2,
    min_p=0.00,
    top_p=0.95,
    audio_prompt_path=None,     # file path — NOT used; use model.conds instead
    exaggeration=0.0,           # NOT supported (logs warning, ignored)
    cfg_weight=0.0,             # NOT supported (logs warning, ignored)
    temperature=0.8,
    top_k=1000,                 # TURBO-SPECIFIC — passed to t3.inference_turbo()
    norm_loudness=True,         # TURBO-SPECIFIC — applied during prepare_conditionals()
)
# Returns: torch.Tensor shape [1, samples] at model.sr (S3GEN_SR = 24000 Hz)

model.prepare_conditionals(wav_fpath, exaggeration=0.5, norm_loudness=True)
# wav_fpath: accepts BytesIO (librosa.load accepts file-like objects)
# norm_loudness: applied here via pyloudnorm, NOT post-generation

model.conds  # Conditionals object (can be set directly)
model.sr     # sample rate (24000)
```

**Important for Turbo:** `norm_loudness` is a **param in `prepare_conditionals()`**, not in `generate()` per se. However `generate()` also accepts `norm_loudness` — it routes it through to `prepare_conditionals()` when `audio_prompt_path` is provided. When using pre-computed `Conditionals` (injected via `model.conds`), `norm_loudness` from the input tuple should still be forwarded to `generate()` as it's accepted.

### ChatterboxMultilingualTTS (`chatterbox.mtl_tts`)

```python
model.generate(
    text,
    language_id,                # required — e.g., "en", "de", "ja"
    audio_prompt_path=None,     # file path — NOT used; use model.conds instead
    exaggeration=0.5,
    cfg_weight=0.5,
    temperature=0.8,
    repetition_penalty=2.0,
    min_p=0.05,
    top_p=1.0,
)
# Returns: torch.Tensor shape [1, samples] at model.sr

model.prepare_conditionals(wav_fpath, exaggeration=0.5)  # accepts BytesIO
model.conds  # Conditionals object
model.sr     # sample rate (24000)
```

### ChatterboxVC (`chatterbox.vc`)

```python
model.generate(
    audio,                      # source audio — file path or BytesIO
    target_voice_path=None,     # target voice file path — NOT used; use set_target_voice or ref_dict
)
# Returns: torch.Tensor shape [1, samples] at model.sr (24000 Hz)

model.set_target_voice(wav_fpath)
# Sets model.ref_dict from target voice wav
# wav_fpath: accepts BytesIO via librosa.load

model.ref_dict  # internal reference dict (can be set if bypassing set_target_voice)
model.sr        # sample rate (24000)
```

---

## Resolved Gray Areas

### Gray Area 1: Audio Bytes vs. File Path API

**Problem:** `generate()` and `prepare_conditionals()` accept file paths. Phase 3 returns raw bytes.

**Resolution:** `librosa.load()` accepts `BytesIO` objects. Wrap audio bytes in `io.BytesIO(audio_bytes)` before passing to any model method that calls `librosa.load()` internally.

**Pattern for TTS (when audio bytes provided as audio prompt):**
```python
audio_buf = io.BytesIO(audio_bytes)
self.model.prepare_conditionals(audio_buf, exaggeration=exaggeration)
```

**Pattern for VC (source audio):**
```python
source_buf = io.BytesIO(source_bytes)
wav_output = self.model.generate(audio=source_buf, ...)
```

**Pattern for VC (target voice):**
```python
target_buf = io.BytesIO(target_audio_bytes)
self.model.set_target_voice(target_buf)
```

### Gray Area 2: Pre-computed Conditionals Injection

**Problem:** `generate()` has no `conditionals=` parameter.

**Resolution:** Move Conditionals to device, then assign to `self.model.conds` before calling `generate()`. Since the model runner is single-threaded (no concurrent inference per worker), this is safe.

```python
if conditionals is not None:
    self.model.conds = conditionals.to(self.device)
    wav = self.model.generate(text, ...)
else:
    # audio bytes must have been provided — prepare them
    audio_buf = io.BytesIO(audio_bytes_from_request)
    self.model.prepare_conditionals(audio_buf, ...)
    wav = self.model.generate(text, ...)
```

**Note:** When using pre-computed Conditionals, do NOT pass `audio_prompt_path` to `generate()` — leave it as `None`. The `assert self.conds is not None` check in `generate()` will pass.

### Gray Area 3: Turbo-Specific Parameters

**Problem:** `ChatterboxTurboTTS` is a **separate class** in `chatterbox.tts_turbo`, not `ChatterboxTTS(turbo=True)`.

**Resolutions:**
1. **Fix [`ChatterboxTurboTTSModel.from_pretrained()`](harmonyspeech/modeling/models/chatterbox/chatterbox.py:52)** to use `ChatterboxTurboTTS.from_pretrained(device=device)` from `chatterbox.tts_turbo`.
2. `top_k` is a real parameter on `ChatterboxTurboTTS.generate()` → pass directly.
3. `norm_loudness` is accepted by `ChatterboxTurboTTS.generate()` → pass directly.
4. The Turbo `generate()` ignores `exaggeration`, `cfg_weight`, `min_p` (logs a warning) — since Phase 3 already validates against passing these, they will never reach the execute function for Turbo.

### Gray Area 4: Conditionals Serialization (Embedding Model)

**Problem:** `Conditionals.save(fpath)` only accepts a file path.

**Resolution:** `torch.save()` (called internally by `save()`) also accepts file-like objects. Build the arg dict and save to BytesIO directly:

```python
# Compute Conditionals from audio bytes
audio_buf = io.BytesIO(audio_bytes)
self.model.prepare_conditionals(audio_buf)  # sets self.model.conds
conditionals = self.model.conds

# Serialize to BytesIO (bypassing Conditionals.save(fpath))
buf = io.BytesIO()
arg_dict = dict(t3=conditionals.t3.__dict__, gen=conditionals.gen)
torch.save(arg_dict, buf)
buf.seek(0)
encoded = base64.b64encode(buf.getvalue()).decode('utf-8')
```

---

## Execute Function Signatures and Logic

### `_execute_chatterbox_tts(self, inputs, requests_to_batch)`

Input tuple from prepare: `(text, conditionals_or_None, exaggeration, cfg_weight, temperature, repetition_penalty, top_p, min_p)`

Logic:
1. If `conditionals` is not None: `self.model.conds = conditionals.to(self.device)`
2. Else: input_audio was provided (raw bytes in `initial_request.request_data.input_audio`, base64-decoded) → `prepare_conditionals(BytesIO(...))`
3. Call `self.model.generate(text, exaggeration=..., cfg_weight=..., temperature=..., repetition_penalty=..., top_p=..., min_p=...)`
4. Encode result tensor to WAV base64 via soundfile
5. Return `TextToSpeechRequestOutput`

⚠️ **Note on audio source:** `prepare_chatterbox_tts_inputs()` does NOT include raw audio bytes in its output tuple (it only passes Conditionals when embedding is provided, or `None` when audio should be used). When `conditionals is None` AND `input_audio` is present, the execute function must re-read from `initial_request.request_data.input_audio` (base64-decode it). This is an intentional design since the prepare function's job was to deserialize the Conditionals object, not the raw audio.

### `_execute_chatterbox_turbo_tts(self, inputs, requests_to_batch)`

Input tuple from prepare: `(text, conditionals_or_None, temperature, repetition_penalty, top_p, top_k, norm_loudness)`

Logic:
1. If `conditionals` is not None: `self.model.conds = conditionals.to(self.device)`
2. Else: `self.model.prepare_conditionals(BytesIO(base64.b64decode(request.input_audio)), norm_loudness=norm_loudness)`
3. Call `self.model.generate(text, temperature=..., repetition_penalty=..., top_p=..., top_k=..., norm_loudness=norm_loudness)`
4. Encode result tensor to WAV base64
5. Return `TextToSpeechRequestOutput`

### `_execute_chatterbox_multilingual_tts(self, inputs, requests_to_batch)`

Input tuple from prepare: `(text, language_id, conditionals_or_None, exaggeration, cfg_weight, temperature, repetition_penalty, top_p, min_p)`

Logic:
1. Same Conditionals injection pattern as base TTS
2. Call `self.model.generate(text, language_id=language_id, exaggeration=..., cfg_weight=..., temperature=..., repetition_penalty=..., top_p=..., min_p=...)`
3. Return `TextToSpeechRequestOutput`

### `_execute_chatterbox_embedding(self, inputs, requests_to_batch)`

Input from prepare: `[audio_bytes, ...]` (list of raw bytes per request)

Logic:
1. `audio_buf = io.BytesIO(audio_bytes)`
2. `self.model.prepare_conditionals(audio_buf)` → sets `self.model.conds`
3. Serialize: `buf = io.BytesIO(); torch.save(dict(t3=..., gen=...), buf); buf.seek(0)`
4. `encoded = base64.b64encode(buf.getvalue()).decode('utf-8')`
5. Return `SpeechEmbeddingRequestOutput`

### `_execute_chatterbox_vc(self, inputs, requests_to_batch)`

Input tuple from prepare: `(source_bytes, target_conditionals_or_None, target_audio_bytes_or_None)`

Logic:
1. If `target_conditionals` is not None: `self.model.ref_dict = target_conditionals.to(self.device).gen`
2. Else: `self.model.set_target_voice(io.BytesIO(target_audio_bytes))`
3. Call `self.model.generate(audio=io.BytesIO(source_bytes))`
4. Encode result tensor to WAV base64
5. Return `VoiceConversionRequestOutput`

---

## WAV Encoding Pattern

All 3 TTS/VC functions must encode the returned `torch.Tensor` to base64 WAV. Canonical reference from [`_execute_kittentts_synthesizer()`](harmonyspeech/task_handler/model_runner_base.py:570):

```python
import soundfile as sf

audio_array = wav_tensor.squeeze(0).numpy()  # or .detach().cpu().numpy() if needed
with io.BytesIO() as wav_buffer:
    sf.write(wav_buffer, audio_array, self.model.sr, format='WAV')
    wav_bytes = wav_buffer.getvalue()
encoded_audio = base64.b64encode(wav_bytes).decode('utf-8')
```

Sample rate: all Chatterbox models output at `model.sr` which equals `S3GEN_SR = 24000`.

---

## execute_model() Dispatch Additions

Add the following branches in [`execute_model()`](harmonyspeech/task_handler/model_runner_base.py:49) before the `else: raise NotImplementedError`:

```python
elif model_type == "ChatterboxTTS":
    outputs = self._execute_chatterbox_tts(inputs, requests_to_batch)
elif model_type == "ChatterboxTurboTTS":
    outputs = self._execute_chatterbox_turbo_tts(inputs, requests_to_batch)
elif model_type == "ChatterboxMultilingualTTS":
    outputs = self._execute_chatterbox_multilingual_tts(inputs, requests_to_batch)
elif model_type == "ChatterboxEmbedding":
    outputs = self._execute_chatterbox_embedding(inputs, requests_to_batch)
elif model_type == "ChatterboxVC":
    outputs = self._execute_chatterbox_vc(inputs, requests_to_batch)
```

---

## Files to Modify

| File | Change |
|---|---|
| `harmonyspeech/task_handler/model_runner_base.py` | Add 5 `_execute_chatterbox_*` methods; add 5 dispatch branches in `execute_model()` |
| `harmonyspeech/modeling/models/chatterbox/chatterbox.py` | Fix `ChatterboxTurboTTSModel.from_pretrained()` to use `ChatterboxTurboTTS` from `chatterbox.tts_turbo` |

---

## Imports Needed in model_runner_base.py

The file already imports `base64`, `io`, `soundfile as sf`, `torch`. Additionally need:

```python
from chatterbox.tts_turbo import ChatterboxTurboTTS  # NOT needed for execute, only for wrapper fix
```

For `execute_model_runner_base.py` specifically — no new imports needed, all required (`base64`, `io`, `torch`, `sf`) are already present.

---

## Success Criteria (from ROADMAP)

1. `_execute_chatterbox_tts()` generates audio and returns base64 WAV in `TextToSpeechRequestOutput` ✓
2. `_execute_chatterbox_turbo_tts()` applies Turbo-specific params (`norm_loudness`, `top_k`) ✓
3. `_execute_chatterbox_multilingual_tts()` forwards `language_id` to model ✓
4. `_execute_chatterbox_embedding()` computes Conditionals from audio and returns base64 serialized embedding ✓
5. `_execute_chatterbox_vc()` performs voice conversion and returns output audio ✓

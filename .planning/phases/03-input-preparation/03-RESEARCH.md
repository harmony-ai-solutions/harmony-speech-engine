# Phase 3: Input Preparation - Research

**Researched:** 2026-03-13
**Domain:** Input preparation pipeline, dataclass/Pydantic extension, BytesIO deserialization, language registration
**Confidence:** HIGH

## Summary

Phase 3 extends two data-model layers (the `TextToSpeechGenerationOptions` dataclass and `GenerationOptions` Pydantic model) with 8 Chatterbox-specific fields, then implements 5 prepare functions in `inputs.py`. All patterns are established and verifiable from existing codebase code — no external library research is required. The main complexity is (1) variant-specific `ValueError` validation (new pattern, deliberate departure from existing silent-ignore), (2) embedding deserialization via `BytesIO + torch.load` (established OpenVoice pattern), and (3) registering 23 language codes as `LanguageOptions` entries in the `ChatterboxMultilingualTTSModel` wrapper.

All source signatures have been confirmed from `.current_work/chatterbox-tts/src/chatterbox/`. The Chatterbox library's `SUPPORTED_LANGUAGES` dict already lives in `mtl_tts.py` (23 entries) and is re-exported by `chatterbox/__init__.py`. The `Conditionals` class has `save(fpath)` / `load(fpath, map_location)` methods; the BytesIO substitution for `fpath` works because `torch.save`/`torch.load` accept file-like objects.

**Primary recommendation:** Follow `prepare_kittentts_synthesizer_inputs` structure exactly (inner closure + `ThreadPoolExecutor.map`); use `prepare_openvoice_tone_converter_inputs` as the BytesIO deserialization reference; add `SUPPORTED_LANGUAGES` as a class constant on `ChatterboxMultilingualTTSModel` and register each entry as a `LanguageOptions` row during model card construction.

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

#### Unsupported Parameter Validation
- **Raise `ValueError`** when a caller explicitly passes a non-None value for a parameter the target model doesn't support.
- This applies symmetrically: Turbo-only params (`norm_loudness`, `top_k`) sent to base ChatterboxTTS raise `ValueError`; base-TTS params (`exaggeration`, `cfg_weight`, `min_p`) sent to ChatterboxTurboTTS raise `ValueError`.
- Only validate when the field is **non-None** — absent/default params do not trigger errors. Callers sending a full options object with None fields must not be broken.
- This is a deliberate departure from the existing silent-ignore pattern elsewhere in the codebase. Chatterbox establishes the new standard.
- Error messages should name the specific param(s) that are unsupported and the model type.

#### Language Validation (Multilingual Model)
- `prepare_chatterbox_multilingual_tts_inputs()` does **not** duplicate language validation from the serving engine — the serving engine (`endpoints/openai/serving_engine.py`) already validates `language` against `model.languages` before the request reaches the prepare function. Phase 3 relies on this.
- If `language_id` is absent/None, **default to English** (`"en"`).
- The 23 supported Chatterbox Multilingual language codes live in the **model wrapper class** (in `harmonyspeech/modeling/models/chatterbox/chatterbox.py`), not in the prepare function. The prepare function imports the list from the model.
- Language discoverability at the API level must align with the existing `LanguageOptions` / `model.languages` pattern in `serving_engine.py`. Chatterbox Multilingual registers its supported languages into that existing structure so they surface via `/v1/models`.
- Languages are **not** added as "voices" in the model card — see Language Registration decision below (REQ-INPUT-06).

#### VC Input Conflict Resolution
- `prepare_chatterbox_vc_inputs()` requires EITHER `target_audio` OR `target_embedding` (not both, not neither).
  - Both provided → **raise `ValueError`**: "Provide either target_audio or target_embedding, not both."
  - Neither provided → **raise `ValueError`**: "ChatterboxVC requires either target_audio or target_embedding."
- Same rule for TTS voice cloning: if both `input_audio` AND `input_embedding` are provided → **raise `ValueError`**.
- `source_audio` presence is NOT validated in the prepare function — trust that the API/request schema layer enforces it upstream.

#### Default Values Ownership
- Chatterbox-specific generation option defaults are applied **inside each prepare function** when the field is None.
- Each prepare function defines its own model-specific defaults inline:
  - `prepare_chatterbox_tts_inputs()`: `exaggeration=0.5`, `cfg_weight=0.5`, `temperature=0.8`, `repetition_penalty=1.2`, `top_p=1.0`, `min_p=0.05`
  - `prepare_chatterbox_turbo_tts_inputs()`: `temperature=0.8`, `repetition_penalty=1.2`, `top_p=0.95`, `top_k=1000`, `norm_loudness=True`
  - `prepare_chatterbox_multilingual_tts_inputs()`: `exaggeration=0.5`, `cfg_weight=0.5`, `temperature=0.8`, `repetition_penalty=2.0`, `top_p=1.0`, `min_p=0.05`
- `TextToSpeechGenerationOptions` is extended (not a separate class) with all 8 Chatterbox fields, all typed `Optional[T] = None`. The dataclass does NOT set model-specific defaults — those remain in the prepare functions.
- `GenerationOptions` (Pydantic wire model in `protocol.py`) also gets the corresponding 8 fields added with `= None` defaults.

#### Generation Option Validation (REQ-INPUT-05)
- Each prepare function validates generation options against the params its model variant supports.
- Raise `ValueError` when a **non-None** unsupported param is explicitly passed:
  - ChatterboxTTS / ChatterboxMultilingualTTS: reject `top_k`, `norm_loudness`
  - ChatterboxTurboTTS: reject `exaggeration`, `cfg_weight`, `min_p`
- Validation is symmetric — works both directions.
- Error messages name the specific param and the model type.
- This is a conscious departure from the existing silent-ignore pattern; existing models (KittenTTS, OpenVoice, MeloTTS) are NOT changed.

#### Language Registration for Multilingual Model (REQ-INPUT-06)
- `ChatterboxMultilingualTTSModel` defines a `SUPPORTED_LANGUAGES` constant (23 language codes) in the model wrapper class.
- During model card construction, each language is registered as a `LanguageOptions` entry so they appear via `/v1/models` and are validated by the serving engine automatically.
- Languages are **not** added as voices — research the MeloTTS / OpenVoice model card pattern and follow it (or extend it to work for all architectures if needed).
- The prepare function relies on the serving engine's existing language validation; it does NOT duplicate it.

### OpenCode's Discretion
- Exact `BytesIO` wrapping approach for embedding deserialization — follow the existing `torch.load` pattern from OpenVoice as reference.
- `ThreadPoolExecutor` parallelism — follow the KittenTTS pattern (all existing prepare functions use it).

### Deferred Ideas (OUT OF SCOPE)
- Shared language registry across all model families — broader than Phase 3. The existing `LanguageOptions` pattern in `serving_engine.py` covers Phase 3 needs.
- Extending the unsupported-param validation pattern to existing models (KittenTTS, OpenVoice, MeloTTS) — this would change established behavior and belongs in a separate refactoring phase.
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| REQ-INPUT-01 | Extend `TextToSpeechGenerationOptions` dataclass with 8 optional Chatterbox fields (all `Optional[T] = None`) | Dataclass currently has 5 fields in `harmonyspeech/common/inputs.py:8`; add 8 fields with `= None` default per lock decision |
| REQ-INPUT-02 | Implement 3 TTS prepare functions (base, multilingual, turbo) | Patterns from `prepare_kittentts_synthesizer_inputs` + `prepare_openvoice_tone_converter_inputs`; signatures confirmed from `tts.py`, `tts_turbo.py`, `mtl_tts.py` |
| REQ-INPUT-03 | Implement `prepare_chatterbox_embedding_inputs()` for embedding endpoint | Simplest function: base64-decode `input_audio`, return raw bytes; no `librosa`, no temp files |
| REQ-INPUT-04 | Implement `prepare_chatterbox_vc_inputs()` for ChatterboxVC | `VoiceConversionRequestInput` has `source_audio`, `target_audio`, `target_embedding`; conflict logic raises `ValueError` |
| REQ-INPUT-05 | Variant-specific generation option validation — `ValueError` for non-None unsupported params | New pattern; symmetric validation table documented below; 3 variants × 2–3 rejected fields each |
| REQ-INPUT-06 | ChatterboxMultilingualTTS registers 23 languages as `LanguageOptions` in model card | `SUPPORTED_LANGUAGES` dict in `mtl_tts.py:24`; add as class constant to `ChatterboxMultilingualTTSModel`; register via `model_card_from_config` extension or new method |
</phase_requirements>

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `io.BytesIO` | stdlib | In-memory file-like object for audio/embedding data | Already used in `inputs.py` for LibROSA loading and OpenVoice embedding deserialization |
| `base64` | stdlib | Decode base64-encoded audio/embedding strings | Used in every existing prepare function |
| `torch` | ≥2.7.1 | `torch.load` / `torch.save` for Conditionals deserialization | Already in requirements; Conditionals serialization uses `torch.save`/`torch.load` |
| `concurrent.futures.ThreadPoolExecutor` | stdlib | Batch parallelism for prepare functions | Pattern used by ALL existing prepare functions |
| `dataclasses.dataclass` | stdlib | `TextToSpeechGenerationOptions` is a `@dataclass` | Already the pattern in `common/inputs.py` |
| `pydantic.BaseModel` | ~2.x | `GenerationOptions` is a Pydantic model | Already in `protocol.py`; all fields use `Optional[T] = None` |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `chatterbox.mtl_tts.SUPPORTED_LANGUAGES` | installed | 23-entry dict of language codes | Import in `ChatterboxMultilingualTTSModel` for `SUPPORTED_LANGUAGES` constant |
| `chatterbox.tts.Conditionals` | installed | Serialize/deserialize Chatterbox Conditionals objects | Deserialization in TTS/VC prepare functions when `input_embedding` is present |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `torch.load` into BytesIO | Custom pickle/JSON serialization | torch.load is the native path used by `Conditionals.load`; don't deviate |
| Inline SUPPORTED_LANGUAGES in prepare function | Import from model wrapper class | Locked decision: language list lives in model wrapper, not prepare function |

---

## Architecture Patterns

### Pattern 1: Prepare Function Structure (KittenTTS reference)
**What:** Inner closure `prepare(request)` inside a named `prepare_X_inputs(requests_to_batch)` function; `ThreadPoolExecutor.map` runs the closure across all requests in batch.  
**When to use:** All 5 Chatterbox prepare functions must follow this pattern.  
**Source:** `harmonyspeech/task_handler/inputs.py:587-612`

```python
def prepare_chatterbox_tts_inputs(requests_to_batch: List[Union[
    TextToSpeechRequestInput,
    SynthesisRequestInput
]]):
    def prepare(request):
        # 1. Validate unsupported params (ValueError if non-None)
        # 2. Decode input (base64 → bytes or Conditionals)
        # 3. Apply defaults for None generation options
        # 4. Return parameter tuple
        ...
    
    with ThreadPoolExecutor() as executor:
        inputs = list(executor.map(prepare, requests_to_batch))
    
    return inputs
```

### Pattern 2: BytesIO + torch.load for Embedding Deserialization
**What:** Decode base64 → bytes → `io.BytesIO` → `torch.load(bytes_obj, ...)` to reconstruct a `Conditionals` object without touching the filesystem.  
**When to use:** `prepare_chatterbox_tts_inputs` and `prepare_chatterbox_multilingual_tts_inputs` when `input_embedding` is provided; `prepare_chatterbox_vc_inputs` when `target_embedding` is provided.  
**Source:** `harmonyspeech/task_handler/inputs.py:434-456` (OpenVoice pattern for BytesIO); `Conditionals.load` in `tts.py:98-103` / `mtl_tts.py:127-132` for the load API.

```python
# Decode and deserialize a Conditionals object from base64 string
embedding_bytes = base64.b64decode(request.input_embedding)
embedding_buf = io.BytesIO(embedding_bytes)
# Conditionals.load accepts a file-like object (BytesIO works because torch.load does)
conditionals = Conditionals.load(embedding_buf, map_location="cpu")
```

**Critical note:** `Conditionals.load` calls `torch.load(fpath, map_location=..., weights_only=True)`. `torch.load` accepts any file-like object (not just path strings), so `BytesIO` works directly. No temp file needed.

### Pattern 3: model_card_from_config — Single Language Registration
**What:** `OpenAIServing.model_card_from_config` reads `config.language` (a single string) and creates one `LanguageOptions` entry. Voices are then appended from `config.voices`.  
**Source:** `harmonyspeech/endpoints/openai/serving_engine.py:131-143`

```python
# Current pattern — single language only:
if config.language is not None and config.language != "":
    lang_option = LanguageOptions(language=config.language)
    card.languages.append(lang_option)
    for voice in config.voices:
        lang_option.voices.append(VoiceOptions(voice=voice))
```

**Problem for Multilingual:** `model_card_from_config` only handles a single `config.language` string — it cannot register 23 languages from a list. The `ChatterboxMultilingualTTSModel` needs a different registration path.

### Pattern 4: Language Registration for ChatterboxMultilingualTTS
**What:** `ChatterboxMultilingualTTSModel` defines `SUPPORTED_LANGUAGES` as a class constant mirroring `chatterbox.mtl_tts.SUPPORTED_LANGUAGES`. During model card construction, the loader (or a new `model_card_from_config` branch) iterates `SUPPORTED_LANGUAGES.keys()` and appends a `LanguageOptions(language=code, voices=[])` entry for each.

**Where to add the hook:** The model card construction in `serving_engine.py:model_card_from_config` reads from `ModelConfig`. The cleanest approach is:
1. Add `SUPPORTED_LANGUAGES: dict = {...}` to `ChatterboxMultilingualTTSModel` in `chatterbox.py`
2. In `loader.py` (where model cards are built or model configs populated), detect `model_type == "ChatterboxMultilingualTTS"` and call `model_card_add_language_list(card, languages)` — a new static helper or inline loop.

**Alternative (simpler):** Extend `model_card_from_config` / `model_card_add_config` to accept an optional `language_list: List[str]` parameter, or check for a `config.languages` (plural) attribute set by the loader before card construction.

**Confirmed LanguageOptions schema** (`protocol.py:26-28`):
```python
class LanguageOptions(BaseModel):
    language: str = "default"
    voices: Optional[List[VoiceOptions]] = Field(default_factory=list)
```

Chatterbox Multilingual has no voices per language (voice cloning via audio, not voice IDs), so each `LanguageOptions` entry has `voices=[]` — that's correct and matches the model's capabilities.

### Pattern 5: Generation Option Validation (new pattern)
**What:** Before applying defaults, each prepare function checks the `generation_options` object for explicitly-set (non-None) fields that the target model does not support. Raise `ValueError` naming the param and model type.

```python
# Example for prepare_chatterbox_tts_inputs (rejects top_k, norm_loudness)
opts = request.generation_options
if opts is not None:
    if opts.top_k is not None:
        raise ValueError("top_k is not supported by ChatterboxTTS")
    if opts.norm_loudness is not None:
        raise ValueError("norm_loudness is not supported by ChatterboxTTS")
```

**Validation matrix (confirmed from CONTEXT.md + model generate() signatures):**

| Model | Rejects (non-None → ValueError) | Accepts |
|-------|--------------------------------|---------|
| ChatterboxTTS | `top_k`, `norm_loudness` | `exaggeration`, `cfg_weight`, `temperature`, `repetition_penalty`, `top_p`, `min_p` |
| ChatterboxMultilingualTTS | `top_k`, `norm_loudness` | `exaggeration`, `cfg_weight`, `temperature`, `repetition_penalty`, `top_p`, `min_p` |
| ChatterboxTurboTTS | `exaggeration`, `cfg_weight`, `min_p` | `temperature`, `repetition_penalty`, `top_p`, `top_k`, `norm_loudness` |

Note: `tts_turbo.py:266-267` shows the turbo `generate()` silently ignores `cfg_weight`, `min_p`, `exaggeration` internally (with a warning log). HSE replaces that silent ignore with an explicit `ValueError` per the locked decision.

### Pattern 6: prepare_inputs() Dispatch Extension
**What:** The top-level `prepare_inputs()` function in `inputs.py` dispatches to the correct prepare function based on `model_config.model_type`. New elif branches must be added for all 4 Chatterbox model types.

```python
elif model_config.model_type == "ChatterboxTTS":
    for r in requests_to_batch:
        if isinstance(r.request_data, (TextToSpeechRequestInput, SynthesisRequestInput)):
            inputs.append(r.request_data)
        else:
            raise ValueError(...)
    return prepare_chatterbox_tts_inputs(inputs)
# ... repeat for ChatterboxTurboTTS, ChatterboxMultilingualTTS, ChatterboxVC, ChatterboxEmbedding
```

### Anti-Patterns to Avoid
- **Temp files:** No `NamedTemporaryFile` for Chatterbox (REQ-PERF-01). Use `io.BytesIO` throughout.
- **Duplicating language validation:** The serving engine validates `language` before the prepare function runs. Don't re-check in the prepare function.
- **Setting defaults in the dataclass:** `TextToSpeechGenerationOptions` fields must all be `= None`. Defaults live in prepare functions.
- **Treating `torch.load` as path-only:** `Conditionals.load` calls `torch.load(fpath, ...)` — in Python, `torch.load` accepts both path strings AND file-like objects (BytesIO). Confirmed from PyTorch docs.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Conditionals deserialization | Custom pickle/JSON parser | `Conditionals.load(BytesIO(...), map_location="cpu")` | `Conditionals.save/load` is the canonical API; torch handles tensor layout |
| Audio bytes decoding | Custom encoding | `base64.b64decode(request.input_audio)` | Established pattern in every existing prepare function |
| Batch parallelism | Manual threading | `ThreadPoolExecutor` inner closure pattern | All existing prepare functions use this; executor handles errors cleanly |
| Language list construction | Hardcoded 23 strings | Import `SUPPORTED_LANGUAGES` from `chatterbox.mtl_tts` via model wrapper constant | Single source of truth; auto-updates if chatterbox adds languages |

---

## Common Pitfalls

### Pitfall 1: None-checking Generation Options Object Itself
**What goes wrong:** `request.generation_options` may itself be `None` (not just the individual fields). If you access `opts.top_k` without checking `if opts is not None` first, you get `AttributeError`.  
**Why it happens:** The field is `Optional[GenerationOptions]` — callers who omit generation options entirely send `None`.  
**How to avoid:** Always guard: `if request.generation_options is not None:` before accessing any field.  
**Warning signs:** `AttributeError: 'NoneType' object has no attribute 'top_k'` in tests.

### Pitfall 2: ValueError Too Early (Before None Guard)
**What goes wrong:** Validation fires for ALL requests, even those that never set the option, if the None check is done in the wrong order.  
**Why it happens:** Checking `opts.top_k is not None` must also be inside `if opts is not None`.  
**How to avoid:** Structure validation as:
```python
opts = request.generation_options
if opts is not None:
    if opts.top_k is not None:
        raise ValueError("top_k is not supported by ChatterboxTTS")
```

### Pitfall 3: BytesIO Position After Read
**What goes wrong:** After calling `io.BytesIO(data)`, if you read from it (e.g. for inspection) the file pointer is at the end. Passing it to `torch.load` then reads 0 bytes.  
**Why it happens:** BytesIO tracks position; `torch.load` reads from current position.  
**How to avoid:** Create the BytesIO object fresh from bytes right before passing to `torch.load` (don't reuse after seeking/reading), or call `buf.seek(0)` before loading.

### Pitfall 4: SUPPORTED_LANGUAGES Key vs Value
**What goes wrong:** `SUPPORTED_LANGUAGES` in `mtl_tts.py` maps `{"en": "English", "fr": "French", ...}` — keys are language codes, values are human-readable names. `LanguageOptions.language` takes the code (`"en"`), not the name.  
**Why it happens:** Easy to iterate `.values()` instead of `.keys()`.  
**How to avoid:** Iterate `SUPPORTED_LANGUAGES.keys()` to get the 2-letter codes. Confirmed: `mtl_tts.py:24-48`.

### Pitfall 5: Chatterbox model_type vs wrapper class name
**What goes wrong:** The HSE `model_type` strings registered in Phase 2 are `"ChatterboxTTS"`, `"ChatterboxTurboTTS"`, `"ChatterboxMultilingualTTS"`, `"ChatterboxVC"`. The `prepare_inputs()` dispatch must use exactly these strings.  
**Why it happens:** Easy to use the wrapper class name (`ChatterboxTTSModel`) vs the registered model_type string.  
**How to avoid:** Check `harmonyspeech/modeling/models/__init__.py` for the exact registered strings from Phase 2 completion.

### Pitfall 6: VC prepare function — target_embedding is a pre-serialized Conditionals
**What goes wrong:** For ChatterboxVC, `target_embedding` is base64-encoded serialized `Conditionals` data (same format as TTS embedding). The prepare function must deserialize it to a `Conditionals` object (or pass raw bytes for the execution step to handle).  
**Why it happens:** `ChatterboxVC.generate()` accepts `target_voice_path` (a file path), not a `Conditionals` object directly. The execution layer (Phase 4) will call `set_target_voice()`. So the prepare function should return bytes/BytesIO for audio, or a deserialized `Conditionals` for embeddings.  
**How to avoid:** Phase 3 CONTEXT.md says: prepare function returns `source_audio bytes + target Conditionals (or bytes to compute)`. Clarify: return the deserialized `Conditionals` object when `target_embedding` is provided; return raw audio bytes when `target_audio` is provided (execution step will call `set_target_voice`).

---

## Code Examples

### Extended TextToSpeechGenerationOptions (REQ-INPUT-01)
Source: `harmonyspeech/common/inputs.py:7-14` + decisions

```python
@dataclass
class TextToSpeechGenerationOptions:
    seed: Optional[int]
    style: Optional[int]
    speed: Optional[float]
    pitch: Optional[float]
    energy: Optional[float]
    # Chatterbox-specific fields (all None = model default)
    exaggeration: Optional[float] = None
    cfg_weight: Optional[float] = None
    temperature: Optional[float] = None
    repetition_penalty: Optional[float] = None
    top_p: Optional[float] = None
    min_p: Optional[float] = None
    top_k: Optional[int] = None
    norm_loudness: Optional[bool] = None
```

### Extended GenerationOptions Pydantic model (REQ-INPUT-01)
Source: `harmonyspeech/endpoints/openai/protocol.py:52-57` + decisions

```python
class GenerationOptions(BaseModel):
    seed: Optional[int] = None
    style: Optional[int] = None
    speed: Optional[float] = None
    pitch: Optional[float] = None
    energy: Optional[float] = None
    # Chatterbox-specific fields
    exaggeration: Optional[float] = None
    cfg_weight: Optional[float] = None
    temperature: Optional[float] = None
    repetition_penalty: Optional[float] = None
    top_p: Optional[float] = None
    min_p: Optional[float] = None
    top_k: Optional[int] = None
    norm_loudness: Optional[bool] = None
```

### prepare_chatterbox_tts_inputs (REQ-INPUT-02)
Tuple returned: `(text, conditionals_or_None, exaggeration, cfg_weight, temperature, repetition_penalty, top_p, min_p)`

```python
def prepare_chatterbox_tts_inputs(requests_to_batch: List[Union[
    TextToSpeechRequestInput,
    SynthesisRequestInput
]]):
    def prepare(request):
        opts = request.generation_options

        # 1. Validate unsupported params
        if opts is not None:
            if opts.top_k is not None:
                raise ValueError("top_k is not supported by ChatterboxTTS")
            if opts.norm_loudness is not None:
                raise ValueError("norm_loudness is not supported by ChatterboxTTS")

        # 2. Conflict check: both input_audio AND input_embedding
        if request.input_audio is not None and request.input_embedding is not None:
            raise ValueError("Provide either input_audio or input_embedding, not both.")

        # 3. Deserialize embedding if provided
        conditionals = None
        if request.input_embedding is not None:
            embedding_bytes = base64.b64decode(request.input_embedding)
            embedding_buf = io.BytesIO(embedding_bytes)
            conditionals = Conditionals.load(embedding_buf, map_location="cpu")

        # 4. Apply model-specific defaults
        exaggeration = (opts.exaggeration if opts and opts.exaggeration is not None else 0.5)
        cfg_weight = (opts.cfg_weight if opts and opts.cfg_weight is not None else 0.5)
        temperature = (opts.temperature if opts and opts.temperature is not None else 0.8)
        repetition_penalty = (opts.repetition_penalty if opts and opts.repetition_penalty is not None else 1.2)
        top_p = (opts.top_p if opts and opts.top_p is not None else 1.0)
        min_p = (opts.min_p if opts and opts.min_p is not None else 0.05)

        return request.input_text, conditionals, exaggeration, cfg_weight, temperature, repetition_penalty, top_p, min_p

    with ThreadPoolExecutor() as executor:
        inputs = list(executor.map(prepare, requests_to_batch))
    return inputs
```

### prepare_chatterbox_embedding_inputs (REQ-INPUT-03)
Returns raw audio bytes. Execution step calls `prepare_conditionals_from_audio()`.

```python
def prepare_chatterbox_embedding_inputs(requests_to_batch: List[SpeechEmbeddingRequestInput]):
    def prepare(request):
        audio_bytes = base64.b64decode(request.input_audio)
        return audio_bytes  # raw bytes; execution step wraps in BytesIO

    with ThreadPoolExecutor() as executor:
        inputs = list(executor.map(prepare, requests_to_batch))
    return inputs
```

### prepare_chatterbox_vc_inputs (REQ-INPUT-04)
Tuple returned: `(source_audio_bytes, target_conditionals_or_None, target_audio_bytes_or_None)`

```python
def prepare_chatterbox_vc_inputs(requests_to_batch: List[VoiceConversionRequestInput]):
    def prepare(request):
        # Conflict validation
        if request.target_audio is not None and request.target_embedding is not None:
            raise ValueError("Provide either target_audio or target_embedding, not both.")
        if request.target_audio is None and request.target_embedding is None:
            raise ValueError("ChatterboxVC requires either target_audio or target_embedding.")

        source_audio_bytes = base64.b64decode(request.source_audio)

        target_conditionals = None
        target_audio_bytes = None
        if request.target_embedding is not None:
            emb_bytes = base64.b64decode(request.target_embedding)
            emb_buf = io.BytesIO(emb_bytes)
            target_conditionals = Conditionals.load(emb_buf, map_location="cpu")
        else:
            target_audio_bytes = base64.b64decode(request.target_audio)

        return source_audio_bytes, target_conditionals, target_audio_bytes

    with ThreadPoolExecutor() as executor:
        inputs = list(executor.map(prepare, requests_to_batch))
    return inputs
```

### SUPPORTED_LANGUAGES constant on ChatterboxMultilingualTTSModel (REQ-INPUT-06)
Add to `harmonyspeech/modeling/models/chatterbox/chatterbox.py`:

```python
from chatterbox.mtl_tts import SUPPORTED_LANGUAGES as _CHATTERBOX_SUPPORTED_LANGUAGES

class ChatterboxMultilingualTTSModel:
    # 23 supported language codes for language registration and validation
    SUPPORTED_LANGUAGES: dict = _CHATTERBOX_SUPPORTED_LANGUAGES.copy()
    ...
```

### Language Registration in model card construction (REQ-INPUT-06)
In `serving_engine.py` or loader, after standard `model_card_from_config`:

```python
# For ChatterboxMultilingualTTS: register all 23 languages
from harmonyspeech.modeling.models.chatterbox.chatterbox import ChatterboxMultilingualTTSModel

if config.model_type == "ChatterboxMultilingualTTS":
    # Override: create card with all supported languages, no voices per language
    card = ModelCard(id=config.name, root=config.name)
    for lang_code in ChatterboxMultilingualTTSModel.SUPPORTED_LANGUAGES.keys():
        card.languages.append(LanguageOptions(language=lang_code, voices=[]))
    return card
```

**Note on placement:** The cleanest location is a new branch in `model_card_from_config` (static method on `OpenAIServing`) that checks `config.model_type`. This keeps card construction centralized. Alternatively, it can live in the loader where `ModelConfig` is processed — this is the pattern used for existing models. The planner should decide based on where Phase 2's loader changes landed.

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Temp files for audio (OpenVoice encoder) | BytesIO in-memory | Existing in most prepare functions | REQ-PERF-01: no temp files for Chatterbox |
| Silent ignore for unsupported params | `ValueError` for explicit non-None | Phase 3 (new standard) | Callers get clear errors instead of silently wrong output |
| Single-language model cards | Multi-language via SUPPORTED_LANGUAGES | Phase 3 | 23-language registration for multilingual model |

**Deprecated/outdated:**
- `NamedTemporaryFile` usage in OpenVoice encoder: still there for legacy, but NOT to be replicated for Chatterbox

---

## Open Questions

1. **Where exactly to put the multilingual language registration logic**
   - What we know: `model_card_from_config` (static method in `serving_engine.py`) only handles `config.language` (single string). The loader (`loader.py`) calls `model_cards_from_config_groups()` which calls `model_card_from_config`.
   - What's unclear: Did Phase 2 add any loader-level card construction hooks? Need to check `loader.py` for Phase 2 changes.
   - Recommendation: Check `harmonyspeech/modeling/loader.py` for the model card construction path used by Phase 2. If the static method is the right injection point, add a new branch for "ChatterboxMultilingualTTS" that iterates `SUPPORTED_LANGUAGES`. If the loader populates `ModelConfig.language` before card construction, set `config.language` to each language in a loop — but that won't work with single-string semantics. A direct card construction branch is cleaner.

2. **Return type of prepare_chatterbox_vc_inputs — Conditionals vs raw bytes**
   - What we know: `ChatterboxVC.generate()` accepts `target_voice_path` (file path); `set_target_voice(wav_fpath)` takes a path too. Neither method accepts a pre-loaded `Conditionals` directly.
   - What's unclear: Phase 4 (execution) will need to either (a) accept a deserialized `Conditionals` object and set `model.ref_dict` directly, or (b) write audio to a temp file and pass the path. Option (a) is consistent with REQ-PERF-01 (no temp files).
   - Recommendation: Return the deserialized `Conditionals` object from the prepare function; Phase 4 execution accesses `model.ref_dict` directly (it's a public attribute on `ChatterboxVC`). Planner should note this dependency between Phase 3 output tuple and Phase 4 input consumption.

3. **ChatterboxTurboTTS — does it accept audio_prompt_path or Conditionals for voice cloning?**
   - What we know: `tts_turbo.py:ChatterboxTurboTTS.generate()` has `audio_prompt_path` parameter; `prepare_conditionals()` method exists. No voice cloning via pre-computed Conditionals in the generate() signature.
   - What's unclear: Should Turbo TTS support `input_embedding` (pre-computed Conditionals) in the prepare function?
   - Recommendation: Yes — implement the same pattern as base TTS (deserialize Conditionals from `input_embedding` if provided). The execution step can set `model.conds` directly (it's a public attribute). This matches the existing Conditionals pattern in both `tts.py` and `tts_turbo.py`.

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (configured in `pyproject.toml`) |
| Config file | `pyproject.toml` → `[tool.pytest.ini_options]` |
| Quick run command | `pytest tests/unit/inference_flow/test_chatterbox_inputs.py -x -v` |
| Full suite command | `pytest tests/unit/ -v` |
| Estimated runtime | ~5-10 seconds (fully mocked, no real models) |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| REQ-INPUT-01 | `TextToSpeechGenerationOptions` has 8 new optional fields | unit | `pytest tests/unit/inference_flow/test_chatterbox_inputs.py::test_generation_options_fields -x` | ❌ Wave 0 gap |
| REQ-INPUT-01 | `GenerationOptions` Pydantic model has 8 new optional fields | unit | `pytest tests/unit/inference_flow/test_chatterbox_inputs.py::test_protocol_generation_options_fields -x` | ❌ Wave 0 gap |
| REQ-INPUT-02 | `prepare_chatterbox_tts_inputs()` returns correct parameter tuple | unit | `pytest tests/unit/inference_flow/test_chatterbox_inputs.py::test_prepare_chatterbox_tts_inputs -x` | ❌ Wave 0 gap |
| REQ-INPUT-02 | `prepare_chatterbox_multilingual_tts_inputs()` defaults language to "en" | unit | `pytest tests/unit/inference_flow/test_chatterbox_inputs.py::test_multilingual_defaults_to_en -x` | ❌ Wave 0 gap |
| REQ-INPUT-02 | `prepare_chatterbox_turbo_tts_inputs()` returns correct turbo tuple | unit | `pytest tests/unit/inference_flow/test_chatterbox_inputs.py::test_prepare_turbo_inputs -x` | ❌ Wave 0 gap |
| REQ-INPUT-03 | `prepare_chatterbox_embedding_inputs()` returns raw audio bytes | unit | `pytest tests/unit/inference_flow/test_chatterbox_inputs.py::test_embedding_inputs_returns_bytes -x` | ❌ Wave 0 gap |
| REQ-INPUT-04 | `prepare_chatterbox_vc_inputs()` raises ValueError on both target_audio + target_embedding | unit | `pytest tests/unit/inference_flow/test_chatterbox_inputs.py::test_vc_conflict_both -x` | ❌ Wave 0 gap |
| REQ-INPUT-04 | `prepare_chatterbox_vc_inputs()` raises ValueError on neither | unit | `pytest tests/unit/inference_flow/test_chatterbox_inputs.py::test_vc_conflict_neither -x` | ❌ Wave 0 gap |
| REQ-INPUT-05 | ValueError raised for `top_k` passed to ChatterboxTTS | unit | `pytest tests/unit/inference_flow/test_chatterbox_inputs.py::test_tts_rejects_top_k -x` | ❌ Wave 0 gap |
| REQ-INPUT-05 | ValueError raised for `norm_loudness` passed to ChatterboxTTS | unit | `pytest tests/unit/inference_flow/test_chatterbox_inputs.py::test_tts_rejects_norm_loudness -x` | ❌ Wave 0 gap |
| REQ-INPUT-05 | ValueError raised for `exaggeration` passed to ChatterboxTurboTTS | unit | `pytest tests/unit/inference_flow/test_chatterbox_inputs.py::test_turbo_rejects_exaggeration -x` | ❌ Wave 0 gap |
| REQ-INPUT-05 | ValueError raised for `cfg_weight` passed to ChatterboxTurboTTS | unit | `pytest tests/unit/inference_flow/test_chatterbox_inputs.py::test_turbo_rejects_cfg_weight -x` | ❌ Wave 0 gap |
| REQ-INPUT-05 | ValueError raised for `min_p` passed to ChatterboxTurboTTS | unit | `pytest tests/unit/inference_flow/test_chatterbox_inputs.py::test_turbo_rejects_min_p -x` | ❌ Wave 0 gap |
| REQ-INPUT-05 | No error when same params are None (backward compat) | unit | `pytest tests/unit/inference_flow/test_chatterbox_inputs.py::test_validation_none_fields_no_error -x` | ❌ Wave 0 gap |
| REQ-INPUT-06 | `ChatterboxMultilingualTTSModel.SUPPORTED_LANGUAGES` has exactly 23 entries | unit | `pytest tests/unit/inference_flow/test_chatterbox_inputs.py::test_supported_languages_count -x` | ❌ Wave 0 gap |
| REQ-INPUT-06 | Model card for ChatterboxMultilingualTTS has 23 LanguageOptions entries | unit | `pytest tests/unit/inference_flow/test_chatterbox_inputs.py::test_multilingual_model_card_languages -x` | ❌ Wave 0 gap |

### Nyquist Sampling Rate
- **Minimum sample interval:** After every committed task → run: `pytest tests/unit/inference_flow/test_chatterbox_inputs.py -x -v`
- **Full suite trigger:** Before merging final task of any plan wave
- **Phase-complete gate:** Full suite green before `/gsd-verify-work` runs
- **Estimated feedback latency per task:** ~5 seconds

### Wave 0 Gaps (must be created before implementation)
- [ ] `tests/unit/inference_flow/test_chatterbox_inputs.py` — covers all REQ-INPUT-01 through REQ-INPUT-06 test cases listed above
- [ ] No new conftest.py needed — `tests/unit/conftest.py` already provides `mock_model_loader` and `mock_hf_downloader` fixtures

*(Framework is installed and configured. Only the test file is missing.)*

---

## Sources

### Primary (HIGH confidence)
- Direct file reads of `harmonyspeech/task_handler/inputs.py` — all prepare function patterns confirmed
- Direct file reads of `harmonyspeech/common/inputs.py` — `TextToSpeechGenerationOptions` dataclass structure confirmed
- Direct file reads of `harmonyspeech/endpoints/openai/protocol.py` — `GenerationOptions` Pydantic model confirmed
- Direct file reads of `harmonyspeech/endpoints/openai/serving_engine.py` — language validation pipeline (lines 89-122) and `model_card_from_config` confirmed
- Direct file reads of `.current_work/chatterbox-tts/src/chatterbox/tts.py` — `ChatterboxTTS.generate()` signature confirmed
- Direct file reads of `.current_work/chatterbox-tts/src/chatterbox/tts_turbo.py` — `ChatterboxTurboTTS.generate()` signature + turbo-specific params confirmed
- Direct file reads of `.current_work/chatterbox-tts/src/chatterbox/mtl_tts.py` — `SUPPORTED_LANGUAGES` (23 entries) + `ChatterboxMultilingualTTS.generate()` signature confirmed
- Direct file reads of `.current_work/chatterbox-tts/src/chatterbox/vc.py` — `ChatterboxVC.generate()` signature confirmed
- Direct file reads of `harmonyspeech/modeling/models/chatterbox/chatterbox.py` — wrapper class structure from Phase 2 confirmed
- `.planning/phases/03-input-preparation/03-CONTEXT.md` — all locked decisions

### Secondary (MEDIUM confidence)
- `torch.load` accepting file-like objects (BytesIO) — standard PyTorch behavior, confirmed by project's own usage of BytesIO + librosa in existing prepare functions and the `Conditionals.load` source reading from `fpath` which torch accepts as file-like

### Tertiary (LOW confidence)
- None

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries already in use in the codebase
- Architecture: HIGH — patterns confirmed directly from source code; no inference required
- Pitfalls: HIGH — identified from direct code inspection + locked decisions in CONTEXT.md
- Open questions: MEDIUM — questions identified but require checking Phase 2 loader output or Phase 4 design decisions

**Research date:** 2026-03-13
**Valid until:** 2026-04-13 (stable codebase; only affected by Phase 2/4 execution decisions)

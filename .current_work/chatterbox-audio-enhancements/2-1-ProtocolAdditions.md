# Phase 2-1: Protocol Additions

## Objective

Extend [`harmonyspeech/endpoints/openai/protocol.py`](../../harmonyspeech/endpoints/openai/protocol.py) to add new optional fields to `TextToSpeechRequest` and `GenerationOptions` that expose the text chunking and audio post-processing controls to API clients.

All new fields are **optional with backward-compatible defaults** so existing API clients are not broken.

## Background

The current `TextToSpeechRequest` has no chunking controls. The `GenerationOptions` has a `speed` field but it is not used by any model runner. We add dedicated fields for chunking and post-processing.

## Files to Modify

- `harmonyspeech/endpoints/openai/protocol.py`

## Detailed Implementation Steps

### Step 1: Add chunking fields to `TextToSpeechRequest`

Open [`harmonyspeech/endpoints/openai/protocol.py`](../../harmonyspeech/endpoints/openai/protocol.py).

Locate the `TextToSpeechRequest` class (currently at approximately line 138).

Add the following two new **optional** fields after the existing fields, before `generation_options`:

```python
split_text: bool | None = Field(
    default=False,
    description=(
        "If True, automatically split long input text into sentence-level chunks "
        "and stitch them together after synthesis. Recommended for inputs longer "
        "than ~120 characters. Default: False (single inference call)."
    ),
)
chunk_size: int | None = Field(
    default=120,
    ge=50,
    le=500,
    description=(
        "Target character length per text chunk when split_text=True. "
        "Chunks are split at sentence boundaries, so actual chunk lengths "
        "may be shorter. Range: 50–500. Default: 120."
    ),
)
```

**Placement:** Insert these two fields between `input_embedding` and `generation_options` in `TextToSpeechRequest`.

### Step 2: Add audio post-processing fields to `GenerationOptions`

Locate the `GenerationOptions` class (currently around line 51).

Add the following new fields at the **end** of the class (after all existing fields).

**Important:** `norm_loudness` (Chatterbox Turbo-only loudness normalization) is **replaced** by the more general `normalize_audio` field. The rename makes the purpose clear for all models and avoids confusion with Turbo-specific behavior. Update all references to `norm_loudness` in `task_handler/inputs.py` and the protocol accordingly.

```python
# Audio post-processing fields (model-agnostic, applied after synthesis)
normalize_audio: bool | None = Field(
    default=True,
    description=(
        "Apply peak normalization to prevent clipping after synthesis (and after chunk stitching). "
        "Normalizes output to 0.95 peak if audio exceeds 0.99. "
        "Replaces the previous Turbo-only norm_loudness field. Default: True."
    ),
)
crossfade_ms: int | None = Field(
    default=20,
    ge=0,
    le=200,
    description=(
        "Crossfade duration in milliseconds used when stitching text chunks. "
        "Only relevant when split_text=True. Range: 0–200 ms. Default: 20 ms."
    ),
)
sentence_pause_ms: int | None = Field(
    default=200,
    ge=0,
    le=2000,
    description=(
        "Silence gap inserted between stitched text chunks in milliseconds. "
        "Only relevant when split_text=True. Range: 0–2000 ms. Default: 200 ms."
    ),
)
```

**Additionally:** Remove `norm_loudness` from `GenerationOptions` and update the `prepare_chatterbox_turbo_tts_inputs()` function in [`harmonyspeech/task_handler/inputs.py`](../../harmonyspeech/task_handler/inputs.py) to use `normalize_audio` instead of `norm_loudness`. The Turbo model's `norm_loudness` parameter maps to the `normalize_audio` field — this is a rename, not a behavioral change.

**Note on `speed`:** The existing `speed: float | None` field in `GenerationOptions` already exists (line ~54 in current protocol.py). **Reuse it** as the unified speed control. Update its docstring to clarify that when a model does not support native speed control, it will be applied as a post-synthesis pitch-preserving time-stretch via librosa. No new `speed_factor` field is needed.

Update the existing `speed` field description to:
```python
speed: float | None = Field(
    default=None,
    description=(
        "Speed adjustment factor. 1.0 = normal speed. "
        "For models that support native speed control, this is passed directly. "
        "For all other models, post-synthesis pitch-preserving time-stretching is "
        "applied via librosa. Range: 0.1–4.0."
    ),
)
```

### Step 3: Update `TextToSpeechRequestInput` in `harmonyspeech/common/inputs.py`

Open [`harmonyspeech/common/inputs.py`](../../harmonyspeech/common/inputs.py).

The `TextToSpeechRequestInput` class needs to carry the new chunking fields so the serving layer can access them after mapping from the OpenAI request.

Add the following parameters to `TextToSpeechRequestInput.__init__()`:

```python
split_text: bool | None = None,
chunk_size: int | None = None,
```

Add them to the constructor body:
```python
self.split_text = split_text
self.chunk_size = chunk_size
```

Update the `from_openai()` classmethod to pass these from the request:
```python
split_text=getattr(request, "split_text", None),
chunk_size=getattr(request, "chunk_size", None),
```

---

## Expected Final State of `GenerationOptions` (key fields)

```python
class GenerationOptions(BaseModel):
    seed: int | None = None
    style: int | None = None
    speed: float | None = None           # unified speed — native where supported, post-synthesis otherwise
    pitch: float | None = None
    energy: float | None = None
    # Chatterbox-specific
    exaggeration: float | None = ...
    cfg_weight: float | None = ...
    temperature: float | None = ...
    repetition_penalty: float | None = ...
    top_p: float | None = ...
    min_p: float | None = ...
    top_k: int | None = ...
    # norm_loudness REMOVED — replaced by normalize_audio below
    # Post-processing (model-agnostic) — NEW
    normalize_audio: bool | None = Field(default=True, ...)
    crossfade_ms: int | None = Field(default=20, ge=0, le=200, ...)
    sentence_pause_ms: int | None = Field(default=200, ge=0, le=2000, ...)
```

---

## Progress Checklist

- [ ] Read current `protocol.py` to confirm exact line numbers and existing `speed` field usage
- [ ] Add `split_text` and `chunk_size` fields to `TextToSpeechRequest`
- [ ] Remove `norm_loudness` from `GenerationOptions`
- [ ] Add `normalize_audio`, `crossfade_ms`, `sentence_pause_ms` fields to `GenerationOptions`
- [ ] Update docstring on existing `speed` field in `GenerationOptions` to clarify dual-use (native + post-synthesis fallback)
- [ ] Update `TextToSpeechGenerationOptions` dataclass in `common/inputs.py`: remove `norm_loudness`, add `normalize_audio`
- [ ] Update `prepare_chatterbox_turbo_tts_inputs()` in `task_handler/inputs.py` to use `normalize_audio` instead of `norm_loudness`
- [ ] Update validation in `prepare_chatterbox_tts_inputs()` and `prepare_chatterbox_multilingual_tts_inputs()` — these previously rejected `norm_loudness`; update to reject nothing (normalize_audio is now handled in serving layer, not passed to Turbo model)
- [ ] Update `TextToSpeechRequestInput.__init__()` in `common/inputs.py` to carry `split_text` and `chunk_size`
- [ ] Update `TextToSpeechRequestInput.from_openai()` to map new fields
- [ ] Verify Pydantic validation: run `python -c "from harmonyspeech.endpoints.openai.protocol import TextToSpeechRequest, GenerationOptions; print('OK')"`

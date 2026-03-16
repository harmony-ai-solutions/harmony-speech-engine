# Phase 7-1: Protocol Additions — Phase B

## Objective

Extend the protocol and internal data structures to support the three new Phase B post-processing options:

1. `remove_dc_offset` — remove DC offset per chunk before stitching
2. `trim_silence` — trim leading/trailing silence on final audio
3. `fix_internal_silence` — shorten unnaturally long internal pauses

All three are **boolean opt-in flags** in `GenerationOptions`. No additional parameter fields are added at this stage — the functions use sensible defaults matching the Chatterbox TTS Server behaviour.

## Prerequisites

- Phase 2-1 complete: `GenerationOptions` already has `normalize_audio`, `crossfade_ms`, `sentence_pause_ms`
- Phase 6-1 complete: `audio_utils.py` has the three new functions

## Files to Modify

- `harmonyspeech/endpoints/openai/protocol.py` — add 3 new fields to `GenerationOptions`
- `harmonyspeech/common/inputs.py` — add 3 new fields to `TextToSpeechGenerationOptions`

---

## Detailed Implementation Steps

### Step 1: Add fields to `GenerationOptions` in `protocol.py`

Read [`harmonyspeech/endpoints/openai/protocol.py`](../../harmonyspeech/endpoints/openai/protocol.py) and locate the `GenerationOptions` class.

After the existing `normalize_audio` field (added in Phase 2-1), add:

```python
remove_dc_offset: bool | None = Field(
    default=None,
    description=(
        "When true, applies a high-pass filter (2nd-order Butterworth at 15 Hz) "
        "to remove DC offset from each audio chunk before stitching. "
        "Only relevant when split_text=True. Requires scipy."
    ),
)
trim_silence: bool | None = Field(
    default=None,
    description=(
        "When true, trims leading and trailing silence from the final synthesized audio. "
        "Uses librosa.effects.trim with a -40 dBFS threshold. Requires librosa."
    ),
)
fix_internal_silence: bool | None = Field(
    default=None,
    description=(
        "When true, shortens unnaturally long internal silence gaps "
        "(pauses longer than ~700ms are reduced to ~300ms). "
        "Uses librosa.effects.split. Requires librosa."
    ),
)
```

### Step 2: Add fields to `TextToSpeechGenerationOptions` in `common/inputs.py`

Read [`harmonyspeech/common/inputs.py`](../../harmonyspeech/common/inputs.py) and locate the `TextToSpeechGenerationOptions` dataclass.

After the `normalize_audio: bool | None = None` field (added in Phase 2-1 / 4-1), add:

```python
remove_dc_offset: bool | None = None
trim_silence: bool | None = None
fix_internal_silence: bool | None = None
```

### Step 3: Update `TextToSpeechRequestInput.from_openai()` in `common/inputs.py`

Check the `from_openai()` classmethod of `TextToSpeechRequestInput` to confirm that `GenerationOptions` fields are passed through to `TextToSpeechGenerationOptions`. If the method explicitly lists fields, add the three new ones. If it passes the entire options object, no change is needed.

---

## Progress Checklist

- [ ] Read `protocol.py` to confirm `GenerationOptions` structure and insertion point
- [ ] Add `remove_dc_offset`, `trim_silence`, `fix_internal_silence` fields to `GenerationOptions`
- [ ] Read `common/inputs.py` to confirm `TextToSpeechGenerationOptions` structure
- [ ] Add three fields to `TextToSpeechGenerationOptions` dataclass
- [ ] Check `TextToSpeechRequestInput.from_openai()` and update if fields are explicitly listed

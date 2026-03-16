# Phase 8-1: Serving Layer — Phase B Post-Processing

## Objective

Wire up the three new Phase B post-processing options in [`harmonyspeech/endpoints/openai/serving_text_to_speech.py`](../../harmonyspeech/endpoints/openai/serving_text_to_speech.py):

1. **`remove_dc_offset`** — applied **per chunk** before stitching (when `split_text=True`)
2. **`trim_silence`** — applied to the **final audio** (after stitching or after single synthesis)
3. **`fix_internal_silence`** — applied to the **final audio** (after stitching or after single synthesis)

The processing order for the complete pipeline (single and chunked paths) is:

```
1. Model inference
2. Decode audio → numpy float32
   --- per-chunk operations (chunked path only) ---
3. [optional] remove_dc_offset per chunk
   --- stitching (chunked path only) ---
4. stitch_audio_chunks (crossfade)
   --- final audio operations (both paths) ---
5. [optional] apply_speed_factor (Phase A)
6. [optional] trim_silence
7. [optional] fix_internal_silence
8. [optional] normalize_peak (Phase A)
9. Encode → base64 WAV → response
```

**Note:** `trim_silence` runs before `normalize_peak` so that any brief transient at the boundary is captured by normalization after trimming.

## Prerequisites

- Phase 3-1 complete: `_create_text_to_speech_chunked()` and `_create_text_to_speech_single()` exist
- Phase 4-1 complete: `_skip_postprocess` flag is on `_create_text_to_speech_single()`
- Phase 6-1 complete: `audio_utils.py` has `remove_dc_offset`, `trim_silence`, `fix_internal_silence`
- Phase 7-1 complete: `GenerationOptions` has the three new boolean fields

## Files to Modify

- `harmonyspeech/endpoints/openai/serving_text_to_speech.py`

---

## Detailed Implementation Steps

### Step 1: Apply `remove_dc_offset` per chunk in `_create_text_to_speech_chunked()`

Read [`serving_text_to_speech.py`](../../harmonyspeech/endpoints/openai/serving_text_to_speech.py) to confirm the chunked path structure from Phase 3-1.

In `_create_text_to_speech_chunked()`, after each chunk's audio is decoded to a numpy array (via `_decode_audio_response()`) and **before** it is appended to the `chunk_arrays` list, apply DC offset removal if the flag is set:

```python
# --- Per-chunk: DC offset removal ---
gen_opts = request.generation_options
if gen_opts and gen_opts.remove_dc_offset:
    from harmonyspeech.common.audio_utils import remove_dc_offset
    chunk_array = remove_dc_offset(chunk_array, sample_rate)
    logger.debug("Per-chunk DC offset removal applied")
```

This step runs on each decoded chunk before stitching, which directly prevents thumps at crossfade boundaries.

### Step 2: Apply `trim_silence` on final audio in post-processing block

In both `_create_text_to_speech_single()` (inside the `if not _skip_postprocess:` block) and `_create_text_to_speech_chunked()` (in the final audio post-processing block), add `trim_silence` **after** `apply_speed_factor` and **before** `normalize_peak`:

Add the following imports at the top of the post-processing block (or inline):
```python
from harmonyspeech.common.audio_utils import trim_silence, fix_internal_silence
```

**In the post-processing block (applies to both paths):**

```python
# --- Post-synthesis: trim leading/trailing silence ---
if gen_opts and gen_opts.trim_silence:
    audio_array = trim_silence(audio_array, sample_rate)
    logger.debug("Leading/trailing silence trimmed")

# --- Post-synthesis: fix internal silence ---
if gen_opts and gen_opts.fix_internal_silence:
    audio_array = fix_internal_silence(audio_array, sample_rate)
    logger.debug("Internal silence shortened")
```

### Step 3: Confirm full post-processing order in both paths

After all Phase A and Phase B changes, the post-processing block in the **single path** (`_create_text_to_speech_single()`) must look like this (inside `if not _skip_postprocess:`):

```python
if not _skip_postprocess:
    gen_opts = request.generation_options
    # 1. Speed factor (Phase A) — only for non-native-speed models
    if gen_opts and gen_opts.speed is not None and gen_opts.speed != 1.0:
        resolved_type = self._get_model_type(request.model)
        if resolved_type not in NATIVE_SPEED_MODEL_TYPES:
            audio_array = apply_speed_factor(audio_array, sample_rate, gen_opts.speed)
    # 2. Trim leading/trailing silence (Phase B)
    if gen_opts and gen_opts.trim_silence:
        audio_array = trim_silence(audio_array, sample_rate)
    # 3. Fix internal silence (Phase B)
    if gen_opts and gen_opts.fix_internal_silence:
        audio_array = fix_internal_silence(audio_array, sample_rate)
    # 4. Peak normalization (Phase A) — last step
    do_normalize = gen_opts.normalize_audio if gen_opts and gen_opts.normalize_audio is not None else True
    if do_normalize:
        audio_array = normalize_peak(audio_array)
```

And in the **chunked path** (`_create_text_to_speech_chunked()`), the final post-processing block after stitching:

```python
# 1. Speed factor — only for non-native-speed models
if gen_opts and gen_opts.speed is not None and gen_opts.speed != 1.0:
    resolved_type = self._get_model_type(request.model)
    if resolved_type not in NATIVE_SPEED_MODEL_TYPES:
        stitched = apply_speed_factor(stitched, detected_sample_rate, gen_opts.speed)
# 2. Trim leading/trailing silence
if gen_opts and gen_opts.trim_silence:
    stitched = trim_silence(stitched, detected_sample_rate)
# 3. Fix internal silence
if gen_opts and gen_opts.fix_internal_silence:
    stitched = fix_internal_silence(stitched, detected_sample_rate)
# 4. Peak normalization — last step
do_normalize = gen_opts.normalize_audio if gen_opts and gen_opts.normalize_audio is not None else True
if do_normalize:
    stitched = normalize_peak(stitched)
```

---

## Progress Checklist

- [ ] Read `serving_text_to_speech.py` to confirm chunk loop structure (Phase 3-1 output)
- [ ] Add `remove_dc_offset` per-chunk in `_create_text_to_speech_chunked()` before appending to `chunk_arrays`
- [ ] Confirm post-processing block order in `_create_text_to_speech_single()` matches: speed → trim_silence → fix_internal_silence → normalize_peak
- [ ] Confirm post-processing block order in `_create_text_to_speech_chunked()` matches the same order on stitched audio
- [ ] Verify that `_skip_postprocess=True` on per-chunk calls correctly skips all post-processing
- [ ] Manual smoke test: `split_text=True, remove_dc_offset=True, trim_silence=True` request produces clean audio

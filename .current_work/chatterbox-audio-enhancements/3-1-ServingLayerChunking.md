# Phase 3-1: Serving Layer — Text Chunking and Stitching

## Objective

Modify [`harmonyspeech/endpoints/openai/serving_text_to_speech.py`](../../harmonyspeech/endpoints/openai/serving_text_to_speech.py) to implement the text chunking pipeline when `split_text=True` on the incoming `TextToSpeechRequest`. Each chunk is submitted as a separate engine request, the resulting audio arrays are collected, then stitched together using the smart crossfade logic from `audio_utils.py`.

This is the core feature of Phase A — it applies to **all TTS models** because it operates entirely in the serving layer on top of whatever the engine returns.

## Prerequisites

- Phase 1-1 complete: `harmonyspeech/common/audio_utils.py` exists
- Phase 2-1 complete: `TextToSpeechRequest` has `split_text` and `chunk_size` fields; `GenerationOptions` has `crossfade_ms` and `sentence_pause_ms`

## Files to Modify

- `harmonyspeech/endpoints/openai/serving_text_to_speech.py`

## Background — Current Serving Logic

Currently [`serving_text_to_speech.py`](../../harmonyspeech/endpoints/openai/serving_text_to_speech.py) has a `create_text_to_speech()` async method that:
1. Validates the request
2. Routes to the engine (possibly with a multi-step pipeline for voice cloning)
3. Returns the audio result as a `TextToSpeechResponse`

The chunking pipeline wraps around this existing flow. **The key design decision** is to split at the serving layer before submitting to the engine, not inside the model runner — this keeps the model runners simple and makes chunking work for every TTS model type.

## Detailed Implementation Steps

### Step 1: Read the current `serving_text_to_speech.py`

Read the full content of [`harmonyspeech/endpoints/openai/serving_text_to_speech.py`](../../harmonyspeech/endpoints/openai/serving_text_to_speech.py) before making any edits to understand the exact method signature and flow.

### Step 2: Add imports to `serving_text_to_speech.py`

At the top of the file, add the following imports (after existing imports):

```python
import base64
import io

import numpy as np
import soundfile as sf

from harmonyspeech.common.audio_utils import (
    chunk_text_by_sentences,
    stitch_audio_chunks,
)
```

Note: `soundfile` is already a dependency (used in audio encoding elsewhere). Confirm `numpy` is also already imported.

### Step 3: Add a private helper `_decode_audio_response()`

Add a private static method to the serving class that decodes a base64 WAV response back to a numpy float32 array + sample rate. This is needed to collect individual chunk outputs for stitching.

```python
@staticmethod
def _decode_audio_response(b64_data: str) -> tuple[np.ndarray, int]:
    """Decode a base64-encoded WAV audio response to numpy array.

    Args:
        b64_data: Base64-encoded WAV bytes.

    Returns:
        Tuple of (audio_array float32, sample_rate int).

    Raises:
        ValueError: If the audio cannot be decoded.
    """
    raw_bytes = base64.b64decode(b64_data)
    buf = io.BytesIO(raw_bytes)
    audio, sr = sf.read(buf, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # Convert stereo to mono
    return audio, sr
```

### Step 4: Add a private helper `_encode_audio_to_b64()`

Add a private static method that encodes a numpy float32 array back to base64 WAV at the given sample rate. This produces the final stitched output.

```python
@staticmethod
def _encode_audio_to_b64(audio: np.ndarray, sample_rate: int, output_format: str = "wav") -> str:
    """Encode a numpy float32 audio array to base64 WAV.

    Args:
        audio: Float32 mono audio array.
        sample_rate: Sample rate in Hz.
        output_format: Currently only "wav" is supported for stitching output.

    Returns:
        Base64-encoded WAV bytes as a string.
    """
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")
```

### Step 5: Modify `create_text_to_speech()` to support chunking

The existing method performs a single engine call. We need to wrap this in a chunking loop when `split_text=True`.

**Structure of the modification:**

```
create_text_to_speech(request, raw_request):
    # Determine if chunking should be applied
    split_text = request.split_text or False
    chunk_size = request.chunk_size or 120

    if split_text and len(request.input) > chunk_size:
        # --- Chunking path ---
        chunks = chunk_text_by_sentences(request.input, chunk_size)
        if len(chunks) <= 1:
            # fallback to single call if chunking yields only one chunk
            pass  # falls through to non-chunking path
        else:
            return await self._create_text_to_speech_chunked(
                request, chunks, raw_request
            )

    # --- Original single-call path (unchanged) ---
    return await self._create_text_to_speech_single(request, raw_request)
```

This is achieved by **refactoring** the existing body of `create_text_to_speech()` into a new private method `_create_text_to_speech_single()`, then adding a new `_create_text_to_speech_chunked()` method. The public `create_text_to_speech()` becomes a thin dispatcher.

### Step 6: Implement `_create_text_to_speech_chunked()`

```python
async def _create_text_to_speech_chunked(
    self,
    request: TextToSpeechRequest,
    chunks: list[str],
    raw_request,
) -> TextToSpeechResponse | ErrorResponse:
    """Run TTS for each text chunk and stitch results together.

    Each chunk is submitted as an independent engine request with the same
    parameters as the original request. The resulting audio arrays are
    decoded from base64 WAV, stitched with optional crossfading, and
    re-encoded to base64 WAV.

    Args:
        request: Original TTS request (used for model, voice, options).
        chunks: List of text chunk strings to synthesize.
        raw_request: FastAPI raw request (for disconnect detection).

    Returns:
        TextToSpeechResponse with stitched audio, or ErrorResponse on failure.
    """
    from harmonyspeech.common.audio_utils import normalize_peak, stitch_audio_chunks

    audio_segments: list[np.ndarray] = []
    detected_sample_rate: int | None = None

    # Extract stitching options from generation_options (with defaults)
    gen_opts = request.generation_options
    crossfade_ms = (gen_opts.crossfade_ms if gen_opts and gen_opts.crossfade_ms is not None else 20)
    sentence_pause_ms = (gen_opts.sentence_pause_ms if gen_opts and gen_opts.sentence_pause_ms is not None else 200)
    do_normalize = (gen_opts.normalize_audio if gen_opts and gen_opts.normalize_audio is not None else True)

    for i, chunk_text in enumerate(chunks):
        # Build a per-chunk request by shallow-copying the original and replacing input
        chunk_request = request.model_copy(update={"input": chunk_text})

        # Call the single-call path for this chunk
        chunk_response = await self._create_text_to_speech_single(chunk_request, raw_request)

        if isinstance(chunk_response, ErrorResponse):
            logger.error(f"Chunk {i+1}/{len(chunks)} failed: {chunk_response.message}")
            return chunk_response  # propagate the error

        # Decode the WAV bytes back to numpy
        try:
            chunk_audio, chunk_sr = self._decode_audio_response(chunk_response.data)
            audio_segments.append(chunk_audio)
            if detected_sample_rate is None:
                detected_sample_rate = chunk_sr
            elif detected_sample_rate != chunk_sr:
                logger.warning(
                    f"Inconsistent sample rate in chunk {i+1}: "
                    f"expected {detected_sample_rate}Hz, got {chunk_sr}Hz"
                )
        except Exception as e:
            logger.error(f"Failed to decode audio for chunk {i+1}: {e}")
            return ErrorResponse(
                message=f"Audio decode failed for chunk {i+1}: {e}",
                type="internal_error",
                code=500,
            )

    if not audio_segments:
        return ErrorResponse(
            message="No audio segments generated from chunks.",
            type="internal_error",
            code=500,
        )

    # Stitch all segments
    stitched = stitch_audio_chunks(
        audio_segments,
        sample_rate=detected_sample_rate,
        crossfade_ms=crossfade_ms,
        sentence_pause_ms=sentence_pause_ms,
        use_crossfade=True,
    )

    # Peak normalization (applied after stitching)
    if do_normalize:
        from harmonyspeech.common.audio_utils import normalize_peak
        stitched = normalize_peak(stitched, threshold=0.99, target=0.95)

    # Re-encode stitched audio to base64 WAV
    b64_result = self._encode_audio_to_b64(stitched, detected_sample_rate)

    # Return a TextToSpeechResponse with the same structure as single-call responses
    return TextToSpeechResponse(
        model=request.model,
        data=b64_result,
    )
```

### Step 7: Logging

Add `logger.info()` calls for observability:
- When chunking is triggered: `f"Splitting text into {len(chunks)} chunks (chunk_size={chunk_size})"`
- When stitching completes: `f"Stitched {len(audio_segments)} audio chunks into final output"`

Use `from loguru import logger` (already the HSE convention per ARCHITECTURE.md).

---

## Notes

- **Multi-step pipelines (voice cloning):** The chunking wraps the full `_create_text_to_speech_single()` path including the embed+synthesize multi-step flow. Each chunk will trigger its own embed step if `input_audio` is provided. This is correct but slightly inefficient — a future optimization could pre-compute the embedding once and reuse it across chunks. That optimization is out of scope for Phase A.
- **Output format:** The chunking path always produces WAV output internally for stitching, then re-encodes. If the client requests `output_options.format = "opus"`, the final re-encoding step must respect that. The `_encode_audio_to_b64()` helper currently only supports WAV. For Phase A, this is acceptable — WAV output is the primary format. Format conversion can be addressed in a follow-up.
- **Backward compatibility:** When `split_text=False` (default), the code is completely unchanged — `_create_text_to_speech_single()` is just the renamed original method body.

---

## Progress Checklist

- [ ] Read full content of `serving_text_to_speech.py` before editing
- [ ] Add `import` statements for `numpy`, `soundfile`, `io`, `base64`, `audio_utils`
- [ ] Refactor existing `create_text_to_speech()` body into `_create_text_to_speech_single()`
- [ ] Add `_decode_audio_response()` static helper
- [ ] Add `_encode_audio_to_b64()` static helper
- [ ] Add `_create_text_to_speech_chunked()` method
- [ ] Update `create_text_to_speech()` to be the chunking dispatcher
- [ ] Add `logger.info()` calls at chunk-split and stitch-complete points
- [ ] Manually verify with a single-chunk input (no stitching): response identical to before
- [ ] Manually verify with a multi-chunk input: response contains stitched audio

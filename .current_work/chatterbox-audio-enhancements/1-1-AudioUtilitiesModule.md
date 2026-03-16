# Phase 1-1: Audio Utilities Module

## Objective

Create a new shared audio post-processing utilities module at [`harmonyspeech/common/audio_utils.py`](../../harmonyspeech/common/audio_utils.py). This module will contain all reusable audio processing functions that are not model-specific, making them available to the serving layer and any future model runners that need them.

## Background

The Chatterbox TTS Server implements these functions in its `utils.py`. For HSE we follow the existing convention of placing shared, cross-cutting utilities in `harmonyspeech/common/`. See [`harmonyspeech/common/utils.py`](../../harmonyspeech/common/utils.py) for the existing pattern.

## Files to Create

- `harmonyspeech/common/audio_utils.py` — **NEW**

## Files to Modify

- `requirements-common.txt` — ensure `librosa` and `scipy` are listed

## Detailed Implementation Steps

### Step 1: Check `requirements-common.txt` for librosa

Read [`requirements-common.txt`](../../requirements-common.txt) and confirm `librosa` is already present. If missing, add it. Also confirm `scipy` is present or add it (needed for DC offset removal / Butterworth filter).

### Step 2: Create `harmonyspeech/common/audio_utils.py`

Create the file with the following structure and content:

```python
"""Audio post-processing utilities for Harmony Speech Engine.

Functions in this module are model-agnostic and operate on raw numpy float32
audio arrays. They are used by the serving layer after model inference.

All functions accept and return numpy float32 arrays.
Sample rate (int, Hz) must be passed explicitly when relevant.
"""

from __future__ import annotations

import re
from typing import List, Tuple

import numpy as np
from loguru import logger


# ---------------------------------------------------------------------------
# Text Utilities (sentence splitting / chunking)
# ---------------------------------------------------------------------------


def split_into_sentences(text: str) -> List[str]:
    """Split a text string into individual sentences.

    Uses a regex-based approach that handles common sentence endings:
    period, question mark, exclamation mark, followed by whitespace or
    end-of-string. Preserves the sentence-ending punctuation.

    Args:
        text: Input text to split.

    Returns:
        List of sentence strings, stripped of leading/trailing whitespace.
        Empty strings are filtered out.
    """
    # Split on sentence-ending punctuation followed by whitespace or end of string.
    # Uses lookahead to keep the punctuation with the preceding sentence.
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in parts if s.strip()]


def chunk_text_by_sentences(text: str, chunk_size: int = 120) -> List[str]:
    """Group sentences into chunks not exceeding chunk_size characters.

    Sentences are grouped greedily: a sentence is added to the current chunk
    if it fits (with a space separator). When it would exceed chunk_size, the
    current chunk is committed and a new chunk starts with that sentence.

    Single sentences longer than chunk_size are not split further — they are
    returned as a single chunk of their own.

    Args:
        text: Input text.
        chunk_size: Approximate maximum character length per chunk.

    Returns:
        List of text chunk strings. Each chunk ends with sentence-level
        punctuation (from the last sentence in that chunk).
    """
    sentences = split_into_sentences(text)
    if not sentences:
        return [text] if text.strip() else []

    chunks: List[str] = []
    current_chunk = ""

    for sentence in sentences:
        if not current_chunk:
            current_chunk = sentence
        elif len(current_chunk) + 1 + len(sentence) <= chunk_size:
            current_chunk = current_chunk + " " + sentence
        else:
            chunks.append(current_chunk)
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


# ---------------------------------------------------------------------------
# Audio Stitching Utilities
# ---------------------------------------------------------------------------


def generate_equal_power_curves(n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate equal-power crossfade curves using cos²/sin² functions.

    Equal-power curves maintain perceptually constant loudness during
    transitions, avoiding the "dip" that occurs with linear crossfades.

    Args:
        n_samples: Number of samples in the crossfade region.

    Returns:
        Tuple of (fade_out, fade_in) float32 numpy arrays, each of length
        n_samples. fade_out goes 1 → 0, fade_in goes 0 → 1.
    """
    t = np.linspace(0, np.pi / 2, n_samples, dtype=np.float32)
    fade_out = np.cos(t) ** 2  # 1 → 0
    fade_in = np.sin(t) ** 2   # 0 → 1
    return fade_out, fade_in


def crossfade_with_overlap(
    chunk_a: np.ndarray,
    chunk_b: np.ndarray,
    fade_samples: int,
) -> np.ndarray:
    """Perform a true equal-power crossfade by overlapping and summing regions.

    Creates a seamless transition by:
    1. Taking the tail of chunk_a and head of chunk_b
    2. Applying equal-power fade curves to each
    3. Summing the overlapped regions

    Result length = len(chunk_a) + len(chunk_b) - fade_samples

    Args:
        chunk_a: First audio chunk (float32 numpy array).
        chunk_b: Second audio chunk (float32 numpy array).
        fade_samples: Number of samples to overlap for the crossfade.

    Returns:
        Crossfaded audio as a float32 numpy array.
    """
    # Clamp fade_samples to the minimum of both chunk lengths
    fade_samples = min(fade_samples, len(chunk_a), len(chunk_b))
    if fade_samples <= 0:
        return np.concatenate([chunk_a, chunk_b])

    fade_out, fade_in = generate_equal_power_curves(fade_samples)

    a_tail = chunk_a[-fade_samples:]
    b_head = chunk_b[:fade_samples]
    crossfaded_region = (a_tail * fade_out) + (b_head * fade_in)

    return np.concatenate(
        [chunk_a[:-fade_samples], crossfaded_region, chunk_b[fade_samples:]]
    )


def apply_edge_fades(
    chunk: np.ndarray,
    fade_samples: int,
    fade_in: bool = True,
    fade_out: bool = True,
) -> np.ndarray:
    """Apply minimal linear edge fades for click protection.

    Used in fallback mode (no crossfading) to prevent audio clicks at
    chunk boundaries. Linear fades are acceptable for very short durations
    (2–3 ms) where the psychoacoustic difference is imperceptible.

    Args:
        chunk: Audio chunk (numpy array).
        fade_samples: Number of samples to fade at each edge.
        fade_in: Apply linear fade-in at the start of the chunk.
        fade_out: Apply linear fade-out at the end of the chunk.

    Returns:
        Audio with edge fades applied as a float32 numpy array.
    """
    if len(chunk) < fade_samples * 2:
        return chunk.astype(np.float32, copy=False)

    result = chunk.astype(np.float32, copy=True)
    if fade_in:
        result[:fade_samples] *= np.linspace(0, 1, fade_samples, dtype=np.float32)
    if fade_out:
        result[-fade_samples:] *= np.linspace(1, 0, fade_samples, dtype=np.float32)
    return result


def stitch_audio_chunks(
    chunks: List[np.ndarray],
    sample_rate: int,
    crossfade_ms: int = 20,
    sentence_pause_ms: int = 200,
    use_crossfade: bool = True,
) -> np.ndarray:
    """Stitch multiple audio chunks into a single array with smooth transitions.

    When use_crossfade=True (default), applies equal-power crossfading with
    silence padding between chunks, matching human perception of sentence
    boundaries.

    When use_crossfade=False, applies minimal linear edge fades (3 ms) for
    click protection without silence gaps.

    Args:
        chunks: List of float32 numpy arrays (one per text chunk).
        sample_rate: Sample rate of all chunks (must be consistent).
        crossfade_ms: Crossfade duration in milliseconds (default 20 ms).
        sentence_pause_ms: Target silence gap between sentences in ms
            (default 200 ms). Only used when use_crossfade=True.
        use_crossfade: Enable smart crossfading with silence gaps (default True).

    Returns:
        Single float32 numpy array containing all stitched audio.
    """
    if not chunks:
        return np.array([], dtype=np.float32)
    if len(chunks) == 1:
        return chunks[0].astype(np.float32, copy=False)

    if use_crossfade:
        fade_samples = int(crossfade_ms / 1000.0 * sample_rate)
        # The silence buffer is padded to compensate for crossfade overlap removal
        desired_silence = int(sentence_pause_ms / 1000.0 * sample_rate)
        silence_buffer_samples = desired_silence + (fade_samples * 2)

        result = chunks[0].astype(np.float32, copy=True)
        for i in range(1, len(chunks)):
            silence = np.zeros(silence_buffer_samples, dtype=np.float32)
            # Crossfade: current result → silence
            result = crossfade_with_overlap(result, silence, fade_samples)
            # Crossfade: silence → next chunk
            result = crossfade_with_overlap(result, chunks[i].astype(np.float32), fade_samples)

        logger.debug(
            f"Smart stitching applied: {len(chunks)} chunks, "
            f"{crossfade_ms}ms crossfades, {sentence_pause_ms}ms pauses"
        )
        return result
    else:
        # Fallback: minimal linear safety fades, no silence gap
        safety_fade_samples = int(3 / 1000.0 * sample_rate)  # 3 ms
        num_chunks = len(chunks)
        processed: List[np.ndarray] = []
        for i, chunk in enumerate(chunks):
            processed.append(
                apply_edge_fades(
                    chunk,
                    safety_fade_samples,
                    fade_in=(i > 0),
                    fade_out=(i < num_chunks - 1),
                )
            )
        logger.debug(f"Safety edge fades applied: {num_chunks} chunks, 3ms linear fades")
        return np.concatenate(processed)


# ---------------------------------------------------------------------------
# Audio Post-Processing
# ---------------------------------------------------------------------------


def normalize_peak(
    audio: np.ndarray,
    threshold: float = 0.99,
    target: float = 0.95,
) -> np.ndarray:
    """Normalize audio peak amplitude to prevent clipping.

    Only normalizes if the peak amplitude exceeds `threshold`.
    Returns audio unchanged (with dtype cast to float32) if peak is below
    threshold.

    Args:
        audio: Input audio as float32 numpy array.
        threshold: Peak level above which normalization is applied (default 0.99).
        target: Target peak level after normalization (default 0.95).

    Returns:
        Peak-normalized float32 numpy array.
    """
    audio = audio.astype(np.float32, copy=False)
    peak = float(np.abs(audio).max())
    if peak > threshold and peak > 0.0:
        audio = audio * (target / peak)
        logger.debug(f"Peak normalization applied (peak was {peak:.4f}, target {target})")
    return audio


def apply_speed_factor(
    audio: np.ndarray,
    sample_rate: int,
    speed_factor: float,
) -> np.ndarray:
    """Apply pitch-preserving time-stretching to change audio speed.

    Uses librosa's time-stretch implementation when available, which
    preserves pitch while changing duration. Falls back to simple
    sample-rate resampling if librosa is not available (changes pitch).

    Speed factor > 1.0 speeds up (shorter output).
    Speed factor < 1.0 slows down (longer output).
    Speed factor == 1.0 returns input unchanged.

    Args:
        audio: Input audio as float32 numpy array (1D, mono).
        sample_rate: Sample rate of the audio in Hz.
        speed_factor: Multiplicative speed factor. 1.0 = no change.

    Returns:
        Time-stretched float32 numpy array. Sample rate is unchanged
        (caller does not need to resample).
    """
    if speed_factor == 1.0:
        return audio.astype(np.float32, copy=False)

    audio_f32 = audio.astype(np.float32, copy=False)

    try:
        import librosa

        # librosa.effects.time_stretch expects rate = output_duration / input_duration
        # which is the inverse of speed_factor
        rate = speed_factor
        stretched = librosa.effects.time_stretch(audio_f32, rate=rate)
        return stretched.astype(np.float32)
    except ImportError:
        logger.warning(
            "librosa not available for pitch-preserving time-stretch. "
            "Falling back to sample-rate resampling (pitch will change). "
            "Install librosa to enable pitch-preserving speed adjustment."
        )
        # Fallback: change sample rate to alter playback speed
        # Output array is resampled to appear at the original sample rate
        import numpy as np
        original_length = len(audio_f32)
        target_length = int(original_length / speed_factor)
        indices = np.linspace(0, original_length - 1, target_length)
        return np.interp(indices, np.arange(original_length), audio_f32).astype(np.float32)
    except Exception as e:
        logger.error(f"Speed factor application failed: {e}. Returning original audio.")
        return audio_f32
```

### Step 3: Verify `requirements-common.txt`

Check that the following packages are listed in [`requirements-common.txt`](../../requirements-common.txt):
- `librosa` (for `apply_speed_factor` pitch-preserving time-stretch)
- `scipy` (optional, for DC offset removal if added later)

If missing, add them. `librosa` is likely already present based on the Chatterbox TTS Server requirements pattern, but confirm explicitly.

---

## Progress Checklist

- [ ] Read `requirements-common.txt` to verify `librosa` and `scipy` are listed; add if missing
- [ ] Create `harmonyspeech/common/audio_utils.py` with all functions above
- [ ] Verify the file imports cleanly with `python -c "from harmonyspeech.common.audio_utils import chunk_text_by_sentences, stitch_audio_chunks, apply_speed_factor, normalize_peak"`
- [ ] Confirm `split_into_sentences` correctly handles periods, question marks, exclamation marks
- [ ] Confirm `chunk_text_by_sentences` produces expected chunks for sample long text

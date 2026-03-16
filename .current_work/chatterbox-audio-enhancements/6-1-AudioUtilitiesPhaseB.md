# Phase 6-1: Audio Utilities — Phase B Functions

## Objective

Extend [`harmonyspeech/common/audio_utils.py`](../../harmonyspeech/common/audio_utils.py) (created in Phase 1-1) with three additional post-processing utility functions:

1. **`remove_dc_offset()`** — scipy 2nd-order Butterworth high-pass filter at 15 Hz; removes low-frequency thumps at chunk join points
2. **`trim_silence()`** — `librosa.effects.trim` to remove leading/trailing silence from final audio
3. **`fix_internal_silence()`** — `librosa.effects.split` to shorten unnaturally long pauses inside audio

These functions are already implemented in the reference [`Chatterbox-TTS-Server/utils.py`](.current_work/Chatterbox-TTS-Server/utils.py) — see `_remove_dc_offset()`, `trim_lead_trail_silence()`, and `fix_internal_silence()`. The HSE versions follow the same algorithm with HSE coding conventions applied (snake_case, type hints, loguru logger).

## Prerequisite

- Phase 1-1 complete: `harmonyspeech/common/audio_utils.py` exists

## Files to Modify

- `harmonyspeech/common/audio_utils.py` — append new functions

## Detailed Implementation Steps

### Step 1: Read the existing `audio_utils.py`

Read the current file to confirm its structure and insert the new functions in the `Audio Post-Processing` section, after `normalize_peak()` and `apply_speed_factor()`.

### Step 2: Add `remove_dc_offset()`

Append to the `# Audio Post-Processing` section:

```python
def remove_dc_offset(
    audio: np.ndarray,
    sample_rate: int,
    cutoff_hz: float = 15.0,
) -> np.ndarray:
    """Remove DC offset using a 2nd-order Butterworth high-pass filter.

    DC offset causes low-frequency thumps when concatenating audio chunks.
    This applies zero-phase filtering (no phase distortion via filtfilt).

    Requires scipy. If scipy is not available the audio is returned
    unchanged with a warning.

    Args:
        audio: Input audio as float32 numpy array (1D, mono).
        sample_rate: Sample rate in Hz.
        cutoff_hz: High-pass cutoff frequency (default 15 Hz).

    Returns:
        Audio with DC offset removed, as float32 numpy array.
    """
    audio_f32 = audio.astype(np.float32, copy=False)
    try:
        from scipy.signal import butter, filtfilt

        nyquist = sample_rate / 2.0
        normalized_cutoff = cutoff_hz / nyquist
        b, a = butter(2, normalized_cutoff, btype="high")
        filtered = filtfilt(b, a, audio_f32)
        return filtered.astype(np.float32)
    except ImportError:
        logger.warning(
            "scipy not available for DC offset removal. "
            "Install scipy to enable this feature."
        )
        return audio_f32
    except Exception as e:
        logger.error(f"DC offset removal failed: {e}. Returning original audio.")
        return audio_f32
```

### Step 3: Add `trim_silence()`

```python
def trim_silence(
    audio: np.ndarray,
    sample_rate: int,
    silence_threshold_db: float = 40.0,
    min_silence_duration_ms: int = 100,
    padding_ms: int = 50,
) -> np.ndarray:
    """Trim leading and trailing silence from audio.

    Uses librosa.effects.trim to find the first and last non-silent
    samples, then adds a configurable padding to avoid hard cuts.
    If no significant silence is found, the original audio is returned.

    Requires librosa. If not available the audio is returned unchanged.

    Args:
        audio: Input audio as float32 numpy array (1D, mono).
        sample_rate: Sample rate in Hz.
        silence_threshold_db: Threshold in dBFS below which audio is
            considered silent (positive value, default 40.0 → -40 dBFS).
        min_silence_duration_ms: Minimum silence duration to trigger
            trimming (ms). Protects against trimming very short pauses.
        padding_ms: Silence padding to preserve at start/end after
            trimming (ms), to avoid clipping transients.

    Returns:
        Trimmed float32 numpy array. Returns original if trim would
        produce an empty or invalid result.
    """
    if audio is None or audio.size == 0:
        return audio

    audio_f32 = audio.astype(np.float32, copy=False)

    try:
        import librosa

        trimmed, index = librosa.effects.trim(
            y=audio_f32,
            top_db=silence_threshold_db,
            frame_length=2048,
            hop_length=512,
        )
        start_sample, end_sample = int(index[0]), int(index[1])
        original_length = len(audio_f32)

        # Only trim if librosa found silence at start or end
        if start_sample == 0 and end_sample == original_length:
            logger.debug("trim_silence: no leading/trailing silence found.")
            return audio_f32

        # Check minimum silence duration before trimming
        min_samples = int((min_silence_duration_ms / 1000.0) * sample_rate)
        if start_sample < min_samples and (original_length - end_sample) < min_samples:
            logger.debug("trim_silence: silence below minimum duration threshold, skipping.")
            return audio_f32

        # Apply padding
        padding_samples = int((padding_ms / 1000.0) * sample_rate)
        final_start = max(0, start_sample - padding_samples)
        final_end = min(original_length, end_sample + padding_samples)

        if final_end <= final_start:
            logger.warning("trim_silence: result would be empty, returning original.")
            return audio_f32

        logger.debug(
            f"trim_silence: trimmed from {original_length} to "
            f"{final_end - final_start} samples "
            f"(start_cut={final_start}, end_cut={original_length - final_end})"
        )
        return audio_f32[final_start:final_end]

    except ImportError:
        logger.warning(
            "librosa not available for silence trimming. "
            "Install librosa to enable this feature."
        )
        return audio_f32
    except Exception as e:
        logger.error(f"trim_silence failed: {e}. Returning original audio.")
        return audio_f32
```

### Step 4: Add `fix_internal_silence()`

```python
def fix_internal_silence(
    audio: np.ndarray,
    sample_rate: int,
    silence_threshold_db: float = 40.0,
    min_silence_to_fix_ms: int = 700,
    max_allowed_silence_ms: int = 300,
) -> np.ndarray:
    """Shorten unnaturally long internal silences in audio.

    Uses librosa.effects.split to detect non-silent intervals and
    re-assembles audio while replacing long silence gaps with a
    shorter silence of max_allowed_silence_ms.

    Silences shorter than min_silence_to_fix_ms are left untouched
    (natural pauses between words/sentences are preserved).

    Requires librosa. If not available the audio is returned unchanged.

    Args:
        audio: Input audio as float32 numpy array (1D, mono).
        sample_rate: Sample rate in Hz.
        silence_threshold_db: Threshold in dBFS below which audio is
            considered silent (positive value, default 40.0 → -40 dBFS).
        min_silence_to_fix_ms: Minimum silence duration (ms) that will
            be shortened. Silences shorter than this are kept as-is.
        max_allowed_silence_ms: Maximum silence duration (ms) to keep
            when a long silence is found.

    Returns:
        Float32 numpy array with long internal silences shortened.
        Returns original if no long silences found or on error.
    """
    if audio is None or audio.size == 0:
        return audio

    audio_f32 = audio.astype(np.float32, copy=False)

    try:
        import librosa

        non_silent_intervals = librosa.effects.split(
            y=audio_f32,
            top_db=silence_threshold_db,
            frame_length=2048,
            hop_length=512,
        )

        if len(non_silent_intervals) <= 1:
            logger.debug("fix_internal_silence: no significant internal silences found.")
            return audio_f32

        min_fix_samples = int((min_silence_to_fix_ms / 1000.0) * sample_rate)
        max_keep_samples = int((max_allowed_silence_ms / 1000.0) * sample_rate)

        parts: list[np.ndarray] = []
        last_end = 0

        for start_sample, end_sample in non_silent_intervals:
            silence_len = start_sample - last_end

            if silence_len > 0:
                if silence_len >= min_fix_samples:
                    # Long silence: truncate to max_keep_samples
                    parts.append(audio_f32[last_end : last_end + max_keep_samples])
                    logger.debug(
                        f"fix_internal_silence: shortened silence from "
                        f"{silence_len} to {max_keep_samples} samples"
                    )
                else:
                    # Short silence: keep as-is
                    parts.append(audio_f32[last_end:start_sample])

            parts.append(audio_f32[start_sample:end_sample])
            last_end = end_sample

        # Handle any remaining trailing audio/silence
        if last_end < len(audio_f32):
            trailing_len = len(audio_f32) - last_end
            if trailing_len >= min_fix_samples:
                parts.append(audio_f32[last_end : last_end + max_keep_samples])
            else:
                parts.append(audio_f32[last_end:])

        if not parts:
            logger.warning("fix_internal_silence: result empty, returning original.")
            return audio_f32

        return np.concatenate(parts)

    except ImportError:
        logger.warning(
            "librosa not available for internal silence fixing. "
            "Install librosa to enable this feature."
        )
        return audio_f32
    except Exception as e:
        logger.error(f"fix_internal_silence failed: {e}. Returning original audio.")
        return audio_f32
```

### Step 5: Verify `scipy` in `requirements-common.txt`

Check [`requirements-common.txt`](../../requirements-common.txt) and confirm `scipy` is listed. If missing, add it. (`scipy` is needed for `remove_dc_offset`).

---

## Progress Checklist

- [ ] Read `harmonyspeech/common/audio_utils.py` to confirm insertion point
- [ ] Append `remove_dc_offset()` to `audio_utils.py`
- [ ] Append `trim_silence()` to `audio_utils.py`
- [ ] Append `fix_internal_silence()` to `audio_utils.py`
- [ ] Read `requirements-common.txt` and confirm `scipy` is listed; add if missing
- [ ] Verify imports: `python -c "from harmonyspeech.common.audio_utils import remove_dc_offset, trim_silence, fix_internal_silence"`

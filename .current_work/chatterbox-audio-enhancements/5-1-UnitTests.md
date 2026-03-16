# Phase 5-1: Unit Tests for Audio Utils and Existing Test Updates

## Objective

1. Create unit tests for the new `harmonyspeech/common/audio_utils.py` module.
2. Update existing unit tests that are broken by the `norm_loudness` → `normalize_audio` rename.

Tests must run without a GPU and without any TTS model loaded.

## Files to Create

- `tests/unit/test_audio_utils.py` — **NEW**

## Files to Modify (Existing Tests)

- `tests/unit/inference_flow/test_chatterbox_inputs.py` — update `norm_loudness` references

### Changes required in `test_chatterbox_inputs.py`

**1. Update `CHATTERBOX_FIELD_NAMES` list** (around line 45):

Remove `"norm_loudness"` from the list. `normalize_audio` is now a model-agnostic field in `GenerationOptions`, not a Chatterbox-specific field — so it should not be in `CHATTERBOX_FIELD_NAMES`.

Before:
```python
CHATTERBOX_FIELD_NAMES = [
    ...
    "top_k",
    "norm_loudness",
]
```

After:
```python
CHATTERBOX_FIELD_NAMES = [
    ...
    "top_k",
    # norm_loudness removed — replaced by model-agnostic normalize_audio in GenerationOptions
]
```

**2. Remove `test_tts_rejects_norm_loudness()`** (around line 160):

This test verifies that `prepare_chatterbox_tts_inputs()` raises `ValueError` when `norm_loudness` is set. Since:
- The field is renamed to `normalize_audio`
- The validation is removed from `prepare_chatterbox_tts_inputs()` (normalize_audio is handled in serving layer)

Delete the entire `test_tts_rejects_norm_loudness()` test function.

**3. Update turbo TTS test result assertion** (around line 194-203):

The test that checks turbo TTS returns `norm_loudness` in the result tuple must be updated to use `normalize_audio`.

Before:
```python
text, conditionals, temperature, repetition_penalty, top_p, top_k, norm_loudness = results[0]
assert norm_loudness is True
```

After:
```python
text, conditionals, temperature, repetition_penalty, top_p, top_k, normalize_audio = results[0]
assert normalize_audio is True
```

**4. Update any call sites that pass `norm_loudness=...`** — search for `norm_loudness=` in the test file and replace with `normalize_audio=`.

**5. Update conftest comment** in `tests/e2e/conftest.py` (line 504):

Before: `# - chatterbox_turbo (ChatterboxTurboTTS) — faster TTS with top_k/norm_loudness support`
After: `# - chatterbox_turbo (ChatterboxTurboTTS) — faster TTS with top_k/normalize_audio support`

## Detailed Implementation Steps

### Step 1: Create `tests/unit/test_audio_utils.py`

```python
"""Unit tests for harmonyspeech.common.audio_utils.

All tests use synthetic numpy arrays — no model loading required.
"""

import numpy as np
import pytest

from harmonyspeech.common.audio_utils import (
    NATIVE_SPEED_MODEL_TYPES,
    apply_speed_factor,
    chunk_text_by_sentences,
    crossfade_with_overlap,
    generate_equal_power_curves,
    normalize_peak,
    split_into_sentences,
    stitch_audio_chunks,
)

SAMPLE_RATE = 24000  # Hz


# ---------------------------------------------------------------------------
# split_into_sentences
# ---------------------------------------------------------------------------


def test_split_into_sentences_basic():
    text = "Hello world. How are you? I am fine!"
    result = split_into_sentences(text)
    assert result == ["Hello world.", "How are you?", "I am fine!"]


def test_split_into_sentences_single():
    text = "Only one sentence."
    assert split_into_sentences(text) == ["Only one sentence."]


def test_split_into_sentences_empty():
    assert split_into_sentences("") == []
    assert split_into_sentences("   ") == []


# ---------------------------------------------------------------------------
# chunk_text_by_sentences
# ---------------------------------------------------------------------------


def test_chunk_text_short_text_not_split():
    text = "Short text."
    chunks = chunk_text_by_sentences(text, chunk_size=120)
    assert chunks == ["Short text."]


def test_chunk_text_respects_chunk_size():
    # 3 sentences, each ~30 chars
    text = "First sentence here. Second sentence here. Third sentence here."
    chunks = chunk_text_by_sentences(text, chunk_size=40)
    # Each chunk should be <= 40 chars (approximately; sentence boundaries)
    for chunk in chunks:
        assert len(chunk) <= 60  # allow some slack for sentence boundary logic


def test_chunk_text_multiple_sentences_grouped():
    text = "A. B. C. D."
    chunks = chunk_text_by_sentences(text, chunk_size=10)
    # Each sentence is 2 chars, so multiple can fit per chunk
    assert len(chunks) >= 1
    assert all(len(c) <= 12 for c in chunks)


def test_chunk_text_empty():
    assert chunk_text_by_sentences("") == []


# ---------------------------------------------------------------------------
# generate_equal_power_curves
# ---------------------------------------------------------------------------


def test_equal_power_curves_shape():
    fade_out, fade_in = generate_equal_power_curves(100)
    assert fade_out.shape == (100,)
    assert fade_in.shape == (100,)


def test_equal_power_curves_values():
    fade_out, fade_in = generate_equal_power_curves(100)
    # fade_out starts near 1, ends near 0
    assert float(fade_out[0]) == pytest.approx(1.0, abs=0.01)
    assert float(fade_out[-1]) == pytest.approx(0.0, abs=0.01)
    # fade_in starts near 0, ends near 1
    assert float(fade_in[0]) == pytest.approx(0.0, abs=0.01)
    assert float(fade_in[-1]) == pytest.approx(1.0, abs=0.01)


def test_equal_power_curves_sum_near_unity():
    """Equal-power property: fade_out^2 + fade_in^2 ≈ 1 at all points."""
    fade_out, fade_in = generate_equal_power_curves(200)
    power_sum = (fade_out ** 2) + (fade_in ** 2)
    assert np.allclose(power_sum, 1.0, atol=0.01)


# ---------------------------------------------------------------------------
# crossfade_with_overlap
# ---------------------------------------------------------------------------


def test_crossfade_output_length():
    a = np.ones(1000, dtype=np.float32)
    b = np.ones(1000, dtype=np.float32)
    fade = 100
    result = crossfade_with_overlap(a, b, fade)
    assert len(result) == len(a) + len(b) - fade


def test_crossfade_zero_fade_is_concatenate():
    a = np.ones(500, dtype=np.float32)
    b = np.ones(500, dtype=np.float32) * 2
    result = crossfade_with_overlap(a, b, 0)
    assert len(result) == 1000
    assert result[0] == pytest.approx(1.0)
    assert result[-1] == pytest.approx(2.0)


def test_crossfade_no_clipping():
    """Sum of crossfaded equal-amplitude signals should not exceed peak amplitude."""
    a = np.ones(1000, dtype=np.float32) * 0.8
    b = np.ones(1000, dtype=np.float32) * 0.8
    result = crossfade_with_overlap(a, b, 200)
    assert float(np.abs(result).max()) <= 0.9  # no clipping with equal-power


# ---------------------------------------------------------------------------
# stitch_audio_chunks
# ---------------------------------------------------------------------------


def test_stitch_single_chunk():
    chunk = np.ones(1000, dtype=np.float32) * 0.5
    result = stitch_audio_chunks([chunk], SAMPLE_RATE)
    assert np.allclose(result, chunk)


def test_stitch_two_chunks_with_crossfade():
    a = np.ones(SAMPLE_RATE, dtype=np.float32) * 0.3   # 1 second
    b = np.ones(SAMPLE_RATE, dtype=np.float32) * 0.3
    result = stitch_audio_chunks([a, b], SAMPLE_RATE, crossfade_ms=20, sentence_pause_ms=200, use_crossfade=True)
    # Result should be longer than 2 seconds (silence padding added)
    assert len(result) > 2 * SAMPLE_RATE
    assert result.dtype == np.float32


def test_stitch_two_chunks_fallback():
    a = np.ones(SAMPLE_RATE, dtype=np.float32) * 0.3
    b = np.ones(SAMPLE_RATE, dtype=np.float32) * 0.3
    result = stitch_audio_chunks([a, b], SAMPLE_RATE, use_crossfade=False)
    assert len(result) == 2 * SAMPLE_RATE
    assert result.dtype == np.float32


def test_stitch_empty_returns_empty():
    result = stitch_audio_chunks([], SAMPLE_RATE)
    assert len(result) == 0


# ---------------------------------------------------------------------------
# normalize_peak
# ---------------------------------------------------------------------------


def test_normalize_no_op_below_threshold():
    audio = np.ones(1000, dtype=np.float32) * 0.5
    result = normalize_peak(audio, threshold=0.99, target=0.95)
    assert np.allclose(result, audio)


def test_normalize_reduces_clipping():
    audio = np.ones(1000, dtype=np.float32) * 1.1  # above threshold
    result = normalize_peak(audio, threshold=0.99, target=0.95)
    assert float(np.abs(result).max()) == pytest.approx(0.95, abs=0.001)


def test_normalize_preserves_shape():
    audio = np.random.randn(5000).astype(np.float32) * 2.0
    result = normalize_peak(audio)
    assert result.shape == audio.shape
    assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# apply_speed_factor
# ---------------------------------------------------------------------------


def test_speed_factor_identity():
    audio = np.random.randn(SAMPLE_RATE).astype(np.float32)
    result = apply_speed_factor(audio, SAMPLE_RATE, 1.0)
    assert np.allclose(result, audio)


def test_speed_factor_faster_shorter():
    audio = np.ones(SAMPLE_RATE, dtype=np.float32)  # 1 second
    result = apply_speed_factor(audio, SAMPLE_RATE, 2.0)
    # 2x speed → ~0.5 seconds
    assert len(result) < len(audio)
    assert len(result) == pytest.approx(SAMPLE_RATE / 2, rel=0.1)


def test_speed_factor_slower_longer():
    audio = np.ones(SAMPLE_RATE, dtype=np.float32)  # 1 second
    result = apply_speed_factor(audio, SAMPLE_RATE, 0.5)
    # 0.5x speed → ~2 seconds
    assert len(result) > len(audio)
    assert len(result) == pytest.approx(SAMPLE_RATE * 2, rel=0.1)


def test_speed_factor_output_dtype():
    audio = np.ones(SAMPLE_RATE, dtype=np.float32)
    result = apply_speed_factor(audio, SAMPLE_RATE, 1.5)
    assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# NATIVE_SPEED_MODEL_TYPES
# ---------------------------------------------------------------------------


def test_native_speed_models_contains_expected():
    assert "MeloTTSSynthesizer" in NATIVE_SPEED_MODEL_TYPES
    assert "KittenTTSSynthesizer" in NATIVE_SPEED_MODEL_TYPES
    assert "HarmonySpeechSynthesizer" in NATIVE_SPEED_MODEL_TYPES
    assert "OpenVoiceV1Synthesizer" in NATIVE_SPEED_MODEL_TYPES


def test_chatterbox_not_in_native_speed():
    assert "ChatterboxTTS" not in NATIVE_SPEED_MODEL_TYPES
    assert "ChatterboxTurboTTS" not in NATIVE_SPEED_MODEL_TYPES
    assert "ChatterboxMultilingualTTS" not in NATIVE_SPEED_MODEL_TYPES
```

---

## Progress Checklist

- [ ] Create `tests/unit/test_audio_utils.py` with all test cases above
- [ ] Run `pytest tests/unit/test_audio_utils.py -v` and confirm all tests pass
- [ ] Confirm tests run without GPU and without any model loaded
- [ ] Confirm `test_speed_factor_faster_shorter` and `test_speed_factor_slower_longer` pass (requires librosa to be installed)

# Phase 9-1: Unit Tests — Phase B Audio Utilities

## Objective

Extend the existing unit test file [`tests/unit/test_audio_utils.py`](../../tests/unit/test_audio_utils.py) (created in Phase 5-1) with tests for the three new Phase B utility functions:

1. `remove_dc_offset()` — DC offset removal via scipy Butterworth high-pass filter
2. `trim_silence()` — leading/trailing silence trimming via librosa
3. `fix_internal_silence()` — internal silence shortening via librosa

No new files are created — tests are appended to the existing `test_audio_utils.py`.

## Files to Modify

- `tests/unit/test_audio_utils.py` — append new test classes

---

## Shared test helpers

These tests generate synthetic audio with known properties rather than loading real audio files, keeping them fast and deterministic. Two module-level helpers are already defined by Phase 5-1 (`make_audio()` and `make_silent_audio()`). Add the following additional helper if not already present:

```python
def make_audio_with_silence_gap(sr: int = 22050, gap_ms: int = 1500) -> np.ndarray:
    """Generate audio with two speech segments separated by a long silence."""
    speech = np.sin(2 * np.pi * 440 * np.linspace(0, 0.3, int(sr * 0.3))).astype(np.float32) * 0.8
    silence = np.zeros(int(sr * (gap_ms / 1000.0)), dtype=np.float32)
    return np.concatenate([speech, silence, speech])
```

---

## 1. `TestRemoveDcOffset`

```python
class TestRemoveDcOffset:
    """Unit tests for remove_dc_offset()."""

    def test_reduces_dc_component(self):
        """A signal with DC offset has near-zero mean after filtering."""
        from harmonyspeech.common.audio_utils import remove_dc_offset

        sr = 22050
        t = np.linspace(0, 1.0, sr, dtype=np.float32)
        dc = 0.3
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32) + dc

        result = remove_dc_offset(audio, sr)

        assert abs(float(result.mean())) < 0.05, (
            f"DC not removed: mean after filter = {result.mean():.4f}"
        )

    def test_output_is_float32(self):
        """Output dtype is float32 regardless of input dtype."""
        from harmonyspeech.common.audio_utils import remove_dc_offset

        sr = 22050
        audio = np.ones(sr, dtype=np.float64) * 0.1
        result = remove_dc_offset(audio, sr)
        assert result.dtype == np.float32

    def test_output_shape_unchanged(self):
        """Output has the same number of samples as input."""
        from harmonyspeech.common.audio_utils import remove_dc_offset

        sr = 22050
        audio = make_audio(sr=sr, duration=1.0)
        result = remove_dc_offset(audio, sr)
        assert result.shape == audio.shape

    def test_returns_original_when_scipy_unavailable(self, monkeypatch):
        """When scipy is not importable, returns float32 array of original shape."""
        import builtins
        import harmonyspeech.common.audio_utils as au_mod

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if "scipy" in name:
                raise ImportError("scipy mocked as unavailable")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        audio = make_audio(sr=22050, duration=0.5)
        result = au_mod.remove_dc_offset(audio, 22050)
        assert result.shape == audio.shape
        assert result.dtype == np.float32
```

---

## 2. `TestTrimSilence`

```python
class TestTrimSilence:
    """Unit tests for trim_silence()."""

    def test_trims_leading_and_trailing_silence(self):
        """Audio padded with silence at both ends is returned shorter."""
        from harmonyspeech.common.audio_utils import trim_silence

        sr = 22050
        silence = np.zeros(int(sr * 0.5), dtype=np.float32)  # 500ms
        speech = (
            np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, int(sr * 0.5))) * 0.8
        ).astype(np.float32)
        audio = np.concatenate([silence, speech, silence])

        result = trim_silence(audio, sr)

        assert len(result) < len(audio), "trim_silence did not shorten audio"
        assert len(result) > 0

    def test_no_silence_leaves_length_approximately_unchanged(self):
        """All-signal audio is returned at approximately original length."""
        from harmonyspeech.common.audio_utils import trim_silence

        sr = 22050
        audio = make_audio(sr=sr, duration=1.0)
        result = trim_silence(audio, sr)
        # Allow up to 10% reduction (padding effects)
        assert len(result) >= int(len(audio) * 0.9)

    def test_output_is_float32(self):
        from harmonyspeech.common.audio_utils import trim_silence

        audio = np.zeros(2000, dtype=np.float64)
        result = trim_silence(audio, 22050)
        assert result.dtype == np.float32

    def test_empty_input_returns_empty(self):
        from harmonyspeech.common.audio_utils import trim_silence

        result = trim_silence(np.array([], dtype=np.float32), 22050)
        assert result.size == 0

    def test_returns_original_when_librosa_unavailable(self, monkeypatch):
        """When librosa is not importable, returns float32 array of original shape."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "librosa":
                raise ImportError("librosa mocked as unavailable")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        from harmonyspeech.common.audio_utils import trim_silence
        audio = make_audio(sr=22050, duration=0.5)
        result = trim_silence(audio, 22050)
        assert result.shape == audio.shape
        assert result.dtype == np.float32
```

---

## 3. `TestFixInternalSilence`

```python
class TestFixInternalSilence:
    """Unit tests for fix_internal_silence()."""

    def test_shortens_long_internal_silence(self):
        """A silence gap longer than min_silence_to_fix_ms is shortened."""
        from harmonyspeech.common.audio_utils import fix_internal_silence

        sr = 22050
        audio = make_audio_with_silence_gap(sr=sr, gap_ms=1500)  # 1.5s gap

        result = fix_internal_silence(
            audio, sr,
            min_silence_to_fix_ms=700,
            max_allowed_silence_ms=300,
        )

        assert len(result) < len(audio), "fix_internal_silence did not shorten audio"
        assert len(result) > 0

    def test_preserves_short_internal_silence(self):
        """Silence below min_silence_to_fix_ms is kept unchanged."""
        from harmonyspeech.common.audio_utils import fix_internal_silence

        sr = 22050
        speech = make_audio(sr=sr, duration=0.3)
        short_silence = np.zeros(int(sr * 0.2), dtype=np.float32)  # 200ms < 700ms threshold
        audio = np.concatenate([speech, short_silence, speech])

        result = fix_internal_silence(
            audio, sr,
            min_silence_to_fix_ms=700,
            max_allowed_silence_ms=300,
        )

        # Length should be very close to original (within 10% tolerance for librosa frame alignment)
        assert abs(len(result) - len(audio)) < int(sr * 0.1), (
            "fix_internal_silence incorrectly modified a short silence"
        )

    def test_output_is_float32(self):
        from harmonyspeech.common.audio_utils import fix_internal_silence

        audio = np.zeros(4000, dtype=np.float64)
        result = fix_internal_silence(audio, 22050)
        assert result.dtype == np.float32

    def test_empty_input_returns_empty(self):
        from harmonyspeech.common.audio_utils import fix_internal_silence

        result = fix_internal_silence(np.array([], dtype=np.float32), 22050)
        assert result.size == 0

    def test_returns_original_when_librosa_unavailable(self, monkeypatch):
        """When librosa is not importable, returns float32 array of original shape."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "librosa":
                raise ImportError("librosa mocked as unavailable")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        from harmonyspeech.common.audio_utils import fix_internal_silence
        audio = make_audio_with_silence_gap(sr=22050, gap_ms=1000)
        result = fix_internal_silence(audio, 22050)
        assert result.shape == audio.shape
        assert result.dtype == np.float32
```

---

## Progress Checklist

- [ ] Read `tests/unit/test_audio_utils.py` to find insertion point (end of Phase A tests)
- [ ] Add `make_audio_with_silence_gap()` helper if not already present
- [ ] Append `TestRemoveDcOffset` class (4 tests)
- [ ] Append `TestTrimSilence` class (5 tests)
- [ ] Append `TestFixInternalSilence` class (5 tests)
- [ ] Run `pytest tests/unit/test_audio_utils.py -v` — all new tests pass

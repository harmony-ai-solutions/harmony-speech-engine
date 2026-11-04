# Phase 1-1: Boundary Audio Decoder

> **Plan:** [Engine Modernization](summary.md) · **Phase:** 1 — Wire contract
> **Depends on:** Phase 0 complete
> **Codebase mapping consulted:** `.planning/codebase/ARCHITECTURE.md`, `STACK.md`, `CONVENTIONS.md`

## Objective

One centralized, format-aware audio decoder at the engine boundary: decode **once** per request, support the codecs users actually send (WAV, FLAC, OGG, MP3 via libsndfile; AAC and anything else via an **ffmpeg fallback**), resample once with a high-quality resampler, and return clear typed errors for undecodable input. This replaces the scattered `librosa.load(BytesIO(...), sr=...)` calls in `task_handler/inputs.py` and makes input-format support an explicit, documented contract.

## Context (self-contained)

Current input decode sites (verified 2026-11-04, `harmonyspeech/task_handler/inputs.py`):

| Line(s) | Step | Today |
|---|---|---|
| `:251`, `:271-275` | HarmonySpeech encoder / synthesizer prep | `base64.b64decode` → `librosa.load(sr=44100)` |
| `:393-444` | OpenVoice TCE prep | b64 → `librosa.load(sr=hf.sr)`, then float→int16→pydub `AudioSegment` → **NamedTemporaryFile .wav** → `librosa.load` again per segment |
| `:485-508` | OpenVoice tone converter prep | b64 → `librosa.load(sr=hf.sr)` |
| `:516-530` | FasterWhisper prep | b64 → `librosa.load(sr=16000)` |
| `:598-602` | Silero VAD prep | b64 → `librosa.load(sr=16000)` |
| `:827-876` | Chatterbox prep | b64 → raw bytes passed to upstream (`prepare_conditionals` does its own librosa decode) |

Properties of `librosa.load`: delegates to `soundfile`/libsndfile — WAV/FLAC/OGG always; MP3 only with libsndfile ≥ 1.1; **AAC never** (libsndfile does not decode AAC; a fallback to `audioread`/ffmpeg only happens if `audioread` with ffmpeg is installed — not guaranteed). Nothing documents which formats are accepted; failures surface as cryptic `RuntimeError`s mid-pipeline.

Scope note: this task **centralizes** decoding but intentionally does *not* deduplicate per-hop decodes (each step still decodes its own input once) — the decode-once artifact reuse is Phase 2. This keeps the change small and reviewable.

## Implementation steps

### 1. New module `harmonyspeech/common/audio_codec.py`

```python
class AudioDecodeError(ValueError):
    """Input audio could not be decoded. Lists what was tried."""

def decode_audio(data: bytes, target_sr: int | None = None) -> tuple[np.ndarray, int]:
    """Decode arbitrary audio bytes to float32 mono numpy waveform.

    Order: soundfile (WAV/FLAC/OGG/MP3 with libsndfile>=1.1) → ffmpeg subprocess
    fallback (AAC, MP3 without new libsndfile, exotic containers) → AudioDecodeError.
    Resamples to target_sr with soxr (HQ, fast) when given.
    """

def sniff_format(data: bytes) -> str | None:
    """Magic-byte probe: RIFF, OggS, fLaC, ID3/0xFFEx MPEG, ADTS (0xFFF1/0xFFF9), etc."""

def resample(waveform: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    """soxr.ndarray-based resample (fallback: librosa.resample withres_type='soxr_hq')."""

FFMPEG_AVAILABLE: bool  # probed at import (shutil.which("ffmpeg"))
```

ffmpeg fallback: `subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0", "-f", "wav", "-ac", "1", "pipe:1"], input=data, capture_output=True)` then decode the WAV with soundfile. No new Python dependency beyond `soxr` (add to `requirements-common.txt`; if soxr is undesirable, use `librosa.resample(..., res_type="soxr_hq")` which routes through soxr if bundled — pick one and document).

### 2. Replace all boundary decodes

In every `prepare_*` in `inputs.py` (table above), replace `librosa.load(io.BytesIO(x), sr=N)` with:

```python
from harmonyspeech.common.audio_codec import decode_audio, AudioDecodeError
try:
    audio, sr = decode_audio(input_audio_bytes, target_sr=N)
except AudioDecodeError as e:
    raise ValueError(f"input_audio: {e}") from e   # → per-request 400-style error
```

Keep the resulting shapes/dtypes identical to today (`float32`, mono) so downstream code is untouched. For the OpenVoice TCE path (`:393-444`): decode **once** via `decode_audio`, then build the pydub `AudioSegment` from the decoded array (this also removes the double decode but keep the tempfile flow otherwise — full cleanup is Phase 2-3).

For the Chatterbox raw-bytes path (`:827-876`): decode via `decode_audio` (no target sr — upstream re-samples internally; re-read `chatterbox.py` `prepare_conditionals` to see whether it accepts waveforms; if it requires a file-like, wrap the decoded WAV bytes) — the goal is that AAC/MP3 reference audio works for Chatterbox too.

### 3. Wire decode errors into typed request errors

`AudioDecodeError` → `ValueError` inside prepare functions propagates to the existing per-request error path (`_build_error_result`). Verify end-to-end that a garbage `input_audio` produces an HTTP error response (not a 500 crash / not a hang) for each endpoint that accepts audio.

### 4. Document the contract

`docs/` (API reference page for audio inputs): supported formats matrix (native vs ffmpeg-dependent), sample-rate behavior (auto-resample to model rate), size limits if any. Note that ffmpeg presence is optional and extends the format set.

## Files to create/modify

| File | Change |
|---|---|
| `harmonyspeech/common/audio_codec.py` (new) | decode/sniff/resample + ffmpeg fallback + capability probe |
| `harmonyspeech/task_handler/inputs.py` | all boundary decodes routed through `decode_audio` |
| `requirements-common.txt` | `soxr` (if chosen) |
| `tests/unit/` (new) | `test_audio_codec.py` |
| `tests/test-data/samples/` | add small mp3 + aac fixtures (generate with ffmpeg once, commit binaries) |
| `docs/` | input-format contract page |

## Tests

- Unit per format: wav, flac, ogg, mp3 → correct waveform & sr; resample 44.1k→16k correctness vs known-good (librosa) within tolerance.
- Unit: aac fixture decodes when ffmpeg present; if ffmpeg absent on CI, `skipif(not FFMPEG_AVAILABLE)`.
- Unit: garbage bytes → `AudioDecodeError` with message naming the probed format and the supported list.
- Unit: each `prepare_*` function converts `AudioDecodeError` into `ValueError` mentioning the field name.
- E2E (existing suites): voice-clone requests still pass unchanged.

## Checklist

- [ ] `audio_codec.py` implemented with sniff → soundfile → ffmpeg → typed error chain
- [ ] All `inputs.py` boundary decodes replaced; per-step semantics otherwise unchanged
- [ ] Chatterbox reference audio accepts mp3/aac via the same path
- [ ] soxr/dependency decision made and pinned
- [ ] Fixtures + unit tests added; ffmpeg-optional tests skip cleanly
- [ ] Input-format contract documented
- [ ] `summary.md` Implementation Status updated

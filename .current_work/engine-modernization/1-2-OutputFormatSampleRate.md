# Phase 1-2: Output Format & Sample-Rate Honoring

> **Plan:** [Engine Modernization](summary.md) · **Phase:** 1 — Wire contract
> **Depends on:** 1-1 (reuses `audio_codec` resample/encode helpers)
> **Codebase mapping consulted:** `.planning/codebase/ARCHITECTURE.md` (serving layer), `CONVENTIONS.md`

## Objective

Make `output_options.format` and `output_options.sample_rate` real: convert the model's native WAV output **once, at the API boundary**, to the requested format/rate — or reject the request with 400 when the combination is unsupported. Never silently ignore what the caller asked for (research Phase 4: *"either implemented (resample at the boundary) or rejected with 400 — never silently ignored"*).

## Context (self-contained)

- `AudioOutputOptions` (`harmonyspeech/endpoints/openai/protocol.py:92-95`): `format: str | None = "wav"`, `sample_rate: int | None = None`, `stream: bool | None = False`. Parsed — then ignored.
- `serving_text_to_speech.py:96-106` returns `final_res.output` (base64 WAV produced by the executors at e.g. `model_runner_base.py:704-707`) verbatim as `TextToSpeechResponse.data`. Same pattern in the other audio-returning serving modules: `serving_voice_conversion.py`, `serving_audio_conversion.py` (enumerate all `data=` assignments at implementation time).
- Executors always emit **base64 WAV** (`sf.write(..., format="WAV")`) — that stays the internal contract; conversion happens only in the serving layer (consistent with the Phase 2 rule: serialization belongs at the boundary).

Target format set (parity with vllm-omni where sensible): `wav` (default, passthrough), `flac`, `ogg`, `mp3`, `opus`, `pcm` (s16le). **AAC output requires ffmpeg** (libsndfile cannot encode AAC) — supported only when ffmpeg is available, else 400.

## Implementation steps

### 1. Encode/validate helpers in `audio_codec.py` (extend 1-1)

```python
SUPPORTED_OUTPUT_FORMATS: set[str]   # {"wav","flac","ogg","mp3","opus","pcm"} (+ "aac" if ffmpeg)
SUPPORTED_SAMPLE_RATES: set[int]     # {8000,16000,22050,24000,32000,44100,48000} — decide & document policy

def encode_audio(waveform: np.ndarray, sr: int, fmt: str, target_sr: int | None) -> bytes:
    """Resample (soxr) if target_sr set, then encode. fmt=="pcm" → s16le bytes.
    fmt=="aac" → ffmpeg subprocess. Raises ValueError listing supported formats."""

def decode_wav_b64(b64: str) -> tuple[np.ndarray, int]: ...  # used on executor output
```

Validate at request admission (serving layer), not after synthesis: unknown `format`, non-positive/unsupported `sample_rate` → immediate 400 via the existing `create_error_response`.

### 2. Apply in each audio-serving module

For `serving_text_to_speech.py`, `serving_voice_conversion.py`, `serving_audio_conversion.py` (and any other endpoint returning audio — enumerate via grep for `TextToSpeechResponse(`/`VoiceConversionRequestOutput`-to-response sites):

```python
audio_bytes = apply_output_options(final_res.output, request.output_options)
response = TextToSpeechResponse(..., data=audio_bytes)
```

- `apply_output_options(b64_wav, opts)`: no opts or `format=="wav"` and no resample → return input unchanged (byte-identical, zero cost). Otherwise decode → resample/encode → base64. Live in a shared helper (e.g. `harmonyspeech/endpoints/openai/output_encoding.py`) so all endpoints share one implementation.
- `pcm` convention: base64 of s16le mono at (possibly resampled) `sample_rate`; document that `sample_rate` should be set explicitly with `pcm` (default = model rate) and reflect the effective rate in the response — add a `sample_rate` field to the response models if not present (additive, optional field).
- Keep the response schema otherwise unchanged (`data: str` base64) for backward compatibility.

### 3. Startup capability probe

At API server startup, compute the effective supported-format list (`soundfile.available_formats()` ∩ our set, plus ffmpeg-dependent entries if `FFMPEG_AVAILABLE`) and log it once; use the same list for validation so 400 messages never advertise a format the runtime can't produce.

### 4. Documentation

Update the API docs (`docs/`) for every affected endpoint: `output_options.format` values + availability notes (aac = ffmpeg-only, opus = libsndfile ≥ 1.1), `sample_rate` policy, `pcm` layout. Example requests.

## Files to create/modify

| File | Change |
|---|---|
| `harmonyspeech/common/audio_codec.py` | `encode_audio`, `decode_wav_b64`, format/rate whitelists |
| `harmonyspeech/endpoints/openai/output_encoding.py` (new) | `apply_output_options()` shared helper |
| `harmonyspeech/endpoints/openai/protocol.py` | response models gain optional `sample_rate` echo field (additive) |
| `harmonyspeech/endpoints/openai/serving_text_to_speech.py` | validate at admission; apply at response |
| `harmonyspeech/endpoints/openai/serving_voice_conversion.py` | same |
| `harmonyspeech/endpoints/openai/serving_audio_conversion.py` | same |
| `tests/e2e/tts/`, `tests/unit/` | format matrix tests |

## Tests

- Unit: `apply_output_options` passthrough — no options → byte-identical output.
- Unit: wav 24k → mp3 @ 16k: decodes back, correct sr, non-silent audio; same for flac/ogg/opus/pcm.
- Unit: unsupported `format: "wma"` → 400 before any synthesis runs (assert engine never received the request — admission-time check).
- Unit: `sample_rate: 0` / negative → 400.
- E2E: TTS request with `output_options {format: "flac", sample_rate: 16000}` returns decodable FLAC at 16k; voice-convert endpoint ditto.

## Checklist

- [ ] `encode_audio`/`apply_output_options` implemented with capability-aware whitelist
- [ ] All audio-returning endpoints validate at admission and convert at response
- [ ] Default behavior byte-identical (no options = no change)
- [ ] `pcm` layout + `sample_rate` echo defined and documented
- [ ] Format matrix tests (incl. ffmpeg-gated skips) passing
- [ ] API docs updated for all affected endpoints
- [ ] `CHANGELOG.md` entry (user-visible feature)
- [ ] `summary.md` Implementation Status updated

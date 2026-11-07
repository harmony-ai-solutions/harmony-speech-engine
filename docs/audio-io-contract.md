# Audio I/O Contract

This page is the single reference for **how audio crosses the Harmony Speech Engine
HTTP boundary**: which input formats are accepted, which transports carry them,
what `output_options` does, and how failures are reported.

Everything below applies uniformly to all audio endpoints:

| Endpoint | Audio fields |
|---|---|
| `POST /v1/audio/speech` | `input_audio` (voice cloning reference) |
| `POST /v1/audio/transcriptions` | `input_audio` |
| `POST /v1/audio/translations` | `input_audio` |
| `POST /v1/audio/vad` | `input_audio` |
| `POST /v1/embed/speaker` | `input_audio` |
| `POST /v1/voice/convert` | `source_audio`, `target_audio` |
| `POST /v1/audio/convert` | `source_audio` |

## Input transports

Each audio-input endpoint accepts **two request bodies on the same path**. The
server picks the parser from the `Content-Type` header; the request model and
pipeline are identical afterwards.

### 1. JSON with base64 audio (always supported)

```bash
curl -X POST http://localhost:2242/v1/audio/transcriptions \
  -H "Content-Type: application/json" \
  -d '{"model": "faster-whisper", "input_audio": "<base64 audio>"}'
```

### 2. `multipart/form-data` uploads (recommended for files)

This matches the OpenAI API convention and avoids the ~33 % base64 inflation:

```bash
# Transcription — OpenAI-style `file` part
curl -X POST http://localhost:2242/v1/audio/transcriptions \
  -F "file=@speech.mp3" \
  -F "model=faster-whisper"

# Voice cloning — reference audio as a file part
curl -X POST http://localhost:2242/v1/audio/speech \
  -F "input_audio=@reference.wav" \
  -F "model=chatterbox" \
  -F "mode=voice_cloning" \
  -F "input=Hello there"

# Voice conversion — two file parts
curl -X POST http://localhost:2242/v1/voice/convert \
  -F "source_audio=@source.wav" \
  -F "target_audio=@target.flac" \
  -F "model=openvoice_v2"

# Audio restoration
curl -X POST http://localhost:2242/v1/audio/convert \
  -F "file=@noisy.wav" \
  -F "model=voicefixer"
```

Accepted upload part names (aliases in priority order):

| Endpoint | Part names |
|---|---|
| `/v1/audio/speech` | `input_audio`, `file`, `audio` |
| `/v1/audio/transcriptions` | `file`, `audio`, `input_audio` |
| `/v1/audio/translations` | `file`, `audio`, `input_audio` |
| `/v1/audio/vad` | `file`, `audio`, `input_audio` |
| `/v1/embed/speaker` | `file`, `audio`, `input_audio` |
| `/v1/voice/convert` | `source_audio` (aliases `file`, `audio`) and `target_audio` |
| `/v1/audio/convert` | `source_audio` (aliases `file`, `audio`) |

All non-file options are ordinary form fields and are parsed with the same types
as the JSON body. Nested options (`generation_options`, `output_options`,
`pre_processing_filters`, `post_generation_filters`) are sent as JSON strings:

```bash
curl -X POST http://localhost:2242/v1/audio/speech \
  -F "input=Hello" -F "model=kitten-tts-mini" -F "mode=single_speaker_tts" -F "voice=Bella" \
  -F 'output_options={"format": "flac", "sample_rate": 16000}'
```

`/v1/audio/speech` keeps JSON as its primary/documented body because the OpenAI
speech API is JSON; multipart exists for the voice-cloning `input_audio` field
and for symmetry with the other endpoints.

### Upload size limit

Request bodies larger than the configured limit are rejected with **HTTP 413**
before any parsing or inference happens:

- CLI: `--max-upload-size-mb <MiB>`
- Environment: `HARMONYSPEECH_MAX_UPLOAD_MB`
- Default: 50 MiB

The limit applies to the whole request body, so it also caps oversized base64
JSON payloads.

## Input formats

Audio is decoded once at the boundary, mono `float32`, and resampled to the
model's native rate with a high-quality SoXR resampler.

| Container / codec | Requirement |
|---|---|
| WAV (PCM, float, µ-law/A-law) | Native (libsndfile) |
| FLAC | Native (libsndfile) |
| OGG / Vorbis | Native (libsndfile) |
| AIFF / CAF / AU | Native (libsndfile) |
| MP3 | libsndfile ≥ 1.1 (bundled with current `soundfile` wheels) |
| AAC (ADTS), M4A/MP4, WebM, AMR, … | **Requires `ffmpeg` on `PATH`** |

`ffmpeg` is optional. When it is not installed, the native formats above are
still accepted; AAC/MP4 and other exotic containers are rejected with a clear
error. The startup log lists the effective input/output capability set:

```
Audio I/O contract — input formats: aac, flac, mp3, ogg, wav; output formats: aac, flac, mp3, ogg, opus, pcm, wav
```

Sample-rate conversion targets are per model and invisible to callers: the
engine always hands the pipeline the rate it needs.

Undecodable input produces **HTTP 400** with a message naming the request field,
the probed container (from magic bytes) and what was tried:

```json
{
  "object": "error",
  "message": "Unable to decode input audio (probed format: mp3). Tried libsndfile: ...; ffmpeg: ... Supported input formats: wav, flac, ogg, mp3 ...",
  "type": "BadRequestError",
  "code": 400
}
```

## STT response formats

`/v1/audio/transcriptions` and `/v1/audio/translations` accept a
`response_format` field. It is validated at admission: an unknown value is
**HTTP 400** before any inference is queued.

| `response_format` | Content type | Body |
|---|---|---|
| `json` (default) | `application/json` | `SpeechToTextResponse` (`id`, `model`, `created`, `text`, `language`, `timestamps`), unchanged from earlier releases |
| `text` | `text/plain` | The plain transcript/translation |
| `srt` | `text/plain` | SubRip document rendered from the segment timestamps (`HH:MM:SS,mmm --> HH:MM:SS,mmm`, numbered blocks) |
| `vtt` | `text/plain` | WebVTT document (`WEBVTT` header, `HH:MM:SS.mmm --> HH:MM:SS.mmm`, no block numbers) |

Subtitle edge cases:

- Timestamps beyond one hour render as `HH:MM:SS` with the hour field growing as
  needed (`01:02:03,004`, `100:00:00,000`); fractions are rounded half-up to the
  nearest millisecond.
- Empty segment lists render an empty document (`""` for SRT, the `WEBVTT`
  header for WebVTT), never an error.
- Cue text keeps internal line breaks; `\r\n`/`\r` are normalised to `\n`.
- A literal `-->` in the transcript is rendered as `- ->` (neither format allows
  the timing arrow inside cue text).

`language` is a **source-language hint** for both endpoints. On translations it
never selects the output language: the model always translates to English.

## Output format and sample rate

`output_options` is honored by every endpoint that returns audio
(`/v1/audio/speech`, `/v1/voice/convert`, `/v1/audio/convert`).

| Field | Values | Notes |
|---|---|---|
| `format` | `wav` (default), `flac`, `ogg`, `mp3`, `opus`, `pcm`, `aac` | `aac` only when `ffmpeg` is installed; `opus` only when libsndfile supports it (current builds do) |
| `sample_rate` | `8000`, `16000`, `22050`, `24000`, `32000`, `44100`, `48000` | Omitted → model native rate |
| `stream` | `false` | `true` is rejected with 400 (streaming is not implemented yet) |

Behavior:

- **No `output_options`, or `format="wav"` with no `sample_rate`** → the
  executor's WAV is returned **byte-identical**. This is the default path; it
  costs nothing and is fully backward compatible.
- **Any other combination** → the WAV is decoded once, resampled with SoXR when
  a rate is requested, and re-encoded at the API boundary.
- **`pcm`** is raw mono **signed little-endian 16-bit** samples (no container
  header). It is base64-encoded into `data` like every other format. Pair it
  with an explicit `sample_rate`; when omitted the model rate is used.
- Invalid `format`/`sample_rate` values are rejected with **HTTP 400 at
  admission**, before any synthesis begins.
- The response gains an optional `sample_rate` field whenever the engine
  re-encoded/resampled the payload (it tells the client how to interpret `pcm`
  and confirms the resample). It is `null`/absent on passthrough.

The advertised format list is probed at startup from
`soundfile.available_formats()`, so an error message never offers a format the
running build cannot produce.

Example:

```bash
curl -X POST http://localhost:2242/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
        "model": "kitten-tts-mini",
        "mode": "single_speaker_tts",
        "input": "Hello from Harmony Speech Engine",
        "voice": "Bella",
        "output_options": {"format": "flac", "sample_rate": 16000}
      }'
```

```json
{
  "id": "tts-…",
  "model": "kitten-tts-mini",
  "created": 1762300000,
  "data": "<base64 FLAC @ 16 kHz>",
  "sample_rate": 16000
}
```

## Related

- [API reference](api.md) — endpoint parameters and examples
- [Models](models.md) — per-model native sample rates and expected input rates

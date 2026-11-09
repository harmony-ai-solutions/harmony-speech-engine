# API Documentation

The API of Harmony Speech Engine is built using FastAPI, which provides automatic interactive documentation.

## API Endpoint

By default, the API is accessible at:

http://127.0.0.1:12080

If you decide to use a different port or you're hosting the API,
please make sure to use the proper endpoint instead.


## Interactive Documentation

### Swagger UI

Swagger UI is available at, also Providing OpenAPI Spec for client generation:

http://127.0.0.1:12080/docs


### ReDoc

ReDoc documentation is available at:

http://127.0.0.1:12080/redoc


Both Swagger UI and ReDoc provide interactive documentation for exploring and testing the API endpoints.


## Authentication

If the engine is configured with an API key (via `HARMONYSPEECH_API_KEY` environment variable or `--api-keys` CLI argument), all `/v1/*` endpoints require authentication. Two methods are supported:

| Method | Header | Example |
|--------|--------|---------|
| Bearer token | `Authorization: Bearer <key>` | `Authorization: Bearer my-secret-key` |
| API key | `Api-Key: <key>` | `Api-Key: my-secret-key` |

When using Harmony Auth (external key management), the `Api-Key` header is validated against the configured rate-limiting service. Requests without a valid key receive `401 Unauthorized`.

Endpoints outside `/v1/*` (such as `/health`, `/version`, `/metrics`) do not require authentication.


## Common Types

Several request types share nested objects. Their fields are documented once here and referenced in each endpoint.

### GenerationOptions

Controls generation parameters for TTS and voice conversion models. Most fields are model-specific — unsupported fields are silently ignored.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `seed` | integer | null | Random seed for reproducible output |
| `style` | integer | null | Style index (model-dependent) |
| `speed` | float | null | Speech speed multiplier |
| `pitch` | float | null | Pitch adjustment |
| `energy` | float | null | Energy/amplitude adjustment |
| `exaggeration` | float | null | *(Chatterbox TTS/Multilingual)* Emotion exaggeration factor (default: 0.5) |
| `cfg_weight` | float | null | *(Chatterbox TTS/Multilingual)* Classifier-free guidance weight (default: 0.5) |
| `temperature` | float | null | *(Chatterbox TTS/Turbo/Multilingual)* Sampling temperature (default: 0.8) |
| `repetition_penalty` | float | null | *(Chatterbox TTS/Turbo/Multilingual)* Repetition penalty, >1.0 reduces repetition (default: 1.2) |
| `top_p` | float | null | *(Chatterbox TTS/Turbo/Multilingual)* Top-p nucleus sampling probability (default TTS: 1.0, Turbo: 0.95) |
| `min_p` | float | null | *(Chatterbox TTS/Multilingual only)* Minimum probability threshold (default: 0.05) |
| `top_k` | integer | null | *(Chatterbox Turbo only)* Top-k sampling candidates (default: 1000) |
| `norm_loudness` | boolean | null | *(Chatterbox Turbo only)* Apply pyloudnorm loudness normalization (default: true) |
| `split_text` | boolean | false | Split long input into sentence-aware chunks, synthesize them concurrently and stitch the result. A request that yields 0 or 1 chunk is sent through the single-shot path unchanged |
| `chunk_size` | integer | null | Character budget per synthesis chunk. Enables chunking even when `split_text` is omitted. When `split_text: true` and this is null, the server budget is **300 characters** (one to three sentences) for **every** family — the Phase 12-5 per-model Turbo 150 / Multilingual 75 overrides were removed in Phases 12-7/12-8 once the underlying AR/KV and analyzer bugs were fixed. An explicit value always wins, for every model |
| `crossfade_ms` | integer | null | Equal-power crossfade length between stitched chunks (ms). When chunking is on and this is null, the server default is **40 ms**. The fade consumes part of `sentence_pause_ms` rather than adding to it. Explicit `0` disables the fade (joins are still edge-conditioned) |
| `sentence_pause_ms` | integer | null | Silence inserted between stitched sentence chunks (ms). When chunking is on and this is null, the server default is **120 ms**. The audible gap is `sentence_pause_ms - crossfade_ms` (clamped at zero), plus the two 10 ms guard pads kept by edge conditioning |
| `remove_dc_offset` | boolean | false | Remove the constant (DC) offset. In the chunked path each chunk's mean is removed **before** stitching (offsets differ per synthesis run); in the single-shot path the final buffer's mean is removed |
| `trim_silence` | boolean | false | Trim leading/trailing silence from the final output. Energy gate: a frame is silent when its RMS is **40 dB below the loudest frame** (32 ms frames, 8 ms hop); **50 ms** of padding is kept at each end |
| `fix_internal_silence` | boolean | false | Collapse internal pauses longer than **700 ms** to at most **300 ms**. Leading/trailing silence is left to `trim_silence`, so a chunk boundary is never treated as the buffer's edge |

> Long-form requests are split sentence-aware (Latin `.!?` and CJK `。！？…`,
> with abbreviations/numbers/versions/bullets preserved; parenthetical prose and
> inline `[laugh]`/`[gasp]`-style tags stay inside their sentence and are never
> split boundaries) and the chunks are submitted **concurrently** so they can
> share one engine batch; stitching is always by chunk index, never completion
> order. Voice-cloning requests resolve the reference embedding once and reuse
> it for every chunk;
> a watermark is applied exactly once, on the stitched buffer. `speed` is
> applied after stitching for families that do not consume it natively (the
> Chatterbox families); native-speed families keep their existing handling.
> Any chunk failure fails the whole request (HTTP error, no partial audio).
> Omitting all four chunking fields leaves single-shot requests byte-identical.
>
> The three Phase B filters are independent opt-ins (all default `false`, so
> omitting them is byte-identical to the pre-filter behaviour). Ordering is
> fixed: in the chunked path DC removal is **per chunk before stitching**, then
> `trim_silence` and `fix_internal_silence` run on the **final stitched buffer**
> (before the post-stitch `speed` stretch and the single watermark). In the
> single-shot path all three run on the model's final buffer. The filters are
> pure numpy and model-agnostic; the thresholds above are presets (there are no
> per-request knobs).
>
> **Join quality (Phases 12-5 / 12-6).** Before joining, the stitcher conditions
> every chunk's edges: synthesis-native leading/trailing near-silence is trimmed
> to a **10 ms guard pad**, with a gate **45 dB below that chunk's own peak** and
> a hard bound of **150 ms removed per edge**. Joins are therefore uniform and
> independent of how much silence the model happened to emit; a chunk with no
> near-silence (or an all-quiet chunk) is left intact.
>
> When `sentence_pause_ms > 0`, the pause is applied as a **two-sided equal-power
> fade** (Phase 12-6): the outgoing chunk's conditioned tail fades **out into
> silence** over `crossfade_ms` (`cos²`), the pause follows, then the incoming
> chunk's conditioned head fades **in from that silence** (`sin²`). No chunk is
> hard-cut at a join. The audible gap is therefore
> `sentence_pause_ms - crossfade_ms` plus the two guard pads — the fades consume
> the last part of the pause rather than adding to it. With `sentence_pause_ms: 0`
> the two conditioned edges overlap directly in a single equal-power crossfade.
>
> This is **stitcher-internal** and separate from `trim_silence`: conditioning
> normalises joins (and also bounds the first/last chunk's outer edges to the
> guard pad), while `trim_silence` remains an explicit whole-output option. On
> chunked output the guard pad is shorter than the filter's 32 ms analysis frame,
> so `trim_silence` usually has nothing left to remove at the outer edges; use it
> for the single-shot path or for models whose output carries longer outer
> silence. The tuned defaults (40 ms fade / 120 ms pause) apply only when
> chunking is on and the client left the field `null`; an explicit value —
> including `0` — always wins, and non-chunked requests are byte-identical to
> before.
>
> **Unified chunk budget & fidelity gate (Phases 12-5..12-8).** When
> `split_text` is on and `chunk_size` is null, the server budget is
> **300 characters for every family**. The earlier 150-character Turbo and
> 75-character Multilingual overrides were removed after the audit found that
> the missing words were engine bugs, not model long-context limits:
>
> * **Turbo** was silently truncating mid-sentence because the shared
>   autoregressive KV-cache slot cap was **768 tokens**, while Turbo's
>   un-resampled conditioning prompt alone can be **376 tokens**; the remaining
>   budget could not cover a 300-character chunk's speech. Raising the slot
>   capacity (Phase 12-8) restores the full generation budget.
> * **Multilingual** was hitting a premature EOS because the engine passed
>   `repetition_penalty = 1.2` instead of upstream's **2.0** into the
>   `AlignmentStreamAnalyzer` repeat detector, and its leaked attention hooks
>   forced slow eager attention. Restoring the upstream penalty and fixing the
>   hook/EOS path (Phase 12-7) also lets it use fast attention with batched AR
>   decoding and batched vocoding.
>
> The **fidelity gate** (`bench_throughput.py --scenario longform --fidelity`,
> faster-whisper `large-v3-turbo`) transcribes a 138-word real-prose fixture and
> reports word edit distance and words heard, so a text-coverage regression is
> caught automatically. The pre-fix fidelity figures at 300 characters and the
> post-fix Phase 12-9 host re-measurement (every family speaks 138/138 words;
> Chatterbox ON edit distance 2/4/4) are recorded in the
> [throughput benchmarks](benchmarks/throughput.md). An explicit `chunk_size`
> always wins, for every model.

### AudioOutputOptions

Controls the output audio format for TTS, voice conversion, and audio conversion endpoints.
See [Audio I/O contract](audio-io-contract.md) for the complete format/transport matrix.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `format` | string | `"wav"` | `wav`, `flac`, `ogg`, `mp3`, `opus`, `pcm` (raw s16le) or `aac` (only when `ffmpeg` is available). Unsupported values → HTTP 400 |
| `sample_rate` | integer | null | Resample output to this rate (Hz): 8000, 16000, 22050, 24000, 32000, 44100, 48000. If null, uses the model's native rate |
| `stream` | boolean | false | Whether to stream the response (reserved for future use; `true` currently returns 400) |

Notes:

- Omitting `output_options` (or sending `format: "wav"` with no `sample_rate`) returns the
  executor WAV **byte-identical** — no decode, no re-encode.
- `pcm` is mono signed little-endian 16-bit samples, base64-encoded in `data`.
  Set `sample_rate` explicitly for `pcm`; otherwise the model native rate applies.
- Audio responses include an optional `sample_rate` field with the effective rate whenever the
  engine re-encoded or resampled the payload.

### Audio inputs and transports

Every endpoint that accepts audio takes it either as base64 inside JSON **or** as a
`multipart/form-data` file upload on the same path (content-type dispatch). Supported input
formats: WAV, FLAC, OGG/Vorbis, AIFF/CAF/AU natively, MP3 with libsndfile ≥ 1.1, and
AAC/M4A/WebM/AMR when `ffmpeg` is installed on the server. Undecodable audio returns HTTP 400
with a message naming the field and the probed format. Uploads larger than
`--max-upload-size-mb` (default 50 MiB, or `HARMONYSPEECH_MAX_UPLOAD_MB`) return HTTP 413.

See [Audio I/O contract](audio-io-contract.md) for `curl -F` examples per endpoint.


## Endpoint Details

### Health Check

**GET /health**

Returns the health status of all engine subsystems. Does not require authentication.

**Example Response (200):**

```json
{
  "text_to_speech": "healthy",
  "speech_to_text": "healthy",
  "voice_conversion": "healthy",
  "voice_embedding": "healthy",
  "voice_activity_detection": "healthy"
}
```

If any subsystem is unhealthy, the response includes the error message and the HTTP status is `503`.

The payload also carries a `vram` block (per-model memory, device-level used/free headroom, and each Whisper model's measured max-shape probe with a `probe_measured` annotation) and a `whisper_vram` block (the boot-time Whisper budget verdict). `vram.headroom_bytes` is **true free device memory** (`torch.cuda.mem_get_info`); the torch-scoped counters remain available as `allocated_bytes` / `reserved_bytes` / `peak_allocated_bytes`. See [VRAM profiles](benchmarks/vram-profiles.md#phase-10-3--device-level-truth).


### Version

**GET /version**

Returns the running Harmony Speech Engine version.

**Example Response:**

```json
{
  "version": "0.5.0"
}
```


---

### Voices Catalog

**GET /v1/audio/voices**

Lists the selectable voices of every configured model, so clients no longer
need to read the server's configuration to know what they can ask for. Each
entry is a model id you can send as `model`, with its native presets and — on a
voice-cloning-capable id — the standard voice pack names, each labeled with its
`source`. See [standard voice pack names](#standard-voice-pack-names) for what a
pack name means across families.

The response is additive and `ModelList`-consistent (`object: "list"` plus
`data`). `?model=<name>` narrows it to one id; an unknown or non-voice id
returns `400` with the message below.

**Response fields:**

| Field | Type | Description |
|-------|------|-------------|
| `object` | string | `"list"` |
| `data[].model` | string | Model or toolchain id to send as `model` |
| `data[].object` | string | `"model"` for a configured executor, `"toolchain"` for a pipeline alias |
| `data[].model_type` | string \| null | Executor model type; `null` for a toolchain entry |
| `data[].cloning_capable` | boolean | A standard voice pack name is meaningful for this id in some direction (TTS `voice` or VC `target_voice`) |
| `data[].vc_target_kind` | string \| null | How a standard `target_voice` is consumed: `"target_audio"` (the executor takes the clip) or `"target_embedding"` (the server embeds the clip first); `null` when the id is not a voice-conversion target |
| `data[].voices[].name` | string | Voice id |
| `data[].voices[].source` | string | `"native"` (a preset of the configured model) or `"standard"` (a vendored pack identity) |
| `data[].voices[].languages` | array \| null | Configured language(s) of a native voice; `null` for pack clips, which are language-agnostic references |

Toolchain ids (`openvoice_v1`, `harmonyspeech`, …) appear as their own entries
because clients can name them as `model`. When a pipeline id is also a
configured model name (the shipped `chatterbox`, `chatterbox_vc`), it is one
merged entry with `object: "model"` — the same presentation as
`GET /v1/audio/speech/models`. A toolchain's native voices are merged across its
configured variants, so `default` on `openvoice_v1` carries both `"EN"` and
`"ZH"`.

The endpoint is read-only catalog state: it does not wait for pack-voice warm-up,
so standard voices are listable before their embeddings are cached. With
`standard_voice_pack: false`, `voices` contains no `standard` entries; the
capability fields still describe the model, because explicit reference clips
keep working.

**Example — full list:**

```bash
curl "http://localhost:12080/v1/audio/voices" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

```json
{
  "object": "list",
  "data": [
    {
      "model": "ov2-synthesizer-en",
      "object": "model",
      "model_type": "MeloTTSSynthesizer",
      "cloning_capable": false,
      "vc_target_kind": null,
      "voices": [
        {"name": "EN-Newest", "source": "native", "languages": ["EN"]}
      ]
    },
    {
      "model": "chatterbox",
      "object": "model",
      "model_type": "ChatterboxTTS",
      "cloning_capable": true,
      "vc_target_kind": null,
      "voices": [
        {"name": "Abigail", "source": "standard", "languages": null},
        {"name": "Adrian", "source": "standard", "languages": null}
      ]
    },
    {
      "model": "chatterbox_vc",
      "object": "model",
      "model_type": "ChatterboxVC",
      "cloning_capable": true,
      "vc_target_kind": "target_audio",
      "voices": [
        {"name": "Abigail", "source": "standard", "languages": null}
      ]
    },
    {
      "model": "openvoice_v1",
      "object": "toolchain",
      "model_type": null,
      "cloning_capable": true,
      "vc_target_kind": "target_embedding",
      "voices": [
        {"name": "default", "source": "native", "languages": ["EN", "ZH"]},
        {"name": "friendly", "source": "native", "languages": ["EN"]},
        {"name": "Abigail", "source": "standard", "languages": null}
      ]
    }
  ]
}
```

> Entries and pack voices are truncated above for readability; the real response
> lists every configured model and all 28 pack names.

**Example — narrow to one model:**

```bash
curl "http://localhost:12080/v1/audio/voices?model=openvoice_v1" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

```json
{
  "object": "list",
  "data": [
    {
      "model": "openvoice_v1",
      "object": "toolchain",
      "model_type": null,
      "cloning_capable": true,
      "vc_target_kind": "target_embedding",
      "voices": [
        {"name": "default", "source": "native", "languages": ["EN", "ZH"]},
        {"name": "Abigail", "source": "standard", "languages": null}
      ]
    }
  ]
}
```

**Example — unknown model (400):**

```bash
curl "http://localhost:12080/v1/audio/voices?model=openvoice_v3" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

```json
{
  "object": "error",
  "message": "The model `openvoice_v3` has no voice catalog entry. Call GET /v1/audio/voices without `model` to list every model with selectable voices.",
  "type": "BadRequestError",
  "param": "model",
  "code": 400
}
```

**Using a listed name.** A `standard` name is sent back unchanged as `voice`
(TTS) or `target_voice` (voice conversion). A `native` name is a preset of the
selected model.

```bash
# TTS: a standard pack voice on a cloning family
curl -X POST "http://localhost:12080/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{"model": "chatterbox", "mode": "voice_cloning", "input": "Hello.", "voice": "Abigail"}'

# VC target-audio: the pack clip becomes the conversion target directly
curl -X POST "http://localhost:12080/v1/voice/convert" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{"model": "chatterbox_vc", "source_audio": "base64_encoded_source_voice", "target_voice": "Abigail"}'

# VC target-embedding: the server embeds the clip through the pipeline first
curl -X POST "http://localhost:12080/v1/voice/convert" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{"model": "ov2-tone-converter", "source_audio": "base64_encoded_source_voice", "target_voice": "Abigail"}'
```

`target_embedding` needs the OpenVoice pipeline's encoder and VAD configured;
otherwise that id honestly reports `vc_target_kind: null` and rejects a pack
`target_voice`. Native voices are unaffected by the pack switch, and a pack name
cannot be combined with an explicit reference (see
[Voice Conversion](#voice-conversion)).


---

### Text-to-Speech

**POST /v1/audio/speech**

Generates speech audio from text input. Supports multi-language TTS, voice cloning with a reference audio, and style control.

Based on the [OpenAI TTS API](https://platform.openai.com/docs/api-reference/audio/createSpeech), extended with voice cloning and pipeline features.

#### Request Body Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model` | string | Yes | - | Name of the TTS model or toolchain group (see [Model Selection](#tts-model-selection) below) |
| `input` | string | Yes | - | The text to synthesize. Some models (e.g. `chatterbox_turbo`) support [paralinguistic tags](#paralinguistic-tags) embedded in the text |
| `mode` | string | Yes | - | Pipeline mode: `"single_speaker_tts"` or `"voice_cloning"` (see [Mode Selection](#tts-mode-selection) below) |
| `language` | string | Conditional | null | Language ID — required by models with multiple languages (see [Language & Voice](#tts-language-and-voice) below) |
| `voice` | string | Conditional | null | Voice ID — a native preset for the selected language, or a [standard voice pack name](#standard-voice-pack-names) on a cloning-capable model |
| `input_audio` | string | Conditional | null | Base64-encoded reference speaker audio — required for `voice_cloning` mode (unless `input_embedding` is provided) |
| `input_embedding` | string | Conditional | null | Base64-encoded speaker embedding — alternative to `input_audio` (faster, skips VAD/embedding steps) |
| `input_vad_data` | string | No | null | VAD result JSON from a prior `/v1/audio/vad` call. If provided, skips the internal VAD step in voice-cloning pipelines |
| `input_vad_mode` | string | No | null | VAD backend for reference-audio segmentation: `"whisper"` (default) or `"silero"`. `"silero"` falls back to Whisper with a warning when no Silero model is configured. |
| `generation_options` | [GenerationOptions](#generationoptions) | No | null | Generation parameters (speed, pitch, Chatterbox params, etc.) |
| `output_options` | [AudioOutputOptions](#audiooutputoptions) | No | null | Output audio format settings |
| `pre_processing_filters` | list | No | `[]` | Filters to apply before synthesis |
| `post_generation_filters` | list | No | `[]` | Filters to apply after synthesis |

#### TTS Model Selection

The `model` parameter accepts either an **individual model name** or a **toolchain group name**. Toolchain groups automatically chain multiple models to produce the final audio.

| `model` value | Type | Pipeline | Requires `language` | Requires `voice` | Supports `input_embedding` | Description |
|---------------|------|----------|---------------------|------------------|---------------------------|-------------|
| `"kitten-tts-mini"` | Individual | Single-model TTS | No (defaults to `"default"`) | Yes | No | Ultra-lightweight English TTS (ONNX) |
| `"kitten-tts-micro"` | Individual | Single-model TTS | No | Yes | No | Compact English TTS variant |
| `"kitten-tts-nano"` | Individual | Single-model TTS | No | Yes | No | Smallest English TTS variant |
| `"chatterbox"` | Toolchain | Embed → ChatterboxTTS | No | No | Yes | English-only Chatterbox voice cloning |
| `"chatterbox_turbo"` | Toolchain | Embed → ChatterboxTurboTTS | No | No | Yes | English-only Chatterbox voice cloning with [paralinguistic tags](#paralinguistic-tags) |
| `"chatterbox_multilingual"` | Toolchain | Embed → ChatterboxMultilingualTTS | Yes (ISO code) | No | Yes | 23-language Chatterbox |
| `"openvoice_v1"` | Toolchain | VAD → Encoder → Synthesizer → ToneConverter | Yes | Yes | Yes | OpenVoice V1 voice cloning |
| `"openvoice_v2"` | Toolchain | VAD → Encoder → MeloTTS → ToneConverter | Yes | Yes | Yes | OpenVoice V2 voice cloning |
| `"harmonyspeech"` | Toolchain | Encoder → Synthesizer → Vocoder | No | No | Yes | English-only voice cloning (always requires a reference voice; no `language` parameter) |

> **`input_embedding` support:** Models marked "Yes" accept a pre-computed speaker embedding (from `POST /v1/embed/speaker`) via the `input_embedding` field. This skips the VAD and embedding-extraction steps, making voice-cloning requests significantly faster. KittenTTS does not support embeddings — it uses fixed preset voices only.

#### TTS Mode Selection

The `mode` parameter is **required** and controls which pipeline steps execute:

| Mode | Description | `input_audio` / `input_embedding` | Pipeline (OpenVoice example) |
|------|-------------|-----------------------------------|------------------------------|
| `"single_speaker_tts"` | Synthesize using a preset voice. No reference audio needed. | Must **not** be provided | Synthesizer only (skips VAD, embedding, tone conversion) |
| `"voice_cloning"` | Clone a target voice from reference audio or embedding. | **Required** (at least one) | VAD → Embedding → Synthesizer → Tone Conversion |

**Mode rules for Chatterbox models** (enforced by the server):
- `voice_cloning` **requires** `input_audio` or `input_embedding`
- `single_speaker_tts` **rejects** `input_audio` and `input_embedding`

#### TTS Language and Voice

For models with multiple languages, the `language` parameter selects the language, and `voice` selects a preset voice within that language. Both are validated against the model's configuration. Call [`GET /v1/audio/voices`](#voices-catalog) to discover every model's selectable voices — native presets and standard pack names, each labeled by source. `GET /v1/audio/speech/models` keeps the per-model language and voice card view.

**OpenVoice V1 (`"openvoice_v1"`):**

| `language` | Valid `voice` values |
|------------|----------------------|
| `"EN"` | `"default"`, `"whispering"`, `"shouting"`, `"excited"`, `"cheerful"`, `"terrified"`, `"angry"`, `"sad"`, `"friendly"` |
| `"ZH"` | `"default"` |

**OpenVoice V2 / MeloTTS (`"openvoice_v2"`):**

| `language` | Valid `voice` values |
|------------|----------------------|
| `"EN"` | `"EN-Newest"` |
| `"ZH"` | `"ZH"` |
| `"ES"` | `"ES"` |
| `"FR"` | `"FR"` |
| `"JA"` | `"JP"` |

**Chatterbox Multilingual (`"chatterbox_multilingual"`):**

23 languages supported (no preset voices — uses `voice_cloning` with reference audio):

`"ar"` (Arabic), `"da"` (Danish), `"de"` (German), `"el"` (Greek), `"en"` (English), `"es"` (Spanish), `"fi"` (Finnish), `"fr"` (French), `"he"` (Hebrew), `"hi"` (Hindi), `"it"` (Italian), `"ja"` (Japanese), `"ko"` (Korean), `"ms"` (Malay), `"nl"` (Dutch), `"no"` (Norwegian), `"pl"` (Polish), `"pt"` (Portuguese), `"ru"` (Russian), `"sv"` (Swedish), `"sw"` (Swahili), `"tr"` (Turkish), `"zh"` (Chinese)

**KittenTTS models:**

No `language` parameter needed. Valid `voice` values: `"Bella"`, `"Jasper"`, `"Luna"`, `"Bruno"`, `"Rosie"`, `"Hugo"`, `"Kiki"`, `"Leo"`

**Chatterbox, Chatterbox Turbo, and HarmonySpeech:**

These models are **English-only** and use **voice cloning exclusively** — they do not expose a `language` parameter and have no native voice presets. A reference voice must be supplied via `input_audio`, `input_embedding`, or a [standard voice pack name](#standard-voice-pack-names) in `voice` (see [Mode Selection](#tts-mode-selection)).

- `"chatterbox"` — standard Chatterbox voice cloning
- `"chatterbox_turbo"` — faster variant; additionally supports [paralinguistic tags](#paralinguistic-tags)
- `"harmonyspeech"` — Harmony Speech V1 voice cloning pipeline

> Only `"chatterbox_multilingual"`, `"openvoice_v1"`, and `"openvoice_v2"` accept the `language` parameter. All other TTS models are single-language (English) and do not expose it.

#### Standard voice pack names

The engine ships 28 named reference speakers. On a voice-cloning model, `voice`
may be one of these names instead of a native preset; the server resolves it to
the shipped reference clip and enters the normal cloning path. A pack name also
implies `mode: "voice_cloning"`, so it can be combined with either declared mode
and does not need `input_audio`.

```
Abigail  Adrian    Alexander  Alice    Austin   Axel     Connor   Cora
Elena    Eli       Emily      Everett  Gabriel  Gianna   Henry    Ian
Jade     Jeremiah  Jordan     Julian   Layla    Leonardo Michael  Miles
Olivia   Ryan      Taylor     Thomas
```

Accepted by: `chatterbox`, `chatterbox_turbo`, `chatterbox_multilingual`,
`openvoice_v1`, `openvoice_v2`, `harmonyspeech`, and the voice-conversion
endpoint as `target_voice` — both `chatterbox_vc` (target audio) and the
OpenVoice V1/V2 tone converters (target embedding). Single-speaker models
(MeloTTS, KittenTTS, concrete OpenVoice/Melo synthesizers) list native voices
only; a pack name on them is rejected with an error naming their native voices
— it is never silently mapped to a native voice.

The same name is the same *reference clip* everywhere, but each family embeds
it in its own embedding space: expect the same speaker identity, not identical
audio. Do not combine a pack name with `input_audio`/`input_embedding` (or
`target_audio`/`target_embedding`); the server rejects the ambiguity. The
`standard_voice_pack: true` config switch (default on) controls name resolution;
native voices and explicit references are unaffected when it is off. See
[Models](models.md#standard-voice-pack) for the full pack table.

#### Paralinguistic Tags

`"chatterbox_turbo"` supports **paralinguistic tags** — effect and emotion tokens that can be embedded directly in the `input` text to shape the delivery of the synthesized speech. The tags are parsed out of the text before synthesis; no special request field is needed.

Supported tags (19, sourced from the model's tokenizer):

```
[advertisement] [angry]      [chuckle]     [clear throat] [cough]
[crying]        [dramatic]   [fear]        [gasp]         [groan]
[happy]         [laugh]      [narration]   [sarcastic]    [shush]
[sigh]          [sniff]      [surprised]   [whispering]
```

**Example — emotional speech with tags:**

```bash
curl -X POST "http://localhost:12080/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "chatterbox_turbo",
    "mode": "voice_cloning",
    "input": "[whispering] Did you hear that? [gasp]",
    "input_audio": "base64_encoded_reference_audio"
  }'
```

> Tag availability is model-specific. Future models may add support; check the engine catalog (`GET /v1/audio/speech/models`) or your gateway's model metadata to confirm which models accept paralinguistic tags.

#### Examples

**Single-speaker TTS (KittenTTS):**

```bash
curl -X POST "http://localhost:12080/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "kitten-tts-mini",
    "mode": "single_speaker_tts",
    "input": "Hello, this is a test of the speech engine.",
    "voice": "Bella",
    "output_options": {
      "format": "wav",
      "sample_rate": 22050
    }
  }'
```

**Single-speaker TTS (OpenVoice V2 / MeloTTS — English):**

```bash
curl -X POST "http://localhost:12080/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "openvoice_v2",
    "mode": "single_speaker_tts",
    "input": "Hello, this is a test of the speech engine.",
    "language": "EN",
    "voice": "EN-Newest"
  }'
```

**Voice cloning (Chatterbox):**

```bash
curl -X POST "http://localhost:12080/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "chatterbox",
    "mode": "voice_cloning",
    "input": "This voice will be cloned from the reference audio.",
    "input_audio": "base64_encoded_reference_audio",
    "generation_options": {
      "exaggeration": 0.5,
      "cfg_weight": 0.5,
      "temperature": 0.8
    }
  }'
```

**Voice cloning with a multipart upload (no base64):**

```bash
curl -X POST "http://localhost:12080/v1/audio/speech" \
  -F "input_audio=@reference.wav" \
  -F "model=chatterbox" \
  -F "mode=voice_cloning" \
  -F "input=This voice will be cloned from the reference audio."
```

**Voice cloning (OpenVoice V2 — full pipeline with embedding):**

```bash
curl -X POST "http://localhost:12080/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "openvoice_v2",
    "mode": "voice_cloning",
    "input": "This text will be spoken in the cloned voice.",
    "language": "EN",
    "voice": "EN-Newest",
    "input_embedding": "base64_encoded_speaker_embedding"
  }'
```

> **Tip:** Pre-compute embeddings via `POST /v1/embed/speaker` and pass them as `input_embedding` to skip the VAD and embedding extraction steps on every TTS call. This is significantly faster than passing `input_audio` each time.

#### Response (TextToSpeechResponse)

```json
{
  "id": "tts-a1b2c3d4",
  "model": "chatterbox",
  "created": 1720000000,
  "data": "base64_encoded_audio_output"
}
```

**GET /v1/audio/speech/models**

Returns the list of available TTS models configured in the engine, including their supported languages and voices. Use this endpoint to dynamically discover valid `language` and `voice` values for each model.

---

### Speaker Embedding

**POST /v1/embed/speaker**

Creates a speaker embedding from a reference audio file. The resulting embedding can be passed as `input_embedding` to TTS or voice conversion requests, avoiding the need to send the reference audio on every call.

**Request Body Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model` | string | Yes | - | Name of the embedding model (e.g. `"chatterbox_embedding"`, `"ov1-tone-converter-encoder"`) |
| `input_audio` | string | Yes | - | Base64-encoded audio of the speaker to embed |

**Example Request:**

```bash
curl -X POST "http://localhost:12080/v1/embed/speaker" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "chatterbox_embedding",
    "input_audio": "base64_encoded_speaker_audio"
  }'
```

**Multipart alternative:** `curl -X POST .../v1/embed/speaker -F "file=@speaker.wav" -F "model=chatterbox_embedding"`

**Response (EmbedSpeakerResponse):**

```json
{
  "id": "embed-a1b2c3d4",
  "model": "chatterbox_embedding",
  "created": 1720000000,
  "data": "base64_encoded_embedding_data"
}
```

**GET /v1/embed/models**

Returns the list of available speaker embedding models.

---

### Speech-to-Text (Transcription)

**POST /v1/audio/transcriptions**

Transcribes speech from an audio file into text. Supports language detection and word-level timestamps.

Based on the [OpenAI Transcription API](https://platform.openai.com/docs/api-reference/audio/createTranscription).

**Request Body Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model` | string | Yes | - | Name of the transcription model (e.g. `"faster-whisper-large-v3-turbo"`, `"faster-whisper-tiny"`) |
| `input_audio` | string | Yes | - | Base64-encoded audio data to transcribe |
| `language` | string | No | null | Source-language hint (ISO 639-1, e.g. `"en"`, `"de"`). Omitted → detected automatically. |
| `prompt` | string | No | null | Optional initial prompt (vocabulary/style) forwarded to the model. |
| `response_format` | string | No | `json` | `json`, `text`, `srt` or `vtt`. `srt`/`vtt` are served as `text/plain`. Unknown values are rejected with HTTP 400. |
| `get_language` | boolean | No | false | Whether to return the detected source language tag (JSON format only) |
| `get_timestamps` | boolean | No | false | Whether to return word-level timestamps (JSON format only) |

`response_format` values:

| Value | Response |
|---|---|
| `json` | The `SpeechToTextResponse` object below (default; unchanged) |
| `text` | The plain transcript as `text/plain` |
| `srt` | SubRip subtitles (`HH:MM:SS,mmm` timings) as `text/plain` |
| `vtt` | WebVTT subtitles (`WEBVTT` header, `HH:MM:SS.mmm` timings) as `text/plain` |

Subtitle timings are rendered from the model's segment timestamps (rounded to the
nearest millisecond); an empty transcript produces an empty document, not an
error. Cue text keeps its internal line breaks, and a literal `-->` in the
transcript is rendered as `- ->` so it cannot be mistaken for a timing line.


**Example Request:**

```bash
curl -X POST "http://localhost:12080/v1/audio/transcriptions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "faster-whisper-large-v3-turbo",
    "input_audio": "base64_encoded_audio_data",
    "get_language": true,
    "get_timestamps": true
  }'
```

**Multipart alternative (OpenAI style):** `curl -X POST .../v1/audio/transcriptions -F "file=@speech.mp3" -F "model=faster-whisper-large-v3-turbo" -F "get_language=true"`

**Response (SpeechToTextResponse):**

```json
{
  "id": "stt-a1b2c3d4",
  "model": "faster-whisper-large-v3-turbo",
  "created": 1720000000,
  "text": "Hello, this is a test recording.",
  "language": "en",
  "timestamps": [
    {"word": "Hello", "start": 0.0, "end": 0.5},
    {"word": "this", "start": 0.6, "end": 0.8}
  ]
}
```

**GET /v1/audio/transcriptions/models**

Returns the list of available transcription models.

---

### Speech Translation

**POST /v1/audio/translations**

Translates speech in an audio file into **English** text. Mirrors the
transcription endpoint: the same request model, upload aliases (`file`, `audio`,
`input_audio`), `response_format` values and `text/plain` subtitle output. The
same Whisper pipeline runs with `task=translate`; `language` is a *source*
hint only (the output language is always English, as in the OpenAI API).

Based on the [OpenAI Translation API](https://platform.openai.com/docs/api-reference/audio/createTranslation).

**Request Body Parameters:** identical to the transcription endpoint above
(`model`, `input_audio`/file upload, `language` source hint, `prompt`,
`response_format`, `get_language`, `get_timestamps`).

**Example Request:**

```bash
curl -X POST "http://localhost:12080/v1/audio/translations" \
  -F "file=@german_speech.mp3" \
  -F "model=faster-whisper-large-v3-turbo" \
  -F "response_format=vtt"
```

**Response:** the same formats as transcription — `json` returns a
`SpeechToTextResponse` whose `text` is the English translation and whose
`language` (with `get_language`) is the detected source language; `text`, `srt`
and `vtt` return `text/plain`.

---

### Voice Conversion

**POST /v1/voice/convert**

Converts the voice in a source audio to sound like a target speaker. The target can be provided as either a reference audio clip or a pre-computed speaker embedding.

**Request Body Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model` | string | Yes | - | Name of the voice conversion model (e.g. `"ov2-tone-converter"`, `"chatterbox_vc"`) |
| `source_audio` | string | Yes | - | Base64-encoded audio of the voice to convert |
| `target_audio` | string | No | null | Base64-encoded reference speaker audio to clone |
| `target_embedding` | string | No | null | Base64-encoded speaker embedding (faster than `target_audio`) |
| `target_voice` | string | No | null | A [standard voice pack name](#standard-voice-pack-names) (e.g. `"Abigail"`) used as the conversion target. Supported by target-audio VC models (`"chatterbox_vc"`, the clip becomes `target_audio`) and by OpenVoice V1/V2 tone converters (the engine embeds the clip once through the pipeline encoder and uses the result as `target_embedding`); cannot be combined with `target_audio` or `target_embedding` |
| `generation_options` | [GenerationOptions](#generationoptions) | No | null | Generation parameters |
| `output_options` | [AudioOutputOptions](#audiooutputoptions) | No | null | Output audio format settings |
| `pre_processing_filters` | list | No | `[]` | Filters to apply before conversion |

> **Note:** Provide exactly one target: `target_audio`, `target_embedding`, or `target_voice`. `target_voice` needs either a target-audio VC model (`chatterbox_vc`) or an OpenVoice tone-converter pipeline with its encoder and VAD configured. The OpenVoice embedding uses the same encoder artifact (and warm cache) as the TTS clone path, so a pack voice warmed at boot costs no encoder forward here.

**Example Request:**

```bash
curl -X POST "http://localhost:12080/v1/voice/convert" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "ov2-tone-converter",
    "source_audio": "base64_encoded_source_voice",
    "target_audio": "base64_encoded_target_speaker",
    "output_options": {
      "format": "wav"
    }
  }'
```

**Multipart alternative:** `curl -X POST .../v1/voice/convert -F "source_audio=@source.wav" -F "target_audio=@target.wav" -F "model=ov2-tone-converter"`

**Response (VoiceConversionResponse):**

```json
{
  "id": "vc-a1b2c3d4",
  "model": "ov2-tone-converter",
  "created": 1720000000,
  "data": "base64_encoded_converted_audio"
}
```

**GET /v1/voice/convert/models**

Returns the list of available voice conversion models.

---

### Voice Activity Detection (VAD)

**POST /v1/audio/vad**

Detects human speech activity in a provided audio file. Supports both FasterWhisper and Silero VAD models.

**Request Body Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model` | string | Yes | - | The VAD model to use (e.g. `"silero-vad"`, `"faster-whisper"`) |
| `input_audio` | string | Yes | - | Base64-encoded audio data |
| `get_timestamps` | boolean | No | false | Whether to return speech segment timestamps |
| `threshold` | float | No | 0.5 | Speech detection sensitivity (Silero VAD only) |
| `min_speech_duration_ms` | integer | No | 250 | Minimum speech chunk duration in milliseconds (Silero VAD only) |
| `min_silence_duration_ms` | integer | No | 100 | Minimum silence duration to separate speech chunks in milliseconds (Silero VAD only) |
| `speech_pad_ms` | integer | No | 30 | Padding around speech segments in milliseconds (Silero VAD only) |
| `return_seconds` | boolean | No | false | Return timestamps in seconds instead of milliseconds (Silero VAD only) |

**Example Request:**

```bash
curl -X POST "http://localhost:12080/v1/audio/vad" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "silero-vad",
    "input_audio": "base64_encoded_audio_data",
    "get_timestamps": true,
    "threshold": 0.3,
    "min_speech_duration_ms": 500
  }'
```

**Multipart alternative:** `curl -X POST .../v1/audio/vad -F "file=@speech.wav" -F "model=silero-vad" -F "get_timestamps=true"`

**Response (DetectVoiceActivityResponse):**

```json
{
  "id": "stt-a1b2c3d4",
  "model": "silero-vad",
  "created": 1720000000,
  "speech_activity": true,
  "timestamps": [
    {"start": 100, "end": 1500},
    {"start": 2000, "end": 3500}
  ]
}
```

**GET /v1/audio/vad/models**

Returns the list of available VAD models.

---

### Audio Conversion

**POST /v1/audio/convert**

Applies audio processing models (e.g. denoising, enhancement, restoration) to an input audio file. This is the base endpoint for models that transform audio without text or voice conversion semantics.

**Request Body Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model` | string | Yes | - | Name of the audio model (e.g. `"voicefixer-restorer"`, `"voicefixer-vocoder"`) |
| `source_audio` | string | Yes | - | Base64-encoded audio data to process |
| `output_options` | [AudioOutputOptions](#audiooutputoptions) | No | null | Output audio format settings |
| `pre_processing_filters` | list | No | `[]` | Filters to apply before processing |

**Example Request:**

```bash
curl -X POST "http://localhost:12080/v1/audio/convert" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "voicefixer-restorer",
    "source_audio": "base64_encoded_noisy_audio",
    "output_options": {
      "format": "wav"
    }
  }'
```

**Multipart alternative:** `curl -X POST .../v1/audio/convert -F "file=@noisy.wav" -F "model=voicefixer-restorer"`

**Response (AudioConversionResponse):**

```json
{
  "id": "ac-a1b2c3d4",
  "model": "voicefixer-restorer",
  "created": 1720000000,
  "data": "base64_encoded_enhanced_audio"
}
```

**GET /v1/audio/convert/models**

Returns the list of available audio conversion models.


## Error Handling

All endpoints return a standard error response on failure:

```json
{
  "object": "error",
  "message": "Descriptive error message",
  "type": "error_type",
  "param": null,
  "code": 400
}
```

Common HTTP status codes:

| Code | Meaning |
|------|---------|
| 200 | Success |
| 400 | Bad request (validation error, missing required field) |
| 401 | Unauthorized (missing or invalid API key) |
| 404 | Model not found |
| 500 | Internal server error (model inference failure) |
| 503 | Service unavailable (engine unhealthy) |

### Rejected payloads and timeouts

- **Unsupported embeddings.** `input_embedding` / `target_embedding` must be the
  base64 embedding produced by the matching embedding step. Payloads that do not
  deserialize to plain tensors are rejected with `400` — they are never
  interpreted as executable code.
- **Missing voice conditioning.** A Chatterbox TTS request in `voice_cloning`
  mode must include `input_audio` or a precomputed `input_embedding`; otherwise
  it fails with `400` instead of reusing another request's voice.
- **Per-request timeout.** Requests that exceed the engine's `request_timeout_s`
  (default 120 s, override with `--request-timeout-s` or the `request_timeout_s`
  config key) resolve with an error mentioning the timeout instead of hanging.


## Voice Cloning Pipeline

For voice cloning with OpenVoice V2, a typical pipeline chains multiple endpoints:

1. **VAD** (`POST /v1/audio/vad`) — detect speech segments in the reference audio
2. **Embed** (`POST /v1/embed/speaker`) — extract a speaker embedding from the reference
3. **TTS** (`POST /v1/audio/speech`) — synthesize text using the embedding, or
   **VC** (`POST /v1/voice/convert`) — convert an existing audio to the target voice

The `input_vad_data` field on the TTS endpoint accepts the VAD response so the model can properly segment the reference audio for embedding extraction.


## Helpful Resources

A recently generated OpenAPI definition file and scripts to generate our Golang and JavaScript clients can be found
in [docs/api](api)

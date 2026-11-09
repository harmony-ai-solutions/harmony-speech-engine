# Changelog

## Unreleased

### Breaking Changes

*   **VRAM headroom now means true free device memory**: the `hse_vram_headroom_bytes` metric used to be "device total minus PyTorch's reserved memory", which overstated free memory whenever a model that does not use PyTorch's allocator (Faster-Whisper/CTranslate2) held device memory. It now reports the device-level free memory, and a new `hse_vram_device_used_bytes` gauge reports device-level used memory. Dashboards that relied on the old definition should switch to the torch-scoped `hse_vram_reserved_bytes` gauge or the new device-level pair. The per-model `hse_model_weights_bytes` gauge now also reports the checkpoint size for native Whisper models.

### New Features

*   **Voice discovery endpoint**: a new `GET /v1/audio/voices` lists every configured model's selectable voices in one place, so clients no longer need to read the server configuration. Each model or toolchain entry reports whether it can clone from a reference and, for voice conversion, whether a target voice is consumed as audio or as an embedding; every voice is labeled as a native preset or a standard voice pack identity, with its language where applicable. `?model=<name>` narrows the response to one id, and an unknown id is rejected with a clear error. The listing is available immediately at startup — it does not wait for pack-voice warm-up — and a server with the standard voice pack switched off simply omits the pack names. See [API reference](docs/api.md#voices-catalog).

*   **Standard voice pack embeddings survive restarts**: the engine now keeps each standard voice's per-family conditioning artifact in a persistent on-disk cache and restores it into the in-memory cache at startup, so restarting the server no longer re-embeds every pack voice. A configurable background warm-up fills any missing entries after boot without blocking or failing startup, one model at a time; a voice that fails to warm is skipped and computed on demand as before, and a real request that already computed a voice is reused rather than recomputed. The cache is invalidated automatically when a reference clip or an embedding parameter changes, writes are atomic, and a corrupt cache entry is ignored and recomputed. Enabled with `voice_pack_warmup` in both shipped configs; `voice_pack_cache_dir` (or `HSE_VOICE_PACK_CACHE_DIR`) selects the location, defaulting to the platform user-cache directory. See [models](docs/models.md#standard-voice-pack) for how to clear it.

*   **Whisper transcribes concurrent requests in parallel with fair lanes**: a burst of speech-to-text requests for the same Whisper model used to be processed strictly one after another, so short clips queued behind long recordings. Whisper models now run a configurable number of CTranslate2 workers on the shared weights (`num_workers`, initial 2 for large-v3-turbo and 4 for tiny, pending benchmark tuning; default 1 keeps the previous behaviour), dispatch each processing step across exactly that many workers, admit requests from two fairness lanes (clips up to 30 s vs longer audio at a 3:1 ratio) and expose workers, in-flight transcribes, lane depths, queue wait and CTranslate2 queue state on `/metrics`. An aborted request now stops between transcription segments instead of after the whole recording.

*   **Whisper can decode the whole queue in shared window batches (opt-in)**: with `batched_inference: true` on a Whisper model, requests that arrive in the same processing step now share one batched encoder/decoder call instead of decoding one after another, with automatic grouping of their 30-second windows (up to 8 per call) and per-request language prompts. Each request still gets its own transcript, language, segment timestamps and segment text; a request that fails on its own audio does not affect the others. The previous per-request behaviour remains the default and the fallback. Batcher failures and fallback counts are visible on `/metrics` (`hse_whisper_batch_fallbacks_total`), and admission now accounts for a long recording's real number of 30-second windows. Real-model throughput and parity figures are recorded by the benchmark workstream.

*   **Whisper memory is measured and bounded**: Faster-Whisper (CTranslate2) memory was invisible to the engine's VRAM accounting, so the reported headroom could not be trusted while Whisper was resident. The engine now measures each CUDA Whisper config's real max-shape footprint at startup (concurrent workers, window batch and the longest admitted audio), reports it per model on `/health` and `/metrics`, warns (or refuses to start with `whisper_vram_budget_bytes`) when the projected footprint does not fit the free device memory, and throttles concurrent Whisper work at runtime as device memory fills. `whisper_max_audio_seconds` defines the Whisper request-size cap. See [VRAM profiles](docs/benchmarks/vram-profiles.md#phase-10-3--device-level-truth) and [models](docs/models.md#whisper-vram-governance).

*   **Subtitles and translation for speech-to-text**: transcription now accepts a `response_format` (`json`, `text`, `srt`, `vtt`), so captioning clients can receive SubRip or WebVTT subtitle documents directly instead of converting JSON themselves; the default JSON response is unchanged. A new `/v1/audio/translations` endpoint translates speech into English using the same models, request fields and response formats, with `language` acting only as a source-language hint. Unknown response formats are rejected before any inference runs. See [API reference](docs/api.md#speech-translation) and the [audio I/O contract](docs/audio-io-contract.md#stt-response-formats).

*   **Output format and sample rate are now honored**: `output_options.format` (`wav`, `flac`, `ogg`, `mp3`, `opus`, `pcm`, and `aac` when ffmpeg is installed) and `output_options.sample_rate` are applied at the API boundary for speech, voice-conversion and audio-conversion endpoints. Invalid values are rejected before synthesis instead of being silently ignored. Responses now report the effective `sample_rate` when the audio was converted.
*   **Declarative pipeline routing**: every model toolchain is now described once, declaratively, and both request routing and step-to-step forwarding are driven by that description. Adding a new toolchain no longer requires editing routing code, model-advertisement lists and validation in several places, and the routing behaviour is covered by a transcribed parity test suite.
*   **Selectable VAD backend for voice cloning**: `input_vad_mode: "silero"` now genuinely routes the OpenVoice reference-audio segmentation to Silero VAD (falling back to Faster-Whisper with a warning when no Silero model is configured). Previously the field had no effect.
*   **Single-speaker TTS is a proper entry point**: `mode: "single_speaker_tts"` starts at the synthesis step and finishes there instead of relying on field coercion.
*   **Strict pipeline validation**: `--strict-pipelines` (or `HSE_STRICT_PIPELINES=1`) turns "incomplete toolchain" startup warnings into a boot error. Incomplete toolchains are also reported with the exact missing step.
*   **Richer model cards**: toolchain entries on the model endpoints now report their capabilities (speech, embedding, voice conversion, transcription, VAD, restoration), their member executors and the languages of the configured variants. Existing fields are unchanged.
*   **Raw PCM output**: `format: "pcm"` returns base64-encoded signed 16-bit little-endian mono samples for clients that decode audio themselves.
*   **Broader audio input support**: WAV, FLAC, OGG and MP3 are decoded natively, and AAC/M4A/WebM/AMR are accepted when ffmpeg is available. This applies to every audio input, including Chatterbox voice-cloning reference audio.
*   **Multipart audio uploads**: every audio-input endpoint now also accepts `multipart/form-data` file uploads (OpenAI-style `file=@audio.mp3`) on the same path, alongside the existing JSON/base64 bodies. Oversized requests are rejected with HTTP 413, configurable via `--max-upload-size-mb` or `HARMONYSPEECH_MAX_UPLOAD_MB` (default 50 MiB).
*   **Clear decode errors**: undecodable audio now produces a descriptive HTTP 400 naming the request field, the detected container and the supported formats.

*   **Bounded, observable memory for autoregressive speech generation**: token-level autoregressive models need a key/value cache per request, and the engine had no shared budget for it. A configurable, device-wide pool now reserves that memory up front (`ar_inference.kv_cache_budget_bytes`, initial 1 GiB, and `ar_inference.default_cap_tokens`; overridable with `--ar-cache-budget-bytes`, disable-able with `--no-ar-cache`), validates every model's cache layout against the remaining budget at startup, and reports its footprint, slot usage and grant/refusal counts on `/health` and `/metrics`. The pool is the foundation for batched autoregressive decoding; a full pool refuses new grants instead of allocating, so it can never cause an out-of-memory failure. See [AR KV-cache pool](docs/ar-cache.md).

*   **Chatterbox vocodes finished speech together**: after a batch of Chatterbox requests finishes autoregressive decoding, their speech-to-audio conversion used to run one request at a time, keeping throughput flat under concurrent load. Rows that finish in the same processing step now share one padded conversion pass, and each request's audio is reproducible from its seed whether it is generated alone or alongside other requests. A row that fails is isolated to its own request, and the previous per-request conversion remains the fallback. The Turbo model's conversion path stays per request.

*   **Optional Chatterbox mixed-precision vocoder (opt-in experiment)**: `HSE_S3GEN_AUTOCAST=float16` runs the Chatterbox vocoder's flow-matching decoder under CUDA mixed precision while its weights stay float32. Measured on the benchmark host: batched voice conversion rises from 4.10 to 6.19 requests/s at eight concurrent requests, with output inside the audio-regression audibility bounds on all four Chatterbox families. `HSE_S3GEN_AUTOCAST=bfloat16` fails that gate and is not usable for speech. The gate is **off by default** and no shipped config sets it; see [S3Gen autocast](docs/benchmarks/throughput.md#phase-11-2--s3gen-autocast) for the numbers and limits.

*   **S3Gen vocode precision is a per-model config knob (`s3gen_precision`)**: the mixed-precision vocoder above is now a first-class, honest config value instead of only an environment gate. `s3gen_precision: "float16"` opts one Chatterbox model into the float16 S3Gen autocast (weights stay float32), while `"float32"` — the default — keeps the pre-experiment behaviour byte-identical. It applies per config entry to the Chatterbox families whose vocode uses S3Gen (TTS, multilingual, VC, turbo), so a mixed deployment stays honest; `ChatterboxEmbedding` ignores it. The effective value is visible per model on `/health` (`models.<name>.s3gen_precision`). `HSE_S3GEN_AUTOCAST` still overrides the config when set (bench/testing escape hatch). Since the owner-approved flip of 2026-11-08 (seeded listening pass), the four S3Gen-vocode families in `config.gpu.yml` ship `s3gen_precision: "float16"`; the CPU config stays unset. The value vocabulary matches `dtype` (`float32`/`float16`). See [Precision](docs/models.md#s3gen-precision-s3gen_precision-phase-11-3).

*   **Long-form synthesis is split, batched and stitched**: `generation_options` accepts `split_text`, `chunk_size`, `crossfade_ms` and `sentence_pause_ms`, and the serving layer now uses them. A long input is split sentence-aware (Latin and CJK enders, abbreviations, numbers and bullets preserved; parenthetical prose and inline bracketed cues stay inside their sentence rather than forming split boundaries), every chunk is submitted as an ordinary admitted request **concurrently** so one long request can fill an engine batch by itself, and the results are stitched in chunk order with equal-power crossfades and optional sentence pauses. When a pause is set, the outgoing chunk's conditioned tail fades out into the silence (`cos²`) and the incoming chunk's conditioned head fades in from it (`sin²`), so no join is hard-cut. Every chunk's edges are conditioned before joining (near-silence trimmed to a 10 ms guard, gate 45 dB below the chunk's own peak, at most 150 ms per edge), so joins are uniform regardless of synthesis-native edge silence. When chunking is on and the client leaves `crossfade_ms`/`sentence_pause_ms` unset, the tuned defaults are **40 ms** and **120 ms**; an explicit value (including 0) still wins. Voice-cloning requests embed the reference voice once and reuse it for every chunk; the audio watermark is applied exactly once, on the stitched output; `speed` is applied after stitching only for families that do not consume it natively; and a failing chunk fails the whole request instead of returning partial audio. Requests that split into 0 or 1 chunk (and every request that omits the fields) keep the previous single-shot path byte-identically. See [API reference](docs/api.md#generationoptions) and [batching](docs/batching.md).

*   **Opt-in output cleanup filters (DC offset, silence trim, internal pauses)**: `generation_options` accepts three independent model-agnostic booleans, all defaulting to `false` (omitting them is byte-identical to before). `remove_dc_offset` subtracts each channel's mean — **per chunk before stitching** in the long-form path, so per-run offsets never smear a chunk join. `trim_silence` removes leading/trailing silence with an energy gate at 40 dB below the loudest 32 ms frame, keeping 50 ms of padding. `fix_internal_silence` collapses internal pauses longer than 700 ms to at most 300 ms without touching the buffer's ends (a chunk boundary is never mistaken for lead/trail silence). Trim and internal-silence run on the final stitched buffer, before the post-stitch speed stretch and the once-per-request watermark. The filters are pure numpy (no new dependency); thresholds are documented presets with no per-request knobs. See [API reference](docs/api.md#generationoptions).

*   **Named standard voices work on every voice-cloning pipeline**: the engine now ships 28 ready-to-use reference speakers (vendored from the MIT-licensed Chatterbox-TTS-Server project, keeping their original names such as `Abigail` and `Taylor`). On a voice-cloning model, `voice` may be one of these names instead of uploading a reference clip; the same name selects the same reference speaker on Chatterbox English, Turbo and Multilingual, OpenVoice V1/V2 and Harmony Speech, and voice conversion accepts it as `target_voice`. Each family embeds the clip in its own embedding space, so the result is the same speaker identity rather than identical audio. Single-speaker models (MeloTTS, KittenTTS) continue to list native voices only and reject a pack name with a clear error instead of silently substituting one. Explicit references and embeddings keep working unchanged, and the feature can be switched off with `standard_voice_pack: false`. See [Models](docs/models.md#standard-voice-pack) and [API reference](docs/api.md#standard-voice-pack-names).

*   **Standard voice names also work as voice-conversion targets on OpenVoice**: `target_voice` used to be limited to Chatterbox voice conversion. It now also selects a named reference speaker for the OpenVoice V1/V2 tone converters, where the engine derives the required target embedding from the shipped clip automatically. The embedding is shared with the text-to-speech clone path, so a voice warmed at startup (or already used for speech) is reused without extra work. See [Models](docs/models.md#standard-voice-pack) and [API reference](docs/api.md#voice-conversion).

### Security

*   **Voice-biometric retention is now explicit and configurable**: speaker characteristics derived from caller-supplied reference audio (the values that let the engine reproduce a voice) may be retained in process memory for a bounded time to avoid recomputing them. Retention is capped by age and by memory, is never written to disk, and can be tuned or switched off per deployment for environments with strict data-retention requirements. The behaviour and its tuning options are documented in the operations guide.

*   **Hardened payload deserialization**:
    *   Client-supplied voice embeddings are no longer interpreted with arbitrary code execution enabled; unsupported payloads are rejected with a clear error response instead of being executed.
    *   Model checkpoints are loaded in the same restricted mode. A documented opt-in is available for trusted checkpoints that genuinely require the unrestricted format.

### Performance

*   **Concurrent voice cloning and restoration are faster out of the box**: the shipped configurations now admit several requests per processing step for the OpenVoice voice-cloning, Harmony Speech and VoiceFixer toolchains. Previously those steps accepted one request at a time even though they could process a batch, so a burst of cloning or restoration requests is now processed together instead of queueing one by one. The per-model capacity is documented in the batching guide and can be tuned for a given GPU; the included throughput benchmark records the curves.
*   **Repeated voices are no longer recomputed**: cloning the same reference voice over and over (agent floods, batch jobs, a caller re-sending one voice) now reuses the previously computed speaker representation instead of re-running the embedding step on every request, removing that work from steady-state latency. The reuse is bounded by a default five-minute lifetime, an entry cap and a memory ceiling, is observable on `/health` and `/metrics` (hit ratio, misses, evictions) and can be tuned or disabled per deployment. Requests with distinct reference audio behave exactly as before.
*   **Batched synthesis for one-shot models**: OpenVoice V1 and MeloTTS requests that arrive together in one engine step are now synthesized as a single padded batch instead of one at a time, with automatic grouping of similar-length texts. Short-request throughput rises (~14 % at concurrency 8 on a RTX 5060 Ti) and tail latency drops; long-form requests are unaffected. A per-model `batched_inference` switch reverts to the previous behaviour.
*   **Voice-embedding steps batch as well**: when more than one voice-cloning request reaches the same speaker-embedding model in one engine step, their reference audio is embedded as one padded batch instead of one request at a time, for both the OpenVoice and the Harmony Speech toolchains. Each request still gets exactly the embedding it got before (differences are below float32 rounding), and requests whose voice is already cached still skip the step entirely. A per-model `batched_inference` switch reverts to the previous behaviour.
*   **Tone conversion batches too**: when more than one voice-conversion request reaches the same OpenVoice V1/V2 tone converter in one engine step, the clips are tone-converted as one padded batch instead of one at a time, with automatic grouping of similar-length clips. Each request still gets its own converted audio at its own length; padding never reaches the converted samples. A per-model `batched_inference` switch reverts to the previous behaviour.
*   **Vocoding and restoration batch as well**: when more than one request reaches the same Harmony Speech vocoder, VoiceFixer restorer or VoiceFixer vocoder in one engine step, the clips are processed as one padded batch instead of one at a time, with automatic grouping of similar-length clips. Every request keeps its own output length, its own inter-chunk silences and its own loudness normalization; padding never reaches a clip's samples. A per-model `batched_inference` switch reverts to the previous behaviour.
*   **KittenTTS serves concurrent requests in parallel**: KittenTTS cannot batch — its exported models accept exactly one request per run — so concurrent requests for the same model are now synthesized on separate cores through one shared model instead of one after another. Requests per second rise from 0.70 to 1.19 at the shipped admission width (4) and to 1.29 at 8 lanes on a 12-core host, while single-request latency is unchanged. The phoneme backend is not thread-safe, so phonemisation is serialised (a fraction of a millisecond per request), and the first request after startup no longer pays the graph warm-up.
*   **Concurrent voice conversion is processed together**: Chatterbox voice conversion used to tokenize each source clip and convert it one request at a time, so concurrent conversion clients only queued even when the model was configured to batch. Requests that arrive together now share one source-tokenization pass and one padded conversion pass, each still producing its own audio (identical whether generated alone or alongside other requests when a seed is given). A request that arrives alone, cannot be batched, or fails uses the previous one-at-a-time path, and a failed request is isolated to itself. The shipped configurations admit up to eight conversion requests per step.

*   **Concurrent Chatterbox requests are synthesized together**: Chatterbox text-to-speech (English and Turbo) used to decode one request at a time, so extra concurrent clients only queued. Requests that arrive together now share the decoding steps, each still reproducing its own voice and its own audio from its seed, and finished requests have their speech-to-audio conversion batched as well. A request that cannot get a slot, fails, or arrives alone uses the previous one-at-a-time path, so availability and output pairing are unchanged. The shipped configurations admit up to four requests per step, derived from the shared cache budget; `/health` and `/metrics` report the cache's footprint, slot use and grant/refusal counts so capacity can be planned honestly. Real-model throughput curves are recorded by the benchmark workstream.

*   **Truthful capacity signals**: `max_batch_size` now means "requests admitted per step **and processed together**". Steps that batch declare their batch capacity; steps that cannot batch say so and declare 1 (the Harmony Speech synthesizer), while Whisper and Silero declare their real admission/queue width and the lightweight KittenTTS declares its lane count. Chatterbox's value is its real autoregressive admission depth.
*   **Reproducible sampling controls**: OpenVoice V1 and MeloTTS synthesis accept `noise_scale`, `noise_scale_w` and `sdp_ratio` per request (setting them to 0 makes synthesis deterministic).
*   **Chatterbox runs in half precision where it is safe**: the large T3 backbone of the Chatterbox family can now load in float16 (the shipped GPU config does), roughly halving its memory. The vocoder and speaker encoder stay float32 because casting them was measured to audibly degrade synthesis. Other torch families keep float32 where their code is not half-precision safe, and that choice is documented per family.
*   **Chatterbox roles share their backbones**: `chatterbox`, `chatterbox_multilingual`, `chatterbox_vc` and `chatterbox_embedding` can now share one vocoder/speaker-encoder copy (opt in with `backbone_group`), instead of loading a full duplicate of the checkpoint per role. On the shipped GPU config the Chatterbox block drops from ~14.3 GiB to ~6.4 GiB.
*   **VRAM is observable**: `/health` reports per-model and shared-group memory, and `/metrics` exposes Prometheus gauges for per-model VRAM, reservation, resident model count and shared backbone bytes. The included measurement tooling records a full profile of a config.

*   **In-process audio pipeline**: intermediate audio, mel spectrograms, speaker embeddings and Chatterbox conditioning now stay in memory as typed objects between pipeline steps instead of being re-encoded as base64/JSON/WAV at every hop. On the worst hop (VoiceFixer restorer → vocoder) this removes roughly 25 ms of serialization and ~1 MiB of allocations per second of audio; the OpenVoice clone path now decodes the reference audio once instead of three times. API responses are unchanged.

### Fixes

*   **Chunked long-form joins no longer stutter**: the stitcher used to butt-splice chunks (or crossfade over whatever silence each synthesis run happened to emit), so ragged 10–300 ms edge silence produced an audible hiccup at every join. It now conditions each chunk's edges first — trimming near-silence to a 10 ms guard, gated 45 dB below that chunk's own peak and bounded at 150 ms per edge — and the chunked path defaults to a 40 ms equal-power crossfade with a 120 ms sentence pause when the client does not set them. Every join is now speech-end → fade → pause → fade → speech-start, with the gap determined by the pause rather than by the model's edge silence. Explicit values (including 0) are unchanged, and non-chunked requests stay byte-identical. See [API reference](docs/api.md#generationoptions).

*   **Chatterbox long-form output is faithful again**: long prose could lose words even with chunking on, but the cause was batching bugs, not a model context limit. The shared autoregressive KV-cache token capacity was too small for Chatterbox Turbo's long conditioning prompt, so 300-character chunks were truncated mid-sentence; raising it lets the whole chunk be spoken. Chatterbox Multilingual cut off early because the engine used a weaker repetition setting than the model expects and its attention hooks leaked; restoring the upstream repetition penalty and fixing the hook/EOS path removes the premature cutoffs and enables fast attention plus batched decoding and batched vocoding. All models now keep the unified **300-character** default chunk budget (the earlier Turbo 150 / Multilingual 75 workarounds are gone), an explicit `chunk_size` always wins, and a transcription-based text-fidelity gate catches word-dropping automatically. See [text fidelity](docs/benchmarks/throughput.md#text-fidelity-post-fix-phase-12-9--2026-11-09).

*   **VoiceFixer works on the GPU again**: audio restoration previously failed with a device-mismatch error as soon as a clip reached a CUDA-configured VoiceFixer model, because part of its audio-analysis frontend stayed on the CPU. The whole model is now placed on the configured device, so the restoration toolchain runs on GPU; CPU behaviour is unchanged.
*   **Health and metrics report engine telemetry reliably**: `/health` now includes the GPU-memory and cache blocks it advertises, and `/metrics` publishes the cache series even while the cache is disabled (as zeros), so dashboards never have to special-case a missing metric family.
*   **Requests above the batch limit are no longer dropped**: when more requests were waiting for a model than its `max_batch_size`, the surplus left the waiting queue without ever running or reporting an error, leaving the caller waiting until its timeout. They now stay queued for the next engine step.
*   **Mixed batches cannot duplicate a response**: a response is built at most once per request per engine step, and an inconsistent batch fails the affected requests with a clear error instead of corrupting the pairing or crashing the processing loop.
*   **Voice isolation between requests**: Speech conditioning is restored after every request, so a request can no longer be synthesized with the previous caller's voice.
*   **Missing conditioning is rejected**: Voice-cloning requests without reference audio or a precomputed embedding now fail with a clear error instead of reusing earlier state.
*   **Chatterbox default voice**: Text-to-speech requests without a reference now use the voice bundled with the model instead of failing.
*   **Correct batch pairing**: Mixed embedding and speech requests in one batch are executed by the right handler, and each result is always paired with its own request data.
*   **No hanging requests**: Requests for a model toolchain that is missing a configured member, or that exceeds its timeout, resolve with an error instead of waiting indefinitely.
*   **Forwarded requests are released**: requests handled by a single explicit model (or an unknown one) are now marked finished, so they no longer linger in the scheduler's running set after their response was returned.
*   **Per-request timeouts**: A slow request no longer stops the background processing loop; only the affected request errors out.
*   **Health reporting**: The health check now queries every loaded model and reports per-model detail.
*   **Startup toolchain validation**: Incomplete model toolchains are logged at startup and are no longer advertised by the model endpoints.

## v0.2.0

### New Features

*   **Chatterbox TTS Integration**:
    *   Added comprehensive support for Resemble AI's Chatterbox TTS family with 4 model variants: ChatterboxTTS, ChatterboxVC, ChatterboxMultilingualTTS, ChatterboxTurboTTS
    *   23 supported languages (multilingual variant)
    *   Voice cloning with watermarking (resemble-perth)
    *   Implemented embed-dispatch support for embedding-capable TTS models
    *   Added watermarker swap in loader for voice cloning workflows

*   **KittenTTS Integration**:
    *   ultra-lightweight ONNX-based English TTS with 4 model variants: mini (80MB), micro (41MB), nano-fp32 (56MB), nano-int8 (25MB)
    *   8 available voices: Bella, Jasper, Luna, Bruno, Rosie, Hugo, Kiki, Leo
    *   Uses misaki[en] phonemizer for text-to-speech conversion
    *   Native ONNX runtime integration (no torch dependency for inference)

### Improvements

*   **Unified Frontend Tooling Alignment**:
    *   Clarified and standardized frontend mode usage for Harmony Speech Engine across Docker, compose guidance, and frontend documentation.
    *   Updated UI release workflow behavior to explicitly build and publish the Speech Engine UI variant from the canonical unified frontend source.
    *   Added clearer local setup guidance for cloning and running the unified frontend in Speech Engine mode from this repository context.

### Fixes

*   **Container UI Runtime Build Reliability**:
    *   Improved Docker build context handling for the frontend image to avoid local dependency contamination that could break runtime UI builds.

*   **CI/Testing Improvements**:
    *   Fixed e2e patching and monkey patching for watermarking on CI runners
    *   Fixed caching and runner cleanup
    *   Fixed pkg_resources compatibility for Python >= 3.12
    *   Fixed requirement ordering
    *   Fixed missing package installations for chatterbox that break unit tests
    *   Fixed docker images and requirements

## v0.1.1-rc1

### New Features

*   **VoiceFixer Audio Restoration Integration**:
    *   Introduced comprehensive audio restoration capabilities through VoiceFixer model integration, enabling noise reduction and audio enhancement for speech processing workflows.
    *   Implemented two-stage restoration pipeline: VoiceFixerRestorer for audio enhancement and VoiceFixerVocoder for mel-spectrogram to audio conversion.
    *   Added `/v1/audio/convert` endpoint for audio conversion and filtering operations using VoiceFixer models.
    *   Voicefixer processes audio internally at 44.1kHz.
    *   Extended request routing system to support audio restoration workflows in addition to existing TTS, STT, and voice conversion pipelines.

*   **Silero VAD Integration**:
    *   Added Silero VAD as a high-performance alternative to FasterWhisper for Voice Activity Detection tasks.
    *   Implemented native model integration following HSE's established patterns for optimal CPU performance.
    *   Enhanced `/v1/audio/vad` endpoint with dynamic parameter support for fine-tuning detection sensitivity.
    *   Added configurable parameters: threshold, min_speech_duration_ms, min_silence_duration_ms, speech_pad_ms, and return_seconds.
    *   Provides better performance and lower false detection rates compared to adapted transcription models.

### Dependencies

*   **New Dependencies**:
    *   TorchLibrosa for STFT operations and spectrogram conversion
    *   silero-vad to support the silero VAD model

---

## v0.1.0 and Earlier

### Foundation

*   **Core Engine Architecture**:
    *   Established HSE core engine with modular architecture.
    *   Implemented base executor system for CPU and GPU model execution.
    *   Created standardized model loading and configuration system.

*   **Harmony Speech V1 Integration**:
    *   Initial integration of Harmony Speech encoder, synthesizer, and vocoder models.
    *   Basic TTS pipeline implementation with voice cloning support.
    *   Foundation for multi-model orchestration system.

*   **API Framework**:
    *   FastAPI-based REST API with automatic documentation generation.
    *   Basic request handling and response formatting.
    *   Initial OpenAI API compatibility layer.

*   **Frontend Development**:
    *   React-based management interface with modern development stack.
    *   TailwindCSS responsive design system.
    *   Interactive audio player and waveform visualization components.

*   **Multi-Model Request Routing System**:
    *   Implemented sophisticated request routing for complex workflows supporting VAD, embedding, synthesis, and voice conversion.
    *   Added intelligent model selection based on request characteristics and parameters.
    *   Enhanced request forwarding mechanism with proper status lifecycle management.

*   **OpenVoice V2/MeloTTS Integration**:
    *   Added support for multilingual synthesis in English, Chinese, Spanish, French, Japanese, and Korean.
    *   Implemented advanced voice cloning capabilities with tone conversion support.
    *   Added OpenVoice V1 and V2 synthesizer and encoder models.

*   **Faster-Whisper Speech Recognition**:
    *   Integrated Faster-Whisper models for high-performance speech recognition and VAD.
    *   Support for multiple model sizes: tiny, medium, large-v3-turbo.
    *   Optimized batch processing for improved transcription throughput.

*   **Docker-First Deployment**:
    *   Comprehensive Docker containerization for consistent deployment.
    *   Support for both CPU and GPU execution environments.
    *   Simplified dependency management and environment setup.

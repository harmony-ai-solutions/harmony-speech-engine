<!-- refreshed: 2026-11-02 -->
# Architecture

**Analysis Date:** 2026-11-02

## System Overview

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                            External Clients                                   │
│  (Harmony Link, Web UI, curl, other apps)                                     │
└────────────────────────────────┬─────────────────────────────────────────────┘
                                 │ HTTP (base64 audio in JSON)
                                 ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                    API Layer — FastAPI (`harmonyspeech/endpoints/openai`)      │
│  ┌───────────────┐ ┌────────────────┐ ┌────────────────┐ ┌─────────────────┐ │
│  │ TTS Serving   │ │ STT Serving    │ │ VC Serving     │ │ AudioConversion │ │
│  │ Embed Serving │ │ VAD Serving    │ │ ModelCard/     │ │ Auth middleware │ │
│  └───────┬───────┘ └───────┬────────┘ └───────┬────────┘ └─────────────────┘ │
│          │ _*_MODEL_TYPES / _*_MODEL_GROUPS (toolchain "embed-dispatch")     │
└──────────┴─────────────────┴─────────────────┴───────────────────────────────┘
                                 │ EngineRequest(request_data = RequestInput)
                                 ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                 Engine Layer (`harmonyspeech/engine/`)                        │
│  AsyncHarmonySpeech (async wrapper + RequestTracker + background loop)        │
│      └── _AsyncHarmonySpeech (subclass of HarmonySpeechEngine)                │
│  HarmonySpeechEngine: request rerouting + multi-step forward pipeline         │
└────────────────────────────────┬─────────────────────────────────────────────┘
                                 │ per-model queues
                                 ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│         Scheduler (`harmonyspeech/processing/scheduler.py`)                   │
│  one deque of waiting + one of running; per-model batch budget                │
└────────────────────────────────┬─────────────────────────────────────────────┘
                                 │ SchedulerOutputs.scheduled_requests_per_model
                                 ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│              Executor Layer (`harmonyspeech/executor/`)                       │
│  CPUExecutor · GPUExecutor · GPUExecutorAsync   (Ray executor is dead code)   │
└────────────────────────────────┬─────────────────────────────────────────────┘
                                 │ one worker per model config
                                 ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│        Worker / Model Runner (`harmonyspeech/task_handler/`)                  │
│  CPUWorker / GPUWorker → CPUModelRunner / GPUModelRunner                      │
│  ModelRunnerBase.execute_model(): prepare_inputs() → per-model `_execute_*`   │
└────────────────────────────────┬─────────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│            Modeling Layer (`harmonyspeech/modeling/`)                         │
│  loader.get_model() → ModelRegistry (_MODELS) or native loader branches       │
│  models/: harmonyspeech · openvoice · melo · kittentts · voicefixer ·         │
│           chatterbox                                                          │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| FastAPI app / routes | HTTP surface, auth middleware, CORS, OpenAPI customization | `harmonyspeech/endpoints/openai/api_server.py` |
| API protocol | Single source of truth for request/response schemas | `harmonyspeech/endpoints/openai/protocol.py` |
| Base serving class | Error responses, model-card construction from configs, per-model language/voice validation | `harmonyspeech/endpoints/openai/serving_engine.py` |
| TTS serving | `/v1/audio/speech` → `TextToSpeechRequestInput`, returns base64 WAV | `harmonyspeech/endpoints/openai/serving_text_to_speech.py` |
| Embedding serving | `/v1/embed/speaker` → `SpeechEmbeddingRequestInput` ("embed-dispatch") | `harmonyspeech/endpoints/openai/serving_voice_embed.py` |
| STT serving | `/v1/audio/transcriptions` → `SpeechTranscribeRequestInput` | `harmonyspeech/endpoints/openai/serving_speech_to_text.py` |
| VAD serving | `/v1/audio/vad` → `DetectVoiceActivityRequestInput` | `harmonyspeech/endpoints/openai/serving_voice_activity_detection.py` |
| Voice conversion serving | `/v1/voice/convert` → `VoiceConversionRequestInput` | `harmonyspeech/endpoints/openai/serving_voice_conversion.py` |
| Audio conversion serving | `/v1/audio/convert` → `AudioConversionRequestInput` | `harmonyspeech/endpoints/openai/serving_audio_conversion.py` |
| Async engine | Request tracker, background loop, stream fan-out, abort | `harmonyspeech/engine/async_harmonyspeech.py` |
| Core engine | Model rerouting, multi-step forwarding state machine, per-model parallel execution | `harmonyspeech/engine/harmonyspeech_engine.py` |
| Scheduler | Waiting/running queues, per-model batch budgets, status transitions | `harmonyspeech/processing/scheduler.py` |
| Executors | Device backend abstraction and one-worker-per-model ownership | `harmonyspeech/executor/` |
| Model runners | Input preparation dispatch and per-model inference (`_execute_*`) | `harmonyspeech/task_handler/model_runner_base.py` |
| Input preparation | `prepare_inputs()` + all `prepare_*_inputs()` functions | `harmonyspeech/task_handler/inputs.py` |
| Model loader | model class/config/weight/speaker resolution and native model construction | `harmonyspeech/modeling/loader.py` |
| Model registry | `model_type` → (module, class) or `"native"` | `harmonyspeech/modeling/models/__init__.py` |
| HuggingFace downloader | Three-tier file resolution (local → HF cache → network), in-memory caches | `harmonyspeech/modeling/hf_downloader.py` |

## Pattern Overview

**Overall:** Async inference server with layered pipeline architecture (vLLM/Aphrodite-derived skeleton), adapted for many independent speech models that are orchestrated into toolchains.

**Key Characteristics:**
- OpenAI-style REST surface, extended with speech-specific request fields (`mode`, `input_audio`, `input_vad_data`, `input_embedding`, `generation_options`, `post_generation_filters`)
- One executor + worker + loaded model per entry in `model_configs` (`config.yml`)
- A single API call may fan out into several sequential model steps through internal request re-submission (never a synchronous nested call)
- Dispatch is string-keyed on `ModelConfig.model_type` at four independent points: serving model lists, loader, input preparation, and model runner execution
- All audio crosses the API boundary as base64 strings; decoding happens inside `prepare_inputs()` on worker threads
- Batch execution is a Python loop over prepared inputs for every `_execute_*` method (no tensor batching)

## Layers

**API Layer:**
- Purpose: Accept HTTP requests, validate via Pydantic, convert protocol objects to `RequestInput`, consume the async result stream, shape responses
- Location: `harmonyspeech/endpoints/openai/`
- Contains: `api_server.py`, `protocol.py`, `args.py`, `serving_engine.py`, one `serving_*.py` per capability
- Depends on: Engine layer, `auth.apikeys`
- Used by: External HTTP clients and the bundled React frontend

**Engine Layer:**
- Purpose: Own the request lifecycle. Resolve which concrete model config handles the current pipeline step, schedule, execute, and either finish or re-submit the request
- Location: `harmonyspeech/engine/`
- Contains: `harmonyspeech_engine.py` (`HarmonySpeechEngine`, `check_reroute_request_to_model`, `check_forward_processing`, `step`), `async_harmonyspeech.py` (`AsyncHarmonySpeech`, `AsyncStream`, `RequestTracker`, `_AsyncHarmonySpeech`), `args_tools.py`, `metrics.py`
- Depends on: Scheduler, Executors, `common`
- Used by: API layer

**Processing / Scheduling Layer:**
- Purpose: Hold `EngineRequest`s in `waiting` / `running` deques and produce per-model batches
- Location: `harmonyspeech/processing/scheduler.py`
- Contains: `Scheduler`, `SchedulingBudget`, `SchedulerOutputs`, `SchedulerWaitingOutputs`
- Depends on: `common.config`, `common.request`
- Used by: Engine layer

**Executor Layer:**
- Purpose: Own one worker per model and expose a sync/async execute entry point
- Location: `harmonyspeech/executor/`
- Contains: `executor_base.py` (`ExecutorBase`, `ExecutorAsyncBase`), `cpu_executor.py` (`CPUExecutor`), `gpu_executor.py` (`GPUExecutor`, `GPUExecutorAsync`)
- Depends on: Task Handler layer
- Used by: Engine layer
- Note: `ray_gpu_executor.py` is entirely commented out and is not imported anywhere. Multi-GPU Ray dispatch is aspirational, not implemented.

**Task Handler / Worker Layer:**
- Purpose: Load the model onto its device and execute one scheduled batch
- Location: `harmonyspeech/task_handler/`
- Contains: `worker_base.py`, `cpu_worker.py`, `gpu_worker.py`, `model_runner_base.py` (all inference), `cpu_model_runner.py`, `gpu_model_runner.py` (model loading only), `inputs.py`
- Depends on: Modeling layer
- Used by: Executor layer

**Modeling Layer:**
- Purpose: Resolve and instantiate model classes, download weights, and host the model implementations
- Location: `harmonyspeech/modeling/`
- Contains: `loader.py`, `hf_downloader.py`, `utils.py`, `models/{harmonyspeech,openvoice,melo,kittentts,voicefixer,chatterbox}/`
- Depends on: torch, ONNX Runtime, faster-whisper/CTranslate2, silero-vad, chatterbox-tts, HuggingFace Hub
- Used by: Task Handler layer

**Common / Shared Layer:**
- Purpose: Config dataclasses, internal request/output types, status enum, metrics, logging, utilities
- Location: `harmonyspeech/common/`
- Contains: `config.py`, `inputs.py`, `outputs.py`, `request.py`, `metrics.py`, `logger.py`, `utils.py`
- Depends on: Nothing internal except `common.utils` (and `inputs.py` imports protocol types for `from_openai` factories)
- Used by: All layers

## Request Lifecycle

### Stage 1 — Protocol → Internal Input

1. Route handler receives the Pydantic request (`harmonyspeech/endpoints/openai/api_server.py`)
2. Serving module validates the model id, mode, language and voice against the `ModelCard` list built from config (`serving_engine.py::_check_model`, called from each `serving_*.py`)
3. `{Capability}RequestInput.from_openai(request_id, request)` creates the internal object in `harmonyspeech/common/inputs.py`. `requested_model` and `model` both start as the API model id (for toolchains this is a sentinel such as `"chatterbox"` or `"openvoice_v2"`).
4. `AsyncHarmonySpeech.generate()` yields an `AsyncStream`; the serving module consumes it and builds the response

### Stage 2 — Reroute to a Concrete Model

`HarmonySpeechEngine.add_request()` calls `check_reroute_request_to_model()` (`harmonyspeech_engine.py:337`), which dispatches on `request.requested_model`:

| `requested_model` | Handler | Routing rule |
|---|---|---|
| `harmonyspeech` | `reroute_request_harmonyspeech` | embed step → `HarmonySpeechEncoder`; synthesis → `HarmonySpeechSynthesizer`; vocode → `HarmonySpeechVocoder`, decided by which of `input_audio` / `input_embedding` are set |
| `openvoice_v1` | `reroute_request_openvoice_v1` | transcribe (VAD) → `FasterWhisper`; embed → `OpenVoiceV1ToneConverterEncoder`; synthesize → `OpenVoiceV1Synthesizer` matching `language`; convert → `OpenVoiceV1ToneConverter` |
| `openvoice_v2` | `reroute_request_openvoice_v2` | same shape, with `MeloTTSSynthesizer` and `OpenVoiceV2ToneConverter(Encoder)` |
| `voicefixer` | `reroute_request_voicefixer` | no `input_mel_spectrogram` → `VoiceFixerRestorer`; otherwise → `VoiceFixerVocoder` |
| `chatterbox` / `chatterbox_turbo` / `chatterbox_multilingual` | `reroute_request_chatterbox` | embed-ish request → `ChatterboxEmbedding` if configured, else the matching TTS variant; synthesis → matching TTS variant |
| `chatterbox_vc` | `reroute_request_chatterbox_vc` | always `ChatterboxVC` |

Routing overwrites `request.model` with the concrete `ModelConfig.name`; `requested_model` is never mutated, so it remains the toolchain identity.

Chatterbox sentinel → `model_type` map (`harmonyspeech_engine.py:294`):

```python
_CHATTERBOX_TTS_TYPE_MAP = {
    "chatterbox": "ChatterboxTTS",
    "chatterbox_turbo": "ChatterboxTurboTTS",
    "chatterbox_multilingual": "ChatterboxMultilingualTTS",
}
```

### Stage 3 — Schedule and Execute

1. `Scheduler.schedule()` builds a `SchedulingBudget` from `model_configs[].max_batch_size` (`processing/scheduler.py:191`)
2. `_schedule_waiting_requests()` admits requests only for models that have no in-flight batch and still have budget (`scheduler.py:134`)
3. `SchedulerOutputs.scheduled_requests_per_model` maps model name → list of `EngineRequest`
4. Sync `step()` dispatches each model's batch on a `ThreadPoolExecutor` (`harmonyspeech_engine.py:656`); async `step_async()` awaits each executor in sequence (`async_harmonyspeech.py:190`)
5. Executor → worker → `ModelRunnerBase.execute_model()` → `prepare_inputs()` → `_execute_{model_type}()` → `ExecutorResult`

### Stage 4 — Forward or Finish

`check_forward_processing()` (`harmonyspeech_engine.py:363`) is the pipeline state machine. It compares the merged request:

- If `result.result_data.finish_reason == "error"` → mark `FINISHED_STOPPED` and pass the error output through (never forward a failure)
- If the executing config's `name` equals `requested_model` → the caller selected the concrete model directly; finish
- Otherwise, per toolchain, it mutates the same `RequestInput` and calls `add_request(request_id, forwarding_request)` again:

| Toolchain | Step output → next input | Status |
|---|---|---|
| HarmonySpeech / OpenVoice V1 / V2 | `SpeechTranscriptionRequestOutput` → `input_vad_data` | `FINISHED_FORWARDED` |
| HarmonySpeech / OpenVoice V1 / V2 | `SpeechEmbeddingRequestOutput` → `input_embedding`, `input_audio = None` | `FINISHED_FORWARDED` |
| OpenVoice V1 / V2 | `SpeechSynthesisRequestOutput` (voice_cloning) → `input_audio` | `FINISHED_FORWARDED` |
| OpenVoice / HarmonySpeech | final vocode/VC output → wrapped as `TextToSpeechRequestOutput` | `FINISHED_STOPPED` |
| Chatterbox TTS variants | `SpeechEmbeddingRequestOutput` → `input_embedding`, `input_audio = None` | `FINISHED_FORWARDED` |
| VoiceFixer | restorer output → `input_mel_spectrogram` | `FINISHED_FORWARDED` |
| VoiceFixer | vocoder output | `FINISHED_STOPPED` |

Forwarded requests are drained from the scheduler (`free_finished_requests()`), and the async loop only exits its idle wait while `len(request_outputs) + len(forwarded_requests) > 0` (`async_harmonyspeech.py:361`).

## Data Flow

### TextToSpeech — Direct Single-Model (KittenTTS, MeloTTS, Chatterbox Turbo, Chatterbox Multilingual)

1. `POST /v1/audio/speech` → `OpenAIServingTextToSpeech.create_text_to_speech()` (`serving_text_to_speech.py:43`)
2. Streaming output is rejected with a 400 (`output_options.stream` not supported)
3. `TextToSpeechRequestInput` submitted to `AsyncHarmonySpeech.generate()`
4. Reroute by sentinel in `reroute_request_chatterbox` (or no-op for individually-listed synthesizers: `KittenTTSSynthesizer`, `MeloTTSSynthesizer`, `OpenVoiceV1Synthesizer`)
5. `prepare_inputs()` → `prepare_kittentts_synthesizer_inputs` / `prepare_melotts_synthesizer_inputs` / `prepare_chatterbox_*_tts_inputs`
6. `_execute_*` produces a base64 WAV; `TextToSpeechRequestOutput` or `SpeechSynthesisRequestOutput`
7. Serving accepts either output class and returns `TextToSpeechResponse` (`serving_text_to_speech.py:97`)

### HarmonySpeech Full Pipeline (embed → synthesize → vocode)

1. `requested_model="harmonyspeech"`, `mode="voice_cloning"`, `input_audio` set
2. → `HarmonySpeechEncoder` → `SpeechEmbeddingRequestOutput`
3. `check_forward_processing` sets `input_embedding`, clears `input_audio`, re-submits
4. → `HarmonySpeechSynthesizer` → `SpeechSynthesisRequestOutput`
5. `check_forward_processing` sets `input_audio` to the mel output, re-submits
6. → `HarmonySpeechVocoder` → `VocodeRequestOutput` → wrapped as `TextToSpeechRequestOutput`

Passing a pre-computed `input_embedding` skips steps 2–3.

### OpenVoice V1 / V2 Pipeline (VAD → embed → synthesize → tone convert)

1. `FasterWhisper` transcribes and returns segments (`SpeechTranscriptionRequestOutput`) → `input_vad_data`
2. `OpenVoiceV1ToneConverterEncoder` / `OpenVoiceV2ToneConverterEncoder` use audio + VAD segments → embedding
3. `OpenVoiceV1Synthesizer` (language-matched) or `MeloTTSSynthesizer` (language-matched) → mel audio
4. `OpenVoiceV1ToneConverter` / `OpenVoiceV2ToneConverter` applies the embedding → `VoiceConversionRequestOutput` wrapped as TTS output

`mode="single_speaker_tts"` deliberately skips the embedding requirement in the rerouter (`harmonyspeech_engine.py:187`, marked `FIXME`).

### Chatterbox Voice Cloning (embed → synthesize)

1. `requested_model="chatterbox"`, `mode="voice_cloning"`, `input_audio` set
2. If a `ChatterboxEmbedding` model is configured: `_execute_chatterbox_embedding` runs `model.prepare_conditionals()`, serializes `Conditionals` with `torch.save` into `BytesIO`, and returns it base64-encoded
3. Engine sets `input_embedding` (serialized conditionals), clears `input_audio`, re-submits
4. TTS variant assigns `model.conds = conditionals.to(device)` and calls `generate()`
5. Without a dedicated embedding executor, the TTS variant handles `input_audio` directly inside `_execute_chatterbox_*_tts` via `prepare_conditionals(audio_buffer, ...)` — a single-step path because the executing config name equals `requested_model`

The embedding output format is also the intended future cache value; the documented intercept point for an embedding cache is inside `check_forward_processing` between the embed step and TTS re-submission (`model_runner_base.py:806-821`).

### Chatterbox Voice Conversion

`/v1/voice/convert` with `model="chatterbox_vc"` → `ChatterboxVC`. Exactly one of `target_audio` or `target_embedding` must be present (`task_handler/inputs.py:840`). `target_embedding` is deserialized with `Conditionals.load(..., map_location="cpu")` and applied to `model.ref_dict`; `target_audio` goes through `model.set_target_voice()`.

### VoiceFixer Audio Conversion (restore → vocode)

1. `/v1/audio/convert` with `model="voicefixer"`, no `input_mel_spectrogram`
2. `VoiceFixerRestorer` (44.1 kHz internal) → `AudioConversionRequestOutput` → `input_mel_spectrogram`
3. `VoiceFixerVocoder` → final audio

### Voice Activity Detection / Transcription

- `/v1/audio/vad` supports `FasterWhisper` and `SileroVAD`, each with their own tunables (`threshold`, `min_speech_duration_ms`, `min_silence_duration_ms`, `speech_pad_ms`, `return_seconds`, `get_timestamps`)
- `/v1/audio/transcriptions` supports `FasterWhisper` only (`_STT_MODEL_TYPES = ["FasterWhisper"]`)
- VAD output is JSON text in `DetectVoiceActivityRequestOutput.output`; transcription output is JSON in `SpeechTranscriptionRequestOutput.output` and parsed by the serving layer for `text` / `language` / `timestamps`

**State Management:**
- No persistent session state; every request carries its own pipeline cursor in mutable `RequestInput` fields
- Model weights are loaded once per worker at startup and held for the process lifetime
- Queues, running set, and streams live in memory only (`Scheduler`, `RequestTracker`)
- Per-request timing captured in `RequestMetrics` (`common/metrics.py`)

## Embed Dispatch (Embedding-Capable TTS Models)

Two cooperating mechanisms let a TTS model also serve `/v1/embed/speaker`:

1. **Model-card level** — `harmonyspeech/endpoints/openai/serving_voice_embed.py:16`:
   - `_EMBEDDING_MODEL_TYPES` includes `ChatterboxEmbedding` and the three Chatterbox TTS variants
   - `_EMBEDDING_MODEL_GROUPS["chatterbox"] = ["ChatterboxEmbedding", "ChatterboxTTS"]`, so a group card is only emitted when at least one is configured
2. **Execution level** — `ModelRunnerBase.execute_model()` inspects the *request type*, not just the model type. For `ChatterboxTTS` / `ChatterboxTurboTTS` / `ChatterboxMultilingualTTS`, a batch whose first item is `SpeechEmbeddingRequestInput` is routed to `_execute_chatterbox_embedding` (`model_runner_base.py:96-111`)
3. **Input level** — `prepare_inputs()` splits a Chatterbox batch into `embed_inputs` and `tts_inputs` and picks the matching preparation function (`task_handler/inputs.py:174-219`). `ChatterboxEmbedding` accepts both `SpeechEmbeddingRequestInput` and `TextToSpeechRequestInput` because the voice-cloning pipeline routes the TTS request into the embedding step (`inputs.py:227-240`).

Known limitation to respect when extending: the request-type check uses `requests_to_batch[0]`, so a mixed embed + TTS batch is not supported and must not be produced.

## Chatterbox Model Family

**Location:** `harmonyspeech/modeling/models/chatterbox/`

| Wrapper class | Upstream class | `model_type` | Notes |
|---|---|---|---|
| `ChatterboxTTSModel` | `chatterbox.ChatterboxTTS` | `ChatterboxTTS` | `from_pretrained(device)` |
| `ChatterboxTurboTTSModel` | `chatterbox.tts_turbo.ChatterboxTurboTTS` | `ChatterboxTurboTTS` | Downloads `ResembleAI/chatterbox-turbo` itself; float32 patches |
| `ChatterboxMultilingualTTSModel` | `chatterbox.ChatterboxMultilingualTTS` | `ChatterboxMultilingualTTS` | Exposes `SUPPORTED_LANGUAGES` for model cards; forces `attn_implementation="eager"` |
| `ChatterboxVCModel` | `chatterbox.ChatterboxVC` | `ChatterboxVC` | Voice conversion |

`ChatterboxEmbedding` is not a wrapper class — it is a *role* served by whichever Chatterbox model the loader instantiates based on the configured repo id (`loader.py:328-337`: turbo → Turbo wrapper, multilingual → Multilingual wrapper, otherwise standard).

**Module-level compatibility patches** in `chatterbox.py` (apply on import):
- `chatterbox.mtl_tts.torch.load` is patched to default `map_location=cpu` so CUDA-saved checkpoints load on CPU
- `perth.PerthImplicitWatermarker` is replaced with a fallback delegating to `DummyWatermarker` when the optional watermarker is unavailable
- Turbo loader bypasses upstream `token=True` by calling `snapshot_download(token=HF_TOKEN or None)` then `from_local`
- Turbo's `s3tokenizer.log_mel_spectrogram` and `ve.forward` are patched per instance to cast float64 → float32

Watermarking is config-driven: `loader.get_model()` swaps `model.watermarker = perth.DummyWatermarker()` whenever `ModelConfig.watermark` is false (`loader.py:307`, `313`, `319`, `325`).

**Parameter surface** (validated in `task_handler/inputs.py:652-820`):

| Parameter | Standard TTS | Multilingual | Turbo |
|---|---|---|---|
| `exaggeration`, `cfg_weight`, `min_p` | yes | yes | rejected |
| `top_k`, `norm_loudness` | rejected | rejected | yes |
| `temperature`, `repetition_penalty`, `top_p` | yes | yes | yes |

Model-specific defaults are applied in the `prepare_*` functions when the field is `None`.

## Model Type Registry

Execution dispatch is a four-way string switch. Adding a model requires touching all four.

| `model_type` | Loader branch | `prepare_*` | `_execute_*` |
|---|---|---|---|
| `HarmonySpeechEncoder` | registry `SpeakerEncoder` | `prepare_harmonyspeech_encoder_inputs` | `_execute_harmonyspeech_encoder` |
| `HarmonySpeechSynthesizer` | registry `ForwardTacotron` | `prepare_harmonyspeech_synthesizer_inputs` | `_execute_harmonyspeech_synthesizer` |
| `HarmonySpeechVocoder` | registry `MelGANGenerator` | `prepare_harmonyspeech_vocoder_inputs` | `_execute_harmonyspeech_vocoder` |
| `OpenVoiceV1Synthesizer` | registry `SynthesizerTrn` | `prepare_openvoice_synthesizer_inputs` | `_execute_openvoice_synthesizer` |
| `OpenVoiceV1ToneConverter` | registry `SynthesizerTrn` | `prepare_openvoice_tone_converter_inputs` | `_execute_openvoice_tone_converter` |
| `OpenVoiceV1ToneConverterEncoder` | registry `SynthesizerTrn` | `prepare_openvoice_tone_converter_encoder_inputs` | `_execute_openvoice_tone_converter_encoder` |
| `MeloTTSSynthesizer` | registry `SynthesizerTrn` (melo) | `prepare_melotts_synthesizer_inputs` | `_execute_melotts_synthesizer` |
| `OpenVoiceV2ToneConverter` | registry `SynthesizerTrn` | `prepare_openvoice_tone_converter_inputs` | `_execute_openvoice_tone_converter` |
| `OpenVoiceV2ToneConverterEncoder` | registry `SynthesizerTrn` | `prepare_openvoice_tone_converter_encoder_inputs` | `_execute_openvoice_tone_converter_encoder` |
| `FasterWhisper` | native (`WhisperModel`) | `prepare_faster_whisper_inputs` | `_execute_faster_whisper` |
| `SileroVAD` | native (`load_silero_vad`) | `prepare_silero_vad_inputs` | `_execute_silero_vad` |
| `VoiceFixerRestorer` | registry + native config | `prepare_voicefixer_restorer_inputs` | `_execute_voicefixer_restorer` |
| `VoiceFixerVocoder` | registry + native config | `prepare_voicefixer_vocoder_inputs` | `_execute_voicefixer_vocoder` |
| `KittenTTSSynthesizer` | native (`KittenTTSSynthesizer(...)`) | `prepare_kittentts_synthesizer_inputs` | `_execute_kittentts_synthesizer` |
| `ChatterboxTTS` | native wrapper | `prepare_chatterbox_tts_inputs` (+ embedding) | `_execute_chatterbox_tts` / `_execute_chatterbox_embedding` |
| `ChatterboxTurboTTS` | native wrapper | `prepare_chatterbox_turbo_tts_inputs` (+ embedding) | `_execute_chatterbox_turbo_tts` / `_execute_chatterbox_embedding` |
| `ChatterboxMultilingualTTS` | native wrapper | `prepare_chatterbox_multilingual_tts_inputs` (+ embedding) | `_execute_chatterbox_multilingual_tts` / `_execute_chatterbox_embedding` |
| `ChatterboxEmbedding` | native wrapper chosen by repo id | `prepare_chatterbox_embedding_inputs` | `_execute_chatterbox_embedding` |
| `ChatterboxVC` | native wrapper | `prepare_chatterbox_vc_inputs` | `_execute_chatterbox_vc` |

## Key Abstractions

**`ModelConfig`** (`harmonyspeech/common/config.py:28`):
- Declarative description of one served model instance: `name`, `model` (HF repo id or local path), `model_type`, `max_batch_size`, `device_config`, `language`, `voices`, `dtype`, `load_format`, `revision`, `enforce_eager`, `watermark`
- `name` is the scheduler/executor key **and** the value that rerouting writes into `RequestInput.model`

**`EngineConfig`** (`common/config.py:102`):
- Frozen dataclass holding `list[ModelConfig]`; `load_config_from_yaml()` is the only loader, used once at startup

**`RequestInput` hierarchy** (`common/inputs.py`):
- Base carries `request_id`, `requested_model`, `model`, `metrics`
- Subclasses: `TextToSpeechRequestInput`, `SpeechEmbeddingRequestInput`, `SynthesisRequestInput`, `VocodeRequestInput`, `VoiceConversionRequestInput`, `AudioConversionRequestInput`, `SpeechTranscribeRequestInput`, `DetectVoiceActivityRequestInput`
- Each has a `from_openai(request_id, request)` classmethod; toolchain steps mutate these objects in place rather than constructing new ones

**`EngineRequest` / `ExecutorResult` / `RequestStatus`** (`common/request.py`):
- `EngineRequest` pairs id + `RequestInput` + status + metrics
- `RequestStatus` includes the pipeline-specific `FINISHED_FORWARDED`, which is treated as finished by the scheduler but means "re-queued as a new step"
- `ExecutorResult` carries both `input_data` and `result_data` so the engine can decide the next hop

**`ModelCard` / model groups** (`endpoints/openai/serving_engine.py:221`):
- `model_cards_from_config_groups(configured_models, individual_model_types, model_groups)` emits one card per directly-served model and one `object="toolchain"` card per group whose required model types are all present
- Group cards are the ids clients pass as `model` (e.g. `harmonyspeech`, `openvoice_v1`, `openvoice_v2`, `voicefixer`, `chatterbox`, `chatterbox_turbo`, `chatterbox_multilingual`, `chatterbox_vc`)

**`SchedulingBudget`** (`processing/scheduler.py:11`):
- Per-model capacity tracker seeded from `max_batch_size`; consumed by `_schedule_waiting_requests`

**Serving modules** (`endpoints/openai/serving_*.py`):
- Uniform shape: module-level `_X_MODEL_TYPES` / `_X_MODEL_GROUPS`, `models_from_config()`, one public `async` method, one `*_full_generator()` that iterates the stream, checks `finish_reason == "error"`, and asserts the expected output class

## API Surface

| Method | Path | Handler | Output type |
|---|---|---|---|
| GET | `/health` | `health` | per-capability health map (calls `engine.check_health()`) |
| GET | `/version` | `show_version` | `{"version": ...}` |
| GET | `/metrics` | Prometheus ASGI mount | metrics |
| POST | `/v1/audio/speech` | `create_speech` | `TextToSpeechResponse` |
| GET | `/v1/audio/speech/models` | `show_available_speech_models` | `ModelList` |
| POST | `/v1/embed/speaker` | `create_embedding` | `EmbedSpeakerResponse` |
| GET | `/v1/embed/models` | `show_available_embedding_models` | `ModelList` |
| POST | `/v1/audio/transcriptions` | `create_transcription` | `SpeechToTextResponse` |
| GET | `/v1/audio/transcriptions/models` | `show_available_transcription_models` | `ModelList` |
| POST | `/v1/voice/convert` | `convert_voice` | `VoiceConversionResponse` |
| GET | `/v1/voice/convert/models` | `show_available_voice_conversion_models` | `ModelList` |
| POST | `/v1/audio/vad` | `create_vad` | `DetectVoiceActivityResponse` |
| GET | `/v1/audio/vad/models` | `show_available_vad_models` | `ModelList` |
| POST | `/v1/audio/convert` | `convert_audio` | `AudioConversionResponse` |
| GET | `/v1/audio/convert/models` | `show_available_audio_conversion_models` | `ModelList` |

No aggregate `/v1/models` route exists; each capability exposes its own list.

## Entry Points

**HTTP server:**
- `harmonyspeech/endpoints/openai/api_server.py` — `run_server()` loads `EngineConfig.load_config_from_yaml(args.config_file_path or "config.yml")`, constructs `AsyncHarmonySpeech`, builds all six serving objects, then calls `uvicorn.run`
- `build_app()` adds CORS, optional API-key middleware (active only when `HARMONYSPEECH_API_KEY` or `--api-keys` is set), custom OpenAPI metadata for Go client generation, and dynamic middleware from `--middleware`

**CLI:**
- `harmonyspeech/endpoints/cli.py` → `harmonyspeech run [options]`
- Console script: `harmonyspeech = "harmonyspeech.endpoints.cli:main"` (`pyproject.toml:33`)
- Equivalently: `uvicorn harmonyspeech.endpoints.openai.api_server:app` (module-level `__main__` block runs `run_server`)

**Programmatic:**
- `harmonyspeech.HarmonySpeechEngine` (`harmonyspeech/__init__.py`)
- `HarmonySpeechEngine.from_engine_args(engine_args)` for sync use
- `AsyncHarmonySpeech.from_engine_args_and_config(engine_args, engine_config)` for the async wrapper

## Configuration System

**Sources:**
1. `config.yml` — CPU/default runtime model list
2. `config.gpu.yml` — GPU variant (Chatterbox on `cuda`, Whisper large-v3-turbo, MeloTTS on `cuda`)
3. CLI via `harmonyspeech/endpoints/openai/args.py` + `harmonyspeech/engine/args_tools.py`
4. Environment: `HARMONYSPEECH_API_KEY`, `HARMONYSPEECH_ADMIN_KEY`, `SERVICE_NAME`, `HARMONYSPEECH_ENGINE_ITERATION_TIMEOUT_S` (default 120), `HARMONYSPEECH_LOCAL_LOGGING_INTERVAL_SEC` (default 5)

**Config flow:**
```text
config.yml → EngineConfig.load_config_from_yaml()
           → [ModelConfig(device_config=DeviceConfig(...))]
           → AsyncEngineArgs.create_engine_config(base_config)  # currently pass-through
           → HarmonySpeechEngine(model_configs=...) → one Executor per ModelConfig
```

Model list in the shipped CPU config: HarmonySpeech V1 (encoder/synthesizer/vocoder), OpenVoice V1 (EN/ZH synthesizer, converter, converter encoder), MeloTTS (EN/ZH/ES/FR/JP/KR), OpenVoice V2 converter + encoder, FasterWhisper (medium, tiny), SileroVAD, VoiceFixer (restorer, vocoder), KittenTTS mini, Chatterbox (standard, turbo, multilingual, VC). `ChatterboxEmbedding` is present but commented out.

## Model Loading

1. `_get_model_cls()` → `ModelRegistry.load_model_cls(model_type)`; unknown arch raises with the supported list
2. `get_model_flavour()` — only `OpenVoiceV1Synthesizer` uses `language` as flavour; everything else is `"default"`
3. `get_model_config()` — `"native"` short-circuits; otherwise `load_or_download_config()` from HF
4. Native branch constructs directly (`WhisperModel`, `load_silero_vad`, `KittenTTSSynthesizer`, Chatterbox wrappers)
5. Non-native branches instantiate `model_class(**hf_config.model)` (with per-family dynamic fields injected for OpenVoice/MeloTTS) or `model_class(**hf_config)` (VoiceFixer)
6. `get_model_weights()` + `model.load_weights(checkpoint, hf_config)`; `load_format="dummy"` uses `initialize_dummy_weights`
7. Returns `model.eval()`

**Safety rail:** FasterWhisper on `device="cuda"` is pre-checked by `_check_ctranslate2_cuda_available()` (`loader.py:31`) because CTranslate2 calls native `abort()` when cuDNN is unavailable — an uncatchable process kill. On failure it logs a warning and falls back to `cpu` + `int8`.

**HuggingFace resolution** (`modeling/hf_downloader.py`): local filesystem → HF disk cache (`local_files_only=True`) → network. Configs and small static files are memoized in-process (`_config_cache`, `_file_cache`).

## Architectural Constraints

- **Threading:** One background `asyncio` loop per `AsyncHarmonySpeech`. Sync `step()` fans out across models with a `ThreadPoolExecutor`; async `step_async()` awaits models sequentially. Executors wrap the blocking worker call with `make_async()` (`common/utils.py:133`), which runs it in the default thread pool.
- **Model concurrency:** Each model owns exactly one worker and one loaded model instance, so at most one batch per model is in flight. The scheduler enforces this by refusing to add waiting requests for a model that already has running requests.
- **No true tensor batching:** every `_execute_*` method iterates prepared inputs one at a time. `max_batch_size` controls queue admission, not vectorization.
- **Single-process, in-memory state:** scheduler queues, request tracker streams, and model instances are process-local. There is no distributed execution today.
- **`model_type` string coupling:** the same literal appears in `serving_*.py` lists, `loader._MODEL_CONFIGS` / `_MODEL_WEIGHTS`, `models/__init__._MODELS`, `inputs.prepare_inputs()`, and `model_runner_base.execute_model()`. There is no enum or registry tying them together.
- **Import-time side effects:** `harmonyspeech/engine/harmonyspeech_engine.py` calls `setup_logger()` at import (line 702); `harmonyspeech/modeling/models/chatterbox/chatterbox.py` monkey-patches `chatterbox.mtl_tts` and `perth` at import. Importing the modeling layer therefore mutates third-party modules.
- **Mutable pipeline request:** forwarding re-uses the same `RequestInput` object and mutates `input_audio` / `input_embedding` / `input_vad_data` / `input_mel_spectrogram`. Any new step must keep those fields consistent or routing will pick the wrong model.
- **`requested_model` is the toolchain identity and must not be overwritten.** Rerouting writes only `request.model`.
- **Blocking work on the event loop:** `check_health()` calls executor health checks synchronously; `project.model_executors` iteration in `check_health` iterates dict keys (not values) in `harmonyspeech_engine.py:698`.

## Anti-Patterns

### Adding a model type to only one dispatch point

**What happens:** A `model_type` is added to `_MODELS` and `config.yml`, but not to `prepare_inputs()` or `execute_model()`.
**Why it's wrong:** `prepare_inputs()` raises `NotImplementedError`, which `execute_model()` converts into a per-request error result — the endpoint returns 400/500 at runtime with no startup failure.
**Do this instead:** Update all five in one change: `serving_*_MODEL_TYPES/GROUPS` (if user-facing), `loader._MODEL_CONFIGS`, `loader._MODEL_WEIGHTS`, `models/__init__._MODELS`, `task_handler/inputs.py`, `task_handler/model_runner_base.py`.

### Assuming `request.model` is stable

**What happens:** Code reads `request_data.model` expecting the API-supplied id.
**Why it's wrong:** `check_reroute_request_to_model()` overwrites it with the concrete `ModelConfig.name` before scheduling. The API id lives in `requested_model`.
**Do this instead:** Use `requested_model` for toolchain logic and `model` for scheduler/executor selection, as `check_forward_processing()` does.

### Emitting a group model card unconditionally

**What happens:** A toolchain id is added to `model_groups` without ensuring every required `model_type` exists in `config.yml`.
**Why it's wrong:** The group card advertises a toolchain that cannot complete, and requests will fail mid-pipeline after consuming compute.
**Do this instead:** Add the group to `_*_MODEL_GROUPS` in the appropriate serving module and let `model_cards_from_config_groups()` filter it out, as done in `serving_audio_conversion.py:22`.

### Batch-type detection on `requests_to_batch[0]` for Chatterbox

**What happens:** A batch mixes `SpeechEmbeddingRequestInput` and `TextToSpeechRequestInput`.
**Why it's wrong:** `execute_model()` and `prepare_inputs()` both branch on the first element, so mixed batches silently take one path and fail or mis-prepare the rest.
**Do this instead:** Keep one request type per scheduled batch; if mixed batches become necessary, split them before dispatch in `prepare_inputs()` and return per-type sub-batches.

### Reaching into third-party libraries without a wrapper

**What happens:** Inference code touches upstream Chatterbox internals (`model.s3gen`, `model.ve`, `model.ref_dict`, `model.conds`).
**Why it's wrong:** Version upgrades break call sites in `task_handler/` silently at runtime; the compatibility patches in `models/chatterbox/chatterbox.py` are the only seam.
**Do this instead:** Add adapter methods to the wrapper classes in `harmonyspeech/modeling/models/chatterbox/chatterbox.py` and call those from `model_runner_base.py`.

## Error Handling

**Strategy:** Per-request error results inside a resilient async loop; HTTP error shaping at the serving boundary.

**Patterns:**
- `ModelRunnerBase.execute_model()` wraps `prepare_inputs()` and each `_execute_*` item in `try/except` and returns `_build_error_result()` — a `RequestOutput(finish_reason="error", error=str(exc))` wrapped in `ExecutorResult` (`model_runner_base.py:57-68`, `121-137`). One bad request does not kill the batch.
- `check_forward_processing()` short-circuits error results to `FINISHED_STOPPED` so failed steps are never forwarded and never leak in `scheduler.running`.
- Serving modules check `final_res.finish_reason == "error"` and return `ErrorResponse` with the embedded message.
- `AsyncHarmonySpeech.engine_step()` catches uncaught step failures, propagates the exception to every in-flight request stream, aborts them in the scheduler, and returns `False` instead of letting the background loop die (`async_harmonyspeech.py:336-355`).
- `run_engine_loop()` wraps each iteration in `asyncio.wait_for(..., ENGINE_ITERATION_TIMEOUT_S)` and marks the engine errored on timeout.
- `RequestValidationError` is mapped to a 400 with an `ErrorResponse` body (`api_server.py:299`).
- Chatterbox upstream `AssertionError`s (e.g. "Audio prompt must be longer than 5 seconds!") are re-raised as `ValueError` so they become a 400 rather than an unhandled exception (`model_runner_base.py:827-830`).

## Cross-Cutting Concerns

**Logging:** `loguru` everywhere (`from loguru import logger`). `common/logger.py` provides `setup_logger()` and `UVICORN_LOG_CONFIG`. Engine/runner errors use `logger.opt(exception=...)` to preserve stack traces.

**Validation:** Pydantic v2 at the boundary (`protocol.py`), plus hand-written semantic validation in `serving_engine._check_model()` (mode, model existence, language/voice membership, Chatterbox mode↔reference consistency). Internal model-type and option validation lives in `task_handler/inputs.py` and raises `ValueError`.

**Authentication:** Optional bearer/Api-Key middleware in `build_app()`; `auth/apikeys.py::ApiKeyCacheManager` performs rate-limit checks against Harmony Auth and registers usage events on a background thread. Auth only applies to paths starting with `/v1`, and `OPTIONS` requests pass through.

**Batching:** Per-model admission batching (`max_batch_size`) with sequential per-item inference; `ThreadPoolExecutor` inside each `prepare_*_inputs()` parallelizes base64 decode, resampling, and tokenization.

**Audio transport:** Base64 strings in JSON in both directions. Final audio outputs are base64 WAV produced with `soundfile.write` into an in-memory buffer (Chatterbox, KittenTTS, VoiceFixer, OpenVoice, MeloTTS). Pipeline intermediates are also base64: VoiceFixerRestorer emits base64-encoded JSON mel, and Chatterbox embedding emits base64 `torch.save` bytes. VoiceFixer resamples internally to 44.1 kHz.

**Metrics:** `RequestMetrics` (arrival, first scheduled, queue time, finish time) attached to every `EngineRequest`; `StatLogger`/`Stats` exist in `engine/metrics.py` but `stat_logger.log()` is commented out in `step()`.

---

*Architecture analysis: 2026-11-02*

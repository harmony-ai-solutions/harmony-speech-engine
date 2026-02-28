# Architecture

**Analysis Date:** 2026-02-28

## Pattern Overview

**Overall:** Async inference server with layered pipeline architecture — similar in design to vLLM/Aphrodite Engine, adapted for speech AI tasks.

**Key Characteristics:**
- OpenAI-compatible REST API surface (extended for speech-specific operations)
- Request batching scheduler that aggregates concurrent requests before dispatching to model workers
- Pluggable executor backends (CPU, GPU, Ray multi-GPU) — selectable at startup
- Multi-model configuration: multiple named model instances running concurrently, each with its own worker
- Pipeline composition: a single high-level request (e.g., TextToSpeech) may internally fan out to multiple sub-model steps (encode → synthesize → vocode → convert)

## Layers

**API Layer (Endpoint):**
- Purpose: Accept HTTP requests, validate schemas, create `EngineRequest` objects, stream or return results
- Location: `harmonyspeech/endpoints/openai/`
- Contains: FastAPI app (`api_server.py`), Pydantic request/response protocol (`protocol.py`), per-capability serving modules (`serving_text_to_speech.py`, `serving_speech_to_text.py`, `serving_voice_conversion.py`, `serving_audio_conversion.py`, `serving_voice_activity_detection.py`, `serving_voice_embed.py`)
- Depends on: Engine layer
- Used by: External HTTP clients (Harmony Link, other applications)

**Engine Layer:**
- Purpose: Orchestrate request lifecycle — submit to scheduler, dispatch to executor, collect and return outputs
- Location: `harmonyspeech/engine/`
- Contains: Synchronous engine (`harmonyspeech_engine.py`), async wrapper (`async_harmonyspeech.py`), argument parsing (`args_tools.py`), metrics (`metrics.py`)
- Depends on: Processing (Scheduler), Executor layer
- Used by: API layer

**Processing / Scheduling Layer:**
- Purpose: Queue and batch incoming `EngineRequest` objects; determine which requests to dispatch in each scheduling cycle
- Location: `harmonyspeech/processing/scheduler.py`
- Contains: `HarmonySpeechScheduler` — manages per-model request queues, batches up to `max_batch_size`
- Depends on: Common (config, request types)
- Used by: Engine layer

**Executor Layer:**
- Purpose: Abstract the compute backend; dispatch batches to one or more workers
- Location: `harmonyspeech/executor/`
- Contains: `ExecutorBase` (`executor_base.py`), `CPUExecutor` (`cpu_executor.py`), `GPUExecutor` (`gpu_executor.py`), `RayGPUExecutor` (`ray_gpu_executor.py`)
- Depends on: Task Handler layer
- Used by: Engine layer

**Task Handler / Worker Layer:**
- Purpose: Run on the compute device (CPU or GPU); load models, prepare inputs, run inference, return outputs
- Location: `harmonyspeech/task_handler/`
- Contains: `WorkerBase` (`worker_base.py`), `CPUWorker` (`cpu_worker.py`), `GPUWorker` (`gpu_worker.py`), `ModelRunnerBase` (`model_runner_base.py`), `CPUModelRunner` (`cpu_model_runner.py`), `GPUModelRunner` (`gpu_model_runner.py`), `inputs.py` (per-model input preparation)
- Depends on: Modeling layer
- Used by: Executor layer

**Modeling Layer:**
- Purpose: Contain model class implementations and HuggingFace download/loading utilities
- Location: `harmonyspeech/modeling/`
- Contains: `loader.py` (model loading, config resolution, speaker embedding loading), `hf_downloader.py`, `utils.py`, per-model packages under `modeling/models/`
- Depends on: PyTorch, HuggingFace Hub, ONNX Runtime
- Used by: Task Handler layer

**Common / Shared Layer:**
- Purpose: Data types, config, utilities shared across all layers
- Location: `harmonyspeech/common/`
- Contains: `config.py` (ModelConfig, DeviceConfig, EngineConfig), `inputs.py` (RequestInput subclasses), `outputs.py` (RequestOutput, GenerationOutput), `request.py` (EngineRequest), `metrics.py`, `logger.py`, `utils.py`
- Depends on: Nothing internal
- Used by: All other layers

## Data Flow

**Standard TextToSpeech Request Flow:**

1. HTTP `POST /v1/audio/speech` received by FastAPI in `api_server.py`
2. `serving_text_to_speech.py` validates request against `TextToSpeechRequest` (Pydantic model in `protocol.py`)
3. `TextToSpeechRequestInput.from_openai()` creates internal input object (`common/inputs.py`)
4. `AsyncHarmonySpeech.generate()` submits `EngineRequest` to the async engine (`engine/async_harmonyspeech.py`)
5. Engine forwards to `HarmonySpeechScheduler.add_request()` — placed in per-model queue
6. Scheduler batches requests up to `max_batch_size` and yields `ScheduledBatch` objects
7. Executor (`GPUExecutor` / `CPUExecutor`) dispatches batch to corresponding `Worker.execute_model()`
8. Worker calls `ModelRunner.execute_model()`:
   a. `prepare_inputs()` in `task_handler/inputs.py` preprocesses audio/text (base64 decode, resampling, tokenization, mel spectrogram preparation) using `ThreadPoolExecutor` for parallel preparation
   b. Model `forward()` is called (PyTorch / ONNX)
   c. Outputs are post-processed (base64 encode audio, build `RequestOutput`)
9. Output propagates back through Executor → Engine → Serving layer
10. HTTP response returned as `TextToSpeechResponse` (audio bytes base64-encoded in JSON)

**Multi-Step TTS Pipeline (e.g., HarmonySpeech full pipeline):**

1. Request arrives with `mode` field indicating pipeline steps needed
2. Engine orchestrates sequential sub-requests:
   - Step 1: `HarmonySpeechEncoder` — extract speaker embedding from reference audio
   - Step 2: `HarmonySpeechSynthesizer` — convert text + embedding → mel spectrogram
   - Step 3: `HarmonySpeechVocoder` — convert mel spectrogram → waveform audio
3. Optional post-generation filters (`post_generation_filters`) can chain `AudioConversionRequest` steps (e.g., VoiceFixer restoration/enhancement)

**Voice Activity Detection + Voice Conversion Flow:**

1. VAD request (`SileroVAD` or `FasterWhisper`) identifies speech segments in audio
2. VAD result stored as `input_vad_data` (JSON with segments/timestamps)
3. Tone converter encoder uses VAD segments to extract speaker embedding
4. Tone converter applies embedding for voice conversion

**State Management:**
- No persistent session state in the engine; each request is self-contained
- Model weights are loaded once per worker at startup and held in memory
- Request queue state is held in-memory in the `Scheduler` — not persisted

## Key Abstractions

**ModelConfig:**
- Purpose: Declarative configuration for one named model instance
- Examples: `harmonyspeech/common/config.py`
- Pattern: Dataclass with fields `name`, `model` (HuggingFace repo ID or local path), `model_type` (string key used by all dispatch switches), `max_batch_size`, `dtype`, `device_config`, `language`, `voices`, `revision`
- Config source: `config.yml` at repo root (list of `model_configs`)

**RequestInput Hierarchy:**
- Purpose: Typed internal representation of a request destined for a specific model type
- Examples: `harmonyspeech/common/inputs.py`
- Types: `TextToSpeechRequestInput`, `SpeechEmbeddingRequestInput`, `SynthesisRequestInput`, `VocodeRequestInput`, `VoiceConversionRequestInput`, `AudioConversionRequestInput`, `SpeechTranscribeRequestInput`, `DetectVoiceActivityRequestInput`
- Pattern: `RequestInput` base class; each subclass has a `from_openai(request_id, openai_request)` factory classmethod

**EngineRequest:**
- Purpose: Wrapper that pairs a `request_id` with `request_data` (a `RequestInput`) and tracks lifecycle
- Location: `harmonyspeech/common/request.py`

**model_type String Key:**
- Purpose: Central dispatch identifier used in `inputs.py`, `model_runner_base.py`, and `loader.py` to select the correct code path for each model
- Known values: `HarmonySpeechEncoder`, `HarmonySpeechSynthesizer`, `HarmonySpeechVocoder`, `OpenVoiceV1Synthesizer`, `OpenVoiceV1ToneConverter`, `OpenVoiceV1ToneConverterEncoder`, `OpenVoiceV2ToneConverter`, `OpenVoiceV2ToneConverterEncoder`, `MeloTTSSynthesizer`, `FasterWhisper`, `SileroVAD`, `VoiceFixerRestorer`, `VoiceFixerVocoder`, `KittenTTSSynthesizer`
- Pattern: `if model_config.model_type == "X":` chains throughout `task_handler/inputs.py`

**Serving Modules:**
- Purpose: One module per API capability; bridge OpenAI protocol to internal engine calls
- Examples: `harmonyspeech/endpoints/openai/serving_text_to_speech.py`, `serving_speech_to_text.py`, `serving_voice_conversion.py`, `serving_voice_activity_detection.py`, `serving_voice_embed.py`, `serving_audio_conversion.py`
- Pattern: Class with async `generate()` / `detect()` / `embed()` methods; references `AsyncHarmonySpeech` engine instance

## Entry Points

**HTTP API Server:**
- Location: `harmonyspeech/endpoints/openai/api_server.py`
- Triggers: Started via `harmonyspeech/endpoints/cli.py` or directly via `uvicorn`
- Responsibilities: Mounts all route handlers, initializes `AsyncHarmonySpeech` engine with `EngineArgs` parsed from CLI/config, exposes `/v1/audio/speech`, `/v1/audio/transcriptions`, `/v1/audio/voice-activity`, `/v1/audio/embeddings`, `/v1/audio/conversion`, `/v1/audio/voice-conversion`, `/v1/models`

**CLI Entry:**
- Location: `harmonyspeech/endpoints/cli.py`
- Triggers: `python -m harmonyspeech` or `uvicorn harmonyspeech.endpoints.openai.api_server:app`

## Error Handling

**Strategy:** Exception-based with HTTP error responses at the API boundary

**Patterns:**
- Pydantic validation errors surface as 422 responses automatically via FastAPI
- Internal model errors raise `ValueError` or `NotImplementedError` in `task_handler/inputs.py` and propagate up
- `ErrorResponse` Pydantic model defined in `protocol.py` for structured error JSON
- `loguru` logger used throughout for structured logging of errors

## Cross-Cutting Concerns

**Logging:** `loguru` (`from loguru import logger`) used consistently across all layers

**Validation:** Pydantic v2 models for all API request/response types; `protocol.py` is the single source of protocol truth

**Authentication:** Basic auth middleware present (`auth/` directory); applied at FastAPI middleware level in `api_server.py`

**Batching:** `ThreadPoolExecutor` used inside `task_handler/inputs.py` for parallel input preprocessing within a batch; model inference itself is single-threaded per worker (PyTorch/ONNX)

**Model Loading:** HuggingFace Hub `snapshot_download` via `modeling/hf_downloader.py`; models cached locally; supports CPU, CUDA, ROCm device targets

**Audio Data Transport:** All audio transmitted as base64-encoded strings in JSON body; decoded at the worker layer immediately before inference

---

*Architecture analysis: 2026-02-28*

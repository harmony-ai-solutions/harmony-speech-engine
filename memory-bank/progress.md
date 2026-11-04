# Progress: Harmony Speech Engine

## Implementation Status Overview

**Current Version**: 0.1.0-rc9  
**Overall Completion**: ~75%  
**Release Target**: Stable v0.1.0 within 1-2 months  

## Engine Modernization — Phase 3 (COMPLETE, 2026-11-04)

**Plan folder**: `.current_work/engine-modernization/` (Phase 3 of a 6-phase modernization; `summary.md` tracks all phases)

- ✅ **3-1 PipelineSpec model & registry** — `Step`/`PipelineSpec`/`EntryRule`/`PipelineState` plus `next_step()`/`advance()` in `harmonyspeech/processing/pipeline_registry.py`; artifact-first `needs`, declared `produces`/`clears`; `EngineRequest.pipeline_state`.
- ✅ **3-2 Family migration** — registry holds 11 specs (8 client-facing toolchains + 3 single-step executors); both `if/elif` ladders deleted (~250 lines); routing/forwarding spec-driven; error guard, explicit-model short-circuit and Phase 6 extension point preserved.
- ✅ **3-3 Startup validation & model cards** — `validate_registry()` names each missing step; `--strict-pipelines`/`HSE_STRICT_PIPELINES`; cards generated from specs (capabilities/languages/members) and every per-serving-module hand list deleted; `/v1/models` ids unchanged.
- ✅ **3-4 VAD switch & entry points** — Silero emits whisper-schema `VadArtifact` segments (backend-aware encoder filter); `input_vad_mode` honoured with fallback+warning; `EntryRule` set replaces field-presence entry hacks (`single_speaker_tts` FIXME resolved); forwarded requests never re-resolve entry.
- ✅ **3-5 Routing tests & exit** — `tests/unit/inference_flow/test_pipeline_routing.py` parity table (40 cases over every family × entry shape), artifact-flow, per-step error, missing-artifact, disconnect, registry-only extension and strict-boot tests.

**Verification**: 469 unit + 23 integration green; wire-contract e2e 38 passed / 3 model-only skips; real-model e2e green on CPU for all TTS families (HarmonySpeech with `HSE_ALLOW_UNSAFE_CHECKPOINT_LOAD=1`), Whisper STT, Silero/Whisper VAD, VoiceFixer, plus a new Silero-VAD-in-pipeline cloning test.

**Next**: Phase 4 — VRAM (fp16 rollout + Chatterbox weight dedup).

## Engine Modernization — Phase 2 (COMPLETE, 2026-11-04)

**Plan folder**: `.current_work/engine-modernization/` (Phase 2 of a 6-phase modernization; `summary.md` tracks all phases)

- ✅ **2-1 Artifact type system** — `harmonyspeech/common/artifacts.py` (`AudioArtifact`, `MelArtifact`, `SpeakerArtifact`, `VadArtifact`), torch.Tensor waveforms (`[1, samples]` float32, `from_numpy`/`.numpy`), explicit `to(device)`, `EngineRequest.artifacts` + deterministic `clear_artifacts()` release on finish.
- ✅ **2-2 Artifact fields & prepare_inputs** — precedence getters decode each inbound field once and cache it; every `prepare_*` returns artifacts; admission decode-once (`add_request` / `wire.from_wire_audio`); transient resampled views instead of per-rate caches.
- ✅ **2-3 Executor artifact handlers** — all 8 hops converted (VAD→encoder, encoder→synth/TC, synth→TC, Chatterbox embed→TTS/VC, VoiceFixer restorer→vocoder, HarmonySpeech synth→vocoder, final outputs); output classes gained optional artifact slots; forwarding copies artifacts, not strings; device asserts + explicit moves; Chatterbox conditionals by reference. Removed `ensure_librosa_readable_bytes` from the pipeline; fixed Chatterbox's bundled default voice path.
- ✅ **2-4 Boundary serialization & compatibility** — `endpoints/openai/wire.py` (`from_wire_audio`/`to_wire_audio`/`to_wire_embedding`), all six serving modules migrated, engine core free of wire encoding (pytest grep invariant), default WAV responses byte-identical, embed response format unchanged.
- ✅ **2-5 Verification & benchmark** — `scripts/bench_pipeline_hops.py` + `docs/benchmarks/artifact-bus.md`; stub microbenchmark (VoiceFixer mel hop 25.4 ms → 0.005 ms/s of audio) and real-weight CPU runs; determinism: VoiceFixer/HarmonySpeech byte-identical, MeloTTS stochastic (Δduration 1.46 %).

**Verification**: 346 unit + 23 integration green; wire-contract e2e 38 passed / 3 model-only skips. Real-model e2e ran green on CPU for HarmonySpeech, OpenVoice V1/V2, VoiceFixer, Chatterbox/Turbo, KittenTTS, Whisper STT, Silero/Whisper VAD. Outstanding: pre-Phase-2 hash-fixture comparison and CUDA numbers.

**Next**: Phase 3 — declarative `PipelineSpec` routing.

## Engine Modernization — Phase 1 (COMPLETE, 2026-11-04)

**Plan folder**: `.current_work/engine-modernization/` (Phase 1 of a 6-phase modernization; `summary.md` tracks all phases)

- ✅ **1-1 Boundary audio decoder** — `harmonyspeech/common/audio_codec.py` sniffs → decodes with libsndfile → falls back to an optional ffmpeg subprocess → raises typed `AudioDecodeError`. Every `librosa.load` boundary decode in `task_handler/inputs.py` now goes through it, and Chatterbox reference audio accepts mp3/aac too. SoXR resampling (`soxr` pinned).
- ✅ **1-2 Output format & sample rate** — `output_options` is validated at admission (400 before synthesis) and applied once at the response boundary; default path is byte-identical WAV passthrough. `pcm` (raw s16le) and `aac` (ffmpeg-only) supported; responses echo the effective `sample_rate`.
- ✅ **1-3 Binary input envelope** — every audio-input endpoint accepts OpenAI-style `multipart/form-data` uploads on the same path via content-type dispatch, JSON stays primary for `/v1/audio/speech`; configurable upload limit returns 413.
- ✅ **1-4 Wire-contract tests & docs** — parametrized contract matrix (input × transport × output), `docs/audio-io-contract.md`, README/CHANGELOG/CONCERNS updates.

**Verification**: 281 unit + 23 integration green; wire-contract e2e matrix (38 passed / 3 skipped without model weights). Real-model e2e remains manual (`HSE_E2E_MODEL_TESTS=1`).

**Next**: Phase 2 — artifact bus (typed intermediates, decode once per request).

## Engine Modernization — Phase 0 (COMPLETE, 2026-11-04)

**Plan folder**: `.current_work/engine-modernization/` (Phase 0 of a 6-phase modernization; `summary.md` tracks all phases)

- ✅ **0-1 Safe deserialization** — request embeddings and model checkpoints load without executing pickle code; unsupported/hostile payloads fail closed with a typed error.
- ✅ **0-2 Per-request state isolation** — Chatterbox conditioning is restored to the post-load baseline after every request; requests without conditioning fail loudly instead of inheriting the previous voice.
- ✅ **0-3 Batch pairing correctness** — mixed-type batches are partitioned before preparation, worker helpers receive their request explicitly, and alignment is asserted.
- ✅ **0-4 Startup validation & no-hang** — declarative family registry, warnings for incomplete toolchains, immediate error on unresolvable routing, scheduler unknown-model safety net, only complete toolchains advertised.
- ✅ **0-5 Timeouts & health** — per-request deadlines (`request_timeout_s`), recoverable loop watchdog with consecutive-trip escalation, structured `check_health` payload.
- ✅ **0-6 Exit review** — unit (162) and integration (23) suites green; E2E (39) collects but requires model downloads (manual verification outstanding).

**Next**: Phase 2 — artifact bus (typed intermediates between pipeline steps).

## Core System Components

### ✅ Engine Architecture (COMPLETE)

**HarmonySpeechEngine Core**
- ✅ Multi-executor initialization and management
- ✅ Request lifecycle management
- ✅ Model configuration loading and validation
- ✅ Health check and monitoring infrastructure
- ✅ Statistics collection and logging

**Request Scheduler**
- ✅ Request queue management
- ✅ Resource allocation and optimization
- ✅ Status lifecycle tracking (WAITING → RUNNING → FINISHED)
- ✅ Request forwarding for multi-step workflows
- ✅ Batch processing coordination

**Executor Framework**
- ✅ CPU executor implementation
- ✅ GPU executor with async support
- ✅ Parallel execution using ThreadPoolExecutor
- ✅ Resource management and optimization
- ✅ Error handling and recovery

### ✅ Request Routing System (COMPLETE)

**Intelligent Model Selection**
- ✅ Harmony Speech routing (Encoder → Synthesizer → Vocoder)
- ✅ OpenVoice V1 routing (VAD → Embedding → Synthesis → Voice Conversion)
- ✅ OpenVoice V2 routing (VAD → Embedding → MeloTTS → Voice Conversion)
- ✅ Language-based model selection
- ✅ Automatic workflow orchestration

**Multi-Step Processing**
- ✅ Request forwarding between models
- ✅ Status management for complex workflows
- ✅ Data transformation between processing steps
- ✅ Error propagation and handling

### ✅ API Layer (COMPLETE)

**FastAPI Implementation**
- ✅ OpenAI-compatible endpoints
- ✅ Extended functionality for voice cloning
- ✅ Automatic OpenAPI/Swagger documentation
- ✅ Request validation with Pydantic
- ✅ Async request handling

**Endpoint Coverage**
- ✅ `/v1/audio/speech` - Text-to-speech synthesis
- ✅ `/v1/audio/transcriptions` - Speech-to-text
- ✅ `/v1/audio/embeddings` - Speaker embeddings
- ✅ `/v1/audio/voice-conversion` - Voice conversion
- ✅ `/v1/audio/convert` - Audio restoration and filtering
- ✅ `/docs` - Interactive API documentation
- ✅ `/redoc` - Alternative documentation interface

## Model Integration Status

### ✅ Harmony Speech V1 (COMPLETE)
- ✅ HarmonySpeechEncoder - Speaker embedding generation
- ✅ HarmonySpeechSynthesizer - Text-to-speech synthesis
- ✅ HarmonySpeechVocoder - Audio vocoding
- ✅ Multi-step pipeline integration
- ✅ Configuration and device management

### ✅ OpenVoice V1 (COMPLETE)
- ✅ OpenVoiceV1Synthesizer - English and Chinese TTS
- ✅ OpenVoiceV1ToneConverter - Voice conversion
- ✅ OpenVoiceV1ToneConverterEncoder - Speaker embedding
- ✅ Emotion support (whispering, shouting, excited, etc.)
- ✅ Multi-step workflow integration

### ✅ OpenVoice V2 / MeloTTS (COMPLETE)
- ✅ MeloTTSSynthesizer - Multilingual synthesis
- ✅ Language support: English, Chinese, Spanish, French, Japanese, Korean
- ✅ OpenVoiceV2ToneConverter - Advanced voice conversion
- ✅ OpenVoiceV2ToneConverterEncoder - Improved embeddings
- ✅ Cross-lingual voice cloning capabilities

### ✅ Faster-Whisper (COMPLETE)
- ✅ Speech recognition and transcription
- ✅ Voice Activity Detection (VAD)
- ✅ Multiple model sizes (tiny, medium, large-v3-turbo)
- ✅ Language detection and multilingual support
- ✅ Integration with voice cloning workflows

### ✅ VoiceFixer (COMPLETE)
- ✅ VoiceFixerRestorer - Audio denoising and enhancement
- ✅ VoiceFixerVocoder - Mel-spectrogram to audio conversion
- ✅ Audio restoration for degraded speech
- ✅ Support for noise, reverberation, and clipping removal
- ✅ Integration with HSE model loading system

### ✅ Silero VAD (COMPLETE)
- ✅ High-performance voice activity detection
- ✅ CPU-optimized ONNX implementation
- ✅ Dynamic parameter support for fine-tuning
- ✅ Enhanced `/v1/audio/vad` endpoint with configurable parameters
- ✅ Native model integration following HSE patterns

### ✅ KittenTTS (COMPLETE)
- ✅ KittenTTSSynthesizer — ONNX-based English TTS
- ✅ 4 model variants: mini (80MB), micro (41MB), nano-fp32 (56MB), nano-int8 (25MB)
- ✅ 8 voices: Bella, Jasper, Luna, Bruno, Rosie, Hugo, Kiki, Leo
- ✅ Phonemizer: misaki[en]
- ✅ 4 e2e tests validated and passing on CPU

### ✅ Chatterbox TTS (COMPLETE — All 7 Phases)
- ✅ Phase 1: Dependencies installed — chatterbox-tts==0.1.6 (--no-deps), resemble-perth, pyloudnorm, einops, omegaconf, conformer, s3tokenizer, spacy-pkuseg; all 7 import tests pass
- ✅ Phase 2: Model Registration — ChatterboxTTS, ChatterboxVC, ChatterboxMultilingualTTS, ChatterboxTurboTTS in ModelRegistry
- ✅ Phase 3: Input Preparation — speed, normalize_audio (renamed from norm_loudness), split_text parameters
- ✅ Phase 4: Model Execution — execute methods and dispatch
- ✅ Phase 5: Request Routing — Chatterbox request routing to engine and serving layer, embed-dispatch support
- ✅ Phase 6: Configuration & Performance — Chatterbox entries in config.yml/config.gpu.yml, watermarker swap
- ✅ Phase 7: Testing & Documentation — Comprehensive unit and e2e tests for all 4 variants

### 🔄 Planned Model Integrations (IN PROGRESS)

**StyleTTS 2** (Priority: High)
- ❌ Text-to-speech synthesis
- ❌ Voice conversion capabilities
- ❌ Few-shot voice cloning
- ❌ Integration with existing workflows

**XTTS V2** (Priority: High)
- ❌ Multilingual zero-shot voice cloning
- ❌ Real-time synthesis capabilities
- ❌ Cross-lingual voice transfer
- ❌ Streaming support

**Vall-E-X** (Priority: Medium)
- ❌ Advanced zero-shot voice cloning
- ❌ Multilingual support
- ❌ High-quality synthesis
- ❌ Integration framework

**Additional Models** (Priority: Low)
- ❌ CosyVoice - Multilingual TTS
- ❌ EmotiVoice - Emotional TTS with multiple speakers
- ❌ SenseVoice - Advanced speech recognition

## Frontend Development

### ✅ React Application (COMPLETE)
- ✅ Modern React 18 + Vite setup
- ✅ TypeScript integration
- ✅ TailwindCSS styling
- ✅ Responsive design implementation

### ✅ Core Components (COMPLETE)
- ✅ TTS module with voice selection
- ✅ Audio player with waveform visualization
- ✅ Settings management interface
- ✅ Performance metrics display
- ✅ Interactive API testing interface

### 🔄 Frontend Enhancements (PLANNED)
- ❌ Real-time processing status updates
- ❌ Advanced audio editing capabilities
- ❌ Batch processing interface
- ❌ Model performance comparison tools

## Infrastructure and Deployment

### ✅ Docker Support (COMPLETE)
- ✅ Multi-stage Dockerfile for different targets
- ✅ Docker Compose configurations
- ✅ NVIDIA GPU support (docker-compose.nvidia.yml)
- ✅ AMD GPU support (docker-compose.amd.yml)
- ✅ CPU-only deployment option

### ✅ Configuration Management (COMPLETE)
- ✅ YAML-based model configuration
- ✅ Environment variable support
- ✅ Device assignment and resource allocation
- ✅ Flexible deployment options

### 🔄 Production Features (IN PROGRESS)
- 🔄 Comprehensive monitoring and logging
- ❌ Health check endpoints
- ❌ Horizontal scaling support
- ❌ Load balancing configuration
- ❌ Performance optimization guides

## Testing and Quality Assurance

### ✅ Testing Infrastructure (ESTABLISHED)
- ✅ 162 unit tests — config parsing, dtype resolution, engine init, model loader registry, safe deserialization, state isolation, batch pairing, startup validation, routing/timeout resilience
- ✅ 23 integration tests — all 7 HTTP endpoints via mocked serving layer
- ✅ 39 e2e tests — full model pipelines for KittenTTS, MeloTTS, OpenVoice V1/V2, HarmonySpeech, FasterWhisper, SileroVAD, VoiceFixer (require model downloads)
- ✅ 80% overall coverage
- ❌ GPU executor path untested (gpu_model_runner.py, gpu_worker.py — 0% coverage)
- ❌ Performance benchmarking suite
- ❌ Load testing framework

### 🔄 Documentation (PARTIAL)
- ✅ API documentation (auto-generated via Swagger/ReDoc)
- ✅ Setup and installation guides
- 🔄 Model configuration examples
- ❌ Advanced usage tutorials
- ❌ Troubleshooting guides
- ❌ Performance optimization documentation

## Performance and Optimization

### ✅ Core Optimizations (COMPLETE)
- ✅ Parallel model execution
- ✅ Resource-aware scheduling
- ✅ Memory management for GPU/CPU
- ✅ Request batching capabilities

### 🔄 Advanced Optimizations (IN PROGRESS)
- 🔄 Model caching and sharing
- ❌ Dynamic model loading/unloading
- ❌ Request prioritization
- ❌ Adaptive quality settings
- ❌ Streaming capabilities

### ❌ Performance Features (PLANNED)
- ❌ TTS streaming for real-time applications
- ❌ Voice conversion post-processing pipelines
- ❌ Batch processing optimization
- ❌ Memory pressure handling

## Known Issues and Limitations

### Current Limitations

**Model Memory Usage**
- Large models require significant GPU/CPU memory
- Limited concurrent model loading capacity
- No automatic model unloading based on usage

**Windows Support**
- Limited Windows compatibility
- CUDA support tested but not fully optimized
- Recommendation to use WSL for Windows users

**Error Handling**
- Complex workflow debugging can be challenging
- Limited error recovery for multi-step processes
- Need for enhanced debugging interfaces

**Performance**
- No streaming support for real-time applications
- Batch processing not fully optimized
- Memory management could be improved

### Known Bugs

**Minor Issues**
- Occasional GPU memory leaks with repeated requests
- Configuration validation could be more comprehensive
- Frontend error handling needs improvement

**Documentation Gaps**
- Missing advanced configuration examples
- Limited troubleshooting documentation
- Need for more integration examples

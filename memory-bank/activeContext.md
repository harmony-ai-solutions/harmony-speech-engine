# Active Context: Harmony Speech Engine

## Current Development Phase

**Release Status**: Release Candidate 9 (v0.1.0-rc9)
**Development Focus**: Chatterbox TTS integration and feature completion for v0.1.0 release
**Target Timeline**: Preparing for stable v0.1.0 release

## Recent Changes and Developments

### Engine Modernization — Phase 3 (COMPLETE, 2026-11-04)

**Current workstream**: `.current_work/engine-modernization/` — foundation-first modernization. Phases 0–3 are complete; Phases 4–6 are queued per `summary.md`.

- Routing is declarative: `harmonyspeech/processing/pipeline_registry.py` holds `PIPELINE_SPECS` (11 specs: 8 client-facing toolchains + 3 single-step executors). `next_step()` resolves the model for the request's `PipelineState`, `advance()` maps typed artifacts onto the next step. The `reroute_request_*` ladders and the output-class-sniffing forwarding branches were deleted (~250 lines).
- Startup validation is spec-driven (`validate_registry`) and names the missing step; `--strict-pipelines` / `HSE_STRICT_PIPELINES=1` turns it into a boot error. `/v1/models` cards are generated from specs (capabilities, member executors, variant languages) — every per-serving-module hand list is gone; the advertised id set on `config.yml` is unchanged.
- `input_vad_mode: "silero"` now selects Silero VAD for OpenVoice cloning (with a warning + Whisper fallback when it is not configured); Silero emits whisper-schema segments (seconds, empty text) and the encoder's text-length filter is backend aware.
- `mode: "single_speaker_tts"` is an entry rule (start and stop at synthesis) — the field-coercion FIXME is gone. Forwarded requests carry their state and never re-resolve entry.
- New routing suite `tests/unit/inference_flow/test_pipeline_routing.py`: 40-case parity table transcribed from the old ladders before deletion, artifact-flow assertions, per-step error/missing-artifact/disconnect cases, a registry-only toy toolchain, and strict-boot coverage.
- Recorded deviation: without a dedicated Chatterbox embedding executor, a cloning request is a **single** TTS hop (the executor embeds inline) — the old engine behaved the same, the docs claimed two hops.

Next up when work resumes: Phase 4 — VRAM (fp16 rollout + Chatterbox backbone dedup).

### Engine Modernization — Phase 2 (COMPLETE, 2026-11-04)

**Current workstream**: `.current_work/engine-modernization/` — foundation-first modernization. Phases 0 (correctness & security), 1 (wire contract) and 2 (artifact bus) are complete; Phases 3–6 are queued per `summary.md`.

- New typed artifact bus (`common/artifacts.py`): `AudioArtifact`, `MelArtifact`, `SpeakerArtifact`, `VadArtifact`; `EngineRequest.artifacts` store with deterministic release on finish; explicit `artifact.to(device)` moves (handlers assert the device tag).
- Precedence getters decode each inbound field exactly once and cache it; all `prepare_*` functions return artifacts; admission decode in `add_request` / `wire.from_wire_audio`.
- All eight internal hops converted: Whisper→encoder (VAD+audio artifacts), encoder→synth/TC (embedding tensor), synth→tone converter (waveform), Chatterbox embed→TTS/VC (Conditionals by reference), VoiceFixer restorer→vocoder and HarmonySpeech synth→vocoder (mel tensor + breaks), final outputs via `audio_artifact`.
- Serialization now exists only in `endpoints/openai/wire.py` (+ the codec module); a pytest invariant enforces it. `encode_artifact` keeps default WAV responses byte-identical.
- Benchmarked on CPU with real weights: VoiceFixer mel hop 25.4 ms → 0.005 ms per second of audio; OpenVoice V2 clone decodes once instead of three times; VoiceFixer/HarmonySpeech outputs byte-identical across runs (MeloTTS is sampling-stochastic). See `docs/benchmarks/artifact-bus.md`.
- Real-model e2e ran green for HarmonySpeech, OpenVoice V1/V2, VoiceFixer, Chatterbox/Turbo, KittenTTS, Whisper STT and Silero/Whisper VAD. Fixed a regression where Chatterbox `single_speaker_tts` ignored the model's bundled default voice.

Next up when work resumes: Phase 3 — declarative `PipelineSpec` routing (routing helpers `_present`/`_missing` are the seam).

### Engine Modernization — Phase 1 (COMPLETE, 2026-11-04)

**Current workstream**: `.current_work/engine-modernization/` — foundation-first modernization. Phase 0 (correctness & security) and Phase 1 (wire contract) are complete; Phases 2–6 are queued per `summary.md`.

- Centralized boundary decoder (`common/audio_codec.py`): sniff → libsndfile → optional ffmpeg → typed `AudioDecodeError`; SoXR resample; all `inputs.py` decode sites migrated; Chatterbox reference audio accepts mp3/aac.
- `output_options.format`/`sample_rate` are now real: admission-time 400, response-time conversion, byte-identical default passthrough, `pcm` s16le and ffmpeg-gated `aac`, `sample_rate` echo field.
- Multipart uploads on all six audio-input endpoints (content-type dispatch on the same path), JSON remains primary for `/v1/audio/speech`, configurable size limit → 413.
- Docs: `docs/audio-io-contract.md`; contract-matrix tests in `tests/e2e/wire_contract/` plus unit coverage in `tests/unit/wire_contract/`.

Next up when work resumes: Phase 2 — artifact bus (typed intermediates between pipeline steps).

### Engine Modernization — Phase 0 (COMPLETE, 2026-11-04)

**Current workstream**: `.current_work/engine-modernization/` — foundation-first modernization of the engine internals. Phase 0 (Correctness & security foundation) is complete; Phases 1–5 are queued per `summary.md`.

- Safe deserialization of request embeddings and checkpoints (no pickle code execution), typed 400-style rejections.
- Chatterbox per-request conditioning restored to the post-load baseline after every request.
- Type-partitioned batches, explicit request passing and alignment invariants in every executor handler.
- Declarative model-toolchain registry with startup validation; unresolvable routing now resolves as an error instead of hanging.
- Per-request deadlines, a recoverable engine-loop watchdog, and a structured health payload.

Next up when work resumes: Phase 2 — artifact bus (typed intermediates between pipeline steps).

### Chatterbox TTS Integration (MAJOR FEATURE - COMPLETE)

**Overview**
- Resemble AI's open-source TTS family: 4 model variants (ChatterboxTTS, ChatterboxVC, ChatterboxMultilingualTTS, ChatterboxTurboTTS)
- 23 supported languages (multilingual variant)
- Voice cloning with watermarking (resemble-perth)

**Phase Status (All 7 Phases Complete)**
- ✅ Phase 1: Dependencies and setup
- ✅ Phase 2: Model Registration — ChatterboxTTS, ChatterboxVC, ChatterboxMultilingualTTS, ChatterboxTurboTTS registered in ModelRegistry
- ✅ Phase 3: Input Preparation — Added speed, normalize_audio, split_text parameters
- ✅ Phase 4: Model Execution — Implemented execute methods and dispatch
- ✅ Phase 5: Request Routing — Added Chatterbox request routing to engine and serving layer
- ✅ Phase 6: Configuration & Performance — Added Chatterbox entries to config files, embed-dispatch and cache support
- ✅ Phase 7: Testing & Documentation — Comprehensive unit and e2e tests added

**Key Technical Changes**
- Implemented watermarker swap in loader for voice cloning workflows
- Added embed-dispatch support for embedding-capable TTS models

### Future: Audio Enhancements Plan (PLANNED - NOT YET IMPLEMENTED)

**Planned Features (Phase A + Phase B)**
- Text chunking + smart crossfade stitching (model-agnostic)
- Speed factor post-processing (librosa pitch-preserving)
- Peak normalization / clip prevention (model-agnostic)
- DC offset removal, leading/trailing silence trimming, internal silence reduction

**Plan Document**: `.current_work/chatterbox-audio-enhancements/summary.md`

### Core Engine Improvements

**Multi-Model Request Routing**
- Implemented sophisticated request routing system for complex workflows
- Added support for multi-step processing pipelines (VAD → Embedding → Synthesis → Voice Conversion → Audio Restoration)
- Enhanced request forwarding mechanism with proper status lifecycle management
- Improved error handling and request status tracking

**Performance Optimizations**
- Implemented parallel model execution using ThreadPoolExecutor
- Added resource-aware scheduling and batching capabilities
- Optimized memory management for GPU and CPU executors
- Enhanced request processing throughput

### Model Integration Status

**Currently Supported Models:**
- ✅ Harmony Speech V1 (Encoder, Synthesizer, Vocoder)
- ✅ OpenVoice V1 (Synthesizer EN/ZH, Tone Converter, Encoder)
- ✅ OpenVoice V2/MeloTTS (Multilingual synthesis: EN, ZH, ES, FR, JP, KR)
- ✅ Faster-Whisper (Speech recognition and VAD: tiny, medium, large-v3-turbo)
- ✅ VoiceFixer (Audio restoration: Restorer, Vocoder)
- ✅ Silero VAD (High-performance voice activity detection)
- ✅ KittenTTS (Ultra-lightweight ONNX TTS: mini, micro, nano, nano-int8)
- ✅ Chatterbox TTS (4 variants: TTS, VC, Multilingual, Turbo) — COMPLETE

**Model Integration Patterns:**
- Standardized model configuration via YAML
- Flexible device assignment (CPU/CUDA)
- Language-specific model routing
- Voice selection and emotion support
- Audio restoration and enhancement pipelines

### KittenTTS Integration - COMPLETED

**Overview**
- Ultra-lightweight ONNX-based English TTS with 4 model variants
- 8 voices: Bella, Jasper, Luna, Bruno, Rosie, Hugo, Kiki, Leo
- Phonemizer: `misaki[en]`
- Implementation: `harmonyspeech/modeling/models/kittentts/`

**Model Variants**
- `kitten-tts-mini`: Highest quality, 80MB
- `kitten-tts-micro`: Good quality, 41MB
- `kitten-tts-nano`: Lightweight, 56MB fp32
- `kitten-tts-nano-int8`: Most compact, 25MB (quantized)

**Integration**
- Native ONNX runtime via `onnxruntime` (no torch dependency for inference)
- Full integration with HSE model loading, routing and executor system
- 4 e2e tests validated and passing on CPU

### Chatterbox TTS Integration - Phase 1 Complete

**Overview**
- Resemble AI's open-source TTS family: 4 model variants (ChatterboxTTS, ChatterboxVC, ChatterboxMultilingualTTS, ChatterboxTurboTTS)
- 23 supported languages (multilingual variant)
- Voice cloning with watermarking (resemble-perth)

**Dependency Approach — Key Decision**
- `chatterbox-tts==0.1.6` declares hard version pins (torch==2.6.0, numpy<1.26.0, transformers==4.46.3, etc.) that conflict with HSE's stack
- The Python code is compatible with newer versions; the pins are overcautious
- Solution: `chatterbox-tts` installed via `pip install --no-deps -r requirements-chatterbox.txt`
- Safe transitive deps (`resemble-perth`, `pyloudnorm`, `einops`, `omegaconf`, `conformer`, `s3tokenizer`, `spacy-pkuseg`) listed in `requirements-common.txt`
- Upgrade checklist documented in `.planning/codebase/CONCERNS.md` § "Chatterbox TTS Dependency Pinning"

**Phase Status**
- ✅ Phase 1: Dependencies installed and verified — all 7 import tests pass
- ❌ Phase 2: Model Registration
- ❌ Phase 3: Input Preparation
- ❌ Phase 4: Model Execution
- ❌ Phase 5: Request Routing
- ❌ Phase 6: Configuration & Performance
- ❌ Phase 7: Testing & Documentation

### VoiceFixer Integration - COMPLETED

**Architecture Implementation**
- Successfully ported original VoiceFixer models to HSE architecture
- Implemented VoiceFixerRestorer for audio enhancement and noise reduction
- Implemented VoiceFixerVocoder for mel-spectrogram to audio conversion
- Maintained exact compatibility with original VoiceFixer preprocessing logic

**Key Technical Achievements**
- Resolved tensor dimension compatibility issues between Restorer and Vocoder
- Fixed pipeline data format consistency (Restorer outputs [batch, 1, time, 128] format)
- Implemented strict input validation matching original VoiceFixer assertions
- Removed interpolation fallbacks to maintain original behavior
- Corrected weight tensor handling and mel preprocessing steps

**Integration Features**
- Two-stage audio restoration pipeline (Restorer → Vocoder)
- Support for 44.1kHz audio processing with 128 mel bins
- Automatic model loading and weight management
- Memory-efficient segmented processing for long audio files
- Full integration with HSE request routing and executor system

**API Endpoints**
- `/v1/audio/convert` - New Audio conversion endpoint for filter models

### API Development

**OpenAI Compatibility Layer**
- Implemented OpenAI-compatible endpoints for easy migration
- Extended API with voice cloning and advanced TTS features
- Added support for complex request parameters (mode, language_id, voice selection)
- Integrated automatic model selection based on request characteristics
- Added Audio conversion endpoint for filter models

**Interactive Documentation**
- FastAPI automatic OpenAPI/Swagger generation
- ReDoc documentation interface
- Generated client libraries for Go and JavaScript
- Comprehensive API examples and usage patterns
- Updated documentation with Audio conversion endpoint

### Frontend Development

**React-based Management Interface**
- Modern React 18 + Vite development stack
- TailwindCSS for responsive design
- Interactive audio player components
- Real-time processing status and metrics display

**Key Components:**
- TTS module with voice selection and cloning options
- Audio player with waveform visualization
- Settings management with tooltips and validation
- Performance metrics and system health monitoring
- Audio restoration module configuration

## Current Work Focus

**1. Chatterbox TTS Integration (COMPLETE — All 7 Phases Done)**
- ✅ Phase 1: Dependencies and setup complete
- ✅ Phase 2: Model Registration — ChatterboxTTS, ChatterboxVC, ChatterboxMultilingualTTS, ChatterboxTurboTTS registered
- ✅ Phase 3: Input Preparation — speed, normalize_audio, split_text parameters
- ✅ Phase 4: Model Execution — execute methods and dispatch
- ✅ Phase 5: Request Routing — Chatterbox request routing to engine and serving layer
- ✅ Phase 6: Configuration & Performance — config entries, embed-dispatch, cache support
- ✅ Phase 7: Testing & Documentation — comprehensive unit and e2e tests

**2. Future: Audio Enhancements Plan (PLANNED)**
- See `.current_work/chatterbox-audio-enhancements/summary.md` for details
- Phase A: Text chunking, speed factor, peak normalization
- Phase B: DC offset removal, silence trimming, internal silence reduction

**3. Model Integration Completion (Backlog)**
- [ ] StyleTTS 2 integration for advanced voice cloning
- [ ] XTTS V2 multilingual support
- [ ] Vall-E-X zero-shot voice cloning

**4. Performance and Reliability**
- [x] Comprehensive error handling and recovery mechanisms
- [ ] Memory optimization for large model loading
- [ ] Request batching optimization for improved throughput
- [ ] GPU memory management improvements

**5. Testing and Quality Assurance**
- 80+ tests across unit / integration / e2e tiers
- [ ] GPU executor path coverage (currently 0%)
- [x] Multi-step pipeline integration tests
- [ ] Performance benchmarking

**6. Advanced Features**
- [ ] TTS streaming capabilities for real-time applications
- [ ] Input filters / pre-processing pipelines
- [ ] Voice conversion post-processing pipelines
- [ ] Advanced audio format support

**7. Production Readiness**
- [ ] Comprehensive monitoring and logging
- [x] Health check endpoints and diagnostics
- [ ] Horizontal scaling capabilities
- [ ] Production deployment guides

**8. Community and Ecosystem**
- [ ] Community model contribution guidelines
- [ ] Third-party integration examples
- [ ] Performance optimization documentation
- [ ] Advanced configuration tutorials

## Active Technical Decisions

### Architecture Decisions

**Request Processing Model**
- **Decision**: Per-request processing instead of token sequence batching
- **Rationale**: Speech processing requires complete audio/text inputs, unlike incremental token generation
- **Impact**: Simplified request handling but requires different optimization strategies

**Multi-Model Orchestration**
- **Decision**: Intelligent request routing with automatic model selection
- **Rationale**: Enables complex workflows without user knowledge of internal model architecture
- **Impact**: Increased system complexity but significantly improved user experience

**Executor Architecture**
- **Decision**: Separate CPU and GPU executors with parallel execution
- **Rationale**: Optimal resource utilization and support for mixed deployment scenarios
- **Impact**: Better performance but increased architectural complexity

### Technology Choices

**FastAPI for API Layer**
- **Decision**: FastAPI over Flask or Django
- **Rationale**: Automatic OpenAPI generation, async support, and modern Python features
- **Impact**: Excellent developer experience and documentation generation

**React + Vite for Frontend**
- **Decision**: React with Vite instead of traditional webpack setup
- **Rationale**: Fast development server, modern build tooling, and excellent TypeScript support
- **Impact**: Improved development velocity and build performance

**Docker-First Deployment**
- **Decision**: Primary focus on Docker containerization
- **Rationale**: Consistent deployment across different environments and simplified dependency management
- **Impact**: Easier deployment but requires Docker knowledge from users

## Current Challenges and Solutions

### Challenge 1: Model Memory Management

**Problem**: Large models consume significant GPU/CPU memory, limiting concurrent model loading
**Current Approach**: Lazy loading and model sharing across requests
**Planned Solution**: Implement model unloading based on usage patterns and memory pressure

### Challenge 2: Complex Workflow Debugging

**Problem**: Multi-step processing pipelines are difficult to debug when failures occur
**Current Approach**: Comprehensive request status tracking and logging
**Planned Solution**: Enhanced debugging interface with step-by-step execution visibility

### Challenge 3: Performance Optimization

**Problem**: Balancing quality and speed for real-time applications
**Current Approach**: Configurable model selection and batch processing
**Planned Solution**: Adaptive quality settings and streaming capabilities

### Challenge 4: Model Integration Complexity

**Problem**: Each new model requires significant integration effort
**Current Approach**: Standardized model configuration and loading patterns
**Planned Solution**: Plugin architecture for community-contributed models

## Development Patterns and Preferences

### Code Organization Principles

**Modular Architecture**
- Clear separation between engine core, executors, and API layers
- Standardized interfaces for model integration
- Plugin-based extensibility for new features

**Configuration-Driven Design**
- YAML-based model configuration
- Environment variable support for deployment settings
- Flexible device and resource allocation

**Error Handling Strategy**
- Comprehensive exception handling with context preservation
- Graceful degradation for non-critical failures
- Detailed logging for debugging and monitoring

### Testing Philosophy

**Test-Driven Development**
- Unit tests for core business logic
- Integration tests for API endpoints
- Performance tests for optimization validation

**Quality Assurance**
- Code formatting with Black and Prettier
- Type checking with MyPy and TypeScript
- Linting with Flake8 and ESLint

## Integration Points

### Project Harmony.AI Ecosystem

**Harmony Link Integration**
- HSE serves as speech processing backend for Harmony Link
- API compatibility for seamless integration
- Shared configuration and deployment patterns

**Community Model Contributions**
- Open architecture for community-contributed models
- Standardized integration guidelines
- Quality assurance processes for new models

### External Dependencies

**HuggingFace Hub**
- Primary source for pre-trained models
- Automatic model downloading and caching
- Version management and updates

**PyTorch Ecosystem**
- Core inference framework
- CUDA and ROCm support for GPU acceleration
- Optimization libraries for performance

**VoiceFixer Dependencies**
- TorchLibrosa for STFT operations
- Original VoiceFixer model architectures
- Mel-scale conversion utilities

## Key Learnings and Insights

### Performance Insights

**Multi-Model Parallelism Benefits**
- Significant throughput improvements with parallel execution
- Resource utilization optimization across CPU and GPU
- Reduced latency for complex workflows

**Request Routing Efficiency**
- Automatic model selection reduces user complexity
- Intelligent workflow orchestration improves user experience
- Proper status tracking essential for debugging

**VoiceFixer Integration Insights**
- Exact compatibility with original implementation crucial for reliability
- Tensor dimension validation prevents runtime errors
- Pipeline format consistency essential for multi-stage processing

### Development Insights

**Docker-First Approach Success**
- Simplified deployment and dependency management
- Consistent behavior across development and production
- Community adoption improved with containerization

**OpenAI API Compatibility Value**
- Familiar interface reduces adoption barriers
- Easy migration from existing solutions
- Extended functionality maintains flexibility

**Model Porting Best Practices**
- Maintain original preprocessing logic exactly
- Implement strict input validation
- Avoid "helpful" interpolation or format conversion
- Test against original implementation thoroughly

### Community Feedback

**User Priorities**
- Easy setup and deployment (5-minute goal)
- Reliable voice cloning capabilities
- Comprehensive documentation and examples
- Performance predictability
- Audio restoration and enhancement features

**Developer Needs**
- Clear integration examples
- Extensible architecture for custom models
- Production-ready deployment guides
- Active community support

## Next Steps and Immediate Actions

### Immediate
1. ✅ Chatterbox Phase 1 complete — dependencies installed, import tests passing
2. → Run `/gsd-plan-phase 2` to plan Chatterbox model registration
3. → Execute Phase 2: register ChatterboxTTS, ChatterboxVC, ChatterboxMultilingualTTS, ChatterboxTurboTTS in ModelRegistry

### Near Term
1. Complete Chatterbox TTS phases 2–7 (model registration → testing)
2. Implement model unloading for memory management
3. Expand e2e test coverage to GPU executor path

### This Month
1. Complete Chatterbox TTS full integration
2. Implement TTS streaming capabilities
3. Add comprehensive monitoring and health checks
4. Prepare for stable v0.1.0 release

## VoiceFixer Technical Implementation Details

### Architecture Overview
- **VoiceFixerRestorer**: Handles audio enhancement and noise reduction
- **VoiceFixerVocoder**: Converts enhanced mel-spectrograms to audio
- **Pipeline Integration**: Seamless integration with HSE request routing

### Key Components
- **FDomainHelper**: STFT operations and spectrogram conversion
- **MelScale**: Mel-spectrogram generation with 128 bins
- **UNetResComplex**: Advanced neural network for audio enhancement
- **Generator**: Original VoiceFixer generator architecture

### Data Flow
1. Audio input → STFT → Mel-spectrogram (128 bins)
2. Mel-spectrogram → VoiceFixerRestorer → Enhanced mel-spectrogram
3. Enhanced mel-spectrogram → VoiceFixerVocoder → Restored audio
4. Format validation at each stage ensures pipeline compatibility

### Configuration
- 44.1kHz sample rate support
- 128 mel bins (strict requirement)
- Configurable device assignment (CPU/CUDA)
- Memory-efficient segmented processing
- Original VoiceFixer weight normalization

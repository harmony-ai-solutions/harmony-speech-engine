# Active Context: Harmony Speech Engine

## Current Development Phase

**Release Status**: Release Candidate 9 (v0.1.0-rc9)  
**Development Focus**: Stabilization and feature completion for v0.1.0 release  
**Target Timeline**: Preparing for stable v0.1.0 release  

## Recent Changes and Developments

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
- ✅ VoiceFixer (Audio restoration: Restorer, Vocoder) - **COMPLETED**

**Model Integration Patterns:**
- Standardized model configuration via YAML
- Flexible device assignment (CPU/CUDA)
- Language-specific model routing
- Voice selection and emotion support
- Audio restoration and enhancement pipelines

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

**1. Model Integration Completion**
- ✅ VoiceFixer integration for audio restoration
- [ ] StyleTTS 2 integration for advanced voice cloning
- [ ] XTTS V2 multilingual support
- [ ] Vall-E-X zero-shot voice cloning
- [ ] Silero VAD as alternative to Faster-Whisper

**2. Performance and Reliability**
- [ ] Comprehensive error handling and recovery mechanisms
- [ ] Memory optimization for large model loading
- [ ] Request batching optimization for improved throughput
- [ ] GPU memory management improvements

**3. Testing and Quality Assurance**
- [ ] Unit test coverage for core components
- [ ] Integration tests for multi-model workflows
- [ ] Performance benchmarking and optimization
- [ ] API compatibility testing

**4. Advanced Features**
- [ ] TTS streaming capabilities for real-time applications
- [ ] Input filters / pre-processing pipelines
- [ ] Voice conversion post-processing pipelines
- [ ] Advanced audio format support

**5. Production Readiness**
- [ ] Comprehensive monitoring and logging
- [ ] Health check endpoints and diagnostics
- [ ] Horizontal scaling capabilities
- [ ] Production deployment guides

**6. Community and Ecosystem**
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

### This Week
1. ✅ Complete VoiceFixer integration and testing
2. Update documentation with VoiceFixer endpoints and usage
3. Add performance benchmarking for audio restoration workflows
4. Prepare release candidate 9 with VoiceFixer integration

### Next Week
1. Begin StyleTTS 2 integration work
2. Implement model unloading for memory management
3. Add integration tests for audio restoration workflows
4. Update API documentation and examples

### This Month
1. Complete major model integrations (StyleTTS 2, XTTS V2)
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

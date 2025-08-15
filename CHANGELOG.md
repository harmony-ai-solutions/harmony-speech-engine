# Changelog

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

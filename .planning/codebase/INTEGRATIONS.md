# External Integrations

**Analysis Date:** 2026-02-28

## APIs & External Services

**Model Hosting / Distribution:**
- **Hugging Face Hub** — Primary source for downloading model weights at runtime
  - SDK/Client: `huggingface_hub` package + `hf_xet` for fast transfer
  - Auth: `HF_TOKEN` env var (optional; needed for gated/private models)
  - Downloader: `harmonyspeech/modeling/hf_downloader.py`
  - Pattern: Models are resolved by HF repo ID (e.g., `harmony-ai/harmony-speech-v1`, `myshell-ai/openvoice`, `myshell-ai/MeloTTS-English-v3`, `KittenML/kitten-tts-mini-0.8`, `jlmarrugom/voice_fixer`)
  - Cache location: `./cache/` directory (Docker volume mount)

## Data Storage

**Databases:**
- None — Harmony Speech Engine is stateless; no database integration

**File Storage:**
- Local filesystem for model cache: `./cache/` mounted as Docker volume at `/app/harmony-speech-engine/cache`
- Models downloaded once and reused across restarts

**Caching:**
- Filesystem cache via `cached_path` package
- Hugging Face Hub cache via `huggingface_hub` snapshot download mechanism

## Authentication & Identity

**Auth Provider:**
- Custom API Key authentication
  - Implementation: `auth/apikeys.py`
  - Pattern: Bearer token / API key header validation on incoming requests
  - Config: API keys configured via `.env` file (environment variable, name not exposed in source)

**Hugging Face Authentication:**
- Optional `HF_TOKEN` environment variable for gated/private model access
- Used by `harmonyspeech/modeling/hf_downloader.py`

## Model Integrations (AI/ML Providers)

All AI models run locally — no cloud AI API calls. Models are downloaded from Hugging Face and served on the local machine.

**Harmony Speech V1 (TTS):**
- Source: `https://huggingface.co/harmony-ai/harmony-speech-v1`
- Implementation: Adapted — model code in `harmonyspeech/modeling/models/harmonyspeech/`
- Components: Encoder, Synthesizer (Forward-Tacotron based), Vocoder (MultiBand-MelGAN)
- Model types in config: `HarmonySpeechEncoder`, `HarmonySpeechSynthesizer`, `HarmonySpeechVocoder`

**OpenVoice V1 (TTS + Voice Conversion):**
- Source: `https://huggingface.co/myshell-ai/OpenVoice` (MyShell AI)
- Implementation: Adapted — model code in `harmonyspeech/modeling/models/openvoice/`
- Languages: EN, ZH
- Model types in config: `OpenVoiceV1Synthesizer`, `OpenVoiceV1ToneConverter`, `OpenVoiceV1ToneConverterEncoder`
- Dependency: Requires Faster-Whisper for embedding step

**OpenVoice V2 / MeloTTS (TTS + Voice Conversion):**
- Source: `https://huggingface.co/myshell-ai/MeloTTS-*` and `https://huggingface.co/myshell-ai/openvoicev2`
- Implementation: Adapted — model code in `harmonyspeech/modeling/models/melo/` and `harmonyspeech/modeling/models/openvoice/`
- Languages: EN, ZH, ES, FR, JP, KR
- Model types in config: `MeloTTSSynthesizer`, `OpenVoiceV2ToneConverter`, `OpenVoiceV2ToneConverterEncoder`
- Dependency: Requires Faster-Whisper for embedding step

**Faster-Whisper / Distil-Whisper (STT + VAD):**
- Source: SYSTRAN's `faster-whisper` library (PyPI: `faster-whisper>=1.1.0`); models from OpenAI via Hugging Face
- Implementation: Native / Third-Party — uses the `faster_whisper` package directly
- Model types in config: `FasterWhisper`
- Supports all Faster-Whisper model tags: `tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3`, `large-v3-turbo`
- Backend: CTranslate2 (`ctranslate2` package)

**Silero VAD (Voice Activity Detection):**
- Source: `https://huggingface.co/onnx-community/silero-vad`; PyTorch source at `snakers4/silero-vad`
- Implementation: Native / Third-Party — uses `silero-vad` package + `onnxruntime`
- Model type in config: `SileroVAD`
- Load format: `onnx` (ONNX version for CPU-optimized inference)

**VoiceFixer (Audio Restoration):**
- Source: `https://huggingface.co/jlmarrugom/voice_fixer` (alternative: `cqchangm/voicefixer`)
- Implementation: Adapted — model code in `harmonyspeech/modeling/models/voicefixer/`
- Components: Restorer (denoising) + Vocoder (mel-spectrogram to audio)
- Model types in config: `VoiceFixerRestorer`, `VoiceFixerVocoder`

**KittenTTS (ONNX TTS):**
- Source: `https://huggingface.co/KittenML/kitten-tts-mini-0.8` (and micro/nano variants)
- Implementation: Adapted — model code in `harmonyspeech/modeling/models/kittentts/`
- Runtime: ONNX via `onnxruntime`
- Phonemizer: `misaki[en]>=0.9.4`
- Model type in config: `KittenTTSSynthesizer`
- Voices: Bella, Jasper, Luna, Bruno, Rosie, Hugo, Kiki, Leo

**Chatterbox (TTS):**
- Implementation: Adapted — model code in `harmonyspeech/modeling/models/chatterbox/`
- Components: s3gen (Matcha-based), s3tokenizer, t3 transformer, voice encoder
- Status: Present in codebase, not yet listed in default `config.yml`

## API Protocol

**Exposed API (Outward-facing):**
- Protocol: OpenAI-compatible REST API
- Framework: FastAPI + Uvicorn
- Default port: `12080`
- Base path: `/v1/`
- Documentation: Swagger UI at `/docs`, ReDoc at `/redoc`
- OpenAPI spec: `docs/api/` directory (for client generation)

**Endpoints:**
- `POST /v1/audio/speech` — Text-to-Speech
- `POST /v1/audio/transcriptions` — Speech-to-Text
- `POST /v1/audio/vad` — Voice Activity Detection
- `POST /v1/audio/convert` — Audio Conversion (e.g., VoiceFixer restoration)
- `POST /v1/audio/voice-convert` — Voice Conversion (tone transfer)
- `POST /v1/audio/embed` — Speaker Embedding
- `POST /v1/audio/synthesize` — Low-level Synthesis (synthesizer step only)
- `POST /v1/audio/vocode` — Low-level Vocoding step
- `GET /v1/models` — List loaded models

**Request/Response format:**
- JSON with base64-encoded audio payloads
- Audio format configurable: default `wav`, sample rate optional
- Defined in `harmonyspeech/endpoints/openai/protocol.py`

## Monitoring & Observability

**Metrics:**
- `prometheus_client` — Prometheus metrics endpoint
- Metrics implementation: `harmonyspeech/common/metrics.py` and `harmonyspeech/engine/metrics.py`

**Logs:**
- `loguru` — Structured logging throughout application
- Logger setup: `harmonyspeech/common/logger.py`

## CI/CD & Deployment

**Hosting:**
- Docker containers published to Docker Hub: `harmonyai/harmonyspeech-engine-cpu:latest`, `harmonyai/harmonyspeech-engine-nvidia:latest`
- UI container: `harmonyai/harmonyspeech-ui:latest`

**CI Pipeline:**
- GitHub Actions (`.github/` directory present); workflows not examined in detail

**Container Variants:**
- CPU: `docker-compose.yml` → `harmonyai/harmonyspeech-engine-cpu:latest`
- NVIDIA GPU: `docker-compose.nvidia.yml` → `harmonyai/harmonyspeech-engine-nvidia:latest` (requires `nvidia` driver capability)
- AMD GPU: `docker-compose.amd.yml` (ROCm-based)
- Docker build contexts: `docker/` directory

## Environment Configuration

**Required env vars (from `.env`):**
- `HF_TOKEN` — Optional Hugging Face token for private model access
- API key configuration for endpoint auth (configured in `auth/apikeys.py`)

**Secrets location:**
- `.env` file (local, not committed to git; `.gitignore` enforced)
- `.env-amd` for AMD GPU variant

## Webhooks & Callbacks

**Incoming:** None — purely request/response REST API

**Outgoing:** None — all processing is local inference only

---

*Integration audit: 2026-02-28*

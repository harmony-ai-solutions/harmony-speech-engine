# Technology Stack

**Analysis Date:** 2026-03-11

## Languages

**Primary:**
- Python 3.12+ — All inference engine code, API server, model implementations
- YAML — Model configuration (`config.yml`, `config.gpu.yml`)

## Runtime

**Environment:**
- Python ≥ 3.12 (enforced in `pyproject.toml` via `requires-python = ">=3.12"`)

**Package Manager:**
- pip with `requirements-*.txt` split files + `pyproject.toml`
- Lockfile: Not present (no `requirements.lock` or `poetry.lock`)

**Requirement Files:**
- `requirements-common.txt` — Base dependencies for all platforms (63 lines)
- `requirements-cpu.txt` — Extends common, for CPU-only deployments
- `requirements-cuda.txt` — Extends common, for NVIDIA GPU deployments
- `requirements-rocm.txt` — Extends common, for AMD GPU deployments

## Frameworks

**Core Web / API:**
- `fastapi` — HTTP API server (OpenAI-compatible REST API)
- `uvicorn` — ASGI server running FastAPI
- `pydantic` — Request/response models and validation (used throughout `harmonyspeech/endpoints/openai/protocol.py`)

**Deep Learning:**
- `torch` ≥ 2.7.1 — Primary tensor computation framework (required at build time in `pyproject.toml`)
- `torchaudio` — Audio tensor processing
- `torchlibrosa` — Librosa-compatible audio feature extraction on PyTorch

**Inference Acceleration:**
- `onnxruntime` — ONNX model inference (used for Silero VAD and KittenTTS)
- `ctranslate2` == 4.4.0 — Quantized fast inference backend (used by Faster-Whisper)
- `faster-whisper` ≥ 1.1.0 — High-performance Whisper wrapper using CTranslate2
- `safetensors` — Safe model weight loading format

**Audio Processing:**
- `librosa` — Audio analysis and feature extraction
- `numba` — JIT-compiled numerical code (accelerates librosa)
- `scipy` — Scientific computing and signal processing
- `soundfile` — Reading/writing audio files
- `pydub` — Audio manipulation (format conversion, slicing)
- `webrtcvad` — WebRTC Voice Activity Detection
- `silero-vad` — State-of-the-art VAD model

**TTS Engines / Providers:**
- **KittenTTS** — Ultra-lightweight ONNX-based TTS (English only)
  - Runtime: ONNX via `onnxruntime`
  - Phonemizer: `misaki[en]>=0.9.4`
  - Models: `KittenML/kitten-tts-mini-0.8`, `kitten-tts-micro-0.8`, `kitten-tts-nano-0.8-fp32`, `kitten-tts-nano-0.8-int8`
  - Implementation: `harmonyspeech/modeling/models/kittentts/`
- **MeloTTS** — Multi-lingual TTS (EN, ZH, ES, FR, JP, KR)
- **OpenVoice V1/V2** — TTS and voice conversion
- **HarmonySpeech V1** — Custom TTS model
- **VoiceFixer** — Audio restoration/denoising

**STT / ASR Engines:**
- **Faster-Whisper** — Whisper-based speech-to-text with CTranslate2 backend
- **Silero VAD** — Voice activity detection

**NLP / Phonemization / G2P:**
- `transformers` — Hugging Face transformer models (BERT-based text modules in MeloTTS)
- `g2p_en` — English grapheme-to-phoneme
- `g2pkk` — Korean G2P
- `gruut[de,es,fr]` — Multilingual phonemizer (German, Spanish, French)
- `eng_to_ipa` — English to IPA conversion
- `misaki[en]` ≥ 0.9.4 — Phonemizer for KittenTTS
- `espeakng_loader` — eSpeak-NG wrapper
- `nltk` — Natural Language Toolkit
- `spacy` — Industrial-strength NLP
- `mecab-python3` + `unidic-lite` + `pykakasi` — Japanese text processing
- `eunjeon` + `fugashi` — Korean text processing
- `cn2an` + `pypinyin` + `jieba` — Chinese text processing
- `num2words` — Number to words conversion

**Model Distribution:**
- `huggingface_hub` — Download models from Hugging Face Hub
- `hf_xet` — High-speed Hugging Face transfer protocol
- `cached_path` — Resolve and cache local/remote paths
- `h5py` — HDF5 file format (model weight storage)

**Utilities:**
- `loguru` — Structured application logging
- `prometheus_client` — Metrics exposition
- `psutil` — System resource monitoring
- `packaging` — Version parsing utilities
- `numpy` — Numerical arrays
- `matplotlib` — Plotting (used internally)
- `inflect` + `anyascii` + `unidecode` + `num2words` — Text normalization helpers
- `jamo` — Korean jamo character handling
- `pyyaml` — YAML config parsing

## Testing Stack

**Test Framework:**
- `pytest` >= 8.0.0 — Test runner
- `pytest-asyncio` >= 0.23.0 — Async test support
- `pytest-mock` >= 3.12.0 — Mocking utilities
- `pytest-cov` >= 5.0.0 — Coverage reporting

**Linting / Code Quality:**
- `black` >= 24.0.0 — Code formatting (line-length: 100, target: py312)
- `flake8` >= 7.0.0 — Code style checking (max-line-length: 100, ignores: E203, W503)

**Pytest Configuration (in `pyproject.toml`):**
- Test paths: `tests/`
- Async mode: `auto`
- Coverage: `--cov=harmonyspeech`, `--cov-report=term-missing`, `--cov-report=xml`
- Markers:
  - `unit` — Fast, fully mocked unit tests
  - `integration` — Component interaction tests, partially mocked
  - `e2e` — Real model tests, slow, CPU-only
  - `slow` — Tests taking >30 seconds
- Filter warnings: DeprecationWarning, UserWarning ignored

## CI/CD Stack

**GitHub Actions:**
- **Test workflow** (`.github/workflows/test.yml`):
  - Unit & Integration Tests: `python -m pytest --device=cpu --dtype=float32 -v --tb=short -m "not e2e"`
  - Lint: black + flake8 checks
  - E2E tests by model type:
    - E2E KittenTTS (CPU)
    - E2E MeloTTS and OpenVoice V2 (CPU)
    - E2E OpenVoice V1 (CPU)
    - E2E HarmonySpeech (CPU)
    - E2E Whisper STT (CPU)
    - E2E SileroVAD (CPU)
    - E2E VoiceFixer (CPU)

- **Docker Release workflow** (`.github/workflows/docker-release-engine.yml`):
  - Builds and pushes to Docker Hub
  - Builds for: CPU, AMD, AMD-WSL, NVIDIA
  - Tags: versioned + `latest`

**Container Registry:**
- Docker Hub: `harmonyai/harmonyspeech-engine-{variant}:{version}`
- Variants: `cpu`, `nvidia`, `amd`, `amd-wsl`
- UI: `harmonyai/harmonyspeech-ui:latest`

## Docker / Deployment Stack

**Docker Compose:**
- `docker-compose.yml` — CPU deployment (port 12080)
- `docker-compose.nvidia.yml` — NVIDIA GPU deployment
- `docker-compose.amd.yml` — AMD ROCm deployment

**Deployment Configuration:**
- API exposed on port `12080`
- UI exposed on port `8080`
- Model config: mounted at `/app/harmony-speech-engine/config.yml`
- Model cache: `./cache/` directory (Docker volume mount)

## Key Dependencies

**Critical:**
- `torch` ≥ 2.7.1 — All neural network inference depends on PyTorch
- `fastapi` + `uvicorn` — Entire HTTP interface
- `faster-whisper` ≥ 1.1.0 — STT and VAD via Whisper models
- `onnxruntime` — KittenTTS and Silero VAD ONNX models
- `huggingface_hub` — Model auto-download on first use
- `pydantic` — All request/response schema validation

**Infrastructure:**
- `prometheus_client` — Exposes `/metrics` endpoint for monitoring
- `loguru` — Replaces standard Python logging throughout `harmonyspeech/common/logger.py`
- `psutil` — Used in `harmonyspeech/common/metrics.py` for system stats

## Model Weight Files

Models are downloaded from Hugging Face Hub at runtime and cached locally in `./cache/` (mapped as Docker volume).

Supported weight formats:
- `safetensors` — Preferred format
- HDF5 / `.h5` — Legacy
- ONNX `.onnx` — For Silero VAD and KittenTTS models

## Configuration

**Environment:**
- `.env` file (exists, contents private) — Used by Docker Compose for API key and runtime configuration
- `.env-amd` — AMD GPU variant environment config

**Model Config:**
- `config.yml` — CPU model definitions (used by default Docker Compose)
- `config.gpu.yml` — GPU model definitions (used by NVIDIA Docker Compose)
- YAML config specifies: model name, model type, Hugging Face repo ID, dtype, device, batch size

**Entry Point:**
- CLI entrypoint: `harmonyspeech/endpoints/cli.py` — `harmonyspeech run [options]`
- Registered as package script `harmonyspeech` in `pyproject.toml`

**Build:**
- `pyproject.toml` — Project metadata, build system (`setuptools`), Python version requirements
- `setup.py` — Additional build configuration with CUDA/ROCm support detection

## Platform Requirements

**Development:**
- Python 3.12+
- pip
- Platform-specific: CPU, CUDA (NVIDIA), or ROCm (AMD) PyTorch

**Production:**
- Docker + Docker Compose
- CPU image: `harmonyai/harmonyspeech-engine-cpu:latest`
- NVIDIA GPU image: `harmonyai/harmonyspeech-engine-nvidia:latest`
- AMD GPU image: (separate `docker-compose.amd.yml`)
- API exposed on port `12080`
- UI image `harmonyai/harmonyspeech-ui:latest` exposed on port `8080`
- Requires `config.yml` mounted at `/app/harmony-speech-engine/config.yml`
- Requires `./cache/` directory mounted for model storage

---

*Stack analysis: 2026-03-11*
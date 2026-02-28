# Technology Stack

**Analysis Date:** 2026-02-28

## Languages

**Primary:**
- Python 3.12+ — All inference engine code, API server, model implementations

**Secondary:**
- YAML — Model configuration (`config.yml`, `config.gpu.yml`)

## Runtime

**Environment:**
- Python ≥ 3.12 (enforced in `pyproject.toml` via `requires-python = ">=3.12"`)

**Package Manager:**
- pip with `requirements-*.txt` split files + `pyproject.toml`
- Lockfile: Not present (no `requirements.lock` or `poetry.lock`)

**Requirement Files:**
- `requirements-common.txt` — Base dependencies for all platforms
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
- `ctranslate2` — Quantized fast inference backend (used by Faster-Whisper)
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
- `setup.py` — Additional build configuration

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

*Stack analysis: 2026-02-28*

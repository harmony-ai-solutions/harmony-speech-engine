<!-- refreshed: 2026-11-02 -->
# Technology Stack

**Analysis Date:** 2026-11-02

## Languages

**Primary:**
- Python 3.12 — Inference engine, FastAPI server, model wrappers, executors
- YAML — Model configuration (`config.yml`, `config.gpu.yml`)
- Bash — Container entrypoints (`docker/entrypoint.sh`, `docker/amd-wsl/entrypoint.sh`)

**Secondary:**
- JavaScript / JSX — Bundled operator UI (`frontend/src/`, React 19 + Vite 7)
- Dockerfile — Four build variants under `docker/` (`cpu`, `nvidia`, `amd`, `amd-wsl`)

## Runtime

**Environment:**
- Python ≥ 3.12 (enforced by `pyproject.toml` → `requires-python = ">=3.12"`)
- PyTorch ≥ 2.7.1 required at build time (`[build-system].requires` in `pyproject.toml`)
- `HARMONYSPEECH_TARGET_DEVICE` env var selects build target in `setup.py`: `cuda` (default), `rocm`, `cpu`
- Version strings are inconsistent and must be read from the right file:
  - `pyproject.toml` → `0.1.1-rc2`
  - `harmonyspeech/version.py` → `v0.1.1-rc1` (reported by `GET /version`)
  - `CHANGELOG.md` → newest documented release is `v0.2.0` (Chatterbox + KittenTTS)

**Package Manager:**
- pip with split `requirements-*.txt` files + editable install of the package (`pip install -e .`)
- Build backend: `setuptools>=61.0` (`build-backend = "setuptools.build_meta"`)
- Lockfile: Not present (no `requirements.lock`, `poetry.lock`, or `uv.lock`)

**Requirement Files:**
- `requirements-common.txt` — All shared dependencies (82 lines)
- `requirements-cpu.txt` — `--extra-index-url https://download.pytorch.org/whl/cpu` + `-r requirements-common.txt`
- `requirements-cuda.txt` — `-r requirements-common.txt` (CUDA wheels come from the Docker build's `cu129` index)
- `requirements-rocm.txt` — `-r requirements-common.txt` (ROCm wheels from `rocm6.2` index in Docker)
- `requirements-chatterbox.txt` — Contains only `chatterbox-tts==0.1.6`; **must** be installed with `pip install --no-deps -r requirements-chatterbox.txt` because upstream pins `torch==2.6.0`, `numpy<1.26.0`, `transformers==4.46.3`, `safetensors==0.5.3`, `diffusers==0.29.0`
- Install order used everywhere (Dockerfiles, CI, README): tier file → `requirements-common.txt` → `pip install --no-deps -r requirements-chatterbox.txt`

## Frameworks

**Core Web / API:**
- `fastapi` — OpenAI-style REST API (`harmonyspeech/endpoints/openai/api_server.py`)
- `uvicorn` — ASGI server
- `pydantic` — Request/response models in `harmonyspeech/endpoints/openai/protocol.py`
- `CORSMiddleware` — Origin/method/header configuration via CLI args (`harmonyspeech/endpoints/openai/args.py`)

**Deep Learning:**
- `torch` / `torchaudio` — Tensor computation and audio I/O
- `torchlibrosa` — Librosa-compatible feature extraction on PyTorch
- `transformers` — BERT/T3/Llama text modules (MeloTTS, Chatterbox)
- `diffusers` — Diffusion components pulled in as a Chatterbox transitive dependency
- `peft` — Adapter support required by Chatterbox
- `safetensors` — Weight loading format

**Inference Acceleration:**
- `onnxruntime` — KittenTTS and Silero VAD ONNX inference
- `ctranslate2` == 4.4.0 — Faster-Whisper backend (pinned, see `requirements-common.txt` comment re: SYSTRAN/faster-whisper#1086)
- `faster-whisper` ≥ 1.1.0 — Whisper / Distil-Whisper wrapper

**Audio Processing:**
- `librosa`, `numba`, `scipy`, `soundfile`, `pydub`, `webrtcvad-wheels`, `silero-vad`
- `pyloudnorm` — Loudness normalization (Chatterbox Turbo)

**TTS / VC Engines:**
- **Chatterbox** (Resemble AI) — Native integration, 4 variants
  - `ChatterboxTTS` — voice cloning, 8 languages, perth watermarking
  - `ChatterboxTurboTTS` — faster variant with `top_k` / `norm_loudness`, no `exaggeration`/`cfg_weight`/`min_p`
  - `ChatterboxMultilingualTTS` — 23 languages
  - `ChatterboxVC` — voice conversion
  - `ChatterboxEmbedding` — embedding-only executor for `/v1/embed/speaker`
  - Implementation: `harmonyspeech/modeling/models/chatterbox/chatterbox.py`
  - Wrappers monkey-patch upstream defects: CPU `map_location` for multilingual loading, `PerthImplicitWatermarker` fallback, Turbo `snapshot_download` token bug, float64 → float32 casts in `s3gen.tokenizer.log_mel_spectrogram` and `ve.forward`, and `attn_implementation="eager"` for the multilingual T3 `LlamaModel` so `output_attentions` works
- **KittenTTS** — Ultra-lightweight ONNX English TTS
  - Variants: `kitten-tts-mini-0.8` (80 MB), `kitten-tts-micro-0.8` (41 MB), `kitten-tts-nano-0.8-fp32` (56 MB), `kitten-tts-nano-0.8-int8` (25 MB)
  - Phonemizer: `misaki[en]>=0.9.4`; espeak via `espeakng_loader`
  - Voices: Bella, Jasper, Luna, Bruno, Rosie, Hugo, Kiki, Leo
  - Implementation: `harmonyspeech/modeling/models/kittentts/`
- **MeloTTS** — Multi-lingual TTS (EN, ZH, ES, FR, JP, KR); implementation `harmonyspeech/modeling/models/melo/`
- **OpenVoice V1 / V2** — Tone conversion + encoder; implementation `harmonyspeech/modeling/models/openvoice/`
- **HarmonySpeech V1** — Encoder / Forward-Tacotron synthesizer / MultiBand-MelGAN vocoder; implementation `harmonyspeech/modeling/models/harmonyspeech/`
- **VoiceFixer** — Restorer + vocoder audio restoration; implementation `harmonyspeech/modeling/models/voicefixer/`

**STT / ASR / VAD:**
- **Faster-Whisper** — `FasterWhisper` model type, CTranslate2 backend
- **Silero VAD** — `SileroVAD` model type, ONNX load format (`load_format: "onnx"`)

**NLP / Phonemization / G2P:**
- `g2p_en`, `g2pkk`, `gruut[de,es,fr]`, `eng_to_ipa`, `misaki[en]`, `espeakng_loader`, `nltk`, `spacy`, `spacy-pkuseg`
- `mecab-python3` + `unidic-lite` + `pykakasi` (Japanese, also Chatterbox MTL tokenizer)
- `eunjeon` + `fugashi` (Korean)
- `cn2an` + `pypinyin` + `jieba` (Chinese)
- `num2words`, `inflect`, `anyascii`, `unidecode`, `jamo`

**Model Distribution:**
- `huggingface_hub` + `hf_xet` — snapshot/download, `HF_TOKEN` support
- `cached_path` — path resolution/caching
- `h5py` — HDF5 weight files

**Utilities:**
- `loguru` — logging (`harmonyspeech/common/logger.py`)
- `prometheus_client` — `/metrics` exposition
- `psutil`, `packaging`, `numpy`, `matplotlib`, `pyyaml`, `omegaconf`, `einops`, `conformer==0.3.2`, `s3tokenizer`, `resemble-perth==1.0.1`

**Planned (NOT yet implemented):**
- **vllm-omni** — GPU-only TTS backend planned for Qwen3-TTS and later CosyVoice3
  - Plan documents: `.current_work/vllm-omni-integration-plan.md`, `.current_work/vllm-omni-analysis.md`
  - Target design: new `VllmOmniExecutor` in `harmonyspeech/executor/vllm_omni_executor.py` wrapping `Omni` (multi-process pipeline orchestrator spawned via `multiprocessing.spawn`/Ray, ZMQ/mp.Queue transport)
  - New model type: `VllmOmniTTS`; new requirement added to `requirements-cuda.txt` and `requirements-rocm.txt` only (never CPU)
  - Analyzed upstream version: vllm-omni v0.16.0 (rebased on vLLM v0.16.0)
  - Status: no `vllm_omni_executor.py`, no `VllmOmniTTS` registry entry, no `vllm-omni` requirement exists in the repo yet

## Testing Stack

**Test Framework:**
- `pytest` ≥ 8.0.0 — runner
- `pytest-asyncio` ≥ 0.23.0 — `asyncio_mode = "auto"`
- `pytest-mock` ≥ 3.12.0 — mocking
- `pytest-cov` ≥ 5.0.0 — coverage (not enabled by default; `addopts` only sets `--strict-markers -ra`)

**Pytest Configuration (`pyproject.toml`):**
- `testpaths = ["tests"]`
- Markers: `unit`, `integration`, `e2e`, `slow`
- Coverage `omit = ["harmonyspeech/modeling/models/*"]`
- Warnings filtered: `DeprecationWarning`, `UserWarning`

**Linting / Formatting:**
- `ruff` is the enforced linter/formatter (CI job "Linting & Formatting" runs `ruff format --check` and `ruff check`)
- Config lives in `pyproject.toml` → `[tool.ruff]`: `line-length = 120`, `target-version = "py312"`, `select = ["E","F","W","I","N","UP","B","C4","SIM"]`, `ignore = ["E501"]`, double quotes
- `black` ≥ 24.0.0 and `flake8` ≥ 7.0.0 remain listed in `requirements-common.txt` but are no longer the CI gate; ruff config states Black-compatible formatting

## CI/CD Stack

**GitHub Actions (`.github/workflows/test.yml`):**
- Jobs: `test`, `lint`, `e2e-kittentts`, `e2e-melotts-openvoice-v2`, `e2e-openvoice-v1`, `e2e-harmonyspeech`, `e2e-whisper`, `e2e-vad`, `e2e-voicefixer`, `e2e-chatterbox`, `e2e-chatterbox-turbo`, `e2e-chatterbox-multilingual`
- System packages installed per job: `libmecab-dev`, `ffmpeg`, `libsndfile1`, `espeak-ng`
- Python install sequence per job: `pip install -r requirements-cpu.txt`, `pip install -r requirements-common.txt`, `pip install --no-deps -r requirements-chatterbox.txt`
- NLTK corpora downloaded with `python -m nltk.downloader all` for OpenVoice/MeloTTS E2E
- HF model cache keyed on `requirements-common.txt`, `requirements-cpu.txt`, `requirements-chatterbox.txt`

**Docker release workflows:**
- `.github/workflows/docker-release-engine.yml` — builds/pushes `harmonyspeech-engine-{cpu,amd,amd-wsl,nvidia}:latest` (+ version tag) from `docker/<variant>`
- `.github/workflows/docker-release-ui.yml` — builds `frontend/Dockerfile.speech-engine` from `./frontend`, pushes `harmonyspeech-ui`

## Docker / Deployment Stack

**Base Images:**
- CPU: `ubuntu:24.04` + `--extra-index-url https://download.pytorch.org/whl/cpu`
- NVIDIA: `nvidia/cuda:12.9.1-devel-ubuntu24.04` + `https://download.pytorch.org/whl/cu129`
- AMD: `rocm/pytorch:rocm6.4.4_ubuntu24.04_py3.12_pytorch_release_2.7.1`, `PYTORCH_ROCM_ARCH="gfx908;gfx90a;gfx942;gfx1100"`, `rocm6.2` wheel index
- AMD-WSL: `harmonyai/base-pytorch-rocm-wsl2:2.7.1-latest`, RDNA arches `gfx1010;gfx1012;gfx1030;gfx1031;gfx1100;gfx1200;gfx1201`

**Runtime Env in Images:**
- `HOME=/app/harmony-speech-engine`, `HF_HOME=/app/harmony-speech-engine/cache`, `NUMBA_CACHE_DIR=${HF_HOME}/numba_cache`
- NVIDIA: `CUDA_HOME=/usr/local/cuda`, `TORCH_CUDA_ARCH_LIST="6.1 7.0 7.5 8.0 8.6 8.9 9.0+PTX"`, runs as UID 1000
- Entrypoint `docker/entrypoint.sh` runs `python3 -m harmonyspeech.endpoints.openai.api_server --host 0.0.0.0 --port 12080 ${CMD_ADDITIONAL_ARGUMENTS}`

**Docker Compose:**
- `docker-compose.yml` — `harmonyai/harmonyspeech-engine-cpu:latest`, mounts `./config.yml`, port `12080`
- `docker-compose.nvidia.yml` — `...-nvidia:latest`, mounts `./config.gpu.yml`, `deploy.resources.reservations.devices` with `driver: nvidia`
- `docker-compose.amd.yml` — `...-amd:latest`, mounts `./config.gpu.yml`, `env_file: ./.env-amd`, `device: /dev/dxg`
- UI service in all three: `harmonyai/harmonyspeech-ui:latest` on port `8080`

## Key Dependencies

**Critical:**
- `torch` / `torchaudio` — all neural inference
- `fastapi` + `uvicorn` — entire HTTP surface
- `pydantic` — request/response validation
- `chatterbox-tts==0.1.6` — Chatterbox model family (installed `--no-deps`)
- `faster-whisper>=1.1.0` + `ctranslate2==4.4.0` — STT
- `onnxruntime` — KittenTTS + Silero VAD
- `huggingface_hub` + `hf_xet` — weight acquisition
- `misaki[en]>=0.9.4` — KittenTTS phonemization
- `resemble-perth==1.0.1` — Chatterbox watermarking (swapped to `DummyWatermarker` when `watermark: false`)

**Infrastructure:**
- `prometheus_client` — `/metrics` (`harmonyspeech/common/metrics.py`, `harmonyspeech/engine/metrics.py`)
- `loguru` — logging; `HARMONYSPEECH_LOG_LEVEL` (default `INFO`)
- `psutil` — process/system stats
- `requests` — Harmony API-key service client (`auth/apikeys.py`)

## Model Weight Files

Fetched at runtime from Hugging Face; cached under `HF_HOME` (`./cache/` in Compose).

Supported formats: `safetensors` (preferred), `.pt`/`.pth`, HDF5 `.h5`, ONNX `.onnx`.
`ModelConfig.load_format` accepts `auto`, `pt`, `pth`, `safetensors`, `npcache`, `dummy`, `onnx` (`harmonyspeech/common/config.py`).

## Configuration

**Environment:**
- `.env` — Compose env file (present; contents private)
- `.env-amd` — AMD variant Compose env file (present; contents private)
- `.env.example` — committed template (present)

**Runtime env vars read by code:**
- `HARMONYSPEECH_API_KEY` — shared/static API key; enables the auth middleware when set
- `HARMONYSPEECH_ADMIN_KEY` — admin key (admin ops disabled if unset)
- `HARMONYSPEECH_LOG_LEVEL` — log level (`harmonyspeech/common/logger.py`)
- `HARMONYSPEECH_LOCAL_LOGGING_INTERVAL_SEC` — stats log cadence, default 5
- `HARMONYSPEECH_ENGINE_ITERATION_TIMEOUT_S` — engine step timeout, default 120
- `SERVICE_NAME` — service identifier sent to the API-key service, default `harmony-speech-engine`
- `ENDPOINT_URL` — base URL of the Harmony API-key/rate-limit service; when empty, key checks are disabled
- `HF_TOKEN` — optional Hugging Face token (Chatterbox Turbo path treats empty string as unset)
- `XDG_CACHE_HOME` — base for `~/.cache/harmony/locks/` file locks
- `CUDA_HOME`, `CUDA_VISIBLE_DEVICES` — read in `harmonyspeech/common/utils.py`
- `TORCH_NCCL_AVOID_RECORD_STREAMS` — set by `harmonyspeech/task_handler/gpu_worker.py`

**Model Config:**
- `config.yml` — CPU definitions; Chatterbox TTS/Turbo/Multilingual/VC enabled on `cpu`
- `config.gpu.yml` — CUDA definitions; adds `faster-whisper-large-v3-turbo`, all four KittenTTS variants, Chatterbox on `cuda`, and `chatterbox_embedding`; MeloTTS Japanese is defined as `ov2-synthesizer-ja` (language `JA`) while Korean appears only in `config.yml`
- YAML keys per entry: `name`, `model`, `model_type`, `max_batch_size`, `dtype`, `device_config.device`, plus optional `language`, `voices`, `load_format`, `watermark`, `revision`, `trust_remote_code`, `seed`, `enforce_eager`

**Entry Point:**
- CLI: `harmonyspeech/endpoints/cli.py` → `main`; package script `harmonyspeech` in `pyproject.toml`
- Server module: `python -m harmonyspeech.endpoints.openai.api_server`
- CLI defaults: `--port 2242` in `harmonyspeech/endpoints/openai/args.py`, but Docker entrypoint, README, and Compose all use `12080`

**Build:**
- `pyproject.toml` — metadata, setuptools, pytest/ruff/coverage config
- `setup.py` — CUDA/ROCm/CPU detection (`HARMONYSPEECH_TARGET_DEVICE`, `MAIN_CUDA_VERSION = "12.4"`), commits `harmonyspeech/commit_id.py`

## Platform Requirements

**Development:**
- Python 3.12, pip
- Linux recommended; `setup.py` warns "Only CUDA backend is tested on Windows"
- System packages: `ffmpeg`, `libsndfile1`, `libmecab-dev`, `espeak-ng`

**Production:**
- Docker + Docker Compose
- `harmonyai/harmonyspeech-engine-cpu:latest`, `...-nvidia:latest`, `...-amd:latest`, `...-amd-wsl:latest`
- `harmonyai/harmonyspeech-ui:latest` on port `8080`; engine on `12080`
- Mount `config.yml`/`config.gpu.yml` at `/app/harmony-speech-engine/config.yml`
- Mount `./cache/` for model storage
- GPU variants need NVIDIA driver capability or `/dev/dxg` (AMD/WSL)

---

*Stack analysis: 2026-11-02*

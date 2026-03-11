# Codebase Structure

**Analysis Date:** 2026-03-11

## Directory Layout

```
harmony-speech-engine/
├── harmonyspeech/              # Main Python package — all application source code
│   ├── __init__.py
│   ├── version.py              # Package version constant
│   ├── common/                 # Shared data types, config, utilities (no internal deps)
│   ├── endpoints/              # API surface — HTTP entry points
│   │   └── openai/             # OpenAI-compatible REST API implementation
│   ├── engine/                 # Core orchestration — request lifecycle management
│   ├── executor/               # Compute backend abstraction (CPU / GPU / Ray)
│   ├── modeling/               # Model loading, downloading, and ML model implementations
│   │   └── models/             # Per-model packages (one subdirectory per model family)
│   ├── processing/             # Scheduler — request queuing and batching
│   └── task_handler/           # Worker layer — model inference execution on device
├── auth/                       # HTTP authentication middleware
├── docker/                     # Docker build assets and helper scripts
├── docs/                       # Documentation (API specs, usage guides)
├── frontend/                   # Frontend assets (web UI)
├── models/                     # Local model weight cache directory (not committed)
├── tests/                      # Test suite (3-tier: unit/integration/e2e)
│   ├── unit/                   # Unit tests (fast, fully mocked)
│   │   ├── initialization/     # Engine initialization tests
│   │   ├── inference_flow/     # Model loader tests
│   │   ├── error_handling/     # Config error tests
│   │   └── conftest.py         # Unit test fixtures
│   ├── integration/            # Integration tests (component interaction)
│   │   ├── test_cli.py         # CLI tests
│   │   ├── test_api_endpoints.py # API endpoint tests
│   │   └── conftest.py         # Integration fixtures
│   ├── e2e/                    # End-to-end tests (real models, CPU-only, slow)
│   │   ├── tts/                # TTS model tests
│   │   │   ├── test_melotts.py
│   │   │   ├── test_kittentts.py
│   │   │   ├── test_openvoice_v1.py
│   │   │   └── test_harmonyspeech.py
│   │   ├── stt/                # STT/Whisper tests
│   │   │   └── test_whisper.py
│   │   ├── vad/                # Voice activity detection tests
│   │   │   ├── test_silero_vad.py
│   │   │   └── test_whisper_vad.py
│   │   ├── audio_restoration/  # VoiceFixer tests
│   │   │   └── test_voicefixer.py
│   │   └── conftest.py         # E2E fixtures
│   ├── conftest.py             # Root test configuration
│   └── test-data/              # Test audio samples
│       └── samples/            # WAV files (wanda4.wav, wanda5.wav, wanda6.wav)
├── .planning/                  # GSD planning documents (not shipped)
│   └── codebase/               # Codebase analysis documents
├── memory-bank/                # Roo memory bank (project context, not shipped)
├── config.yml                  # Primary runtime configuration (model_configs list)
├── config.gpu.yml              # GPU-targeted runtime configuration variant
├── docker-compose.yml          # CPU deployment
├── docker-compose.nvidia.yml   # NVIDIA GPU deployment
├── docker-compose.amd.yml      # AMD ROCm GPU deployment
├── pyproject.toml              # Build metadata and tool configuration
├── setup.py                    # Package setup
├── requirements-common.txt     # Base Python dependencies
├── requirements-cpu.txt        # CPU-only torch variant
├── requirements-cuda.txt       # NVIDIA CUDA torch variant
├── requirements-rocm.txt       # AMD ROCm torch variant
└── README.md
```

## Directory Purposes

**`harmonyspeech/common/`:**
- Purpose: Foundation types and utilities shared by all layers; has no imports from other harmonyspeech subpackages
- Contains: `config.py` (ModelConfig, DeviceConfig, EngineConfig dataclasses), `inputs.py` (all RequestInput subclasses), `outputs.py` (RequestOutput, GenerationOutput), `request.py` (EngineRequest), `metrics.py`, `logger.py`, `utils.py`
- Key files: `harmonyspeech/common/config.py`, `harmonyspeech/common/inputs.py`, `harmonyspeech/common/outputs.py`, `harmonyspeech/common/request.py`

**`harmonyspeech/endpoints/openai/`:**
- Purpose: FastAPI application and all HTTP route handlers; maps OpenAI-style requests to internal engine calls
- Contains: `api_server.py` (FastAPI app, route registration, startup), `protocol.py` (all Pydantic request/response models), `args.py` (CLI/startup argument definitions), `serving_engine.py` (base serving class), `serving_text_to_speech.py`, `serving_speech_to_text.py`, `serving_voice_activity_detection.py`, `serving_voice_conversion.py`, `serving_audio_conversion.py`, `serving_voice_embed.py`
- Key files: `harmonyspeech/endpoints/openai/api_server.py`, `harmonyspeech/endpoints/openai/protocol.py`

**`harmonyspeech/engine/`:**
- Purpose: Owns the request lifecycle — accepts requests from serving layer, drives the scheduler/executor loop, returns outputs
- Contains: `harmonyspeech_engine.py` (synchronous engine), `async_harmonyspeech.py` (async wrapper used by API), `args_tools.py` (EngineArgs dataclass and parsing helpers), `metrics.py`
- Key files: `harmonyspeech/engine/async_harmonyspeech.py`, `harmonyspeech/engine/harmonyspeech_engine.py`

**`harmonyspeech/executor/`:**
- Purpose: Abstract the compute backend; concrete subclasses route work to one or many workers
- Contains: `executor_base.py` (abstract `ExecutorBase`), `cpu_executor.py`, `gpu_executor.py`, `ray_gpu_executor.py`
- Key files: `harmonyspeech/executor/executor_base.py`, `harmonyspeech/executor/gpu_executor.py`

**`harmonyspeech/processing/`:**
- Purpose: Request scheduling and batching — maintains per-model queues and decides which requests are ready to dispatch each cycle
- Contains: `scheduler.py` (`HarmonySpeechScheduler`)
- Key files: `harmonyspeech/processing/scheduler.py`

**`harmonyspeech/task_handler/`:**
- Purpose: The worker layer that actually loads and runs models; runs inside the executor's worker process/thread
- Contains: `worker_base.py`, `cpu_worker.py`, `gpu_worker.py`, `model_runner_base.py`, `cpu_model_runner.py`, `gpu_model_runner.py`, `inputs.py` (all `prepare_*_inputs()` functions), and `inputs.py` also imports from modeling layer
- Key files: `harmonyspeech/task_handler/inputs.py`, `harmonyspeech/task_handler/gpu_model_runner.py`

**`harmonyspeech/modeling/`:**
- Purpose: Model weight loading, HuggingFace Hub integration, and PyTorch/ONNX model class implementations
- Contains: `loader.py` (get_model_class, get_model_config, get_model_flavour, get_model_speaker), `hf_downloader.py`, `utils.py`, `models/` subdirectory with one package per model family
- Key files: `harmonyspeech/modeling/loader.py`

**`harmonyspeech/modeling/models/`:**
- Purpose: One subdirectory per supported model family; each contains the PyTorch nn.Module classes and model-specific utilities
- Subdirectories:
  - `harmonyspeech/` — Harmony Speech v1 (encoder, synthesizer, vocoder + parallel_wavegan)
  - `openvoice/` — OpenVoice V1/V2 (ToneConverter, Synthesizer, Encoder)
  - `melo/` — MeloTTS / OpenVoice V2 (multi-language TTS synthesizer)
  - `kittentts/` — KittenTTS (ultra-lightweight ONNX TTS) ⭐ NEW
  - `voicefixer/` — VoiceFixer (audio restoration and vocoder)
- Key files: `harmonyspeech/modeling/models/harmonyspeech/harmonyspeech.py`, `harmonyspeech/modeling/models/openvoice/openvoice.py`, `harmonyspeech/modeling/models/melo/melo.py`, `harmonyspeech/modeling/models/kittentts/kittentts.py`, `harmonyspeech/modeling/models/voicefixer/voicefixer.py`

**`tests/`:**
- Purpose: Comprehensive test suite with 3-tier architecture
- Structure:
  - `tests/unit/` — Fast, fully mocked unit tests
  - `tests/integration/` — Component interaction tests
  - `tests/e2e/` — Real model tests (CPU-only, slow)
- Run with: `pytest tests/unit/`, `pytest tests/integration/`, `pytest tests/e2e/`
- Key fixtures: `tests/conftest.py` (root), `tests/unit/conftest.py`, `tests/integration/conftest.py`, `tests/e2e/conftest.py`

**`tests/test-data/`:**
- Purpose: Audio samples for testing
- Contains: `samples/` directory with WAV files (wanda4.wav, wanda5.wav, wanda6.wav)
- Generated: No (committed)
- Committed: Yes

**`auth/`:**
- Purpose: HTTP basic authentication middleware for the FastAPI server
- Key files: `auth/` (applied via FastAPI middleware in `api_server.py`)

**`docker/`:**
- Purpose: Dockerfile variants and helper scripts for container builds

**`models/`:**
- Purpose: Local on-disk cache for downloaded model weights; populated at runtime by HuggingFace Hub download
- Generated: Yes (populated at runtime)
- Committed: No (in `.gitignore`)

## Key File Locations

**Entry Points:**
- `harmonyspeech/endpoints/cli.py`: CLI entry point for starting the server
- `harmonyspeech/endpoints/openai/api_server.py`: FastAPI application — primary server entry point

**Configuration:**
- `config.yml`: Runtime model configuration — defines all `model_configs` entries with model types, batch sizes, device assignments
- `config.gpu.yml`: GPU-variant runtime configuration
- `harmonyspeech/common/config.py`: Python dataclasses for all configuration types (`ModelConfig`, `DeviceConfig`, `EngineConfig`)
- `harmonyspeech/engine/args_tools.py`: `EngineArgs` — CLI argument definitions and conversion to config objects

**Protocol / API Contract:**
- `harmonyspeech/endpoints/openai/protocol.py`: All Pydantic request and response types — single source of truth for the API contract

**Internal Request Types:**
- `harmonyspeech/common/inputs.py`: All `RequestInput` subclasses — internal typed representations of requests

**Model Dispatch:**
- `harmonyspeech/task_handler/inputs.py`: `prepare_inputs()` — central dispatch by `model_type` string, all model-specific input preparation
- `harmonyspeech/modeling/loader.py`: `get_model_class()` — model class resolution by `model_type` string

**Core Logic:**
- `harmonyspeech/engine/async_harmonyspeech.py`: Async engine — used by the API server
- `harmonyspeech/processing/scheduler.py`: Request scheduler and batcher

## Tests Directory Deep Dive

**Test Structure (3-Tier):**

```
tests/
├── conftest.py              # Root fixtures (shared across all test tiers)
├── unit/                    # Unit tests - FAST, fully mocked
│   ├── conftest.py          # Unit-specific fixtures
│   ├── initialization/      # Engine initialization tests
│   │   ├── test_engine.py
│   │   └── test_config.py
│   ├── inference_flow/      # Model loader tests
│   │   └── test_loader.py
│   └── error_handling/      # Config error validation
│       └── test_config_errors.py
├── integration/             # Integration tests - COMPONENT INTERACTION
│   ├── conftest.py          # Integration-specific fixtures
│   ├── test_cli.py          # CLI entry point tests
│   └── test_api_endpoints.py # API endpoint tests
└── e2e/                     # End-to-end tests - REAL MODELS, CPU-ONLY, SLOW
    ├── conftest.py          # E2E-specific fixtures
    ├── tts/                 # Text-to-Speech model tests
    │   ├── test_melotts.py
    │   ├── test_kittentts.py
    │   ├── test_openvoice_v1.py
    │   └── test_harmonyspeech.py
    ├── stt/                 # Speech-to-Text tests
    │   └── test_whisper.py
    ├── vad/                 # Voice Activity Detection tests
    │   ├── test_silero_vad.py
    │   └── test_whisper_vad.py
    └── audio_restoration/   # Audio restoration tests
        └── test_voicefixer.py
```

**Test Configuration (pyproject.toml):**
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
addopts = ["--strict-markers", "-ra", "--cov=harmonyspeech", ...]

[tool.pytest.markers]
unit = "marks tests as unit tests (fast, fully mocked)"
integration = "marks tests as integration tests (component interaction, partially mocked)"
e2e = "marks tests as end-to-end tests (real models, slow, CPU-only)"
slow = "marks tests that take > 30 seconds to run"
```

**Running Tests:**
```bash
pytest tests/unit/              # Run all unit tests (fast)
pytest tests/integration/       # Run integration tests
pytest tests/e2e/               # Run e2e tests (slow, requires models)
pytest tests/                   # Run all tests
pytest -m unit                  # Run only unit tests by marker
pytest -m e2e                   # Run only e2e tests by marker
```

## Naming Conventions

**Files:**
- Snake case: `api_server.py`, `async_harmonyspeech.py`, `gpu_model_runner.py`
- Serving modules prefixed: `serving_text_to_speech.py`, `serving_speech_to_text.py`
- Worker/runner pairs mirror executor names: `cpu_worker.py` / `cpu_model_runner.py`, `gpu_worker.py` / `gpu_model_runner.py`
- Model packages match model family name: `openvoice/`, `melo/`, `kittentts/`, `harmonyspeech/`, `voicefixer/`
- Test files prefixed: `test_{module}.py`

**Directories:**
- All lowercase, underscore-separated: `task_handler/`, `modeling/`, `endpoints/`
- Model family directories match the HuggingFace repo slug where possible

**Classes:**
- PascalCase: `HarmonySpeechEngine`, `AsyncHarmonySpeech`, `TextToSpeechRequest`, `ModelConfig`
- Request inputs follow `{Capability}RequestInput` pattern: `TextToSpeechRequestInput`, `SpeechEmbeddingRequestInput`
- Response types follow OpenAI naming with `Response` suffix: `TextToSpeechResponse`, `SpeechToTextResponse`

**model_type Strings:**
- PascalCase model type identifiers used as string keys: `"HarmonySpeechEncoder"`, `"MeloTTSSynthesizer"`, `"FasterWhisper"`, `"SileroVAD"`, `"KittenTTSSynthesizer"`

## Where to Add New Code

**New Model Support:**
- Add model class implementation: `harmonyspeech/modeling/models/{model_family}/{model_family}.py`
- Register model class in: `harmonyspeech/modeling/loader.py` (`get_model_class()`)
- Add input preparation: `harmonyspeech/task_handler/inputs.py` (new `elif` branch in `prepare_inputs()` + new `prepare_{model}_inputs()` function)
- Add model runner inference logic: `harmonyspeech/task_handler/gpu_model_runner.py` and/or `cpu_model_runner.py`
- Add model_type entry to: `config.yml`

**New TTS Provider (like KittenTTS):**
1. Create `harmonyspeech/modeling/models/{provider}/` with model implementation
2. If ONNX-based: add `onnx_model.py` wrapper
3. Register in `harmonyspeech/modeling/loader.py`:
   - Add `"provider_type": {"default": "native"}` to `_MODEL_CONFIGS`
   - Add `"provider_type": {"default": "native"}` to `_MODEL_WEIGHTS`
   - Add model instantiation logic in `get_model()` with `"native"` bailout
4. Add input preparation in `task_handler/inputs.py`
5. Add runner logic (ONNX or PyTorch path)
6. Add e2e test in `tests/e2e/tts/test_{provider}.py`

**New API Endpoint / Capability:**
- Add Pydantic request/response types: `harmonyspeech/endpoints/openai/protocol.py`
- Add `RequestInput` subclass: `harmonyspeech/common/inputs.py`
- Create serving module: `harmonyspeech/endpoints/openai/serving_{capability}.py`
- Register route in: `harmonyspeech/endpoints/openai/api_server.py`

**New Configuration Option:**
- Add to: `harmonyspeech/common/config.py` (ModelConfig or EngineConfig)
- Expose via CLI: `harmonyspeech/engine/args_tools.py`

**Tests:**
- Unit tests: `tests/unit/{category}/test_{feature}.py`
- Integration tests: `tests/integration/test_{feature}.py`
- E2E tests: `tests/e2e/{capability}/test_{model}.py`
- Test data: `tests/test-data/samples/`

## Special Directories

**`.planning/`:**
- Purpose: GSD planning and codebase analysis documents
- Generated: No (written by GSD agents)
- Committed: Yes

**`memory-bank/`:**
- Purpose: Roo AI agent memory bank — project context files
- Generated: Yes (written by AI sessions)
- Committed: Yes (provides persistent project context)

**`models/`:**
- Purpose: On-disk HuggingFace model weight cache
- Generated: Yes (populated at first run by `hf_downloader.py`)
- Committed: No

**`.github/`:**
- Purpose: GitHub Actions CI/CD workflows
- Committed: Yes

**`docs/`:**
- Purpose: API documentation and usage guides
- Generated: No (manually written)
- Committed: Yes

---

*Structure analysis: 2026-03-11*
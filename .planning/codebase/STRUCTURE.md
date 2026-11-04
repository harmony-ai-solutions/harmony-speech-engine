# Codebase Structure

**Analysis Date:** 2026-11-02

## Directory Layout

```text
harmony-speech-engine/
├── harmonyspeech/                  # Main Python package — all application source
│   ├── __init__.py                 # Exports HarmonySpeechEngine, __version__, commit ids
│   ├── version.py                  # Version + commit metadata
│   ├── commit_id.py                # Generated commit hash
│   ├── common/                     # Shared config, request/response types, metrics, logging
│   ├── endpoints/                  # API surface
│   │   ├── cli.py                  # `harmonyspeech run` entry point
│   │   └── openai/                 # FastAPI app, protocol, serving modules
│   ├── engine/                     # Request lifecycle, rerouting, async loop, metrics
│   ├── executor/                   # CPU / GPU executor backends
│   ├── modeling/                   # Loader, HF downloader, model implementations
│   │   └── models/                 # One package per model family
│   │       ├── harmonyspeech/      # HarmonySpeech V1 (encoder, synthesizer, vocoder)
│   │       ├── openvoice/          # OpenVoice V1/V2 synthesizer + tone converter
│   │       ├── melo/               # MeloTTS multi-language synthesizer
│   │       ├── kittentts/          # KittenTTS ONNX synthesizer
│   │       ├── voicefixer/         # VoiceFixer restorer + vocoder
│   │       └── chatterbox/         # Chatterbox TTS/Turbo/Multilingual/VC wrappers
│   ├── processing/                 # Scheduler and batching budget
│   └── task_handler/               # Workers, model runners, input preparation
├── auth/                           # Harmony Auth API key cache + rate limiting
│   └── apikeys.py
├── docker/                         # Dockerfiles per backend
│   ├── cpu/Dockerfile
│   ├── nvidia/Dockerfile
│   ├── amd/Dockerfile
│   ├── amd-wsl/Dockerfile
│   └── entrypoint.sh
├── docs/                           # Hand-written docs + generated OpenAPI artifacts
│   ├── api.md
│   ├── audio-restoration.md
│   ├── docker.md
│   ├── models.md
│   ├── routing.md
│   ├── testing.md
│   ├── api/                        # openapi.json + Go/JS client generators
│   └── images/
├── frontend/                       # Unified React/Vite UI (speech-engine + harmony-link modes)
│   ├── src/
│   │   ├── main.jsx
│   │   ├── HarmonySpeechEngineApp.jsx
│   │   ├── HarmonyLinkApp.jsx
│   │   ├── assets/ components/ config/ constants/ contexts/ hooks/
│   │   ├── i18n/ services/ store/ styles/ utils/
│   │   └── constants.jsx, style.css
│   ├── Dockerfile, Dockerfile.speech-engine, Dockerfile.harmony-link
│   └── package.json, vite.config.js, tailwind.config.js
├── tests/                          # 3-tier test suite
│   ├── unit/                       # Fast, fully mocked
│   ├── integration/                # Component interaction
│   ├── e2e/                        # Real models, CPU-only, slow
│   ├── conftest.py
│   └── test-data/samples/          # WAV fixtures
├── .planning/                      # GSD planning + codebase analysis (committed)
│   └── codebase/
├── memory-bank/                    # Roo memory bank (committed project context)
├── models/                         # Local model weight cache (generated, gitignored)
├── cache/                          # HF hub / numba / xet caches (generated, gitignored)
├── plans/                          # Empty scratch directory
├── config.yml                      # Default (CPU) model_configs
├── config.gpu.yml                  # GPU model_configs variant
├── requirements-common.txt         # Base dependencies
├── requirements-cpu.txt            # torch CPU variant
├── requirements-cuda.txt           # torch CUDA variant
├── requirements-rocm.txt           # torch ROCm variant
├── requirements-chatterbox.txt     # chatterbox-tts pin (install with --no-deps)
├── pyproject.toml                  # Build metadata, ruff/pytest/coverage config
├── setup.py
├── docker-compose.yml              # CPU deployment
├── docker-compose.nvidia.yml       # NVIDIA deployment
├── docker-compose.amd.yml          # AMD ROCm deployment
├── AGENTS.md / opencode.json       # Agent instructions and tool config
├── CHANGELOG.md
├── README.md
└── LICENSE
```

## Directory Purposes

**`harmonyspeech/common/`:**
- Purpose: Foundation types shared by every layer; imports no other `harmonyspeech` subpackage except `common.utils`
- Contains:
  - `config.py` — `DeviceConfig`, `ModelConfig`, frozen `EngineConfig` with `load_config_from_yaml()`, dtype coercion
  - `inputs.py` — `RequestInput` base + all eight subclasses and `TextToSpeechGenerationOptions`
  - `outputs.py` — `RequestOutput` base + all output subclasses, including `finish_reason` / `error`
  - `request.py` — `EngineRequest`, `ExecutorResult`, `RequestStatus` (incl. `FINISHED_FORWARDED`)
  - `metrics.py` — `RequestMetrics` dataclass
  - `logger.py` — `setup_logger()`, `UVICORN_LOG_CONFIG`
  - `utils.py` — `make_async`, `random_uuid`, device/ROCm/WSL probes, `CudaMemoryProfiler`, `LRUCache`
- Key files: `harmonyspeech/common/config.py`, `harmonyspeech/common/inputs.py`, `harmonyspeech/common/request.py`

**`harmonyspeech/endpoints/openai/`:**
- Purpose: HTTP boundary — protocol schemas, validation, serving modules, app wiring
- Contains: `api_server.py` (routes, auth middleware, CORS, `build_app`, `run_server`), `protocol.py` (all Pydantic models), `args.py` (`make_arg_parser`), `serving_engine.py` (base class, model cards, `_check_model`), and one `serving_*.py` per capability
- Key files: `harmonyspeech/endpoints/openai/api_server.py`, `harmonyspeech/endpoints/openai/protocol.py`, `harmonyspeech/endpoints/openai/serving_engine.py`

**`harmonyspeech/engine/`:**
- Purpose: Request lifecycle ownership — rerouting, multi-step forwarding, async stream fan-out
- Contains:
  - `harmonyspeech_engine.py` — `HarmonySpeechEngine`, `reroute_request_*`, `check_forward_processing`, `step`, `_process_model_outputs`
  - `async_harmonyspeech.py` — `AsyncHarmonySpeech`, `_AsyncHarmonySpeech`, `AsyncStream`, `RequestTracker`, `run_engine_loop`
  - `args_tools.py` — `EngineArgs`, `AsyncEngineArgs`
  - `metrics.py` — `StatLogger`, `Stats`
- Key files: `harmonyspeech/engine/harmonyspeech_engine.py`, `harmonyspeech/engine/async_harmonyspeech.py`

**`harmonyspeech/executor/`:**
- Purpose: Compute-backend abstraction; one executor instance per model config
- Contains: `executor_base.py` (`ExecutorBase`, `ExecutorAsyncBase`), `cpu_executor.py`, `gpu_executor.py` (`GPUExecutor`, `GPUExecutorAsync`), `ray_gpu_executor.py` (fully commented out, unused)
- Key files: `harmonyspeech/executor/executor_base.py`, `harmonyspeech/executor/gpu_executor.py`

**`harmonyspeech/processing/`:**
- Purpose: Request admission and per-model batching
- Contains: `scheduler.py` — `Scheduler`, `SchedulingBudget`, `SchedulerOutputs`, `SchedulerWaitingOutputs`
- Key files: `harmonyspeech/processing/scheduler.py`

**`harmonyspeech/task_handler/`:**
- Purpose: Runs on the compute device; loads models and performs inference
- Contains:
  - `worker_base.py` — `WorkerBase` ABC
  - `cpu_worker.py`, `gpu_worker.py` — device setup + delegation to the runner
  - `model_runner_base.py` — **all** inference logic: `execute_model()` dispatch and every `_execute_*` method
  - `cpu_model_runner.py`, `gpu_model_runner.py` — thin subclasses that only implement `load_model()`
  - `inputs.py` — `prepare_inputs()` dispatch and all `prepare_*_inputs()` functions
- Key files: `harmonyspeech/task_handler/model_runner_base.py`, `harmonyspeech/task_handler/inputs.py`

**`harmonyspeech/modeling/`:**
- Purpose: Model class/config/weight resolution and model implementations
- Contains: `loader.py` (`_MODEL_CONFIGS`, `_MODEL_WEIGHTS`, `_MODEL_SPEAKERS`, `get_model`, `get_model_config`, `get_model_weights`, `get_model_speaker`, `get_model_flavour`, CTranslate2 CUDA pre-check), `hf_downloader.py` (three-tier resolution + in-memory caches), `utils.py` (`set_random_seed`), `models/`
- Key files: `harmonyspeech/modeling/loader.py`, `harmonyspeech/modeling/hf_downloader.py`, `harmonyspeech/modeling/models/__init__.py`

**`harmonyspeech/modeling/models/`:**
- Purpose: One package per model family. `__init__.py` holds `ModelRegistry._MODELS` mapping `model_type` → `(module, class)` or `"native"`.
- Subdirectories:

| Package | Files | Model types | Backend |
|---|---|---|---|
| `harmonyspeech/` | `harmonyspeech.py`, `common.py`, `encoder/`, `synthesizer/` (cleaners, numbers, symbols, text), `vocoder/parallel_wavegan/` | `HarmonySpeechEncoder`, `HarmonySpeechSynthesizer`, `HarmonySpeechVocoder` | PyTorch |
| `openvoice/` | `openvoice.py`, `modules.py`, `attentions.py`, `commons.py`, `transforms.py`, `mel_processing.py`, `utils.py`, `inputs.py`, `text/` | `OpenVoiceV1Synthesizer`, `OpenVoiceV1/V2ToneConverter(Encoder)` | PyTorch |
| `melo/` | `melo.py`, `modules.py`, `attentions.py`, `commons.py`, `transforms.py`, `utils.py`, `split_utils.py`, `inputs.py`, `monotonic_align/`, `text/` (multilingual BERT/phonemizers) | `MeloTTSSynthesizer` | PyTorch |
| `kittentts/` | `kittentts.py`, `onnx_model.py`, `preprocess.py` | `KittenTTSSynthesizer` | ONNX Runtime |
| `voicefixer/` | `voicefixer.py`, `modules.py`, `utils.py` | `VoiceFixerRestorer`, `VoiceFixerVocoder` | PyTorch |
| `chatterbox/` | `chatterbox.py`, `__init__.py` | `ChatterboxTTS`, `ChatterboxTurboTTS`, `ChatterboxMultilingualTTS`, `ChatterboxEmbedding`, `ChatterboxVC` | PyTorch + upstream `chatterbox-tts` |

- Key file: `harmonyspeech/modeling/models/chatterbox/chatterbox.py` — the only Chatterbox code in-repo; it wraps the external library, applies module-level monkey-patches (CPU `torch.load`, perth watermarker fallback) and Turbo-specific dtype/attention patches.

**`tests/`:**
- Purpose: Three-tier test suite (unit / integration / e2e)
- Structure:
  - `tests/unit/` — mocked, fast
  - `tests/integration/` — serves two layers together with mocked engines/models
  - `tests/e2e/` — real models, CPU-only, marked `e2e`/`slow`
- Key fixtures: `tests/conftest.py`, `tests/{unit,integration,e2e}/conftest.py`

**`auth/`:**
- Purpose: Harmony Auth integration — `ApiKeyCacheManager` caches per-key service stats and validates rate limits; usage events are registered from a separate thread in `api_server.py`
- Key file: `auth/apikeys.py`

**`frontend/`:**
- Purpose: Unified React + Vite + Tailwind UI that can boot in speech-engine or harmony-link mode (`HarmonySpeechEngineApp.jsx`, `HarmonyLinkApp.jsx`, selected in `main.jsx`)
- Contains: `src/{components,services,hooks,contexts,store,config,i18n,assets,styles,utils,constants}`, three Dockerfiles for the different deployment targets
- Key files: `frontend/src/main.jsx`, `frontend/src/HarmonySpeechEngineApp.jsx`, `frontend/vite.config.js`

**`docker/`:**
- Purpose: Per-backend images (`cpu/`, `nvidia/`, `amd/`, `amd-wsl/`) and the shared `entrypoint.sh`
- Key files: `docker/cpu/Dockerfile`, `docker/nvidia/Dockerfile`, `docker/amd/Dockerfile`, `docker/entrypoint.sh`

**`docs/`:**
- Purpose: Hand-written feature documentation plus generated client artifacts
- Contains: `api.md`, `models.md`, `routing.md`, `docker.md`, `testing.md`, `audio-restoration.md`, `api/openapi.json`, `api/generate_openapi_go.sh`, `api/generate_openapi_js.sh`, `images/`
- Note: `docs/api/openapi.json` is generated from the running FastAPI app and must be refreshed when protocol schemas change.

## Key File Locations

**Entry Points:**
- `harmonyspeech/endpoints/cli.py` — `harmonyspeech run` console script (`pyproject.toml:33`)
- `harmonyspeech/endpoints/openai/api_server.py` — FastAPI app, `build_app()`, `run_server()`, module `__main__` block

**Configuration:**
- `config.yml` — default model list (CPU)
- `config.gpu.yml` — GPU model list (Chatterbox and MeloTTS on `cuda`)
- `harmonyspeech/common/config.py` — `DeviceConfig`, `ModelConfig`, `EngineConfig.load_config_from_yaml()`
- `harmonyspeech/endpoints/openai/args.py` — server CLI (`--host`, `--port`, `--config-file-path`, `--api-keys`, `--admin-key`, `--middleware`, `--root-path`, SSL)
- `harmonyspeech/engine/args_tools.py` — `EngineArgs` / `AsyncEngineArgs`
- `requirements-chatterbox.txt` — isolated chatterbox pin; must be installed with `--no-deps`

**Protocol / API Contract:**
- `harmonyspeech/endpoints/openai/protocol.py` — every request/response Pydantic model, including Chatterbox generation fields
- `docs/api/openapi.json` — generated schema used for client generation

**Internal Request Types:**
- `harmonyspeech/common/inputs.py` — all `RequestInput` subclasses and `from_openai()` factories
- `harmonyspeech/common/outputs.py` — all `RequestOutput` subclasses

**Model Dispatch (four switch points):**
- `harmonyspeech/endpoints/openai/serving_{text_to_speech,voice_embed,speech_to_text,voice_activity_detection,voice_conversion,audio_conversion}.py` — `_*_MODEL_TYPES` / `_*_MODEL_GROUPS`
- `harmonyspeech/modeling/models/__init__.py` — `_MODELS`
- `harmonyspeech/modeling/loader.py` — `_MODEL_CONFIGS`, `_MODEL_WEIGHTS`, `_MODEL_SPEAKERS`, `get_model()`
- `harmonyspeech/task_handler/inputs.py` — `prepare_inputs()`
- `harmonyspeech/task_handler/model_runner_base.py` — `execute_model()`

**Request Routing / Pipeline:**
- `harmonyspeech/engine/harmonyspeech_engine.py` — `check_reroute_request_to_model()` (line ~337), `reroute_request_*` (lines ~109-335), `check_forward_processing()` (line ~363)
- `harmonyspeech/processing/scheduler.py` — `_schedule_waiting_requests()` (line ~134), `get_scheduling_budget()` (line ~191)

**Core Logic:**
- `harmonyspeech/engine/async_harmonyspeech.py` — `AsyncHarmonySpeech.generate/abort`, `run_engine_loop`, `engine_step`
- `harmonyspeech/task_handler/model_runner_base.py` — all `_execute_*` inference paths
- `harmonyspeech/modeling/models/chatterbox/chatterbox.py` — Chatterbox wrappers and compatibility patches

**Version:**
- `harmonyspeech/version.py` — `__version__` (currently `v0.1.1-rc1`)
- `pyproject.toml` — `project.version` (currently `0.1.1-rc2`); both must be kept in sync

## Tests Directory Deep Dive

```text
tests/
├── conftest.py
├── unit/                                  # FAST, fully mocked
│   ├── conftest.py
│   ├── initialization/
│   │   ├── test_engine.py
│   │   ├── test_config.py
│   │   └── test_chatterbox_imports.py     # import-time monkey-patch safety
│   ├── inference_flow/
│   │   ├── test_loader.py
│   │   ├── test_chatterbox_inputs.py      # prepare_chatterbox_* param validation
│   │   ├── test_chatterbox_routing.py     # reroute_request_chatterbox decisions
│   │   └── test_chatterbox_no_tempfile.py # embedding path avoids filesystem I/O
│   ├── modeling/
│   │   ├── test_hf_downloader.py
│   │   └── test_chatterbox_registry.py    # _MODELS contains all Chatterbox types
│   └── error_handling/
│       ├── test_config_errors.py
│       └── test_engine_error_resilience.py # error results, embed-dispatch fallback
├── integration/                           # COMPONENT INTERACTION
│   ├── conftest.py
│   ├── test_cli.py
│   ├── test_api_endpoints.py
│   └── test_chatterbox_flow.py            # serving-layer Chatterbox lifecycle
└── e2e/                                   # REAL MODELS, CPU-ONLY, SLOW
    ├── conftest.py
    ├── tts/
    │   ├── test_harmonyspeech.py
    │   ├── test_melotts.py
    │   ├── test_openvoice_v1.py
    │   ├── test_kittentts.py
    │   ├── test_chatterbox.py
    │   ├── test_chatterbox_turbo.py
    │   └── test_chatterbox_multilingual.py
    ├── stt/test_whisper.py
    ├── vad/test_silero_vad.py, test_whisper_vad.py
    └── audio_restoration/test_voicefixer.py
```

**pyproject configuration (`pyproject.toml:46`):**
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
addopts = ["--strict-markers", "-ra"]      # coverage is opt-in for speed
markers = ["unit", "integration", "e2e", "slow"]
```

```bash
pytest tests/unit/                     # fast, mocked
pytest tests/integration/              # serving + engine wiring
pytest tests/e2e/                      # real models, slow
pytest -m unit                         # by marker
pytest --cov=harmonyspeech --cov-report=term-missing --cov-report=xml
```

Coverage omits `harmonyspeech/modeling/models/*` (`pyproject.toml:62`), so vendor model code is not measured.

## Naming Conventions

**Files:**
- Snake case throughout: `api_server.py`, `async_harmonyspeech.py`, `gpu_model_runner.py`
- Serving modules: `serving_{capability}.py` (`serving_text_to_speech.py`, `serving_voice_activity_detection.py`, `serving_voice_embed.py`, `serving_audio_conversion.py`)
- Worker/runner pairs mirror each other: `cpu_worker.py` ↔ `cpu_model_runner.py`, `gpu_worker.py` ↔ `gpu_model_runner.py`
- Model package names match the family slug: `harmonyspeech/`, `openvoice/`, `melo/`, `kittentts/`, `voicefixer/`, `chatterbox/`
- Tests: `test_{module_or_behavior}.py`; e2e grouped by capability directory

**Directories:**
- Lowercase with underscores for infrastructure (`task_handler/`, `modeling/`, `processing/`)
- Lowercase family names for model packages; `parallel_wavegan/` keeps upstream naming

**Classes:**
- PascalCase: `HarmonySpeechEngine`, `AsyncHarmonySpeech`, `RequestTracker`, `Scheduler`, `ChatterboxTurboTTSModel`
- Request inputs: `{Capability}RequestInput`; outputs: `{Capability}RequestOutput`
- Serving classes: `OpenAIServing{Capability}`

**`model_type` strings:**
- PascalCase identifiers used as dictionary/string keys: `"HarmonySpeechEncoder"`, `"MeloTTSSynthesizer"`, `"FasterWhisper"`, `"SileroVAD"`, `"KittenTTSSynthesizer"`, `"ChatterboxTurboTTS"`
- Toolchain/API ids are lowercase snake case: `harmonyspeech`, `openvoice_v1`, `openvoice_v2`, `voicefixer`, `chatterbox`, `chatterbox_turbo`, `chatterbox_multilingual`, `chatterbox_vc`
- `ModelConfig.name` values in `config.yml` mix both: concrete names (`chatterbox`, `chatterbox_turbo`, `chatterbox_vc`, `kitten-tts-mini`) and role names (`hs1-encoder`, `ov2-tone-converter`)

**Test naming:**
- Behaviour-focused: `test_chatterbox_routing.py`, `test_engine_error_resilience.py`, `test_chatterbox_no_tempfile.py`

## Where to Add New Code

**New Model Implementation (PyTorch, registry-based):**
1. `harmonyspeech/modeling/models/{family}/{family}.py` — model class with `load_weights(checkpoint, config)`
2. `harmonyspeech/modeling/models/__init__.py` — add `"{model_type}": ("{family}", "{ClassName}")` to `_MODELS`
3. `harmonyspeech/modeling/loader.py` — add config/weight paths to `_MODEL_CONFIGS` and `_MODEL_WEIGHTS` (use `"native"` to bypass HF config loading); add speaker entries to `_MODEL_SPEAKERS` if applicable
4. `harmonyspeech/task_handler/inputs.py` — new branch in `prepare_inputs()` + `prepare_{family}_inputs()`
5. `harmonyspeech/task_handler/model_runner_base.py` — new `elif model_type == ...` in `execute_model()` + `_execute_{family}()`
6. `config.yml` and `config.gpu.yml` — add a `model_configs` entry
7. `tests/e2e/{capability}/test_{family}.py`

**New Native Model (ONNX / external library, like KittenTTS or Chatterbox):**
1. `harmonyspeech/modeling/models/{family}/` with the wrapper class
2. `_MODELS` entry mapped to `"native"`
3. `loader.py` — `"native"` entries in `_MODEL_CONFIGS` and `_MODEL_WEIGHTS`, plus an instantiation branch inside the `model_class == "native" and hf_config == "native"` block (see `loader.py:264-337`)
4. Input preparation + runner execution as above
5. If the library needs compatibility shims, keep them in the family package, not in `task_handler/`

**New Toolchain (multi-model pipeline, like Chatterbox or VoiceFixer):**
1. Add sentinel handling to `check_reroute_request_to_model()` and a `reroute_request_{toolchain}()` in `harmonyspeech/engine/harmonyspeech_engine.py`
2. Add the forwarding branch in `check_forward_processing()`; use `RequestStatus.FINISHED_FORWARDED` when re-submitting and `FINISHED_STOPPED` on the final step
3. Register the toolchain in the relevant `_*_MODEL_GROUPS` dicts:
   - `serving_text_to_speech.py::_TTS_MODEL_GROUPS`
   - `serving_voice_embed.py::_EMBEDDING_MODEL_GROUPS`
   - `serving_voice_conversion.py::_VOICE_CONVERSION_MODEL_GROUPS`
   - `serving_audio_conversion.py::_AUDIO_CONVERSION_MODEL_GROUPS`
4. Add or extend the `RequestInput` fields the pipeline needs in `harmonyspeech/common/inputs.py` and, if user-facing, `protocol.py`
5. Add `model_configs` entries for every step in `config.yml` / `config.gpu.yml`

**New Embedding-Capable TTS Model (embed-dispatch):**
1. Add the `model_type` to `_EMBEDDING_MODEL_TYPES` in `harmonyspeech/endpoints/openai/serving_voice_embed.py`
2. Add it to the relevant `_EMBEDDING_MODEL_GROUPS` list (order defines fallback priority)
3. Accept both `SpeechEmbeddingRequestInput` and `TextToSpeechRequestInput` in `prepare_inputs()`
4. Route `SpeechEmbeddingRequestInput` batches to an `_execute_*_embedding()` in `execute_model()`
5. Return a serializable embedding from `SpeechEmbeddingRequestOutput.output`

**New API Endpoint / Capability:**
1. `harmonyspeech/endpoints/openai/protocol.py` — request + response Pydantic models
2. `harmonyspeech/common/inputs.py` / `outputs.py` — `RequestInput` / `RequestOutput` subclass with `from_openai()`
3. `harmonyspeech/endpoints/openai/serving_{capability}.py` — `_*_MODEL_TYPES`, `_*_MODEL_GROUPS`, `models_from_config()`, async public method, `*_full_generator()`
4. `harmonyspeech/endpoints/openai/api_server.py` — import the serving class, add a module-level singleton, register routes in `router`, instantiate in `run_server()`, and add the capability to `/health`
5. `harmonyspeech/engine/harmonyspeech_engine.py` — routing + forwarding support for the new request type

**New Configuration Option:**
- Model-scoped: add a parameter to `ModelConfig.__init__` in `harmonyspeech/common/config.py` (and to `config.yml` entries)
- Engine-scoped: add to `EngineConfig`
- CLI-scoped: `harmonyspeech/endpoints/openai/args.py` and `harmonyspeech/engine/args_tools.py`

**New Audio Post-Processing Filter:**
- `post_generation_filters` and `pre_processing_filters` already exist on the protocol and on `TextToSpeechRequestInput`
- Implement the filter as an `AudioConversion` toolchain model and add execution to the `# TODO: Post-Generation Processing` block in `check_forward_processing()` (`harmonyspeech_engine.py:504`)

**Tests:**
- Unit: `tests/unit/{initialization,inference_flow,modeling,error_handling}/test_{feature}.py`
- Integration: `tests/integration/test_{feature}.py`
- E2E: `tests/e2e/{tts,stt,vad,audio_restoration}/test_{model}.py`
- Fixtures/audio: `tests/test-data/samples/`

**Documentation:**
- Feature docs: `docs/{topic}.md`
- Regenerate API schema after protocol changes: `docs/api/generate_openapi_go.sh` / `generate_openapi_js.sh` (both consume `docs/api/openapi.json`)
- Changelog entry: `CHANGELOG.md`

## Special Directories

**`.planning/`:**
- Purpose: GSD planning documents and codebase analysis
- Generated: No (written by GSD agents)
- Committed: Yes

**`memory-bank/`:**
- Purpose: Roo AI agent memory bank (`activeContext.md`, `progress.md`, etc.)
- Generated: Yes (written by AI sessions)
- Committed: Yes

**`models/`:**
- Purpose: Local HuggingFace weight cache populated at first run
- Generated: Yes
- Committed: No (gitignored)

**`cache/`:**
- Purpose: Runtime caches (`hub/`, `numba_cache/`, `xet/`); safe to delete
- Generated: Yes
- Committed: No

**`undefined/`:**
- Purpose: Accidental artifact containing only `.cache/`; not referenced by any code
- Generated: Yes
- Committed: No — safe to remove

**`plans/`:**
- Purpose: Empty scratch directory
- Committed: Present but empty

**`.current_work/`:**
- Purpose: Scratch notes and working configs (e.g. `vllm-omni-analysis.md`, `vllm-omni-integration-plan.md`, a local `config.yml`)
- Generated: Yes (manual)
- Committed: Currently present; treat as non-shipping scratch

**`docs/api/`:**
- Purpose: Generated OpenAPI schema plus client-code generation scripts
- Generated: `openapi.json` yes; scripts no
- Committed: Yes

**`frontend/`:**
- Purpose: Single React source tree built in two modes; has its own `AGENTS.md`, `package.json`, lockfile, and Dockerfiles
- Generated: `node_modules/` yes (not committed); `src/` no
- Committed: Source yes, `node_modules/` no

**`.github/workflows/`:**
- Purpose: CI/CD — `test.yml`, `docker-release-engine.yml`, `docker-release-ui.yml`

**`.claude/`, `.roo/`, `.clinerules/`, `.gitnexus/`, `.idea/`, `.vscode/`:**
- Purpose: Tooling/agent configuration and local IDE settings
- Committed: Mixed; `.gitnexus/` and caches are local

---

*Structure analysis: 2026-11-02*

<!-- refreshed: 2026-11-02 -->
# Testing Patterns

**Analysis Date:** 2026-11-02

## Test Framework

**Runner:** pytest (`pyproject.toml` → `[tool.pytest.ini_options]`)

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
addopts = [
    "--strict-markers",
    "-ra",
]
markers = [
    "unit: marks tests as unit tests (fast, fully mocked)",
    "integration: marks tests as integration tests (component interaction, partially mocked)",
    "e2e: marks tests as end-to-end tests (real models, slow, CPU-only)",
    "slow: marks tests that take > 30 seconds to run",
]
```

**Key dependencies** (`requirements-common.txt`):
- `pytest>=8.0.0`
- `pytest-asyncio>=0.23.0` (mode: auto)
- `pytest-mock>=3.12.0`
- `pytest-cov>=5.0.0`
- `torch` — device/dtype fixtures and tensor assertions
- Chatterbox dependencies are installed separately: `pip install --no-deps -r requirements-chatterbox.txt` (`chatterbox-tts==0.1.6`)

**Notes:**
- Coverage is **not** in `addopts`; default runs are fast. Pass `--cov=...` explicitly when coverage is needed.
- `--strict-markers` means an unregistered marker aborts the run. Register new markers in `pyproject.toml` before using them.
- `asyncio_mode = "auto"` makes `async def` tests work without a marker. Some integration tests still include `@pytest.mark.asyncio`; that is harmless and accepted.

## Test Directory Structure

```
tests/
├── conftest.py                          # CLI options (--device/--dtype), shared fixtures
├── test-data/
│   └── samples/                         # WAV assets used by E2E voice-cloning tests
│       ├── wanda4.wav                   # short (<5 s)
│       ├── wanda5.wav                   # short (<5 s)
│       ├── wanda6.wav                   # short (<5 s)
│       └── jerry_seinfeld_prompt.wav    # long (~32 s)
├── unit/
│   ├── conftest.py                      # mock_model_loader, mock_hf_downloader
│   ├── error_handling/
│   │   ├── test_config_errors.py
│   │   └── test_engine_error_resilience.py
│   ├── inference_flow/
│   │   ├── test_chatterbox_inputs.py
│   │   ├── test_chatterbox_no_tempfile.py
│   │   ├── test_chatterbox_routing.py
│   │   └── test_loader.py
│   ├── initialization/
│   │   ├── test_chatterbox_imports.py
│   │   ├── test_config.py
│   │   └── test_engine.py
│   └── modeling/
│       ├── test_chatterbox_registry.py
│       └── test_hf_downloader.py
├── integration/
│   ├── conftest.py                      # test_app, mock_engine_app
│   ├── test_api_endpoints.py
│   ├── test_chatterbox_flow.py
│   └── test_cli.py
└── e2e/
    ├── conftest.py                      # engine fixtures, model cache, helpers, auto e2e marker
    ├── stt/test_whisper.py
    ├── tts/
    │   ├── test_chatterbox.py
    │   ├── test_chatterbox_multilingual.py
    │   ├── test_chatterbox_turbo.py
    │   ├── test_harmonyspeech.py
    │   ├── test_kittentts.py
    │   ├── test_melotts.py
    │   └── test_openvoice_v1.py
    ├── vad/
    │   ├── test_silero_vad.py
    │   └── test_whisper_vad.py
    └── audio_restoration/test_voicefixer.py
```

**Current size (2026-11-02):** 174 test functions — 112 unit, 23 integration, 39 E2E. (Count excludes fixtures/helpers named `test_*` in `conftest.py`.)

## Test Tiers

| Tier | Speed | Mocking | Use Case |
|------|-------|---------|----------|
| `unit` | Fast (<1 s) | Full mocking | Routing, config parsing, input preparation, serialization, registry |
| `integration` | Medium (1–30 s) | Serving layer mocked, no real models | API endpoints, CLI, serving-handler lifecycles |
| `e2e` | Slow (minutes) | None (real models) | Full inference with real weights and real audio |

**Markers:**
- `@pytest.mark.unit` — unit tests
- `@pytest.mark.integration` — integration tests
- `@pytest.mark.e2e` — E2E tests; auto-applied by `pytest_collection_modifyitems` in `tests/e2e/conftest.py:472-476`
- `@pytest.mark.slow` — tests > 30 s

Newer E2E files use a module-level marker list instead of decorating each test:

```python
pytestmark = [pytest.mark.e2e, pytest.mark.slow]
```

(`tests/e2e/tts/test_chatterbox.py`, `test_chatterbox_turbo.py`, `test_chatterbox_multilingual.py`)

## Running Tests

### Basic commands

```bash
# All tests (CPU, no coverage)
pytest

# One tier
pytest tests/unit/
pytest tests/integration/
pytest tests/e2e/

# By marker
pytest -m unit
pytest -m "not e2e"
pytest -m e2e
pytest -m "not slow"
```

### Device and dtype

```bash
pytest --device=cpu --dtype=float32      # defaults
pytest --device=cuda                     # fails fast if CUDA is unavailable
pytest --device=cuda --dtype=float16
```

| Flag | Default | Values | Description |
|------|---------|--------|-------------|
| `--device` | `cpu` | `cpu`, `cuda` | Target device; used by Chatterbox engine fixtures and future device-aware fixtures |
| `--dtype` | `float32` | `float32`, `float16` | Default tensor dtype |

`pytest_configure` in `tests/conftest.py:28-34` calls `pytest.exit(..., returncode=1)` when `--device=cuda` is requested without CUDA. Do not weaken this guard.

### Coverage

```bash
pytest --cov=harmonyspeech --cov-report=term-missing --cov-report=xml
pytest --cov=harmonyspeech --cov-report=html
```

### Targeted Chatterbox runs

```bash
# Unit only
pytest tests/unit/inference_flow/test_chatterbox_routing.py
pytest tests/unit/inference_flow/test_chatterbox_inputs.py
pytest tests/unit/error_handling/test_engine_error_resilience.py

# Integration
pytest tests/integration/test_chatterbox_flow.py

# E2E by variant (mirrors CI selectors)
pytest -k "chatterbox and not turbo and not multilingual" tests/e2e/ --device=cpu --dtype=float32
pytest -k "chatterbox_turbo" tests/e2e/ --device=cpu --dtype=float32
pytest -k "chatterbox_multilingual" tests/e2e/ --device=cpu --dtype=float32
```

`--no-cov` appears in some CI commands for legacy reasons. Because coverage is no longer in `addopts`, it is a no-op unless `pytest-cov` options are supplied.

## Root Fixtures (`tests/conftest.py`)

| Fixture | Scope | Description |
|---------|-------|-------------|
| `device` | session | `--device` value (`"cpu"` / `"cuda"`) |
| `dtype` | session | `--dtype` value (`"float32"` / `"float16"`) |
| `tests_root` | session | `Path` to `tests/` |
| `test_data_dir` | session | `Path` to `tests/test-data/` |
| `sample_config` | function | Minimal valid model config dict for parser tests |

**CLI hooks:** `pytest_addoption` registers `--device`/`--dtype`; `pytest_configure` validates CUDA availability.

## Unit Fixtures (`tests/unit/conftest.py`)

| Fixture | Description |
|---------|-------------|
| `mock_model_loader` | Patches `harmonyspeech.modeling.loader.get_model_class` with a `MagicMock` |
| `mock_hf_downloader` | Patches `harmonyspeech.modeling.hf_downloader.download_model` to avoid network calls |

Additional local fixtures are defined per file, e.g. `cpu_device` in `tests/unit/initialization/test_config.py`, and cache-clearing fixtures in `tests/unit/modeling/test_hf_downloader.py`:

```python
@pytest.fixture(autouse=True)
def _clear_caches():
    """Clear caches before and after each test for isolation."""
    _config_cache.clear()
    _file_cache.clear()
    yield
    _config_cache.clear()
    _file_cache.clear()
```

## Integration Fixtures (`tests/integration/conftest.py`)

| Fixture | Scope | Description |
|---------|-------|-------------|
| `test_app` | module | `TestClient` built from the real `api_server.app`; no model loading |
| `mock_engine_app` | module | `TestClient` built with all serving-module globals patched |

`mock_engine_app` patches eight `harmonyspeech.endpoints.openai.api_server` globals:
- `openai_serving_tts`, `openai_serving_stt`, `openai_serving_vc`
- `openai_serving_embedding`, `openai_serving_vad`, `openai_serving_ac`
- `engine`, `engine_args`

Each mocked serving object exposes async `create_*`, `get_model_list`, and `show_available_models`, plus `engine.check_health`. The app is built with `build_app(parser.parse_args([]))`, so CLI defaults must remain valid for an empty argument list.

## E2E Fixtures (`tests/e2e/conftest.py`)

### Engine fixtures

| Fixture | Scope | Returns | Notes |
|---------|-------|---------|-------|
| `kittentts_mini_engine` | session | `(engine, serving_tts)` | `KittenML/kitten-tts-mini-0.8` |
| `kittentts_micro_engine` | session | `(engine, serving_tts)` | micro variant |
| `kittentts_nano_engine` | session | `(engine, serving_tts)` | nano fp32 variant |
| `kittentts_nano_int8_engine` | session | `(engine, serving_tts)` | nano int8 variant |
| `melotts_en_engine` | module | `(engine, serving_tts, serving_embed, serving_vc)` | MeloTTS + OpenVoice V2 (4 models) |
| `openvoice_v1_en_engine` | module | `(engine, serving_tts, serving_embed, serving_vc)` | OpenVoice V1 (4 models) |
| `harmonyspeech_engine` | module | `(engine, serving_tts, serving_embed)` | encoder + synthesizer + vocoder |
| `whisper_engine` | module | `(engine, serving_stt)` | `Systran/faster-whisper-tiny` |
| `vad_engine` | module | `(engine, serving_vad)` | SileroVAD |
| `whisper_vad_engine` | module | `(engine, serving_vad)` | FasterWhisper used for VAD |
| `voicefixer_engine` | module | `(engine, serving_audio)` | restorer + vocoder |
| `chatterbox_engine` | module | `(engine, serving_tts, serving_embed, serving_vc)` | `ChatterboxTTS` + `ChatterboxVC` |
| `chatterbox_turbo_engine` | module | `(engine, serving_tts, serving_embed)` | `ChatterboxTurboTTS` + `ChatterboxEmbedding` |
| `chatterbox_multilingual_engine` | module | `(engine, serving_tts, serving_embed)` | `ChatterboxMultilingualTTS` + `ChatterboxEmbedding` |

All engine fixtures construct `ModelConfig` → `EngineConfig` → `AsyncEngineArgs` → `AsyncHarmonySpeech.from_engine_args_and_config(...)`, then instantiate the serving handler with `models_from_config(...)`. Follow this shape for a new model.

**Device behavior:** the three Chatterbox fixtures consume the `device` fixture (`DeviceConfig(device=device)`), so they run on CPU or CUDA. Older model fixtures hardcode `DeviceConfig(device="cpu")`; changing them to device-aware is a per-model decision, not a blanket change.

### Helper fixtures and functions

| Name | Scope | Description |
|------|-------|-------------|
| `models_cache_dir` | session | Returns `HF_HOME` (default `~/.cache/huggingface`) and ensures `<HF_HOME>/hub` exists; does not create a temp dir |
| `mock_raw_request` | session | `MagicMock` FastAPI request with async `is_disconnected()` returning `False` |
| `make_silent_wav_b64(sample_rate=24000, duration_secs=1)` | function | Generates a silent mono 16-bit WAV as base64 |
| `load_sample_audio_b64(sample_name="wanda4")` | function | Reads `tests/test-data/samples/<name>.wav` and returns base64 |

E2E tests import helpers explicitly:

```python
from tests.e2e.conftest import load_sample_audio_b64
```

This requires the repository root on `sys.path` (running pytest from the repo root).

## Test Structure Patterns

### Unit: routing without real models

`tests/unit/inference_flow/test_chatterbox_routing.py` builds an engine shell and `MagicMock(spec=ModelConfig)` configs. This is the standard way to test engine dispatch cheaply:

```python
def _make_model_config(name: str, model_type: str) -> ModelConfig:
    cfg = MagicMock(spec=ModelConfig)
    cfg.name = name
    cfg.model_type = model_type
    return cfg


def _make_engine(*model_configs) -> HarmonySpeechEngine:
    engine = HarmonySpeechEngine.__new__(HarmonySpeechEngine)
    engine.model_configs = list(model_configs)
    engine.scheduler = MagicMock()
    return engine
```

Rules:
- Use `__new__` to bypass `__init__` when the constructor would load models.
- Set only the attributes the method under test reads.
- Use `MagicMock(spec=...)` so typos and unexpected attributes fail.

### Unit: input preparation and option validation

`tests/unit/inference_flow/test_chatterbox_inputs.py` covers defaults, unsupported-option rejection, conflict validation, and language codes. It uses `MagicMock` requests plus real `TextToSpeechGenerationOptions` / `GenerationOptions` objects:

```python
def test_tts_rejects_top_k():
    opts = GenerationOptions(top_k=500)
    req = _make_tts_request(generation_options=opts)
    with pytest.raises(ValueError, match="top_k"):
        prepare_chatterbox_tts_inputs([req])
```

The file also demonstrates the optional-import guard used when a test must be collectible before an implementation exists:

```python
try:
    from harmonyspeech.task_handler.inputs import prepare_chatterbox_tts_inputs
    PREPARE_FUNCTIONS_AVAILABLE = True
except ImportError:
    PREPARE_FUNCTIONS_AVAILABLE = False

@pytest.mark.skipif(not PREPARE_FUNCTIONS_AVAILABLE, reason="prepare functions not yet implemented")
```

Do not add new guards for APIs that already exist.

### Unit: asserting no filesystem I/O

`tests/unit/inference_flow/test_chatterbox_no_tempfile.py` patches `builtins.open`, `tempfile.NamedTemporaryFile`, and `tempfile.mkstemp`, then asserts none were called. It also patches `io.BytesIO` with a tracking wrapper to prove in-memory processing:

```python
with (
    patch("builtins.open") as mock_open,
    patch("tempfile.NamedTemporaryFile") as mock_tmp,
    patch("tempfile.mkstemp") as mock_mkstemp,
):
    try:
        runner._execute_chatterbox_embedding(inputs, requests)
    except Exception:
        pass  # only file I/O behavior is under test
    mock_open.assert_not_called()
    mock_tmp.assert_not_called()
    mock_mkstemp.assert_not_called()
```

Swallowing exceptions is acceptable only when the assertion target is a side effect (I/O, call count). Do not swallow exceptions when asserting output values.

### Integration: serving-layer lifecycle with AsyncMock

`tests/integration/test_chatterbox_flow.py` mocks the serving handlers and asserts response types/data:

```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_chatterbox_tts_direct(mock_chatterbox_serving):
    serving_tts, _, _ = mock_chatterbox_serving
    request = TextToSpeechRequest(model="chatterbox", input="Hello world", mode="single_speaker_tts")
    response = await serving_tts.create_text_to_speech(request, mock_raw_request)
    assert response is not None
    assert hasattr(response, "data")
```

Note: this file mocks the serving object itself, so it verifies the request/response contract, not routing. Routing is covered in unit tests.

### E2E: real inference, response-type assertions

Sync tests call async serving methods with `asyncio.run` and validate both type and payload:

```python
def test_chatterbox_tts_no_cloning(chatterbox_engine, mock_raw_request):
    engine, serving_tts, _, _ = chatterbox_engine
    request = TextToSpeechRequest(model="chatterbox", input=TEXT_INPUT, mode="single_speaker_tts")
    response = asyncio.run(serving_tts.create_text_to_speech(request, mock_raw_request))
    assert isinstance(response, TextToSpeechResponse), f"Got: {response}"
    assert response.data
    assert len(base64.b64decode(response.data)) > 0
```

Always include the actual response in the assertion message — E2E failures frequently return an `ErrorResponse` instead of audio.

### E2E: regression tests for upstream failures

Newer Chatterbox E2E tests document the bug they lock down in the docstring and, where relevant, assert engine recovery:

```python
def test_chatterbox_turbo_embedding_short_audio_returns_error(chatterbox_turbo_engine, mock_raw_request):
    """Turbo embedding with audio < 5s returns 400 ErrorResponse; engine survives to serve next request."""
    ...
    assert isinstance(bad_response, ErrorResponse)
    recovery_response = asyncio.run(serving_tts.create_text_to_speech(recovery_request, mock_raw_request))
    assert isinstance(recovery_response, TextToSpeechResponse)
```

When fixing a production bug, add a test in the tier where it can fail deterministically, and include the original exception text in the docstring.

## Chatterbox Test Coverage Map

| File | Protects |
|------|----------|
| `tests/unit/initialization/test_chatterbox_imports.py` | `perth`, `pyloudnorm`, `ChatterboxTTS`, `ChatterboxVC`, `ChatterboxMultilingualTTS`, `ChatterboxTurboTTS`, 23+ language codes importable |
| `tests/unit/modeling/test_chatterbox_registry.py` | All five Chatterbox architectures registered; `load_model_cls` returns `"native"` for TTS/VC |
| `tests/unit/inference_flow/test_chatterbox_routing.py` | Sentinel model IDs map to variants; embedding fallback; multi-step embed→TTS forwarding; VC routing |
| `tests/unit/inference_flow/test_chatterbox_inputs.py` | Generation-option fields/defaults; unsupported-option `ValueError`; audio/embedding conflict; language defaults; embedding/VC tuple shapes |
| `tests/unit/inference_flow/test_chatterbox_no_tempfile.py` | No `open()`/`tempfile` in Chatterbox TTS and embedding execution; `io.BytesIO` used |
| `tests/unit/error_handling/test_engine_error_resilience.py` | Embedding accepts `TextToSpeechRequestInput`; prepare failures become per-request errors; error results are not forwarded; `engine_step` survives `step_async` exceptions and aborts failed requests |
| `tests/integration/test_chatterbox_flow.py` | Serving-layer TTS/embed/VC request contracts for Chatterbox model IDs |
| `tests/e2e/tts/test_chatterbox.py` | Real `ChatterboxTTS` + `ChatterboxVC`: direct TTS, precomputed embedding, voice cloning, standalone embedding, VC with target audio/embedding |
| `tests/e2e/tts/test_chatterbox_turbo.py` | `ChatterboxTurboTTS`: direct TTS, ≥5 s embedding requirement, precomputed embedding, short-audio error + engine recovery |
| `tests/e2e/tts/test_chatterbox_multilingual.py` | Multilingual TTS (English), embedding without `language` attribute on request, precomputed embedding path |

## Mocking Patterns

**Framework:** `unittest.mock` + `pytest.monkeypatch`, with `pytest-mock` available but rarely used.

**What to mock:**
- HuggingFace Hub calls (`harmonyspeech.modeling.hf_downloader.download_model`, `snapshot_download`)
- Model classes and instances (`MagicMock`, `MagicMock(spec=...)`)
- Serving module globals in `harmonyspeech.endpoints.openai.api_server`
- File I/O when asserting in-memory behavior (`builtins.open`, `tempfile.*`)
- `torch.cuda.is_available` and device detection helpers with `monkeypatch.setattr`

**What NOT to mock:**
- Input preparation logic and option validation (unit tests exercise the real functions)
- Routing decisions in `HarmonySpeechEngine` (only configs/model execution are mocked)
- Configuration parsing and dtype resolution
- E2E inference output (real models)

**monkeypatch example** (`tests/unit/initialization/test_config.py`):

```python
def test_device_config_auto_no_device_raises(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("harmonyspeech.common.config.is_cpu", lambda: False)
    with pytest.raises(RuntimeError):
        DeviceConfig("auto")
```

**Patch target rule:** patch where the name is looked up, not where it is defined. `mock_model_loader` patches `harmonyspeech.modeling.loader.get_model_class`; `_build_error_result` tests patch `harmonyspeech.task_handler.model_runner_base.prepare_inputs`.

**Async mocks:** use `AsyncMock` for coroutine methods (`check_health`, `create_text_to_speech`, `step_async`) and plain `MagicMock` for sync attributes.

## Coverage

**Requirements:** none enforced. Coverage is collected only when requested and uploaded as an artifact in CI.

**Configuration** (`pyproject.toml`):

```toml
[tool.coverage.run]
omit = [
    "harmonyspeech/modeling/models/*",
]
```

**Coverage exclusions:**
- `harmonyspeech/modeling/models/*` — vendored/adapted model implementations are excluded.
- `.coverage`, `coverage.xml`, and `htmlcov/` are gitignored; prior artifacts exist at the repo root but are not authoritative for current coverage.

**Configuration quirk:** `filterwarnings = ["ignore::DeprecationWarning", "ignore::UserWarning"]` is currently nested under `[tool.coverage.run]` in `pyproject.toml`, where pytest does not read it. If warning filtering is needed, move it into `[tool.pytest.ini_options]`. Do not assume warnings are suppressed.

## CI Pipeline

`.github/workflows/test.yml` runs on every push and pull request.

| Job | Depends on | Command |
|-----|------------|---------|
| `test` | — | `pytest --device=cpu --dtype=float32 -v --tb=short -m "not e2e" --cov=harmonyspeech --cov-report=term-missing --cov-report=xml` |
| `lint` | — | `ruff format --check --diff harmonyspeech/ tests/` then `ruff check harmonyspeech/ tests/` |
| `e2e-kittentts` | `test` | `pytest -k "kittentts" tests/e2e/ --device=cpu --dtype=float32 -v --tb=short` |
| `e2e-melotts-openvoice-v2` | `test` | `pytest -k "melotts or openvoice_v2" tests/e2e/ ...` |
| `e2e-openvoice-v1` | `test` | `pytest -k "openvoice_v1" tests/e2e/ ...` |
| `e2e-harmonyspeech` | `test` | `pytest -k "harmonyspeech" tests/e2e/ ...` |
| `e2e-whisper` | `test` | `pytest -k "whisper" tests/e2e/ ... --no-cov` |
| `e2e-vad` | `test` | `pytest -k "silero_vad or vad" tests/e2e/ ... --no-cov` |
| `e2e-voicefixer` | `test` | `pytest -k "voicefixer" tests/e2e/ ... --no-cov` |
| `e2e-chatterbox` | `test` | `pytest -k "chatterbox and not turbo and not multilingual" tests/e2e/ ... --no-cov` |
| `e2e-chatterbox-turbo` | `test` | `pytest -k "chatterbox_turbo" tests/e2e/ ... --no-cov` |
| `e2e-chatterbox-multilingual` | `test` | `pytest -k "chatterbox_multilingual" tests/e2e/ ... --no-cov` |

All E2E jobs:
- set `HF_HOME: ~/.cache/huggingface`
- cache `~/.cache/huggingface` keyed on `requirements-common.txt`, `requirements-cpu.txt`, and (for Chatterbox jobs) `requirements-chatterbox.txt`
- install system packages `libmecab-dev ffmpeg libsndfile1 espeak-ng`
- install `requirements-cpu.txt`, `requirements-common.txt`, then `pip install --no-deps -r requirements-chatterbox.txt`
- free disk space with `AdityaGarg8/remove-unwanted-software@v5`
- run on `ubuntu-latest`, Python 3.12

The unit/integration job uploads `coverage.xml` with `retention-days: 7`. The `test` job installs Chatterbox dependencies, so `tests/unit/initialization/test_chatterbox_imports.py` must pass there.

## Adding New Tests

### Unit tests
1. Place under `tests/unit/<category>/test_<module>.py` (categories: `initialization`, `inference_flow`, `error_handling`, `modeling`).
2. Import the module under test at module level; use `MagicMock` / `MagicMock(spec=...)` for collaborators.
3. Build engine shells with `HarmonySpeechEngine.__new__(...)` and set only required attributes.
4. Mark with `@pytest.mark.unit`.
5. Add `pytest.raises(ValueError, match="...")` for validation paths, asserting the option name appears in the message.

### Integration tests
1. Place under `tests/integration/test_<feature>.py`.
2. Use `mock_engine_app` for endpoint tests, or mock serving handlers for contract tests.
3. Mark with `@pytest.mark.integration` (async methods may also carry `@pytest.mark.asyncio`).
4. Do not load real models; patch `harmonyspeech.endpoints.openai.api_server` globals.

### E2E tests
1. Place under `tests/e2e/<task>/test_<model>.py`.
2. Add or reuse an engine fixture in `tests/e2e/conftest.py`; return `(engine, serving_*)` tuples in the same order as existing fixtures.
3. Use `models_cache_dir` in the fixture signature so HuggingFace caching is consistent.
4. Declare `pytestmark = [pytest.mark.e2e, pytest.mark.slow]` (preferred) or decorate each test.
5. Use `load_sample_audio_b64()` with the correct sample length:
   - `jerry_seinfeld_prompt` (~32 s) for Turbo embedding, which enforces ≥5 s.
   - `wanda4`/`wanda5`/`wanda6` for short-audio/negative paths and multilingual embedding.
6. Assert `isinstance(response, <Expected>Response)` with the response in the message, then validate non-empty base64 payload.

### Definition of done for a model feature
- Unit coverage: registry, imports, routing, input preparation, and no-I/O guarantees where applicable.
- Integration coverage: serving request/response contract.
- E2E coverage: real weights for each variant, including one negative path that proves the engine survives an inference error.
- `ruff format --check` and `ruff check` pass on touched files.
- New fixtures/markers documented in this file and registered in `pyproject.toml`.

---

*Testing analysis: 2026-11-02*

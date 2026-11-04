<!-- refreshed: 2026-11-02 -->
# External Integrations

**Analysis Date:** 2026-11-02

## APIs & External Services

**Model Hosting / Distribution:**
- **Hugging Face Hub** — sole source of model weights, downloaded at runtime
  - SDK/Client: `huggingface_hub` (`hf_hub_download`, `snapshot_download`) + `hf_xet` fast transfer
  - Auth: `HF_TOKEN` env var (optional; public repos work anonymously — the Chatterbox Turbo wrapper explicitly maps empty string → `None`)
  - Downloader: `harmonyspeech/modeling/hf_downloader.py` (three-tier resolution: local dir → HF disk cache `local_files_only=True` → network), plus direct `hf_hub_download` in `harmonyspeech/modeling/models/kittentts/kittentts.py` and `snapshot_download` in `harmonyspeech/modeling/models/chatterbox/chatterbox.py`
  - Cache: `HF_HOME` (set to `/app/harmony-speech-engine/cache` in containers, `./cache` volume in Compose); file locks under `$XDG_CACHE_HOME/harmony/locks/`

**Harmony API-Key / Rate-Limit Service (optional, external HTTP):**
- Base URL: `ENDPOINT_URL` env var; if unset/empty, key validation and rate limiting are disabled
- Client: `requests` in `auth/apikeys.py` (`ApiKeyCacheManager`)
- Calls:
  - `POST {ENDPOINT_URL}/check` — body `{apiKey, service}`, header `Harmony-Api-Key: $HARMONYSPEECH_API_KEY`; returns per-key stats (`requestsLastMinute`, `requestsLastDay`, `requestsPerMinuteLimit`, `requestsPerDayLimit`)
  - `POST {ENDPOINT_URL}/event` — body `{apiKey, service, eventType, ipAddress}`; returns `eventID`; fired from a background thread per authenticated request
- Timeouts: 10 s; connection/timeout/HTTP/JSON errors are logged, not raised
- In-memory cache of stats keyed by API key (60 s window logic in `check_request_allowed_by_rate_limit`)

## Data Storage

**Databases:**
- None — the engine is stateless; no database integration anywhere in `harmonyspeech/`

**File Storage:**
- Local filesystem only
- Model cache: `./cache/` → `/app/harmony-speech-engine/cache` (`HF_HOME`)
- Numba cache: `${HF_HOME}/numba_cache` (`NUMBA_CACHE_DIR`)
- File locks for concurrent downloads: `$XDG_CACHE_HOME/harmony/locks/`

**Caching:**
- `cached_path` + Hugging Face Hub snapshot/disk cache
- In-process immutable caches in `harmonyspeech/modeling/hf_downloader.py` (`_config_cache`, `_file_cache`) to avoid repeated HF HEAD/ETag checks
- In-process API-key stats cache in `auth/apikeys.py`

## Authentication & Identity

**Inbound API auth (optional):**
- Enabled when `HARMONYSPEECH_API_KEY` is set (env) or `--api-keys` is passed (`harmonyspeech/endpoints/openai/api_server.py:304`)
- Headers accepted: `Api-Key` (Harmony rate-limited keys) and `Authorization` (static shared key)
- Applied only to paths starting with `/v1`; `OPTIONS` requests bypass; `/api` prefix excluded
- `HARMONYSPEECH_ADMIN_KEY` / `--admin-key` gates admin operations; warning logged if absent
- Rate limiting delegated to the external Harmony service (see above); `SERVICE_NAME` (default `harmony-speech-engine`) scopes limits
- Client IP taken from `request.client.host`, with `X-Real-Ip` header honored when present

**Hugging Face auth:**
- `HF_TOKEN` for gated/private repositories

## Model Integrations (AI/ML Providers)

All AI models execute locally — no cloud inference APIs are called.

### TTS / Voice Conversion Providers

**Chatterbox (Resemble AI) — 4 variants:**
- Standard TTS: `https://huggingface.co/resemble-ai/chatterbox` (repo id used verbatim in `config.yml`/`config.gpu.yml`)
- Turbo TTS: `https://huggingface.co/ResembleAI/chatterbox-turbo` (hard-coded `_CHATTERBOX_TURBO_REPO_ID` in `harmonyspeech/modeling/models/chatterbox/chatterbox.py`, fetched via `snapshot_download` with `allow_patterns=["*.safetensors","*.json","*.txt","*.pt","*.model"]`)
- Multilingual TTS: same `resemble-ai/chatterbox` repo, 23 languages (`SUPPORTED_LANGUAGES` mirrored from `chatterbox.mtl_tts`)
- Voice conversion: `resemble-ai/chatterbox` via `ChatterboxVC`
- Implementation: thin wrappers in `harmonyspeech/modeling/models/chatterbox/chatterbox.py` over the `chatterbox` Python package (`chatterbox-tts==0.1.6`, installed `--no-deps`)
- Loader branches: `harmonyspeech/modeling/loader.py:304-336`
- Watermarking: `resemble-perth==1.0.1` implicit watermarker; `watermark: false` in YAML swaps in `perth.DummyWatermarker()` at load time
- Embeddings: `ChatterboxEmbedding` model type reuses one of the TTS wrappers to serve `/v1/embed/speaker`
- CPU caveats handled in code: `map_location=cpu` monkey-patch for multilingual checkpoints; float32 casts for `s3gen` mel filters and `voice_encoder` LSTM; `attn_implementation="eager"` for the multilingual T3

**KittenTTS (ONNX TTS, English):**
- Sources: `KittenML/kitten-tts-mini-0.8`, `KittenML/kitten-tts-micro-0.8`, `KittenML/kitten-tts-nano-0.8-fp32`, `KittenML/kitten-tts-nano-0.8-int8`
- Runtime: `onnxruntime`; downloads `config.json`, the model file named by `config["model_file"]`, and voices file via `hf_hub_download`
- Phonemizer: `misaki[en]>=0.9.4` (+ `espeakng_loader`)
- Model type: `KittenTTSSynthesizer`; voices Bella, Jasper, Luna, Bruno, Rosie, Hugo, Kiki, Leo
- Implementation: `harmonyspeech/modeling/models/kittentts/`

**HarmonySpeech V1:**
- Source: `https://huggingface.co/harmony-ai/harmony-speech-v1`
- Model types: `HarmonySpeechEncoder` (SpeakerEncoder), `HarmonySpeechSynthesizer` (ForwardTacotron), `HarmonySpeechVocoder` (MelGANGenerator)
- Implementation: `harmonyspeech/modeling/models/harmonyspeech/`

**OpenVoice V1:**
- Source: `https://huggingface.co/myshell-ai/openvoice`
- Model types: `OpenVoiceV1Synthesizer`, `OpenVoiceV1ToneConverter`, `OpenVoiceV1ToneConverterEncoder`
- Languages: EN, ZH; voices for EN include default/whispering/shouting/excited/cheerful/terrified/angry/sad/friendly
- Implementation: `harmonyspeech/modeling/models/openvoice/`
- Depends on Faster-Whisper for the embedding/VAD-assisted step

**MeloTTS / OpenVoice V2:**
- Sources: `myshell-ai/MeloTTS-English-v3`, `-Chinese`, `-Spanish`, `-French`, `-Japanese`, `-Korean`; tone converter from `https://huggingface.co/myshell-ai/openvoicev2`
- Model types: `MeloTTSSynthesizer`, `OpenVoiceV2ToneConverter`, `OpenVoiceV2ToneConverterEncoder`
- Implementation: `harmonyspeech/modeling/models/melo/` and `harmonyspeech/modeling/models/openvoice/`
- Depends on Faster-Whisper for the embedding step

**VoiceFixer (audio restoration):**
- Source: `https://huggingface.co/jlmarrugom/voice_fixer` (alternative in config comments: `cqchangm/voicefixer`)
- Model types: `VoiceFixerRestorer`, `VoiceFixerVocoder`
- Processes at 44.1 kHz internally
- Implementation: `harmonyspeech/modeling/models/voicefixer/`

### STT Providers

**Faster-Whisper / Distil-Whisper:**
- Library: `faster-whisper>=1.1.0`; backend `ctranslate2==4.4.0` (pinned)
- Model tags: `tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3`, `large-v3-turbo`
- Config uses `medium` + `tiny` on CPU (`config.yml`) and `large-v3-turbo` + `tiny` on CUDA (`config.gpu.yml`)
- Model type: `FasterWhisper`; implementation is the third-party package (registry entry `("faster-wisper", "native")`)

### VAD Providers

**Silero VAD:**
- Package: `silero-vad`; ONNX weights via `onnx-community/silero-vad`
- Model type: `SileroVAD`, `load_format: "onnx"`, always CPU
- Configurable params surfaced through `/v1/audio/vad`: `threshold`, `min_speech_duration_ms`, `min_silence_duration_ms`, `speech_pad_ms`, `return_seconds`

### Planned Provider Integration (NOT implemented)

**vllm-omni (GPU-only):**
- Design: `VllmOmniExecutor` wrapping `Omni`, which spawns its own per-stage subprocesses (`multiprocessing.spawn` or Ray) communicating over ZMQ/mp.Queue — bypasses `GPUWorker → GPUModelRunner → loader.get_model()`
- Planned model type: `VllmOmniTTS`; planned config entries for `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` and `Qwen/Qwen3-TTS-12Hz-1.7B-Base`
- Planned dependency placement: `requirements-cuda.txt` and `requirements-rocm.txt` only
- Reference docs: `.current_work/vllm-omni-integration-plan.md`, `.current_work/vllm-omni-analysis.md` (upstream analyzed at v0.16.0)
- Current repo state: no `harmonyspeech/executor/vllm_omni_executor.py`, no `VllmOmniTTS` in `harmonyspeech/modeling/models/__init__.py`, no `vllm-omni` requirement

## API Protocol

**Exposed API (outward-facing):**
- Protocol: OpenAI-style REST + JSON (audio returned base64-encoded)
- Framework: FastAPI + Uvicorn; Swagger at `/docs`, ReDoc at `/redoc`
- Port: `12080` (Docker entrypoint, Compose, README). CLI arg default is `2242` in `harmonyspeech/endpoints/openai/args.py`
- OpenAPI artifacts: `docs/api/openapi.json`, generators `docs/api/generate_openapi_go.sh` / `generate_openapi_js.sh`; schema tagged `x-go-client-name: HarmonySpeechAPI`, `x-go-package: harmonyspeech`
- Streaming: not active (streaming responses are commented out in `api_server.py`)

**Routes (from `harmonyspeech/endpoints/openai/api_server.py`):**
- `GET /health` — aggregate health of TTS/STT/VC/embedding/VAD engines
- `GET /version` — `harmonyspeech.__version__`
- `POST /v1/audio/speech` — TTS
- `GET /v1/audio/speech/models`
- `POST /v1/embed/speaker` — speaker embedding
- `GET /v1/embed/models`
- `POST /v1/audio/transcriptions` — STT
- `GET /v1/audio/transcriptions/models`
- `POST /v1/voice/convert` — voice conversion
- `GET /v1/voice/convert/models`
- `POST /v1/audio/vad` — voice activity detection
- `GET /v1/audio/vad/models`
- `POST /v1/audio/convert` — audio restoration/conversion (VoiceFixer)
- `GET /v1/audio/convert/models`
- `GET /metrics` — Prometheus ASGI app mounted on the router
- No `/v1/models`, `/v1/audio/synthesize`, `/v1/audio/vocode`, `/v1/audio/voice-convert`, or `/v1/audio/embed` routes exist (superseded by the per-modality `/models` and `/v1/embed/speaker` routes)

**Request/response models:** `harmonyspeech/endpoints/openai/protocol.py`
**Serving layer:** `harmonyspeech/endpoints/openai/serving_text_to_speech.py`, `serving_speech_to_text.py`, `serving_voice_conversion.py`, `serving_voice_embed.py`, `serving_voice_activity_detection.py`, `serving_audio_conversion.py`

**Frontend integration:**
- Operator UI lives in `frontend/` (React 19 + Vite 7 + Tailwind 4, zustand, i18next)
- Client SDK: `@harmony-ai/harmonyspeech` (`^0.1.1`, `HarmonySpeechEnginePlugin`) and `@harmony-ai-solutions/soulbits-api-client` (pinned GitHub ref)
- Defaults to engine endpoint `http://localhost:12080` (`frontend/src/HarmonySpeechEngineApp.jsx`)
- Mode selectable via `VITE_APP_MODE` (`harmony-link` vs `speech-engine`) in `frontend/src/config/appMode.js`; build with `npm run build:speech-engine`
- Management API client: `VITE_MGMT_API_URL` (default `http://localhost`) and `VITE_MGMT_API_PORT` (default `28081`) in `frontend/src/services/management/baseService.js`

## Monitoring & Observability

**Metrics:**
- `prometheus_client` ASGI app mounted at `/metrics` (`api_server.py:83`)
- Counters/histograms/gauges defined in `harmonyspeech/engine/metrics.py` and `harmonyspeech/common/metrics.py`: scheduler running/swapped/waiting, GPU/CPU cache usage, prompt/generation tokens, time-to-first-token, time-per-output-token, E2E latency, avg prompt/generation throughput
- Periodic stats logging: `HARMONYSPEECH_LOCAL_LOGGING_INTERVAL_SEC` (default 5)

**Logs:**
- `loguru` throughout; `harmonyspeech/common/logger.py`, level from `HARMONYSPEECH_LOG_LEVEL` (default `INFO`); Uvicorn log config `UVICORN_LOG_CONFIG`
- `auth/apikeys.py` uses the stdlib `logging` module

## CI/CD & Deployment

**Hosting:**
- Docker Hub (namespace from `secrets.DOCKER_HUB_USERNAME`)
- Engine images: `harmonyspeech-engine-cpu:latest`, `harmonyspeech-engine-nvidia:latest`, `harmonyspeech-engine-amd:latest`, `harmonyspeech-engine-amd-wsl:latest`
- UI image: `harmonyspeech-ui` built from `frontend/Dockerfile.speech-engine`

**CI Pipeline:**
- `.github/workflows/test.yml` — `test`, `lint` (ruff), and 10 model-scoped E2E jobs including `e2e-chatterbox`, `e2e-chatterbox-turbo`, `e2e-chatterbox-multilingual`
- `.github/workflows/docker-release-engine.yml` — 4 build jobs, one per Docker variant, with buildx caching
- `.github/workflows/docker-release-ui.yml` — single UI build/push job

**Docker build contexts:** `docker/cpu`, `docker/amd`, `docker/amd-wsl`, `docker/nvidia`; `CLONE_FROM_GITHUB=1` build arg switches from local source to a fresh GitHub clone for CI

## Environment Configuration

**Required / meaningful env vars:**
- `ENDPOINT_URL` — Harmony API-key service base URL (omit to disable key checks)
- `HARMONYSPEECH_API_KEY` — static shared key; also enables the auth middleware
- `HARMONYSPEECH_ADMIN_KEY` — admin key
- `SERVICE_NAME` — default `harmony-speech-engine`
- `HF_TOKEN` — optional Hugging Face token
- `HF_HOME` — model cache root (set in images)
- `NUMBA_CACHE_DIR` — numba cache (set in images)
- `HARMONYSPEECH_LOG_LEVEL`, `HARMONYSPEECH_LOCAL_LOGGING_INTERVAL_SEC`, `HARMONYSPEECH_ENGINE_ITERATION_TIMEOUT_S`
- `HARMONYSPEECH_TARGET_DEVICE` — build-time target (`cuda`/`rocm`/`cpu`) read by `setup.py`

**Secrets location:**
- `.env` (CPU/NVIDIA Compose) and `.env-amd` (AMD Compose) — present locally, not committed; `.env.example` is the committed template
- GitHub Actions secrets: `DOCKER_HUB_USERNAME`, Docker Hub token, HF token for E2E caching

## Webhooks & Callbacks

**Incoming:** None — request/response REST only; no webhook receiver routes.

**Outgoing:**
- Hugging Face Hub downloads on first model use
- `POST {ENDPOINT_URL}/check` and `POST {ENDPOINT_URL}/event` to the optional Harmony API-key service
- No other outbound callbacks; all AI inference runs in-process

---

*Integration audit: 2026-11-02*

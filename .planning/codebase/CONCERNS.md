<!-- refreshed: 2026-11-02 -->
# Codebase Concerns

**Analysis Date:** 2026-11-02

## Phase 0 resolutions (2026-11-04)

The following findings were fixed by the engine modernization Phase 0 plan
(`.current_work/engine-modernization/`); each entry names the fixing subtask.

| Finding | Status | Fix |
|---|---|---|
| API-supplied embedding deserialized with `weights_only=False` (critical RCE) | **RESOLVED** (0-1) | request embeddings load with `weights_only=True`; checkpoints hardened with an explicit trusted-only opt-in |
| Checkpoint loading with full pickle (`hf_downloader.py:159-163`) | **RESOLVED** (0-1) | `weights_only=True` + typed `ValueError` + escape hatch |
| Voice conditionals leak between requests | **RESOLVED** (0-2) | baseline snapshot/restore (`try/finally`) in all Chatterbox handlers; missing conditioning fails with 400 |
| Mixed Chatterbox batch misattributes results / drops requests | **RESOLVED** (0-3) | type-partitioned batches, explicit request passing, alignment asserts |
| Closure over the loop variable in `_execute_*` | **RESOLVED** (0-3) | every worker helper receives its request as a parameter |
| Requests can hang forever when a group member is missing | **RESOLVED** (0-4, 3) | spec-driven startup validation naming the missing step, immediate error on unresolvable routing, scheduler unknown-model safety net |
| Engine iteration timeout kills the server loop | **RESOLVED** (0-5) | per-request deadlines; watchdog demoted to recoverable safety net with consecutive-trip escalation |
| Health check iterates dict keys | **RESOLVED** (0-5) | `check_health` iterates `.values()` via `.items()` and returns a structured payload |

Still open after Phase 0: authentication/CORS/rate-limit hardening, request size
limits, real batching/streaming, VRAM residency management, dependency hygiene.

## Phase 5 resolutions (2026-11-04)

| Finding | Status | Fix |
|---|---|---|
| `max_batch_size` was decorative: schedulers admitted 10 requests and handlers looped one at a time | **RESOLVED** (5-2, 5-3) | padded-batch forward for the one-shot synthesizers (`infer_batched`, length-similar sub-batching, token budget); every value in `config.gpu.yml`/`config.yml` now matches executor reality (loops = 1, Chatterbox = 1) |
| Requests above a model's `max_batch_size` were silently dropped by the scheduler | **RESOLVED** (5-1) | `_schedule_waiting_requests` returns the leftover queue instead of the drained input copy; regression test in `tests/integration/test_batching_invariants.py` |
| Batch mispairing could only be caught in review | **RESOLVED** (5-1) | `assert_batch_aligned` + `EngineBatchError` (typed per-request errors) and a per-step request-id uniqueness guard in `_build_result`; engine-level invariant suite |
| "Real batching" listed as an open gap (`gpu_model_runner` loop) | **PARTIALLY RESOLVED** (5-2) | one-shot synthesis families batch for real; Chatterbox stays autoregressive by decision (needs a per-sequence KV cache) and scales by replicas, documented in `docs/batching.md` |

Remaining batching work (explicitly out of scope for Phase 5): Chatterbox AR
batching with a KV cache; batched tone-converter/encoder hops (they are per-item
loops and cap end-to-end clone throughput at ~1.9 req/s).

## Overview

The engine hosts every model in a YAML config as a permanently resident executor, executes batches request-by-request in Python loops, and transports all audio as base64 inside JSON. Three properties dominate the risk profile:

1. **VRAM residency.** `config.gpu.yml` declares 28 models, 20 of them on `cuda`, and every CUDA model is loaded onto the same device at startup.
2. **No real batching or streaming.** `gpu_model_runner.py` contains no batching logic, and `_execute_*` methods loop per request. `output_options.stream=true` is rejected.
3. **Transport and dependency coupling.** Full WAV payloads cross the API as base64 JSON, and a module-level `import chatterbox` makes the optional `--no-deps` Chatterbox install mandatory for the server to boot at all.

## Highest-Risk Items

| Severity | Concern | Primary location |
|----------|---------|------------------|
| Critical | `torch.load(..., weights_only=False)` on API-supplied embeddings (remote code execution) — **resolved by 0-1** | `harmonyspeech/task_handler/model_runner_base.py:332` |
| Critical | All CUDA models resident simultaneously on one GPU; no unload/eviction/memory cap | `harmonyspeech/engine/harmonyspeech_engine.py:69`, `config.gpu.yml` |
| High | Mixed Chatterbox embed + TTS batch causes result misattribution and silently dropped requests — **resolved by 0-3** | `harmonyspeech/task_handler/inputs.py:174-219`, `harmonyspeech/task_handler/model_runner_base.py:96-115` |
| High | Voice conditionals persist on the shared model instance between requests — **resolved by 0-2** | `harmonyspeech/task_handler/model_runner_base.py:683-700` |
| High | No TTS streaming; full synthesis must finish before any byte is returned | `harmonyspeech/endpoints/openai/serving_text_to_speech.py:60` |
| High | Engine iteration hard timeout (120 s default) kills the whole background loop for slow CPU generations — **resolved by 0-5** | `harmonyspeech/engine/async_harmonyspeech.py:15,378-383` |
| High | Chatterbox installed with `pip install --no-deps`; metadata pins bypassed permanently | `requirements-chatterbox.txt` |
| Medium | Auth disabled unless configured; fail-open on empty `ENDPOINT_URL` | `harmonyspeech/endpoints/openai/api_server.py:304`, `auth/apikeys.py:107` |

## Tech Debt

### GPU Model Runner Is a Stub

- **Issue:** `harmonyspeech/task_handler/gpu_model_runner.py` is 334 lines, of which only 36 are neither comments nor blank lines; the commented block is the Aphrodite/vLLM CUDA-graph implementation. The only live code is `GPUModelRunner.load_model()` (memory profiling wrapper) plus unused `BatchType`, `_BATCH_SIZES_TO_CAPTURE`, and `_get_graph_batch_size`.
- **Files:** `harmonyspeech/task_handler/gpu_model_runner.py:1-40`, `harmonyspeech/task_handler/gpu_model_runner.py:42-321`
- **Impact:** Readers assume CUDA graphs, KV-cache profiling, and batch scheduling exist on the GPU path. None do. All GPU inference goes through the same generic Python loop as CPU.
- **Fix approach:** Either delete the dead block and rename the class to reflect its actual role, or implement the intended `profile_run`/`capture_model` path. Do not add features on top of the commented code.

### Ray Executor Is Entirely Dead Code

- **Issue:** `harmonyspeech/executor/ray_gpu_executor.py` is 398 lines with zero uncommented statements and is not imported anywhere.
- **Files:** `harmonyspeech/executor/ray_gpu_executor.py:1-398`
- **Impact:** Multi-GPU / distributed inference appears supported but is unreachable. `GPUExecutor` always creates exactly one `GPUWorker` on `cuda:0`.
- **Fix approach:** Remove the file or move it to a design note; add a real executor only when horizontal scaling is funded.

### Carried-Over Aphrodite/VLLM Semantics

- **Issue:** Docstrings, error messages, and comments still describe token sampling, KV blocks, LoRA, and `gpu_memory_utilization`, none of which exist.
- **Files:** `harmonyspeech/engine/harmonyspeech_engine.py:600-650`, `harmonyspeech/task_handler/gpu_worker.py:107-122`, `harmonyspeech/executor/executor_base.py:1-10`, `harmonyspeech/task_handler/cpu_worker.py:16-21`
- **Impact:** Misleads operators tuning "KV cache" or "GPU memory utilization"; `raise_if_cache_size_invalid` is never called.
- **Fix approach:** Rewrite docstrings to describe the actual request/executor model; delete cache-config helpers and metrics that reference nonexistent caches.

### Prometheus Metrics Are Inherited and Misnamed

- **Issue:** Metric names remain `aphrodite:*` (`aphrodite:num_requests_running`, `aphrodite:gpu_cache_usage_perc`, token counters) and include metrics that are never populated. `StatLogger` is instantiated but `step()` has stats logging commented out.
- **Files:** `harmonyspeech/engine/metrics.py:18-79`, `harmonyspeech/engine/harmonyspeech_engine.py:669-672`
- **Impact:** Dashboards show zeros; `/metrics` implies GPU KV-cache telemetry that does not exist for a TTS engine. Token metrics are meaningless for audio.
- **Fix approach:** Replace with speech-specific metrics (requests by endpoint/model, queue depth, synthesis seconds per audio second, GPU allocated bytes per executor) and rename the namespace.

### Hardcoded String Routing — RESOLVED (Phase 3)

- **Was:** Multi-step pipelines were routed by `model_type` string comparisons and sentinel model IDs in long `if/elif` chains; serving modules kept their own model-type/group lists. Adding a model required edits in five places that could drift.
- **Now:** `harmonyspeech/processing/pipeline_registry.py` declares every toolchain as a `PipelineSpec` (steps with `requires`/`needs`/`produces`, entry rules, capabilities). Routing (`next_step`), forwarding (`advance`), startup validation (`validate_registry`) and `/v1/models` cards all read that one declaration; the per-serving-module lists are deleted. The parity of every old routing decision is locked by `tests/unit/inference_flow/test_pipeline_routing.py`.
- **Remaining:** input preparation and execution dispatch still key on `model_type` (`task_handler/inputs.py`, `task_handler/model_runner_base.py`) — a second place that must learn a new `model_type`. Unifying those behind the registry is future work, not a routing concern.

### MeloTTS Input Preparation

- **Issue:** In-code FIXME states the normalization/Bert path is "super complicated, imperformant and underoptimized", including per-call model allocation.
- **Files:** `harmonyspeech/modeling/models/melo/inputs.py:29-31`, `harmonyspeech/modeling/models/melo/melo.py`
- **Impact:** Extra latency and memory per MeloTTS/OpenVoice V2 request.
- **Fix approach:** Cache the Bert/tokenizer per process; rewrite normalization to avoid per-part allocations.

### Generic Error Handling

- **Issue:** Broad `except Exception` blocks convert every failure into an error `RequestOutput`; domain exceptions do not exist. Serving layers still carry "TODO: Use an aphrodite-specific Validation Error".
- **Files:** `harmonyspeech/task_handler/model_runner_base.py:121-160` (plus every `_execute_*`), `harmonyspeech/endpoints/openai/serving_*.py`
- **Impact:** Callers cannot distinguish bad input (400) from model failure (500) from OOM; logs carry stack traces but API responses collapse to a string.
- **Fix approach:** Add `HarmonySpeechError` subclasses (`InvalidInputError`, `ModelLoadError`, `InferenceError`, `ResourceExhaustedError`) and map them to HTTP status codes in serving.

### Module-Level Import Coupling

- **Issue:** `harmonyspeech/task_handler/model_runner_base.py:31` imports `task_handler.inputs`, which at line 11 imports `chatterbox.tts.Conditionals` at module scope. `harmonyspeech/modeling/loader.py:22-27` also imports the Chatterbox wrappers at module scope, and those wrappers import `chatterbox`, `perth`, and `huggingface_hub` eagerly.
- **Files:** `harmonyspeech/task_handler/inputs.py:11`, `harmonyspeech/modeling/loader.py:22-27`, `harmonyspeech/modeling/models/chatterbox/chatterbox.py:14-17`
- **Impact:** The server cannot start without `chatterbox-tts`, even if no Chatterbox model is configured. `setup.py` does not declare it (`install_requires` reads only the platform requirements file, `setup.py:420-438,494`), so `pip install harmony-speech-engine` yields a package that fails on first import of the engine.
- **Fix approach:** Move Chatterbox imports into the `Chatterbox*` loader branches and `prepare_chatterbox_*` functions; declare `chatterbox-tts` as an optional extra with a clear install error.

### Unused Imports and Dead Request Fields

- **Issue:** `harmonyspeech/task_handler/inputs.py:4-6` still imports `NamedTemporaryFile` and `os` after the tempfile-free rewrite. `TextToSpeechAudioOutputOptions` (`harmonyspeech/common/inputs.py:38-41`) is a plain class with no `__init__` and is only used as a type annotation.
- **Files:** `harmonyspeech/task_handler/inputs.py:1-12`, `harmonyspeech/common/inputs.py:20-41`
- **Impact:** Noise; the output-options type cannot be instantiated safely if future code tries.
- **Fix approach:** Delete unused imports; make the options type a dataclass or reuse the Pydantic model.

## Known Bugs

### API-Supplied Embedding Is Deserialized With `weights_only=False` — RESOLVED (Phase 0-1)

- **Symptoms:** Any caller of `/v1/voice/convert` or a voice-cloning TTS request who supplies `input_embedding` / `target_embedding` controls bytes passed to `torch.load` with pickle execution enabled. A crafted payload executes arbitrary code in the server process.
- **Files:** `harmonyspeech/task_handler/model_runner_base.py:332-333`, decode path `harmonyspeech/task_handler/inputs.py:484-508`, request fields `harmonyspeech/endpoints/openai/protocol.py:167-170,125-131`
- **Trigger:** `POST /v1/voice/convert` with `target_embedding` set to a base64 pickle payload; or `POST /v1/audio/speech` with `mode=voice_cloning` and `input_embedding`.
- **Workaround:** Disable the embedding-input paths or require trusted clients only.
- **Fix:** Load embeddings with `weights_only=True` (they are plain tensors/state dicts), or replace pickle with `safetensors`/`numpy` serialization. Apply the same treatment to model checkpoints in `harmonyspeech/modeling/hf_downloader.py:159-163`.

### Mixed Chatterbox Batch Misattributes Results and Drops Requests — RESOLVED (Phase 0-3)

- **Symptoms:** One executor serves both `/v1/embed/speaker` (`SpeechEmbeddingRequestInput`) and `/v1/audio/speech` (`TextToSpeechRequestInput`) when no dedicated `ChatterboxEmbedding` config exists. `prepare_inputs` collects both, then returns only the embedding-prepared list (`inputs.py:187-188`). `execute_model` dispatches on `requests_to_batch[0]` and calls `_execute_chatterbox_embedding`, whose loop zips the shorter `inputs` list to `requests_to_batch[i]`. TTS requests are never executed; their streams hang. If order differs, embedding/audio results are attached to the wrong request IDs.
- **Files:** `harmonyspeech/task_handler/inputs.py:174-219`, `harmonyspeech/task_handler/model_runner_base.py:96-115`, `harmonyspeech/task_handler/model_runner_base.py:840-849`, `harmonyspeech/processing/scheduler.py:201-230`
- **Trigger:** Default `config.yml` leaves `chatterbox_embedding` commented out; send one `/v1/embed/speaker` and one `/v1/audio/speech` to model `chatterbox` while both are queued in the same scheduling step.
- **Fix:** Split mixed batches at the scheduler or executor into two calls; assert `len(inputs) == len(requests_to_batch)` and key results by request ID rather than index.

### Voice Conditionals Leak Between Requests — RESOLVED (Phase 0-2)

- **Symptoms:** `_execute_chatterbox_tts` / `_execute_chatterbox_turbo_tts` / `_execute_chatterbox_multilingual_tts` assign `self.model.conds` only when a request provides `input_embedding` or `input_audio`. A later `single_speaker_tts` request without a reference reuses the previous request's conditionals, so audio is spoken in the previous caller's voice. `_execute_chatterbox_vc` behaves the same way with `self.model.ref_dict`.
- **Files:** `harmonyspeech/task_handler/model_runner_base.py:683-700`, `harmonyspeech/task_handler/model_runner_base.py:726-742`, `harmonyspeech/task_handler/model_runner_base.py:770-786`, `harmonyspeech/task_handler/model_runner_base.py:857-863`
- **Trigger:** Two sequential requests to the same Chatterbox model: first with `input_audio`, second with `mode=single_speaker_tts` and no reference.
- **Fix:** Reset `self.model.conds`/`ref_dict` at the start of every execution, or hold conditionals per request and pass them explicitly instead of mutating the model.

### Requests Can Hang Forever When a Group Member Is Missing — RESOLVED (Phase 0-4)

- **Symptoms:** If routing found no matching `ModelConfig`, `request.model` stayed at the sentinel and the request was requeued forever.
- **Fix (0-4 + Phase 3):** `validate_registry` reports incomplete pipelines at startup, `next_step` returns `None` for an unresolvable step, and `add_request` rejects the request immediately with an error naming the pipeline and step. The scheduler keeps its unknown-model safety net.

### Rate-Limit Arithmetic Uses Negative Deltas

- **Symptoms:** `_register_cache_stats_event` stores `cache_timestamp - current_timestamp` (negative) into `request_timestamps_delta`; the check then computes `check_window = time_since_cached + 60` from another negative value. The per-minute window therefore never reflects real elapsed time, so rate limiting is either always-pass or always-block depending on server-provided deltas.
- **Files:** `auth/apikeys.py:57-63`, `auth/apikeys.py:116-133`
- **Fix:** Use `current_timestamp - cache_timestamp`, prune events older than 60 s, and unit-test the window logic.

### `len(ENDPOINT_URL)` Crashes When the Variable Is Unset

- **Symptoms:** `check_request_allowed_by_rate_limit` and `register_rate_limiting_event` call `len(ENDPOINT_URL)`. When `ENDPOINT_URL` is not defined, `ENDPOINT_URL` is `None` and the middleware raises `TypeError`, producing a 500 for any request carrying a non-default `Api-Key` header.
- **Files:** `auth/apikeys.py:107`, `auth/apikeys.py:135`
- **Fix:** `if not ENDPOINT_URL: ...` and decide explicitly between fail-open and fail-closed.

### Health Check Iterates Dict Keys — RESOLVED (Phase 0-5)

- **Symptoms:** `HarmonySpeechEngine.check_health` does `for model_executor in self.model_executors:` and then calls `.check_health()` on the string key.
- **Files:** `harmonyspeech/engine/harmonyspeech_engine.py:697-699`
- **Impact:** Latent `AttributeError`; currently masked because `AsyncHarmonySpeech` calls `check_health_async`, which does iterate `.values()` (`harmonyspeech/engine/async_harmonyspeech.py:214-216`).
- **Fix:** Iterate `.values()` in both places.

### Language Codes Are Inconsistent Between CPU and GPU Configs

- **Symptoms:** `config.yml` declares MeloTTS Japanese as `language: "JP"` and Korean as `"KR"`; `config.gpu.yml` declares Japanese as `"JA"` (with `voices: ["JP"]`). Routing matches `cfg.language == request.language_id` exactly (`harmonyspeech_engine.py:196-200`), and model cards expose whatever the config says. Clients therefore need different language codes for the same model depending on deployment.
- **Files:** `config.yml:102-120`, `config.gpu.yml:102-112`, `harmonyspeech/engine/harmonyspeech_engine.py:196-200`
- **Fix:** Adopt one code set (BCP-47: `ja`, `ko`, `zh`, …) in configs, routing, and docs; add a config-load validation that language codes come from a known enum.

## Security Considerations

### Unauthenticated Deserialization of Request Data (Critical) — RESOLVED (Phase 0-1)

- **Risk:** Remote code execution via `torch.load(weights_only=False)` on base64 input from `/v1/voice/convert` and voice-cloning TTS. Also applies to model checkpoint loading (`harmonyspeech/modeling/hf_downloader.py:163`), where a malicious HF repo or local path executes code.
- **Files:** `harmonyspeech/task_handler/model_runner_base.py:332-333`, `harmonyspeech/modeling/hf_downloader.py:159-163`
- **Current mitigation:** None. `trust_remote_code` is exposed in `ModelConfig` but unused (`harmonyspeech/common/config.py:54,71`), so it provides no gate.
- **Recommendations:** Switch embeddings to `safetensors`/`torch.load(weights_only=True)`; reject unknown pickle globals; add an allowlist of model repositories; document that arbitrary model IDs are trusted input.

### Authentication Is Opt-In and Has a Fail-Open Branch

- **Risk:** Middleware is only installed when `HARMONYSPEECH_API_KEY` or `--api-keys` is set (`api_server.py:304`). With no key configured, every endpoint is open. When a key is set and Harmony auth is used, `ENDPOINT_URL` empty means `check_request_allowed_by_rate_limit` returns `(True, "apikey check disabled")` (`auth/apikeys.py:107-108`) — any `Api-Key` value is accepted.
- **Files:** `harmonyspeech/endpoints/openai/api_server.py:304-348`, `auth/apikeys.py:105-133`
- **Current mitigation:** Depends on `.env` being configured by the operator (`docker-compose*.yml` load `./.env`).
- **Recommendations:** Fail closed by default when binding non-loopback; emit a startup warning/health flag when auth is disabled; treat an unset `ENDPOINT_URL` as an error, not a bypass.

### Permissive CORS Defaults

- **Risk:** `--allowed-origins`, `--allowed-methods`, and `--allowed-headers` default to `["*"]`.
- **Files:** `harmonyspeech/endpoints/openai/args.py:12-15`, `harmonyspeech/endpoints/openai/api_server.py:291-297`
- **Recommendations:** Default to same-origin/empty list, and require opt-in for wildcard.

### No Request Size or Text Length Limits

- **Risk:** `input`, `input_audio`, `source_audio`, and `target_audio` are unbounded strings decoded into memory. Base64 decode + `librosa.load` + tensor conversion multiply memory. A single large request can exhaust RAM/VRAM; sustained requests are a trivial DoS.
- **Files:** `harmonyspeech/endpoints/openai/protocol.py:146-182`, `harmonyspeech/task_handler/inputs.py:484-523,827-876`
- **Recommendations:** Enforce maximum body size at the ASGI layer, maximum decoded audio duration, and maximum text length per model; return 413/422.

### Unauthenticated Metrics and Error Detail

- **Risk:** `/metrics` is mounted outside `/v1` and therefore bypasses the auth middleware (`api_server.py:82-84,315-316`). Validation errors return `str(exc)` to clients (`api_server.py:299-302`).
- **Files:** `harmonyspeech/endpoints/openai/api_server.py:82-84`, `harmonyspeech/endpoints/openai/api_server.py:299-302`
- **Recommendations:** Gate `/metrics` behind the same auth or a private listener; return generic validation messages and log details server-side.

### Watermarking Silently Degrades

- **Risk:** If `perth.PerthImplicitWatermarker` is unavailable, the module replaces it with a wrapper around `DummyWatermarker` (`harmonyspeech/modeling/models/chatterbox/chatterbox.py:55-76`), so `watermark: true` produces unwatermarked audio with no warning. `watermark: false` also swaps in `DummyWatermarker` in the loader (`loader.py:307-326`).
- **Files:** `harmonyspeech/modeling/models/chatterbox/chatterbox.py:49-76`, `harmonyspeech/modeling/loader.py:304-326`
- **Recommendations:** Log a warning when the fallback is used; expose effective watermark status in `/health`.

### API Keys Appear in Logs

- **Risk:** `logging.log(logging.WARNING, f"Unable to add event for API Key '{api_key}'")` writes raw keys to logs.
- **Files:** `auth/apikeys.py:54-55`
- **Recommendations:** Log a hash/prefix only.

## Performance Bottlenecks

### All Models Are Residency-Pinned to One GPU

- **Problem:** `init_custom_executors()` creates one executor per `ModelConfig`, and each GPU executor loads its model during construction (`gpu_executor.py:7-27`). Nothing unloads, evicts, or lazily loads models; there is no `gpu_memory_utilization`, memory budget, or OOM preflight.
- **Files:** `harmonyspeech/engine/harmonyspeech_engine.py:69-87`, `harmonyspeech/executor/gpu_executor.py:7-31`, `harmonyspeech/task_handler/gpu_worker.py:46-75`, `config.gpu.yml`
- **Current capacity:** `config.gpu.yml` declares 28 models, of which 20 target `cuda`: 2 OpenVoice V1 synthesizers, 2 OpenVoice V1 converter roles, 5 MeloTTS languages, 2 OpenVoice V2 converter roles, Faster-Whisper `large-v3-turbo` and `tiny`, VoiceFixer restorer + vocoder, 4 Chatterbox variants, and `chatterbox_embedding`.
- **Limit (measured after Phase 4, 2026-11-04):** the full catalogue now fits: 8.52 GiB of torch allocations (8.83 GiB via `nvidia-smi`) on a 15.93 GiB card, ~7 GiB free. Before Phase 4 the same catalogue allocated 16.45 GiB — i.e. it overflowed a 16 GB card and only "worked" by paging into shared system memory under WDDM. See `docs/benchmarks/vram-profiles.md`. The floor of ≥8 GiB headroom is still missed by ~0.7 GiB because OpenVoice/MeloTTS and VoiceFixer remain float32 (upstream fp16 incompatibilities) and Chatterbox Turbo carries ~1.5 GiB of GPT2 causal-mask buffers.
- **Improved by:** Phase 4 — fp16 T3 for the Chatterbox family, plus the ref-counted `WeightRegistry` with `backbone_group: "chatterbox-base"` (one S3Gen/VE for `chatterbox`, `chatterbox_multilingual`, `chatterbox_vc`, `chatterbox_embedding`; the Chatterbox block dropped 14.29 GiB → 6.35 GiB). The remaining gap is upstream-model work, not engine wiring.
- **Still open:** no unload/evict path, no OOM preflight. `/health` and `/metrics` now expose per-model and shared-group VRAM (`hse_model_vram_bytes`, `hse_shared_backbone_bytes`, …), which is the input a future governor needs.
- **Improvement path:** Pick a small "resident set" per deployment; move the rest to lazy load with LRU eviction (the registry's `release()` is already wired for it); finish the fp16 opt-ins once upstream VITS/VoiceFixer code is half-precision safe.

### Per-Request Batching in Name Only

- **Problem:** The scheduler builds per-model batches up to `max_batch_size` (`scheduler.py:191-230`), but every `_execute_*` method loops `for i, x in enumerate(inputs)` and runs the model once per item. `gpu_model_runner.py` has no batching code at all, and `FasterWhisper` is the only path that uses a batched inference pipeline (`BatchedInferencePipeline.transcribe(..., batch_size=16)`, `model_runner_base.py:371-394`).
- **Files:** `harmonyspeech/task_handler/model_runner_base.py:162-884` (8 `# FIXME: This is not properly batched` markers at lines 163, 186, 234, 279, 327, 397, 456, and the KittenTTS/Chatterbox loops at 666-675, 709-718, 751-760, 795-804), `harmonyspeech/task_handler/gpu_model_runner.py:27-40`
- **Impact:** GPU utilization stays low, `max_batch_size` inflates queueing latency without throughput gain, and concurrent requests for one model serialize completely. A batch of 10 TTS requests takes ~10× the single-request time.
- **Improvement path:** Add true tensor batching for MeloTTS/OpenVoice/Chatterbox where the model supports padded batches; otherwise set `max_batch_size: 1` so the scheduler stops pretending, and expose per-model concurrency instead. Remove the `FIXME` markers only when the corresponding loop is gone.

### Scheduler Serializes Models With In-Flight Work

- **Problem:** `_schedule_waiting_requests` refuses to admit new requests for a model while it has running requests (`scheduler.py:172-175`). Because `free_finished_requests()` runs only after a full step, a multi-step pipeline holds the model slot across steps and blocks unrelated requests for that model.
- **Files:** `harmonyspeech/processing/scheduler.py:134-230`
- **Impact:** Head-of-line blocking across multi-step pipelines (VAD → embed → synth → tone convert); a slow Chatterbox request stalls all Chatterbox traffic.
- **Improvement path:** Track per-model in-flight capacity instead of a boolean, and release slots when a pipeline step finishes rather than at step end.

### Base64 JSON Transport Overhead

- **Problem:** Every audio field is a base64 string inside JSON: inputs (`input_audio`, `source_audio`, `target_audio`, `input_mel_spectrogram`, `input_embedding`) and outputs (`data`). Each direction costs ~33% size inflation plus JSON escaping, a full string copy in Pydantic, a decode to `bytes`, a `BytesIO` wrapper, and often a `librosa.load` into float32. WAV is written to `BytesIO` and re-encoded to base64 for every response.
- **Status (Phase 2, 2026-11-04 — internal hops retired):** the base64/pickle/JSON-float **intermediates between pipeline steps are gone**. Requests decode once at admission into typed artifacts (`harmonyspeech/common/artifacts.py`) and steps exchange `AudioArtifact` / `MelArtifact` / `SpeakerArtifact` / `VadArtifact` objects; responses encode once at the serving boundary (`endpoints/openai/wire.py`). A pytest invariant forbids wire encoding outside `endpoints/`, `common/audio_codec.py`, `common/inputs.py` and `common/outputs.py`. Measured on CPU: the VoiceFixer mel hop drops from ~25 ms / ~1 MiB allocations to ~0 ms, and an OpenVoice V2 clone now decodes its reference audio once instead of three times.
- **Remaining:** the **external** wire is still base64-in-JSON on both directions (kept for compatibility with Soulbits' `HSEAdapter` and existing clients), and the multipart envelope still takes its documented in-process base64 round-trip before admission decoding. Binary request/response bodies and streaming are tracked separately.
- **Files:** `harmonyspeech/endpoints/openai/protocol.py`, `harmonyspeech/endpoints/openai/request_envelope.py`, `harmonyspeech/endpoints/openai/wire.py`, `harmonyspeech/common/artifacts.py`, `harmonyspeech/common/audio_codec.py`
- **Impact (remaining):** higher latency and memory per request at the HTTP boundary, no ability to stream or range-request, and no zero-copy path between HTTP and model. Long-form audio (minutes) produces multi-megabyte JSON responses.
- **Improvement path:** Return `audio/wav` (or negotiated format) with `Content-Disposition` and accept binary bodies without the compatibility round-trip; streaming output is deferred to its own workstream.

### GIL and Threading Model

- **Problem:** The synchronous `step()` fans out models on a `ThreadPoolExecutor` (`harmonyspeech_engine.py:655-663`) whose own comment asks whether the GIL defeats it. `AsyncHarmonySpeech.step_async()` instead awaits executors sequentially (`async_harmonyspeech.py:188-193`), so the async server does not overlap models at all. `prepare_*` functions spin up a fresh `ThreadPoolExecutor` per batch, and `make_async` runs each executor call on the default loop executor.
- **Files:** `harmonyspeech/engine/harmonyspeech_engine.py:655-663`, `harmonyspeech/engine/async_harmonyspeech.py:176-203`, `harmonyspeech/common/utils.py` (`make_async`), `harmonyspeech/task_handler/inputs.py:455-456`
- **Impact:** CPU-heavy pre/post-processing (librosa, phonemizers) blocks other models; the async path gains nothing from multiple GPUs/models.
- **Improvement path:** Use one worker process or thread per device with an explicit device queue; use a process pool for CPU preprocessing; reuse a single executor rather than creating one per batch.

### Startup Time and Cold Start

- **Problem:** Server startup loads every configured model synchronously before binding traffic. `ExecutorBase.__init__` calls `_init_executor()` with no error isolation (`executor_base.py:17-24`), so one missing checkpoint aborts the entire server.
- **Files:** `harmonyspeech/engine/harmonyspeech_engine.py:69-87`, `harmonyspeech/executor/executor_base.py:17-24`, `harmonyspeech/endpoints/openai/api_server.py:69-85`
- **Impact:** 28 models means long startup, large HF downloads on first run, and no partial availability: a typo in one model ID takes down every endpoint.
- **Improvement path:** Lazy-load on first routed request; isolate failures per model with a degraded model card; add a readiness endpoint that reports per-model state.

## Fragile Areas

### Chatterbox Wrapper Monkey Patches (Import-Time Side Effects)

- **Files:** `harmonyspeech/modeling/models/chatterbox/chatterbox.py:28-46` (patches `chatterbox.mtl_tts.torch.load` process-wide), `harmonyspeech/modeling/models/chatterbox/chatterbox.py:55-76` (replaces `perth.PerthImplicitWatermarker`), `harmonyspeech/modeling/models/chatterbox/chatterbox.py:116-189` (per-instance float32 patches on `s3gen.tokenizer.log_mel_spectrogram` and `ve.forward`)
- **Why fragile:** The `torch.load` patch is applied to the upstream module globally, with `map_location` defaulted based on argument inspection (`isinstance(args[0], (str, object))` is always true). Any other library using `chatterbox.mtl_tts.torch` gets the patch. The watermark fallback silently changes output provenance.
- **Safe modification:** Keep patches localized to the load path; prefer passing `map_location` through `from_local`/`from_pretrained` arguments; add a regression test that imports the module twice and confirms the patch is idempotent.
- **Test coverage:** `tests/unit/initialization/test_chatterbox_imports.py` covers imports; patch behavior itself is not asserted.

### Chatterbox `model:` Field Is Ignored

- **Files:** `harmonyspeech/modeling/loader.py:304-337`
- **Why fragile:** `ChatterboxTTSModel.from_pretrained(device=device)` and friends never receive `model_config.model`. The Turbo wrapper hardcodes `ResembleAI/chatterbox-turbo` (`harmonyspeech/modeling/models/chatterbox/chatterbox.py:20,145-149`), and `ChatterboxEmbedding` picks a wrapper by substring-matching the configured repo id. Pointing a config at a local path, a fine-tune, or a mirror silently loads the upstream default instead.
- **Safe modification:** Thread `model_config.model` through the wrappers; for embedding, choose the wrapper from an explicit config field rather than a repo-name heuristic.

### Closure over the Loop Variable in `_execute_*` — RESOLVED (Phase 0-3)

- **Files:** `harmonyspeech/task_handler/model_runner_base.py:680-707`, `723-749`, `765-793`, `854-873`
- **Why fragile:** Inner `synthesize`/`convert` closures read `initial_request`, which is assigned in the enclosing loop after the function definition. It works only because the call happens in the same iteration and batches are executed sequentially. Any reordering, retry, or parallelization of the loop silently pairs audio with the wrong request.
- **Safe modification:** Pass `initial_request` as an explicit argument; add an assertion that `len(inputs) == len(requests_to_batch)`.

### Shared Mutable Model State — PARTIALLY RESOLVED (Phase 0-2)

- **Files:** `harmonyspeech/task_handler/model_runner_base.py:685,728,771,831,859`
- **Why fragile:** `self.model.conds` and `self.model.ref_dict` are mutated per request on a process-wide model instance (the watermarker is likewise swapped at load time). Sequential execution prevents races today, but any parallel batch execution (the obvious next performance step) would corrupt voices across requests.
- **Status (Phase 0-2):** every Chatterbox handler now restores the post-load baseline in a `finally` block, and a request without conditioning fails with a 400 instead of inheriting state. The model instance is still mutated *during* a request, so concurrent batch execution (Phase 5) must keep one wrapper instance per in-flight request or stop mutating the model entirely.
- **Safe modification:** Treat the loaded model as read-only after load; carry per-request conditionals in the prepared input and pass them to `generate`.

### Engine Iteration Timeout Kills the Server Loop — RESOLVED (Phase 0-5)

- **Files:** `harmonyspeech/engine/async_harmonyspeech.py:15,375-383`
- **Why fragile:** `asyncio.wait_for(engine_step(), HARMONYSPEECH_ENGINE_ITERATION_TIMEOUT_S)` defaults to 120 s. A single slow request (Chatterbox on CPU, long text, or a cold model) exceeds it; the handler logs "This should never happen!", calls `set_errored`, and re-raises, ending `run_engine_loop`. Subsequent requests receive `AsyncEngineDeadError` until restart.
- **Safe modification:** Apply timeouts per request, not per engine step; on timeout, abort only the offending request and keep the loop alive.

### Multi-Step Forwarding Mutates Request Objects

- **Files:** `harmonyspeech/engine/harmonyspeech_engine.py:363-506`
- **Why fragile:** `check_forward_processing` mutates `input_data` in place (`input_audio = None`, `input_embedding = ...`, `input_vad_data = ...`) and re-adds the same object to the scheduler. Retries, aborts, or duplicate results can corrupt a request that is already queued; status transitions are spread across `if` blocks with `new_status` defaulting to `FINISHED_STOPPED`.
- **Safe modification:** Create an explicit immutable step descriptor per pipeline stage; centralize status transitions in one table; add tests for abort-during-forward and duplicate-output cases.
- **Test coverage:** Routing/forwarding unit tests exist (`tests/unit/inference_flow/test_chatterbox_routing.py`), but abort/duplicate races are untested.

### In-Memory Caches With No Bounds

- **Files:** `auth/apikeys.py:31` (`_api_key_cache` dict, never pruned), `harmonyspeech/modeling/hf_downloader.py:20-21` (`_config_cache`, `_file_cache` grow per repo/revision), `harmonyspeech/common/utils.py:42-105` (`LRUCache` exists but is unused for these)
- **Why fragile:** A long-running server accumulates API-key entries and file blobs indefinitely; a hostile client can rotate keys to grow the cache.
- **Safe modification:** Bound the caches (LRU) and expire rate-limit entries.

### Windows/Path and Cache Oddities

- **Files:** workspace root contains an untracked `undefined/` directory holding a HuggingFace cache (`undefined/.cache/huggingface/...`) and a `nul` entry
- **Why fragile:** Indicates an unset environment path was joined into `"undefined"` during a local run; `nul` is a Windows device-name artifact. Neither is tracked, but both confuse tooling and searches.
- **Safe modification:** Resolve the offending `HF_HOME`/`XDG_CACHE_HOME` default, delete the artifacts, and add `undefined/` to `.gitignore` if it can recur.

## Scaling Limits

### Single-Process, Single-GPU Execution

- **Current capacity:** One process, one `cuda:0`, one worker per model, synchronous per-model execution.
- **Limit:** Vertical only. `ray_gpu_executor.py` is dead, there is no tensor/pipeline parallelism, no multi-GPU placement, and no model sharding. Adding a second GPU has no effect.
- **Scaling path:** Implement model-level placement (one process per GPU, each loading a subset of configs) behind a router; revive a real distributed executor only if a single-GPU split is insufficient.

### No Horizontal State Sharing

- **Current capacity:** Scheduler queues, running set, and `RequestTracker` streams are in-memory (`processing/scheduler.py:74-82`, `engine/async_harmonyspeech.py:76-116`).
- **Limit:** Requests are pinned to the instance that accepted them; no shared queue, no cross-instance abort, no distributed rate limiting.
- **Scaling path:** External queue (Redis/NATS) plus a stateless API tier; move the API-key cache to a shared store.

### VRAM Saturation Prevents Concurrency

- **Current capacity (measured after Phase 4):** full `config.gpu.yml` catalogue = 8.52 GiB torch allocations (8.83 GiB via `nvidia-smi`) on a 15.93 GiB card, ~7 GiB free; before Phase 4 it was 16.45 GiB and overflowed the card.
- **Limit:** Headroom is now sufficient for activations and a second concurrent request on the smallest models, but not for two concurrent large generations (Chatterbox Turbo alone is ~3.4 GiB and its activation peak is several GiB). OOM still appears at inference time, not load time.
- **Scaling path:** Resident-set configs, lazy load/evict (registry `release()` is ready), finishing the fp16 opt-ins for OpenVoice/MeloTTS/VoiceFixer, and the per-model/shared-group gauges now reported by `/health` and `/metrics`.

## Dependencies at Risk

### Chatterbox TTS Is Installed With `--no-deps` (Permanent Metadata Bypass)

- **Package:** `chatterbox-tts==0.1.6` (`requirements-chatterbox.txt`)
- **Risk:** Upstream declares `torch==2.6.0`, `torchaudio==2.6.0`, `transformers==4.46.3`, `numpy<1.26.0`, `safetensors==0.5.3`, `diffusers==0.29.0`. A normal install would downgrade the engine's stack. The workaround (`pip install --no-deps -r requirements-chatterbox.txt`) means pip no longer resolves or verifies any Chatterbox dependency: a new release that adds a runtime import fails at import time, and a release that changes behavior is invisible to the resolver.
- **Files:** `requirements-chatterbox.txt:1-21`, `requirements-common.txt:42-62`, `docker/nvidia/Dockerfile:41-42`, `docker/cpu/Dockerfile:41-42`, `docker/amd/Dockerfile:44-45`, `.github/workflows/test.yml` (every job installs it explicitly)
- **Second-order risk:** Because `task_handler/inputs.py` imports `chatterbox.tts.Conditionals` at module scope, skipping this install step does not degrade Chatterbox — it prevents the server from starting at all, and `setup.py` does not declare the package.
- **Upgrade checklist (must be followed for every version bump, already referenced by the requirements file):**
  1. Inspect the new release's `requires_dist` for new runtime packages or changed pins.
  2. Install with `pip install chatterbox-tts==<version> --no-deps` in a clean environment.
  3. Verify all four classes import: `ChatterboxTTS`, `ChatterboxVC`, `ChatterboxMultilingualTTS`, `ChatterboxTurboTTS`.
  4. Run `pytest tests/unit/initialization/test_chatterbox_imports.py -v` and the three Chatterbox e2e suites.
  5. Update the pin and any new transitive deps in `requirements-common.txt`.
- **Migration plan:** Move the Chatterbox import behind lazy loading and optional extras now; if Resemble relaxes the pins, drop `--no-deps`.

### Pinned `ctranslate2==4.4.0`

- **Risk:** Pinned to work around `faster-whisper` issue #1086. It blocks security and performance updates and constrains `faster-whisper` upgrades.
- **Files:** `requirements-common.txt:34-35`
- **Migration plan:** Track the upstream fix; add a compatibility test matrix before unpinning.

### Unpinned Core ML Stack

- **Risk:** `torch`, `torchaudio`, `transformers`, `numpy`, `diffusers`, `safetensors`, `peft`, `onnxruntime`, `spacy`, and `huggingface_hub` have no upper bounds; `pyproject.toml` only requires `torch>=2.7.1` at build time. A major release can break imports, dtype handling, or Chatterbox patches without warning.
- **Files:** `requirements-common.txt:1-70`, `pyproject.toml:1-30`
- **Migration plan:** Add a lock file (pip-tools/uv) and a CI job that tests the lock plus a "latest" canary.

### Python 3.13 and Library Removals

- **Risk:** `pydub` uses `audioop` and `audioread` uses `aifc`/`sunau`, all removed in Python 3.13; `harmonyspeech/modeling/models/harmonyspeech/common.py:7` uses `scipy.ndimage.morphology` (removed in SciPy 2.0); `harmonyspeech/modeling/models/melo/text/japanese.py:548-551` uses `kakasi.setMode`/`getConverter` (removed in kakasi 3.0); MeloTTS voice cloning uses `torch.stft(return_complex=False)`.
- **Files:** `requirements-common.txt`, `harmonyspeech/modeling/models/harmonyspeech/common.py:7`, `harmonyspeech/modeling/models/melo/text/japanese.py:548-551`
- **Migration plan:** Pin `requires-python = ">=3.12,<3.13"` until replacements land; migrate to `scipy.ndimage.grey_*`, new kakasi API, and `return_complex=True`.

### Complex Phonemizer/NLP Tree

- **Risk:** Three phonemizer families (`gruut`, `misaki`, `espeakng_loader`, `g2p_en`, `g2pkk`) and ten-plus language tokenizers (`spacy`, `nltk`, `mecab-python3`, `unidic-lite`, `pykakasi`, `eunjeon`, `fugashi`, `cn2an`, `pypinyin`, `jieba`, `spacy-pkuseg`) fight over shared transitive pins.
- **Files:** `requirements-common.txt:10-70`
- **Migration plan:** Split per-language extras; test each language in isolation.

### No Lock File

- **Risk:** No `requirements.lock`, `uv.lock`, or `poetry.lock`; Docker images and CI resolve independently and can diverge.
- **Files:** repository root
- **Migration plan:** Generate and commit a lock; make Docker builds install from it.

## Missing Critical Features

### TTS Streaming

- **Problem:** `output_options.stream=true` returns a 400 "Stream output is not yet supported"; the API server's `StreamingResponse` branches are all commented out.
- **Files:** `harmonyspeech/endpoints/openai/serving_text_to_speech.py:60-64`, `harmonyspeech/endpoints/openai/api_server.py:150-153,191-194,210-213,229-232,248-251`
- **Blocks:** Real-time voice agents, low time-to-first-audio, and playback before full synthesis. Chatterbox and MeloTTS must finish the entire utterance first.

### Output Format and Sample Rate Selection (RESOLVED — Phase 1)

- **Former problem:** `output_options.format` and `output_options.sample_rate` were accepted and documented but never read.
- **Status (Phase 1, 2026-11-04):** implemented at the API boundary. `output_options` is validated at admission (400 before synthesis) and applied once at response time (`harmonyspeech/endpoints/openai/output_encoding.py`). Supported: `wav` (byte-identical passthrough), `flac`, `ogg`, `mp3`, `opus`, `pcm` (raw s16le) and `aac` when ffmpeg is present; sample rates 8000–48000. `docs/audio-io-contract.md` documents the contract. Covered by `tests/unit/wire_contract/` and `tests/e2e/wire_contract/test_contract_matrix.py`.
- **Remaining:** `stream=true` is still rejected (see "TTS Streaming").

### Pre-/Post-Processing Filters

- **Problem:** `pre_processing_filters` and `post_generation_filters` are accepted on TTS requests but never executed; only `# TODO: Post-Generation Processing` exists.
- **Files:** `harmonyspeech/endpoints/openai/protocol.py:177-182`, `harmonyspeech/engine/harmonyspeech_engine.py:504`
- **Blocks:** Declarative denoise/restore/format chains; VoiceFixer exists but must be called manually.

### Embedding Cache

- **Problem:** The embedding path documents a future cache ("Cache key: hash(audio_bytes) or voice_id", `model_runner_base.py:809-821`) but no cache exists. Every voice-cloning request re-runs VAD/embedding.
- **Files:** `harmonyspeech/task_handler/model_runner_base.py:806-849`, `harmonyspeech/engine/harmonyspeech_engine.py:467-483`
- **Blocks:** Low-latency repeated voice cloning; unnecessary GPU work.

### `input_vad_mode` Selection — RESOLVED (Phase 3)

- **Was:** The field was documented as a VAD engine hint but was purely informational; routing always picked the first `FasterWhisper` config.
- **Now:** `"silero"` selects a configured Silero VAD executor for the OpenVoice pipelines (with a warning and Faster-Whisper fallback when none is configured); Silero emits the same whisper-schema segment artifact the encoder consumes.
- **Files:** `harmonyspeech/endpoints/openai/protocol.py:160-163`, `harmonyspeech/engine/harmonyspeech_engine.py:167,234`, `docs/api.md:143`
- **Blocks:** Users choosing Silero VAD inside OpenVoice/HarmonySpeech pipelines.

### Generation Options Other Than Speed

- **Problem:** `generation_options.seed`, `style`, `pitch`, and `energy` are documented (`docs/api.md:58-62`) but no code path reads them. `speed` is honored only for HarmonySpeech, MeloTTS, and KittenTTS; Chatterbox honors its sampling fields but not `speed`.
- **Files:** `harmonyspeech/endpoints/openai/protocol.py:51-90`, `harmonyspeech/task_handler/inputs.py:625-800`
- **Blocks:** Reproducible output, style control, and prosody adjustment promised by the API docs.

### Model Unloading / Capacity Management

- **Problem:** No API or runtime mechanism unloads a model or swaps a resident set. Per-model VRAM **is** now reported (Phase 4): `get_vram_stats()` feeds `/health` and `/metrics` (`hse_model_vram_bytes`, `hse_vram_reserved_bytes`, `hse_resident_models`, `hse_shared_backbone_bytes{group}`), and `WeightRegistry.release()` decrements ref counts — but nothing calls it in production yet.
- **Files:** `harmonyspeech/engine/harmonyspeech_engine.py` (`get_vram_stats`), `harmonyspeech/modeling/weight_registry.py`, `harmonyspeech/executor/gpu_executor.py`
- **Blocks:** Hosting more models than fit in VRAM; operating a 16 GB card with a catalogue of 20 CUDA models.

## Test Coverage Gaps

The suite has 224 test functions (162 unit, 23 integration, 39 E2E) and runs CPU-only in CI. The `coverage.xml` present in the workspace is stale (generated 2026-03-15, 16.5% line rate) and does not reflect the current suite.

| Gap | What is untested | Files | Risk | Priority |
|-----|------------------|-------|------|----------|
| GPU execution path | `GPUExecutor`, `GPUWorker`, `GPUModelRunner`, `torch.cuda` init, OOM behavior, multi-model residency | `harmonyspeech/executor/gpu_executor.py`, `harmonyspeech/task_handler/gpu_worker.py`, `harmonyspeech/task_handler/gpu_model_runner.py` | GPU-only regressions ship silently; every CI job is CPU | High |
| Multi-model residency | Loading `config.gpu.yml` (20 CUDA models) on a real GPU; startup memory | `config.gpu.yml` | OOM at deploy, not in CI | High |
| Mixed Chatterbox batches | embed + TTS in one scheduled batch, result attribution — **covered by `tests/unit/inference_flow/test_batch_pairing.py`** (Phase 0-3) | `harmonyspeech/task_handler/inputs.py:174-219`, `model_runner_base.py:96-115` | Hanging requests / wrong audio returned | High |
| Voice state leakage | Sequential voice-cloning → `single_speaker_tts` requests on one model — **covered by `tests/unit/inference_flow/test_chatterbox_state_isolation.py`** (Phase 0-2) | `harmonyspeech/task_handler/model_runner_base.py:683-700` | Wrong speaker audio, privacy issue | High |
| Auth and rate limiting | `auth/apikeys.py` middleware, cache expiry, `ENDPOINT_URL` unset/empty | `auth/apikeys.py`, `harmonyspeech/endpoints/openai/api_server.py:304-348` | Bypass or 500s in production | High |
| Streaming/rejection contract | No test sends `stream=true`; the 400 path is unverified | `serving_text_to_speech.py:60-64` | Feature gaps untracked | Medium |
| Output options | `format`/`sample_rate` now validated at admission and applied at response — **covered by `tests/unit/wire_contract/` and `tests/e2e/wire_contract/test_contract_matrix.py`** (Phase 1) | `harmonyspeech/endpoints/openai/output_encoding.py` | Silent contract violation | Medium |
| Scheduler edge cases | Unknown model → infinite wait; budget accounting; abort during forward — **unknown-model hang covered by `tests/unit/error_handling/test_unresolvable_routing.py`** (Phase 0-4); abort-during-forward still untested | `harmonyspeech/processing/scheduler.py`, `harmonyspeech/engine/harmonyspeech_engine.py:363-506` | Stuck requests, leaked slots | Medium |
| Engine timeout/death | `ENGINE_ITERATION_TIMEOUT_S` path, `AsyncEngineDeadError` recovery — **covered by `tests/unit/error_handling/test_request_timeouts.py`** (Phase 0-5) | `harmonyspeech/engine/async_harmonyspeech.py:375-383` | Full outage from one slow request | Medium |
| Chatterbox compatibility patches | `torch.load` patch, watermark fallback, dtype patches | `harmonyspeech/modeling/models/chatterbox/chatterbox.py:28-189` | Silent behavior drift on dependency upgrade | Medium |
| `check_health` | Base engine health iteration bug — **covered by `tests/unit/error_handling/test_request_timeouts.py`** (Phase 0-5) | `harmonyspeech/engine/harmonyspeech_engine.py:697-699` | Latent 500 if call path changes | Low |
| Frontend integration | No automated tests for the UI against a live engine | `frontend/` | UI regressions found manually | Low |

Also untested by design: E2E suites download real weights and are split across separate CI jobs; none run on CUDA, and the CUDA OpenVoice/MeloTTS fixtures in `tests/e2e/conftest.py:173-210` are commented out.

## Recommendations

### Priority 1 — Security and Correctness (before any new feature)

1. **Remove pickle execution from request paths.** *(Done — Phase 0-1)* Replace `torch.load(..., weights_only=False)` with `weights_only=True` or `safetensors` for embeddings and checkpoints (`model_runner_base.py:332-333`, `hf_downloader.py:163`).
2. **Fix Chatterbox voice leakage.** *(Done — Phase 0-2)* Reset `model.conds`/`ref_dict` per request or carry conditionals in the input (`model_runner_base.py:683-700`).
3. **Reject or split mixed Chatterbox batches.** *(Done — Phase 0-3)* Assert `len(inputs) == len(requests_to_batch)` and dispatch on request type per item (`inputs.py:174-219`, `model_runner_base.py:96-115`).
4. **Fail loudly on unresolvable routing.** *(Done — Phase 0-4, made structural in Phase 3)* Validate every declared pipeline at startup (optionally `--strict-pipelines`) and reject unresolvable requests immediately instead of queueing forever (`pipeline_registry.py`, `scheduler.py`).
5. **Make auth fail closed and fix the rate-limit math** (`api_server.py:304`, `apikeys.py:107,123`).

### Priority 2 — GPU Viability on a 16 GB Card

6. **Introduce a resident-set model config** — split `config.gpu.yml` into a small GPU set plus CPU/on-demand models; do not ship a 20-CUDA-model default. *(Still open — the measured post-Phase-4 catalogue now fits, so this is a latency/headroom choice rather than a hard blocker.)*
7. **Add lazy load with LRU eviction and per-model VRAM reporting**, plus an OOM preflight before loading (`engine/harmonyspeech_engine.py`, `executor/gpu_executor.py`). *(Reporting done — Phase 4-5 `/health` + `/metrics` + `scripts/measure_vram.py`; eviction/preflight still open, `WeightRegistry.release()` is the hook.)*
8. **Share Chatterbox weights** across TTS/embedding/VC roles. *(Done — Phase 4-4 `backbone_group`; block 14.29 GiB → 6.35 GiB.)*
9. **Default GPU configs to `dtype: "auto"` (fp16)** and remove `dtype: "float32"` from model entries that honor dtype. *(Partially done — Phase 4-1: Chatterbox entries are `float16` (T3). OpenVoice/MeloTTS/VoiceFixer stay `float32` because their adapted code is not half-precision safe; see `docs/benchmarks/audio-regression.md`.)*
10. **Replace the 120 s engine-step timeout with per-request timeouts** so one slow generation cannot kill the loop (`async_harmonyspeech.py:375-383`).

### Priority 3 — Throughput and API Contract

11. **Implement real batching or stop advertising it.** Either batch tensors per model or force `max_batch_size: 1` and expose concurrency; delete the stale `gpu_model_runner.py` CUDA-graph block.
12. **Add TTS streaming** (chunked audio over `StreamingResponse` or SSE) and make `stream=true` meaningful.
13. ~~**Honor or reject `format`/`sample_rate`/filters** — no silent acceptance.~~ **Done (Phase 1)** for `format`/`sample_rate`; filters are still ahead (pre/post-processing workstream).
14. ~~**Add binary/multipart audio transport** alongside base64 to cut 33% overhead and copies.~~ **Done (Phase 1)** for inputs on all audio endpoints.
15. ~~**Add request size and duration limits** to protect memory.~~ **Done (Phase 1)** for upload size (`--max-upload-size-mb`, 413); audio *duration* limits remain open.

### Priority 4 — Dependency and Test Hygiene

16. **Make Chatterbox a true optional extra** (lazy imports, optional dependency, clear install error) and keep the `--no-deps` upgrade checklist current (`requirements-chatterbox.txt`).
17. **Add a lock file** and a CI "latest dependencies" canary.
18. **Add a CUDA CI lane** (or a scheduled GPU runner) covering `GPUExecutor`, multi-model startup, and one Chatterbox generation.
19. **Replace Aphrodite metrics** with speech-specific metrics and remove dead metric definitions (`engine/metrics.py`).
20. **Delete dead code** (`ray_gpu_executor.py`, commented CUDA-graph block, unused imports) and remove or regenerate the stale workspace `coverage.xml`.

---

*Concerns audit: 2026-11-02*

# Phase 4: Engine Integration

## Objective

Wire `VllmOmniExecutor` into `HarmonySpeechEngine` by:
1. Adding `VllmOmniTTS` as a recognised model type in `init_custom_executors()`
2. Adding a `reroute_request_vllm_omni_tts()` routing method
3. Registering the routing method in `check_reroute_request_to_model()`

## Background

`HarmonySpeechEngine.init_custom_executors()` currently selects between `CPUExecutor` and `GPUExecutorAsync` based on `device_type`. `VllmOmniExecutor` is a third option — it doesn't use either, managing its own GPU internally.

`check_reroute_request_to_model()` dispatches to model-specific routing methods based on `request.requested_model`. For vllm-omni, the routing is simple: find the first configured `VllmOmniTTS` model and assign the request to it. There is no multi-step forwarding because vllm-omni handles the entire pipeline (LLM → vocoder) internally.

## File to Modify

**`harmonyspeech/engine/harmonyspeech_engine.py`**

## Change 1: `init_custom_executors()`

### Current code

```python
def init_custom_executors(self) -> None:
    for model_cfg in self.model_configs:
        if model_cfg.device_config.device_type == "cpu":
            from harmonyspeech.executor.cpu_executor import CPUExecutor
            executor_class = CPUExecutor
        else:
            from harmonyspeech.executor.gpu_executor import GPUExecutorAsync
            executor_class = GPUExecutorAsync

        executor = executor_class(model_config=model_cfg)
        self.model_executors[model_cfg.name] = executor
```

### Modified code

Add a branch for `VllmOmniTTS` **before** the existing `cpu`/GPU branches so it takes priority:

```python
def init_custom_executors(self) -> None:
    """Initialize custom executors for each provided ModelConfig."""
    for model_cfg in self.model_configs:
        # vllm-omni models use a dedicated executor that manages its own GPU processes
        if model_cfg.model_type == "VllmOmniTTS":
            from harmonyspeech.executor.vllm_omni_executor import VllmOmniExecutor
            executor_class = VllmOmniExecutor
        elif model_cfg.device_config.device_type == "cpu":
            from harmonyspeech.executor.cpu_executor import CPUExecutor
            executor_class = CPUExecutor
        else:
            from harmonyspeech.executor.gpu_executor import GPUExecutorAsync
            executor_class = GPUExecutorAsync

        executor = executor_class(model_config=model_cfg)
        self.model_executors[model_cfg.name] = executor
```

## Change 2: Add `reroute_request_vllm_omni_tts()`

Add this method to `HarmonySpeechEngine`, following the same pattern as other `reroute_request_*` methods (e.g., `reroute_request_voicefixer`). Place it after `reroute_request_voicefixer`:

```python
def reroute_request_vllm_omni_tts(self, request: RequestInput):
    """Route a vllm-omni TTS request to the appropriate VllmOmniTTS executor.

    Single-step routing: vllm-omni handles its full LLM → vocoder pipeline
    internally, so no multi-step forwarding is needed here.

    Selects the first configured VllmOmniTTS model whose voices list either:
    - Is empty (accept all voices)
    - Contains the requested voice_id

    Falls back to the first VllmOmniTTS model if no voice match is found.
    """
    if not isinstance(request, TextToSpeechRequestInput):
        return

    voice_id = getattr(request, "voice_id", None)
    fallback_cfg = None

    for cfg in self.model_configs:
        if cfg.model_type != "VllmOmniTTS":
            continue
        if fallback_cfg is None:
            fallback_cfg = cfg  # Remember first VllmOmniTTS as fallback
        # If voices list is empty or contains the requested voice, use this model
        if not cfg.voices or (voice_id and voice_id in cfg.voices):
            request.model = cfg.name
            return

    # Fallback: use first VllmOmniTTS model regardless of voice match
    if fallback_cfg is not None:
        request.model = fallback_cfg.name
```

## Change 3: Register in `check_reroute_request_to_model()`

### Current code (end of the method)

```python
def check_reroute_request_to_model(self, request: RequestInput):
    if request.requested_model == "harmonyspeech":
        self.reroute_request_harmonyspeech(request)
    if request.requested_model == "openvoice_v1":
        self.reroute_request_openvoice_v1(request)
    if request.requested_model == "openvoice_v2":
        self.reroute_request_openvoice_v2(request)
    if request.requested_model == "voicefixer":
        self.reroute_request_voicefixer(request)
```

### Modified code

Add the vllm-omni routing at the end:

```python
def check_reroute_request_to_model(self, request: RequestInput):
    if request.requested_model == "harmonyspeech":
        self.reroute_request_harmonyspeech(request)
    if request.requested_model == "openvoice_v1":
        self.reroute_request_openvoice_v1(request)
    if request.requested_model == "openvoice_v2":
        self.reroute_request_openvoice_v2(request)
    if request.requested_model == "voicefixer":
        self.reroute_request_voicefixer(request)
    if request.requested_model == "vllm_omni_tts":          # NEW
        self.reroute_request_vllm_omni_tts(request)         # NEW
```

## How Clients Request vllm-omni TTS

Clients send a `TextToSpeechRequest` with `model: "vllm_omni_tts"` (matching `requested_model`). Example:

```json
POST /v1/audio/speech
{
    "model": "vllm_omni_tts",
    "input": "Hello, this is Harmony Speech Engine.",
    "voice": "Vivian",
    "language_id": "EN"
}
```

Or for voice cloning:
```json
{
    "model": "vllm_omni_tts",
    "input": "Hello world",
    "mode": "voice_cloning",
    "input_audio": "<base64-encoded-reference-audio>"
}
```

## No `check_forward_processing` Changes Needed

`VllmOmniExecutor.execute_model()` returns a `TextToSpeechRequestOutput` directly (not an intermediate `SpeechSynthesisRequestOutput`). The existing `check_forward_processing` handles `TextToSpeechRequestOutput` in the final `else` branch:

```python
else:  # Final processing results
    new_status = RequestStatus.FINISHED_STOPPED
    self.scheduler.update_request_status(result.request_id, new_status)
    tts_result = TextToSpeechRequestOutput(...)  # ← already this type, but re-wrapped
    result.result_data = tts_result
```

Wait — this will attempt to re-wrap the output, which is incorrect when the executor already returns `TextToSpeechRequestOutput`. 

**Correction**: The engine currently re-wraps `VocodeRequestOutput` into `TextToSpeechRequestOutput` for multi-step workflows. Since `VllmOmniTTS` returns a `TextToSpeechRequestOutput` **directly** from `execute_model()`, `check_forward_processing` will hit the final `else` branch and **re-wrap** it unnecessarily.

**Fix**: In `check_forward_processing`, add an early-exit check for requests from `VllmOmniTTS`:

```python
# In check_forward_processing, within the TextToSpeechRequestInput block:
if isinstance(input_data, TextToSpeechRequestInput) and requested_model in [
    "harmonyspeech",
    "openvoice_v1",
    "openvoice_v2"
]:
    # ... existing multi-step forwarding logic ...
```

By only entering the multi-step block for known multi-step workflows, `vllm_omni_tts` requests will fall through naturally. Since `requested_model == "vllm_omni_tts"` is not in that list, `check_forward_processing` will return `(RequestStatus.FINISHED_STOPPED, None)` without any re-wrapping.

**Verify** the condition in `check_forward_processing` around line:
```python
if isinstance(input_data, TextToSpeechRequestInput) and requested_model in [
    "harmonyspeech",
    "openvoice_v1",
    "openvoice_v2"
]:
```
If this condition already exists exactly like this, no further change is needed. If it uses `isinstance(input_data, TextToSpeechRequestInput)` without the `requested_model` check, it must be updated.

## Progress Checklist

- [ ] Add `VllmOmniTTS` branch in `init_custom_executors()` (before the cpu/GPU branches)
- [ ] Add `reroute_request_vllm_omni_tts()` method
- [ ] Add `vllm_omni_tts` case in `check_reroute_request_to_model()`
- [ ] Verify `check_forward_processing()` — confirm the multi-step block uses `requested_model in [...]` guard so vllm_omni_tts requests bypass it cleanly
- [ ] If the guard is missing, add `requested_model in ["harmonyspeech", "openvoice_v1", "openvoice_v2"]` condition to the `TextToSpeechRequestInput` block

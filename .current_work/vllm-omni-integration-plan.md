## Revised Integration Plan: vllm-omni into HSE

After studying the actual codebase, my original plan was significantly wrong about the nature of `Omni`. Here's what the codebase actually shows and what needs to happen.

---

### Key Finding: `Omni` is NOT an embeddable model object

`Omni` is a **multi-process pipeline orchestrator** that:
1. Downloads model weights
2. Spawns **separate OS-level subprocesses** for each stage via `multiprocessing.spawn` or Ray
3. Communicates between stages via ZMQ sockets or mp.Queues
4. Manages its own CUDA contexts entirely within its worker subprocesses (no conflict with HSE's `torch.cuda.set_device`)

This means it **cannot** be loaded via HSE's `GPUWorker → GPUModelRunner → loader.get_model()` chain. It IS the worker, model runner, and loader combined — just like how a Docker container cannot be embedded inside another process.

### The Correct Architecture: A New `VllmOmniExecutor`

The right approach is to create a **dedicated executor class** that holds the `Omni` instance, bypassing the entire `GPUWorker/GPUModelRunner` chain:

```
HarmonySpeechEngine.model_executors["qwen3-tts"] = VllmOmniExecutor(model_config)
                                                          |
                                                    Omni(model="Qwen/...")
                                                    ├── Stage-0 worker subprocess (GPU)
                                                    └── Stage-1 worker subprocess (GPU)
```

The executor delegates everything — loading, input prep, execution, output extraction — to `Omni`.

---

### Audio Output Structure (confirmed from codebase)

From `end2end.py` and `serving_speech.py`:
```python
# omni.generate([prompt]) returns List[OmniRequestOutput]
stage_output = omni_outputs[0]  # OmniRequestOutput
mm = stage_output.multimodal_output  # dict: {"audio": tensor_or_list, "sr": int_or_list}

# Extract audio:
audio_data = mm["audio"]  # torch.Tensor or list of tensors (streaming chunks)
sr_raw = mm["sr"]         # sample rate
sr = sr_raw[-1] if isinstance(sr_raw, list) else sr_raw
sr = int(sr.item() if hasattr(sr, 'item') else sr)
audio_tensor = torch.cat(audio_data, dim=-1) if isinstance(audio_data, list) else audio_data
audio_np = audio_tensor.float().cpu().numpy().flatten()
```

### Prompt Input Structure (Qwen3-TTS, confirmed from `end2end.py`)

The prompt is NOT a plain string — it's a dict with placeholder tokens and an `additional_information` payload:
```python
prompt = {
    "prompt_token_ids": [0] * estimated_length,  # placeholder, length matters
    "additional_information": {
        "task_type": ["CustomVoice"],       # or "Base" (voice clone), "VoiceDesign"
        "text": ["Hello world"],
        "language": ["English"],            # "Chinese", "Auto", etc.
        "speaker": ["Vivian"],              # for CustomVoice
        "instruct": [""],                   # style/emotion instructions
        "max_new_tokens": [2048],
        # For Base (voice clone):
        # "ref_audio": [[wav_samples_list, sample_rate]],
        # "ref_text": ["transcript..."],
        # OR "x_vector_only_mode": [True]   (no ref_text needed)
    }
}
```

The `estimated_length` requires loading the model's `AutoTokenizer` and calling a model-specific estimation function. We cache this tokenizer in the executor at init time.

---

### Files to Create / Modify

#### NEW: `harmonyspeech/executor/vllm_omni_executor.py`

This is the core new file. Key responsibilities:
- On `_init_executor()`: Set `VLLM_WORKER_MULTIPROC_METHOD=spawn`, instantiate `Omni(model=...)`, load and cache the tokenizer for prompt length estimation
- On `execute_model(requests_to_batch)`: For each request in the batch, build the `additional_information` dict from `TextToSpeechRequestInput`, call `omni.generate([prompt])`, extract audio from `mm["audio"]`, encode as WAV base64, return `TextToSpeechRequestOutput`

Mapping from HSE's `TextToSpeechRequestInput` to vllm-omni input:
| HSE field | vllm-omni field | Notes |
|---|---|---|
| `input_text` | `text` | Direct |
| `language_id` | `language` | Default "Auto" |
| `voice_id` | `speaker` | For CustomVoice |
| `input_audio` (base64) | `ref_audio` | Decoded → `[[samples, sr]]`, triggers "Base" task |
| `generation_options.speed` | *(not supported by Qwen3-TTS)* | Ignored |

Task type logic:
- If `input_audio` is provided → "Base" with `x_vector_only_mode=True` (no ref_text needed)  
- Else if `voice_id` is set → "CustomVoice" with `speaker=voice_id`
- Else → "CustomVoice" with default speaker "Vivian"

#### MODIFIED: `harmonyspeech/engine/harmonyspeech_engine.py`

1. In `init_custom_executors()`: Add a branch for `VllmOmniTTS` model type to use `VllmOmniExecutor` instead of `GPUExecutorAsync`
2. Add `reroute_request_vllm_omni_tts()` routing method (single-step, no multi-stage forwarding — vllm-omni handles the pipeline internally)
3. Register it in `check_reroute_request_to_model()` under `requested_model == "vllm_omni_tts"`

#### MODIFIED: `harmonyspeech/modeling/models/__init__.py`

Register a single generic model type:
```python
"VllmOmniTTS": ("vllm_omni", "native"),
```
This is only needed for `ModelRegistry` consistency / `get_supported_archs()`. The actual loading is handled entirely by `VllmOmniExecutor`, not by `loader.get_model()`.

#### MODIFIED: `requirements-cuda.txt` and `requirements-rocm.txt`

Add `vllm-omni` to both GPU-tier requirements files. Not in `requirements-common.txt` or `requirements-cpu.txt` — GPU only.

#### MODIFIED: `config.gpu.yml`

Add example entries:
```yaml
# Qwen3-TTS CustomVoice (built-in speakers)
- name: "qwen3-tts-customvoice"
  model: "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
  model_type: "VllmOmniTTS"
  voices: ["Vivian", "Ryan", "Ethan", "Chelsie", "Serena"]
  max_batch_size: 4
  dtype: "bfloat16"
  device_config:
    device: "cuda"

# Qwen3-TTS Base (voice cloning via reference audio)  
- name: "qwen3-tts-base"
  model: "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
  model_type: "VllmOmniTTS"
  voices: []
  max_batch_size: 4
  dtype: "bfloat16"
  device_config:
    device: "cuda"
```

CosyVoice3 can be added later — its input format requires separate investigation (different stage config structure).

---

### Constraint: `loader.py` and `task_handler/` are NOT touched

`VllmOmniExecutor` is completely self-contained — it handles input prep, execution, and output extraction internally. No changes to `loader.py`, `inputs.py`, or `model_runner_base.py`. This is cleaner and avoids coupling vllm-omni's complex input structure into HSE's existing input preparation system.

---

### Summary

| Category | Detail |
|---|---|
| New files | 1 (`vllm_omni_executor.py`) |
| Modified files | 4 (`harmonyspeech_engine.py`, `models/__init__.py`, `requirements-*.txt`, `config.gpu.yml`) |
| New executor pattern | `VllmOmniExecutor` wraps `Omni` — completely self-contained |
| Input format | `additional_information` dict built in executor from `TextToSpeechRequestInput` |
| Output format | `mm["audio"]` + `mm["sr"]` → WAV base64 → `TextToSpeechRequestOutput` |
| Multi-step forwarding needed? | **No** — vllm-omni handles the LLM→Vocoder pipeline internally |
| GPU memory note | vllm-omni uses `gpu_memory_utilization` per stage (0.3 + 0.2 = 50% for Qwen3-TTS) — other HSE models must fit in the remaining GPU memory |

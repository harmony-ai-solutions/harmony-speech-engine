# vllm-omni Integration into Harmony Speech Engine

## Overview

Integrates vllm-omni as a new GPU-accelerated TTS backend within Harmony Speech Engine (HSE), enabling high-quality LLM-based TTS models: **Qwen3-TTS** (CustomVoice, VoiceDesign, Base/voice-clone), **CosyVoice3**, and **Fish Speech**.

### Key Architectural Decision: `VllmOmniExecutor`

vllm-omni's `Omni` class is a **multi-process pipeline orchestrator** — not a simple PyTorch model. It spawns OS-level subprocesses via `multiprocessing.spawn` for each stage (e.g., LLM talker + vocoder) and communicates between them via ZMQ sockets. This makes it fundamentally incompatible with HSE's existing `GPUWorker → GPUModelRunner → loader.get_model()` chain.

**Solution**: A dedicated `VllmOmniExecutor` class that holds the `Omni` instance and handles everything (loading, input preparation, generation, output extraction) internally, bypassing the standard worker chain entirely.

### Architecture Diagram

```
HarmonySpeechEngine.model_executors
  └─ "qwen3-tts" → VllmOmniExecutor
                        └─ Omni(model="Qwen/Qwen3-TTS-...")
                             ├─ Stage-0 subprocess (GPU): LLM Talker → speech tokens
                             └─ Stage-1 subprocess (GPU): Code2Wav → audio waveform
```

### Per-Stage Memory Configuration

Users specify GPU memory per stage in human-readable format (`"3G"`, `"512M"`, `"1.5G"`) in HSE's `config.yml`. The executor converts these to fractional `gpu_memory_utilization` values at runtime using the GPU's actual total VRAM, then patches the vllm-omni stage YAML config before passing to `Omni`.

### Audio Output Flow (confirmed from vllm-omni codebase)

```python
omni_outputs = omni.generate([prompt])  # List[OmniRequestOutput]
mm = omni_outputs[0].multimodal_output  # {"audio": tensor_or_list, "sr": int_or_list}
# audio → concat if list → numpy → WAV → base64 → TextToSpeechRequestOutput
```

### What is NOT changed

- `harmonyspeech/modeling/loader.py` — VllmOmniExecutor handles its own loading
- `harmonyspeech/task_handler/inputs.py` — input prep is internal to the executor
- `harmonyspeech/task_handler/model_runner_base.py` — bypassed entirely
- CPU path is unaffected; vllm-omni is GPU-only

---

## Reference Files

| File | Purpose |
|---|---|
| `.current_work/vllm-omni/vllm_omni/entrypoints/omni.py` | `Omni` class implementation |
| `.current_work/vllm-omni/vllm_omni/entrypoints/utils.py` | `resolve_model_config_path()`, `load_and_resolve_stage_configs()` |
| `.current_work/vllm-omni/vllm_omni/outputs.py` | `OmniRequestOutput` with `multimodal_output` property |
| `.current_work/vllm-omni/examples/offline_inference/qwen3_tts/end2end.py` | Reference for prompt format + audio extraction |
| `.current_work/vllm-omni/vllm_omni/entrypoints/openai/serving_speech.py` | Reference for `_build_tts_params`, `_extract_audio_output` |
| `.current_work/vllm-omni/vllm_omni/model_executor/stage_configs/qwen3_tts.yaml` | Default Qwen3-TTS 2-stage pipeline YAML |
| `.current_work/vllm-omni/vllm_omni/model_executor/stage_configs/cosyvoice3.yaml` | Default CosyVoice3 stage YAML |
| `harmonyspeech/common/config.py` | `ModelConfig` class to extend |
| `harmonyspeech/executor/gpu_executor.py` | Existing executor pattern to follow |
| `harmonyspeech/engine/harmonyspeech_engine.py` | Engine routing and executor init |
| `harmonyspeech/modeling/models/__init__.py` | Model registry |
| `requirements-cuda.txt`, `requirements-rocm.txt` | GPU requirements files |
| `config.gpu.yml` | Example GPU config |

---

## Implementation Status

Track the completion of each phase as implementation progresses:

- [ ] **Phase 1: ModelConfig Extension** ([1-ModelConfigExtension.md](1-ModelConfigExtension.md))
- [ ] **Phase 2: VllmOmniExecutor**
  - [ ] Core Class Structure and Initialization ([2-1-VllmOmniExecutor-CoreClass.md](2-1-VllmOmniExecutor-CoreClass.md))
  - [ ] Memory Parsing and Stage Config Override ([2-2-VllmOmniExecutor-MemoryAndStageConfig.md](2-2-VllmOmniExecutor-MemoryAndStageConfig.md))
  - [ ] Prompt Building for Qwen3-TTS ([2-3-VllmOmniExecutor-PromptBuilding.md](2-3-VllmOmniExecutor-PromptBuilding.md))
  - [ ] Execute Model and Audio Extraction ([2-4-VllmOmniExecutor-ExecuteAndExtract.md](2-4-VllmOmniExecutor-ExecuteAndExtract.md))
- [ ] **Phase 3: Model Registry** ([3-ModelRegistry.md](3-ModelRegistry.md))
- [ ] **Phase 4: Engine Integration** ([4-EngineIntegration.md](4-EngineIntegration.md))
- [ ] **Phase 5: Requirements** ([5-Requirements.md](5-Requirements.md))
- [ ] **Phase 6: Config Examples** ([6-ConfigExamples.md](6-ConfigExamples.md))
- [ ] **Phase 7: Documentation** ([7-Documentation.md](7-Documentation.md))

---

## TODO / Post-Implementation Notes

- **README.md**: Add a section or note that HSE supports additional LLM-based TTS models (Qwen3-TTS, CosyVoice3, Fish Speech) through vllm-omni integration, with a pointer to the models documentation.
- **docs/models.md** (or create `docs/vllm-omni-models.md`): Add a dedicated section explaining:
  - What vllm-omni is and why it's used
  - Which models are supported (`VllmOmniTTS` type)
  - How to configure them in `config.yml` (the `stage_memory` field, `voices`, etc.)
  - Where to find the exact stage YAML configs: `vllm_omni/model_executor/stage_configs/` in the vllm-omni package, or in the model's HuggingFace repo
  - Task types for Qwen3-TTS: `CustomVoice` (named speakers), `VoiceDesign` (instruct-style), `Base` (voice cloning via `input_audio`)
  - GPU memory planning: how to calculate `stage_memory` values based on your GPU VRAM

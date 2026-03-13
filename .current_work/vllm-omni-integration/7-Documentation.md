# Phase 7: Documentation

## Objective

Update HSE's user-facing documentation to:
1. Add a note in `README.md` about vllm-omni backed LLM TTS models
2. Add a dedicated section in `docs/` explaining how vllm-omni integration works, which models are supported, and how to configure them

## Files to Modify / Create

1. **`README.md`** — Add a brief mention of vllm-omni support in the "Supported Models" or "Features" section
2. **`docs/models.md`** (create if it doesn't exist) OR add to an existing models document — Detailed vllm-omni section

## Step 1: Locate existing README sections

Before editing, read `README.md` to identify:
- Where TTS models are listed (likely a table or list of supported models)
- The Features or Capabilities section
- Where GPU requirements are mentioned

## Step 2: README.md Update

Add the following to the README's supported models / capabilities section. Find the section describing TTS models and add an entry for vllm-omni backed models. Example addition:

```markdown
### LLM-Based TTS via vllm-omni (GPU only)

Harmony Speech Engine supports high-quality, LLM-backed TTS models through optional **[vllm-omni](https://github.com/vllm-project/vllm-omni)** integration:

| Model | HuggingFace ID | Task Types |
|---|---|---|
| Qwen3-TTS CustomVoice 1.7B | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | Named speakers |
| Qwen3-TTS Base 1.7B | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | Voice cloning |
| Qwen3-TTS VoiceDesign 1.7B | `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | Voice description |
| Qwen3-TTS CustomVoice 3B | `Qwen/Qwen3-TTS-12Hz-3B-CustomVoice` | Named speakers (higher quality) |
| CosyVoice3 | `FunAudioLLM/CosyVoice3-0.5B` | TTS + voice cloning |
| Fish Speech | `fishaudio/fish-speech-1.5` | TTS + voice cloning |

> **Requirements**: GPU with CUDA or ROCm; install with `pip install vllm-omni`  
> See [docs/vllm-omni-models.md](docs/vllm-omni-models.md) for configuration details.

> **⚠️ Note on long-term roadmap**: The vllm-omni integration is a **pragmatic, interim solution** to expand HSE's model portfolio while native support for LLM-based TTS architectures (Qwen3-TTS, CosyVoice3, Fish Speech) is absent. These models will likely be integrated directly into HSE's native model loader and executor framework in the future, which will provide tighter control, better resource sharing, and full CPU/GPU parity where applicable. The vllm-omni wrapper should be considered a compromise driven by current developer resource constraints, not the intended final architecture.
```

## Step 3: Create `docs/vllm-omni-models.md`

Create this file with the following full content:

```markdown
# vllm-omni Backed TTS Models

Harmony Speech Engine supports LLM-based, high-quality TTS models through
**[vllm-omni](https://github.com/vllm-project/vllm-omni)** integration, adding
next-generation speech synthesis capabilities alongside the existing model portfolio.

## Why vllm-omni?

Traditional TTS models (XTTSv2, StyleTTS2, OpenVoice) use smaller dedicated neural
networks. Modern LLM-based TTS (Qwen3-TTS, CosyVoice3, Fish Speech) chain a large
language model with a vocoder, producing significantly more natural and expressive
speech — at the cost of requiring GPU and more memory.

vllm-omni extends the vLLM inference engine to handle these multi-stage pipelines,
providing efficient batching and KV-cache management for the LLM stage.

## Prerequisites

- NVIDIA GPU with CUDA (or AMD GPU with ROCm)
- vllm-omni installed: `pip install vllm-omni`
- When using CUDA requirements: `pip install -r requirements-cuda.txt`
- When using ROCm requirements: `pip install -r requirements-rocm.txt`

## Supported Models

| Model | HuggingFace ID | Capabilities |
|---|---|---|
| Qwen3-TTS 1.7B CustomVoice | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | Named speakers |
| Qwen3-TTS 1.7B Base | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | Voice cloning from reference audio |
| Qwen3-TTS 1.7B VoiceDesign | `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | Voice described in natural language |
| Qwen3-TTS 3B CustomVoice | `Qwen/Qwen3-TTS-12Hz-3B-CustomVoice` | Named speakers (higher quality) |
| CosyVoice3 | `FunAudioLLM/CosyVoice3-0.5B` | TTS + voice cloning |
| Fish Speech | `fishaudio/fish-speech-1.5` | TTS + voice cloning |

All models use `model_type: "VllmOmniTTS"` in `config.yml`.

## Configuration

### Basic example (Qwen3-TTS CustomVoice on a 24GB GPU)

```yaml
model_configs:
  - name: "qwen3-tts"
    model: "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    model_type: "VllmOmniTTS"
    stage_memory: ["6G", "4G"]
    voices: ["Vivian", "Ryan", "Ethan"]
    max_batch_size: 4
    dtype: "bfloat16"
    device_config:
      device: "cuda"
```

### `stage_memory` — per-stage GPU memory allocation

vllm-omni TTS models use a multi-stage pipeline (e.g., LLM talker → vocoder decoder).
Each stage consumes a portion of GPU VRAM. `stage_memory` lets you control this:

- Each entry corresponds to one stage, in order (Stage 0, Stage 1, ...)
- Format: human-readable strings — `"3G"`, `"1.5G"`, `"512M"`, `"2048M"`
- Values are converted to fractional `gpu_memory_utilization` at runtime based on your GPU's actual total VRAM
- If omitted, vllm-omni uses defaults from its built-in stage YAML config

**Memory guide for Qwen3-TTS 1.7B** (on common GPU sizes):

| GPU VRAM | Stage 0 (LLM) | Stage 1 (Vocoder) | Remaining for other models |
|---|---|---|---|
| 24 GB | 6G | 4G | ~14 GB |
| 16 GB | 4G | 3G | ~9 GB |
| 12 GB | 4G | 3G | ~5 GB (tight) |
| 8 GB | Not recommended | — | — |

### Finding stage configurations

vllm-omni stores its pipeline stage YAML configs in:
```
<vllm_omni_install>/vllm_omni/model_executor/stage_configs/
```

Key files:
- `qwen3_tts.yaml` — Qwen3-TTS 2-stage pipeline config
- `qwen3_tts_no_async_chunk.yaml` — Qwen3-TTS without streaming (slower, simpler)
- `cosyvoice3.yaml` — CosyVoice3 2-stage pipeline config
- `fish_speech_s2_pro.yaml` — Fish Speech pipeline config

These files are read automatically by HSE when `model_type: "VllmOmniTTS"` is used.
The `stage_memory` field in `config.yml` patches the `gpu_memory_utilization` values
in these files at startup.

Alternatively, models in HuggingFace repos may provide their own `pipeline.yaml`;
vllm-omni will use that automatically if found.

## API Usage

Request vllm-omni TTS by setting `model: "vllm_omni_tts"` in your API request:

### CustomVoice (named speaker)

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vllm_omni_tts",
    "input": "Hello, this is a test.",
    "voice": "Vivian",
    "language_id": "EN"
  }'
```

### Base (voice cloning from reference audio)

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vllm_omni_tts",
    "input": "Hello, this is a test.",
    "mode": "voice_cloning",
    "input_audio": "<base64-encoded-reference-WAV-audio>"
  }'
```

## Architecture Notes

Unlike traditional HSE models that use the `GPUWorker → GPUModelRunner → loader` chain,
vllm-omni models use a dedicated `VllmOmniExecutor` that:

1. Instantiates vllm-omni's `Omni` pipeline orchestrator
2. `Omni` spawns separate GPU worker subprocesses per stage via `multiprocessing.spawn`
3. Inter-stage communication uses ZMQ sockets (zero-copy shared memory on a single node)
4. `VllmOmniExecutor.execute_model()` calls `omni.generate()` synchronously and converts the audio output to HSE's standard `TextToSpeechRequestOutput`

This means HSE's existing CPU path, STT, Voice Conversion, and Speech Embedding
capabilities are completely unaffected by vllm-omni.

## GPU Memory Considerations

vllm-omni models run alongside other HSE models on the same GPU. Total GPU memory
consumed is the sum of:
- All HSE model weights (FasterWhisper, OpenVoice, etc.)
- vllm-omni stage allocations (controlled by `stage_memory`)

Plan accordingly: if HSE models consume ~4GB and you want Qwen3-TTS 1.7B with
`stage_memory: ["6G", "4G"]`, you need at least a 16GB GPU.

## Limitations

- **GPU only** — vllm-omni does not support CPU inference
- **VoiceDesign task** — not yet fully exposed through HSE's TTS API (no `instructions` field in the standard request). Can be added by extending `TextToSpeechRequestInput`.
- **CosyVoice3 / Fish Speech** — these models follow the `VllmOmniTTS` path but have different internal prompt formats. The `_estimate_prompt_len` fallback of 2048 is used, which may not be optimal. Per-model estimators can be added in future phases.
- **Streaming** — vllm-omni supports audio chunk streaming internally, but HSE currently waits for the full audio before returning. Streaming support can be added via `AsyncOmni` in a future phase.
```

## Progress Checklist

- [ ] Read `README.md` to identify where to add the vllm-omni mention
- [ ] Add vllm-omni TTS models table/section to `README.md` Features or Supported Models section
- [ ] Add link from README to `docs/vllm-omni-models.md`
- [ ] Create `docs/vllm-omni-models.md` with full content above
- [ ] Verify all HuggingFace model IDs in the docs are correct (check HF hub)
- [ ] Verify the stage YAML file names are accurate (cross-check with `.current_work/vllm-omni/vllm_omni/model_executor/stage_configs/`)

# Phase 5: Requirements

## Objective

Add `vllm-omni` as a dependency to the GPU-specific requirements files. It must NOT be added to `requirements-common.txt` or `requirements-cpu.txt` because vllm-omni is GPU-only (no CPU path, hard CUDA/ROCm requirement).

## Files to Modify

1. **`requirements-cuda.txt`** — NVIDIA GPU requirements
2. **`requirements-rocm.txt`** — AMD ROCm GPU requirements

## Current State

### `requirements-cuda.txt`
```
# Common dependencies
-r requirements-common.txt

# Requirements for Nvidia GPUs
```
(Currently has no GPU-specific packages)

### `requirements-rocm.txt`
```
# Common dependencies
-r requirements-common.txt

# Requirements for AMD GPUs (ROCm)
```
(Currently has no GPU-specific packages)

## Implementation

### `requirements-cuda.txt`

```
# Common dependencies
-r requirements-common.txt

# Requirements for Nvidia GPUs
# vllm-omni: GPU-accelerated multi-model TTS (Qwen3-TTS, CosyVoice3, Fish Speech)
# Requires CUDA and vLLM; installs vLLM as a transitive dependency
vllm-omni
```

### `requirements-rocm.txt`

```
# Common dependencies
-r requirements-common.txt

# Requirements for AMD GPUs (ROCm)
# vllm-omni: GPU-accelerated multi-model TTS (Qwen3-TTS, CosyVoice3, Fish Speech)
# Requires ROCm-compatible vLLM; ensure vLLM ROCm wheels are installed separately
# See: https://github.com/harmony-ai-solutions/vllm-omni#installation
vllm-omni
```

## Notes

- **`vllm-omni` is a pip package** installable from PyPI (or from source). It pulls in `vllm` as a dependency, which in turn installs CUDA/ROCm-specific torch wheels.
- **Version pinning**: For now, leave as unpinned `vllm-omni`. Since vllm-omni is early-stage and moves fast, consider pinning to a specific version once the integration is tested (e.g., `vllm-omni>=0.16.0`).
- **`requirements-cpu.txt`**: Do NOT add vllm-omni here. CPU-only HSE deployments should remain unaffected.
- **`requirements-common.txt`**: Do NOT add vllm-omni here. It would break CPU-only installs.
- **Additional transitive dependencies** automatically installed with vllm-omni:
  - `vllm` (LLM inference engine)
  - `zmq` / `pyzmq` (inter-stage communication)
  - `ray` (optional, for multi-node deployments)
  - `huggingface_hub` (already in requirements-common.txt)
  - `transformers` (already in requirements-common.txt)
  - `msgspec` (vllm-omni serialization)

## Progress Checklist

- [ ] Add `vllm-omni` with comment to `requirements-cuda.txt`
- [ ] Add `vllm-omni` with comment to `requirements-rocm.txt`
- [ ] Confirm `requirements-common.txt` and `requirements-cpu.txt` are NOT modified

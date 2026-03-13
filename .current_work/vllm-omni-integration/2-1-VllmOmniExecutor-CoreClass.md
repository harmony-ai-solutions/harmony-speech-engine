# Phase 2-1: VllmOmniExecutor — Core Class Structure and Initialization

## Objective

Create the file `harmonyspeech/executor/vllm_omni_executor.py` with the core class definition, imports, and `_init_executor()` / `check_health()` methods. This phase establishes the skeleton; subsequent phases (2-2, 2-3, 2-4) add the helper methods.

## Background

`VllmOmniExecutor` extends `ExecutorBase` (not `GPUExecutor`) and holds an `Omni` instance from vllm-omni as its "model". Key differences from `GPUExecutorAsync`:
- No `GPUWorker`, no `GPUModelRunner`, no `loader.get_model()` call
- `Omni` manages its own CUDA subprocesses via `multiprocessing.spawn`
- `VLLM_WORKER_MULTIPROC_METHOD=spawn` must be set before `Omni` is instantiated
- The tokenizer (for Qwen3-TTS prompt length estimation) is loaded and cached during init

**`ExecutorBase` interface** (from `harmonyspeech/executor/executor_base.py`):
```python
class ExecutorBase(ABC):
    def __init__(self, model_config: ModelConfig) -> None: ...
    def _init_executor(self) -> None: ...  # abstract
    def execute_model(self, requests_to_batch: List[EngineRequest]) -> List[ExecutorResult]: ...  # abstract
    def check_health(self) -> None: ...  # abstract
```

## File to Create

**`harmonyspeech/executor/vllm_omni_executor.py`** (new file)

## Implementation

The complete file begins with:

```python
"""
VllmOmniExecutor: Executor for vllm-omni TTS models (Qwen3-TTS, CosyVoice3, Fish Speech).

Architecture:
    VllmOmniExecutor holds a vllm-omni `Omni` instance which internally manages
    multi-stage GPU worker subprocesses. Input preparation, prompt building, and
    audio extraction are handled entirely within this executor — bypassing HSE's
    standard GPUWorker / GPUModelRunner / loader.get_model() chain.

Supported model_type values:
    "VllmOmniTTS" — any vllm-omni TTS model (Qwen3-TTS, CosyVoice3, Fish Speech)

Config example:
    - name: "qwen3-tts-customvoice"
      model: "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
      model_type: "VllmOmniTTS"
      stage_memory: ["6G", "4G"]   # per-stage GPU memory; omit for vllm-omni defaults
      voices: ["Vivian", "Ryan"]
      max_batch_size: 4
      dtype: "bfloat16"
      device_config:
        device: "cuda"
"""

import base64
import io
import os
import re
import tempfile
import time
from typing import List, Optional

import torch
import yaml
from loguru import logger

from harmonyspeech.common.config import ModelConfig
from harmonyspeech.common.request import EngineRequest, ExecutorResult
from harmonyspeech.executor.executor_base import ExecutorBase


# ---------------------------------------------------------------------------
# Module-level memory utility functions (used by executor and accessible for
# testing independently)
# ---------------------------------------------------------------------------

def _parse_memory_bytes(memory_str: str) -> int:
    """Parse a human-readable memory string to bytes.

    Supported formats (case-insensitive):
        "512M", "512MB", "512MiB"  → 512 * 1024^2
        "3G", "3GB", "3GiB"        → 3 * 1024^3
        "1.5G"                     → 1536 * 1024^2
        "2048M"                    → 2 * 1024^3
        "256K"                     → 256 * 1024
        "1T"                       → 1 * 1024^4

    Args:
        memory_str: Human-readable memory string.

    Returns:
        Integer number of bytes.

    Raises:
        ValueError: If the string cannot be parsed.
    """
    # Normalize: strip, uppercase, remove IB/B suffix (KiB→K, MiB→M, GiB→G, etc.)
    s = memory_str.strip().upper()
    s = s.replace("IB", "").replace("B", "")  # KiB→K, MiB→M, GiB→G, KB→K, MB→M, GB→G
    match = re.match(r'^(\d+(?:\.\d+)?)\s*([KMGT]?)$', s)
    if not match:
        raise ValueError(
            f"Cannot parse memory string: {memory_str!r}. "
            "Use formats like '512M', '3G', '1.5G', '2048M', '256K'."
        )
    value = float(match.group(1))
    unit = match.group(2)
    multipliers = {'': 1, 'K': 1024, 'M': 1024 ** 2, 'G': 1024 ** 3, 'T': 1024 ** 4}
    return int(value * multipliers[unit])


def _memory_bytes_to_gpu_fraction(memory_bytes: int, device_idx: int = 0) -> float:
    """Convert a byte count to a fractional GPU memory utilization value (0.0–1.0).

    Args:
        memory_bytes: Requested memory in bytes.
        device_idx: CUDA device index (default 0).

    Returns:
        Float in [0.0, 1.0], rounded to 4 decimal places.
    """
    total_memory = torch.cuda.get_device_properties(device_idx).total_memory
    return round(min(memory_bytes / total_memory, 1.0), 4)


# ---------------------------------------------------------------------------
# VllmOmniExecutor
# ---------------------------------------------------------------------------

class VllmOmniExecutor(ExecutorBase):
    """Executor for vllm-omni TTS models.

    Holds a vllm-omni `Omni` pipeline orchestrator that internally spawns
    CUDA worker subprocesses for multi-stage inference (e.g., LLM talker +
    vocoder). All input preparation, generation, and output extraction are
    self-contained in this class.

    Supported model_type: "VllmOmniTTS"
    """

    def _init_executor(self) -> None:
        """Initialize the vllm-omni Omni pipeline.

        Steps:
        1. Set VLLM_WORKER_MULTIPROC_METHOD=spawn (required by vllm-omni).
        2. If model_config.stage_memory is set, generate a patched stage config
           YAML with per-stage gpu_memory_utilization values.
        3. Pre-load and cache the model tokenizer for prompt length estimation.
        4. Instantiate Omni(model=..., stage_configs_path=...).
        """
        # 1. Required: vllm-omni workers must spawn (not fork) new processes
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

        # 2. Lazy imports (avoid importing torch.cuda before CUDA_VISIBLE_DEVICES is set)
        from vllm_omni.entrypoints.omni import Omni  # noqa: PLC0415

        omni_kwargs: dict = {}

        # 3. Handle per-stage memory override
        stage_memory = getattr(self.model_config, "stage_memory", None)
        if stage_memory:
            patched_path = self._generate_stage_config_with_memory(stage_memory)
            if patched_path:
                omni_kwargs["stage_configs_path"] = patched_path
                logger.info(
                    f"[VllmOmniExecutor] Using patched stage config: {patched_path}"
                )

        # 4. Pre-load tokenizer cache (used later in _estimate_prompt_len)
        self._tokenizer: Optional[object] = None   # lazy-loaded on first call
        self._talker_config: Optional[object] = None  # lazy-loaded on first call

        # 5. Instantiate Omni (this downloads weights and spawns stage workers)
        logger.info(
            f"[VllmOmniExecutor] Initializing Omni pipeline for model: "
            f"{self.model_config.model} (this may take several minutes)"
        )
        self.omni = Omni(model=self.model_config.model, **omni_kwargs)
        logger.info(
            f"[VllmOmniExecutor] Omni pipeline ready for {self.model_config.name} "
            f"({self.model_config.model})"
        )

    def _get_device_index(self) -> int:
        """Return the integer CUDA device index from device_config."""
        device = self.device_config.device
        if hasattr(device, "index") and device.index is not None:
            return device.index
        return 0

    def check_health(self) -> None:
        """Health check — Omni manages its own process health internally."""
        return

    # -----------------------------------------------------------------------
    # Helper methods — implemented in subsequent phases (2-2, 2-3, 2-4)
    # -----------------------------------------------------------------------

    def _generate_stage_config_with_memory(self, stage_memory: List[str]) -> Optional[str]:
        """Generate a patched vllm-omni stage YAML with overridden gpu_memory_utilization.
        Implemented in Phase 2-2.
        """
        raise NotImplementedError("Implemented in Phase 2-2")

    def _get_tokenizer(self):
        """Lazily load and cache the model tokenizer. Implemented in Phase 2-3."""
        raise NotImplementedError("Implemented in Phase 2-3")

    def _get_talker_config(self):
        """Lazily load and cache the Qwen3-TTS talker config. Implemented in Phase 2-3."""
        raise NotImplementedError("Implemented in Phase 2-3")

    def _estimate_prompt_len(self, additional_information: dict) -> int:
        """Estimate Qwen3-TTS prompt token placeholder length. Implemented in Phase 2-3."""
        raise NotImplementedError("Implemented in Phase 2-3")

    def _build_prompt(self, request_data) -> dict:
        """Build vllm-omni prompt dict from TextToSpeechRequestInput. Implemented in Phase 2-3."""
        raise NotImplementedError("Implemented in Phase 2-3")

    def execute_model(self, requests_to_batch: List[EngineRequest]) -> List[ExecutorResult]:
        """Execute a batch of TTS requests. Implemented in Phase 2-4."""
        raise NotImplementedError("Implemented in Phase 2-4")
```

## Notes

- All `raise NotImplementedError` stubs are replaced in subsequent phases (2-2, 2-3, 2-4). The file must be built incrementally: each phase adds its methods.
- `self._tokenizer` and `self._talker_config` are set to `None` in `_init_executor()` and populated lazily on first call to `_get_tokenizer()` / `_get_talker_config()`.
- The `stage_memory` field is read via `getattr(self.model_config, "stage_memory", None)` (safe for models without it).

## Progress Checklist

- [ ] Create `harmonyspeech/executor/vllm_omni_executor.py`
- [ ] Add module docstring with usage example
- [ ] Implement `_parse_memory_bytes()` module-level function
- [ ] Implement `_memory_bytes_to_gpu_fraction()` module-level function
- [ ] Implement `VllmOmniExecutor` class inheriting `ExecutorBase`
- [ ] Implement `_init_executor()` with all 5 steps
- [ ] Implement `_get_device_index()` helper
- [ ] Implement `check_health()` (no-op)
- [ ] Add stub `NotImplementedError` placeholders for phases 2-2, 2-3, 2-4 methods

# Phase 1: ModelConfig Extension

## Objective

Add `stage_memory` as an optional field to `ModelConfig` so that vllm-omni model configurations can specify per-stage GPU memory allocations in human-readable format (e.g., `"3G"`, `"512M"`).

## Background

`EngineConfig.load_config_from_yaml()` constructs `ModelConfig` instances by unpacking the entire YAML dict as `**kwargs`:
```python
model_configs.append(ModelConfig(device_config=device_config, **model_cfg))
```
Any YAML field not declared in `ModelConfig.__init__()` will cause a `TypeError`. Therefore, `stage_memory` must be explicitly declared.

## File to Modify

**`harmonyspeech/common/config.py`**

## Implementation Steps

### Step 1: Add `stage_memory` parameter to `ModelConfig.__init__()`

Locate the `__init__` signature and add the new parameter before the closing `) -> None:`. Place it after the existing `enforce_eager` parameter:

```python
def __init__(
    self,
    name: str,
    model: str,
    model_type: str,
    max_batch_size: int,
    device_config: DeviceConfig,
    language: Optional[str] = None,
    voices: Optional[List[str]] = None,
    trust_remote_code: Optional[bool] = False,
    download_dir: Optional[str] = None,
    load_format: Optional[str] = "auto",
    dtype: Optional[Union[str, torch.dtype]] = "bfloat16",
    seed: Optional[int] = 0,
    revision: Optional[str] = None,
    code_revision: Optional[str] = None,
    enforce_eager: bool = True,
    stage_memory: Optional[List[str]] = None,   # NEW: per-stage GPU memory, e.g. ["3G", "2G"]
) -> None:
```

### Step 2: Assign `stage_memory` in the constructor body

After `self.enforce_eager = enforce_eager`, add:

```python
self.stage_memory = stage_memory  # e.g. ["3G", "2G"] — one entry per vllm-omni stage
```

### Step 3: No validation needed

`stage_memory` is validated and parsed in `VllmOmniExecutor._generate_stage_config_with_memory()`, not here. This keeps `ModelConfig` generic and decoupled from vllm-omni specifics.

## Complete Modified `__init__` Method

```python
def __init__(
    self,
    name: str,
    model: str,
    model_type: str,
    max_batch_size: int,
    device_config: DeviceConfig,
    language: Optional[str] = None,
    voices: Optional[List[str]] = None,
    trust_remote_code: Optional[bool] = False,
    download_dir: Optional[str] = None,
    load_format: Optional[str] = "auto",
    dtype: Optional[Union[str, torch.dtype]] = "bfloat16",
    seed: Optional[int] = 0,
    revision: Optional[str] = None,
    code_revision: Optional[str] = None,
    enforce_eager: bool = True,
    stage_memory: Optional[List[str]] = None,
) -> None:
    self.name = name
    self.model = model
    self.model_type = model_type
    self.max_batch_size = max_batch_size
    self.device_config = device_config
    self.language = language
    self.voices = voices
    self.trust_remote_code = trust_remote_code
    self.download_dir = download_dir
    self.load_format = load_format
    self.dtype = dtype
    self.seed = seed
    self.revision = revision
    self.code_revision = code_revision
    self.enforce_eager = enforce_eager
    self.stage_memory = stage_memory  # NEW

    self.dtype = _get_and_verify_dtype(dtype)
    self._verify_load_format()
```

## Verification

After this change, the following YAML block parses successfully:
```yaml
- name: "qwen3-tts"
  model: "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
  model_type: "VllmOmniTTS"
  stage_memory: ["6G", "4G"]
  max_batch_size: 4
  dtype: "bfloat16"
  device_config:
    device: "cuda"
```

And existing model configs without `stage_memory` continue to work unchanged (it defaults to `None`).

## Progress Checklist

- [ ] Add `stage_memory: Optional[List[str]] = None` parameter to `ModelConfig.__init__()` signature
- [ ] Add `self.stage_memory = stage_memory` assignment in constructor body
- [ ] Verify existing config YAML files still load without errors

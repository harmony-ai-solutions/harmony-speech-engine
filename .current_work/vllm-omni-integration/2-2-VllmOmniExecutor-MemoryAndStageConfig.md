# Phase 2-2: VllmOmniExecutor — Memory Parsing and Stage Config Override

## Objective

Implement `_generate_stage_config_with_memory()` inside `VllmOmniExecutor`. This method:
1. Uses vllm-omni's own `resolve_model_config_path()` to find the base stage YAML for the model
2. Parses each `stage_memory` string (e.g., `"3G"`) to bytes, then to a GPU fraction
3. Patches `gpu_memory_utilization` per stage in the loaded YAML
4. Writes the result to a temp file and returns its path for use as `stage_configs_path` in `Omni()`

## Background

### How vllm-omni finds stage configs

`vllm_omni.entrypoints.utils.resolve_model_config_path(model)` works as follows:
1. Calls `get_config(model, trust_remote_code=True)` to load the model's HuggingFace `config.json`
2. Reads `config.model_type` (e.g., `"qwen3_tts"` for Qwen3-TTS models)
3. Returns the path to `vllm_omni/model_executor/stage_configs/{model_type}.yaml`

This means for `"Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"` it returns the path to `qwen3_tts.yaml`, and for `"FunAudioLLM/CosyVoice3-0.5B"` it returns `cosyvoice3.yaml`.

### Stage YAML structure (from `qwen3_tts.yaml`)

```yaml
async_chunk: true
stage_args:
  - stage_id: 0
    stage_type: llm
    runtime:
      devices: "0"
      max_batch_size: 10
    engine_args:
      gpu_memory_utilization: 0.3   # ← this is what we patch
      ...
  - stage_id: 1
    stage_type: llm
    runtime:
      devices: "0"
    engine_args:
      gpu_memory_utilization: 0.2   # ← and this
      ...
runtime:
  ...
```

We must patch only `stage_args[i]["engine_args"]["gpu_memory_utilization"]` for each stage. Everything else is preserved as-is.

### Example conversion on a 24GB GPU

| Input | Bytes | Fraction on 24GB (25,769,803,776 B) |
|---|---|---|
| `"6G"` | 6,442,450,944 | `0.25` |
| `"4G"` | 4,294,967,296 | `0.1667` |
| `"3G"` | 3,221,225,472 | `0.125` |
| `"512M"` | 536,870,912 | `0.0208` |

## File to Modify

**`harmonyspeech/executor/vllm_omni_executor.py`** — replace the `_generate_stage_config_with_memory` stub

## Implementation

Replace the stub `_generate_stage_config_with_memory` with:

```python
def _generate_stage_config_with_memory(self, stage_memory: List[str]) -> Optional[str]:
    """Load the model's default vllm-omni stage YAML, override gpu_memory_utilization
    per stage with values derived from human-readable memory strings, and write the
    result to a temporary YAML file.

    Args:
        stage_memory: List of memory strings, one per stage. E.g. ["6G", "4G"].
                      Entries beyond the number of stages in the YAML are ignored
                      with a warning.

    Returns:
        Path to the generated temporary YAML file, or None if the base stage config
        could not be found (in which case vllm-omni will use its own defaults).
    """
    from vllm_omni.entrypoints.utils import resolve_model_config_path  # noqa: PLC0415

    # 1. Locate the base stage YAML for this model
    base_yaml_path = resolve_model_config_path(self.model_config.model)
    if not base_yaml_path:
        logger.warning(
            f"[VllmOmniExecutor] Could not resolve stage config path for model "
            f"'{self.model_config.model}'. Ignoring stage_memory — using vllm-omni defaults."
        )
        return None

    logger.info(f"[VllmOmniExecutor] Base stage config: {base_yaml_path}")

    # 2. Convert memory strings to GPU fractions
    device_idx = self._get_device_index()
    fractions = []
    for i, mem_str in enumerate(stage_memory):
        try:
            mem_bytes = _parse_memory_bytes(mem_str)
            fraction = _memory_bytes_to_gpu_fraction(mem_bytes, device_idx)
            fractions.append(fraction)
            logger.info(
                f"[VllmOmniExecutor] stage_memory[{i}]: {mem_str} → "
                f"{mem_bytes / (1024**3):.2f} GiB → gpu_memory_utilization={fraction}"
            )
        except ValueError as e:
            logger.error(
                f"[VllmOmniExecutor] Invalid stage_memory[{i}]='{mem_str}': {e}. "
                "Using vllm-omni default for this stage."
            )
            fractions.append(None)  # None means: don't override this stage

    # 3. Load base YAML
    with open(base_yaml_path, "r") as f:
        config_data = yaml.safe_load(f)

    # 4. Patch gpu_memory_utilization per stage
    stage_args = config_data.get("stage_args", [])
    for i, fraction in enumerate(fractions):
        if i >= len(stage_args):
            logger.warning(
                f"[VllmOmniExecutor] stage_memory[{i}] specified but only "
                f"{len(stage_args)} stage(s) exist in the config — ignoring."
            )
            break
        if fraction is None:
            continue  # Skip stages where parsing failed
        if "engine_args" not in stage_args[i] or stage_args[i]["engine_args"] is None:
            stage_args[i]["engine_args"] = {}
        stage_args[i]["engine_args"]["gpu_memory_utilization"] = fraction

    # Write back (ensure stage_args is updated in config_data)
    config_data["stage_args"] = stage_args

    # 5. Write patched config to a named temp file
    #    delete=False: file persists until process exits or explicit cleanup
    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        prefix="hse_vllm_omni_",
        delete=False,
    )
    try:
        yaml.dump(config_data, tmp, default_flow_style=False, allow_unicode=True)
        tmp_path = tmp.name
    finally:
        tmp.close()

    logger.info(f"[VllmOmniExecutor] Patched stage config written to: {tmp_path}")
    return tmp_path
```

## Edge Cases

| Case | Behaviour |
|---|---|
| `resolve_model_config_path()` returns `None` (unknown model) | Log warning, return `None`, let Omni use built-in defaults |
| `stage_memory` has more entries than stages | Extra entries logged as warning, ignored |
| `stage_memory` has fewer entries than stages | Remaining stages keep their YAML defaults |
| Malformed memory string (e.g., `"abc"`) | Log error, skip that stage (keep YAML default) |
| Stage has no `engine_args` key | `engine_args` dict is created before inserting |
| Memory > total GPU VRAM | Fraction clamped to `1.0` by `_memory_bytes_to_gpu_fraction` |

## Temporary File Lifecycle

The temp file is created with `delete=False`, meaning it persists for the lifetime of the process. It is passed to `Omni(stage_configs_path=...)` which reads it during initialization. After `Omni` is initialized, the file is no longer needed but remains on disk.

**Optional cleanup** (not required for correctness, but good practice): If desired, the temp file path could be stored as `self._tmp_stage_config_path` and deleted in a `__del__` method. This is left as optional since `/tmp` is cleaned on reboot and files are small.

## Progress Checklist

- [ ] Replace `_generate_stage_config_with_memory` stub with full implementation
- [ ] Verify the function correctly parses `stage_memory: ["6G", "4G"]` on a 24GB GPU
- [ ] Verify it handles `stage_memory: ["3G"]` on a 2-stage model (only patches stage 0)
- [ ] Verify fallback when `resolve_model_config_path` returns `None`
- [ ] Verify the temp YAML is valid PyYAML output that vllm-omni can load

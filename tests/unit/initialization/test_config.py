"""Unit tests for harmonyspeech/common/config.py — DeviceConfig, ModelConfig, EngineConfig."""

import os

import pytest
import torch
import yaml

from harmonyspeech.common.config import DeviceConfig, EngineConfig, ModelConfig
from harmonyspeech.modeling.loader import _MODEL_CONFIGS
from harmonyspeech.task_handler.batching import (
    BATCHED_INFERENCE_FAMILIES,
    batched_inference_enabled,
    supports_batched_inference,
)

# ── DeviceConfig ──────────────────────────────────────────────────────────────


def test_device_config_explicit_cpu():
    cfg = DeviceConfig("cpu")
    assert cfg.device_type == "cpu"
    assert cfg.device == torch.device("cpu")


def test_device_config_auto_falls_back_to_cpu(monkeypatch):
    """When CUDA unavailable and is_cpu() returns True, device_type is 'cpu'."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("harmonyspeech.common.config.is_cpu", lambda: True)
    cfg = DeviceConfig("auto")
    assert cfg.device_type == "cpu"


def test_device_config_auto_no_device_raises(monkeypatch):
    """When neither CUDA nor CPU available, RuntimeError is raised."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("harmonyspeech.common.config.is_cpu", lambda: False)
    with pytest.raises(RuntimeError):
        DeviceConfig("auto")


# ── ModelConfig ───────────────────────────────────────────────────────────────


@pytest.fixture
def cpu_device():
    return DeviceConfig("cpu")


def test_model_config_valid_attributes(cpu_device):
    cfg = ModelConfig(
        name="test-tts",
        model="path/to/model",
        model_type="MeloTTSSynthesizer",
        max_batch_size=4,
        device_config=cpu_device,
        dtype="float32",
    )
    assert cfg.name == "test-tts"
    assert cfg.model == "path/to/model"
    assert cfg.model_type == "MeloTTSSynthesizer"
    assert cfg.max_batch_size == 4
    assert cfg.dtype == torch.float32
    assert cfg.load_format == "auto"
    assert cfg.enforce_eager is True


def test_model_config_dtype_float16(cpu_device):
    cfg = ModelConfig(
        name="t",
        model="m",
        model_type="MeloTTSSynthesizer",
        max_batch_size=1,
        device_config=cpu_device,
        dtype="float16",
    )
    assert cfg.dtype == torch.float16


def test_model_config_dtype_bfloat16(cpu_device):
    cfg = ModelConfig(
        name="t",
        model="m",
        model_type="MeloTTSSynthesizer",
        max_batch_size=1,
        device_config=cpu_device,
        dtype="bfloat16",
    )
    assert cfg.dtype == torch.bfloat16


def test_model_config_invalid_dtype_raises(cpu_device):
    with pytest.raises(ValueError, match="Unknown dtype"):
        ModelConfig(
            name="t",
            model="m",
            model_type="MeloTTSSynthesizer",
            max_batch_size=1,
            device_config=cpu_device,
            dtype="invalid_dtype",
        )


def test_model_config_invalid_load_format_raises(cpu_device):
    with pytest.raises(ValueError, match="Unknown load format"):
        ModelConfig(
            name="t",
            model="m",
            model_type="MeloTTSSynthesizer",
            max_batch_size=1,
            device_config=cpu_device,
            dtype="float32",
            load_format="garbage",
        )


def test_model_config_load_format_uppercase_normalized(cpu_device):
    """load_format is lowercased during validation."""
    cfg = ModelConfig(
        name="t",
        model="m",
        model_type="MeloTTSSynthesizer",
        max_batch_size=1,
        device_config=cpu_device,
        dtype="float32",
        load_format="AUTO",
    )
    assert cfg.load_format == "auto"


# ── EngineConfig ──────────────────────────────────────────────────────────────


def test_engine_config_to_dict(cpu_device):
    model_cfg = ModelConfig(
        name="t",
        model="m",
        model_type="MeloTTSSynthesizer",
        max_batch_size=1,
        device_config=cpu_device,
        dtype="float32",
    )
    eng_cfg = EngineConfig(model_configs=[model_cfg])
    d = eng_cfg.to_dict()
    assert "model_configs" in d
    assert len(d["model_configs"]) == 1


def test_engine_config_load_yaml_single_model(tmp_path):
    yaml_content = """
model_configs:
  - name: test-tts
    model: some/model
    model_type: MeloTTSSynthesizer
    max_batch_size: 4
    dtype: float32
    device_config:
      device: cpu
"""
    yaml_file = tmp_path / "config.yml"
    yaml_file.write_text(yaml_content)
    cfg = EngineConfig.load_config_from_yaml(str(yaml_file))
    assert len(cfg.model_configs) == 1
    mc = cfg.model_configs[0]
    assert mc.name == "test-tts"
    assert mc.model_type == "MeloTTSSynthesizer"
    assert mc.device_config.device_type == "cpu"
    assert mc.dtype == torch.float32


def test_engine_config_load_yaml_multiple_models(tmp_path):
    yaml_content = """
model_configs:
  - name: tts-model
    model: path/to/tts
    model_type: MeloTTSSynthesizer
    max_batch_size: 4
    dtype: float32
    device_config:
      device: cpu
  - name: stt-model
    model: path/to/stt
    model_type: FasterWhisper
    max_batch_size: 2
    dtype: float32
    device_config:
      device: cpu
"""
    yaml_file = tmp_path / "config_multi.yml"
    yaml_file.write_text(yaml_content)
    cfg = EngineConfig.load_config_from_yaml(str(yaml_file))
    assert len(cfg.model_configs) == 2
    names = [m.name for m in cfg.model_configs]
    assert "tts-model" in names
    assert "stt-model" in names


# ── Shipped config honesty: the execution-model class table (Phase 10-5) ─────
#
# One declarative table maps every registered model type to exactly one execution
# class (owner decision 2026-11-07, authoritative definition in 10-5). Each class
# has a semantic validator, so a new model type cannot ship without a deliberate
# capacity decision, and a shipped config cannot claim a capacity its executor
# does not honour. Phase 9-6 extends the table (it moves the English/Turbo
# Chatterbox roles into `ar_decode` when the shared KV pool lands).

#: The five execution classes and what their `max_batch_size` really means.
EXECUTION_CLASS_SEMANTICS: dict[str, dict] = {
    "padded_batch": {
        "execution": "one padded forward per length-similar sub-batch",
        "max_batch_size": "sub-batch width",
        "knobs": ("batched_inference", "batch_token_budget"),
    },
    "loop": {
        "execution": "per-request sequential",
        "max_batch_size": "must be 1",
        "knobs": (),
    },
    "queue_depth": {
        "execution": "admission/backpressure only",
        "max_batch_size": "queue depth",
        "knobs": (),
    },
    "lanes_windows": {
        "execution": "M parallel lanes; Whisper adds opt-in cross-request window batches",
        "max_batch_size": "admission/lane depth",
        "knobs": ("num_workers", "window batch cap"),
    },
    "ar_decode": {
        "execution": "token-level AR decode on the shared KV pool",
        "max_batch_size": "admission depth",
        "knobs": ("pool slots", "token budget"),
    },
}

#: Members whose capacity is a **lane count**, not a queue depth (KittenTTS's
#: ONNX graphs have a static batch dimension of 1; Phase 8-5).
LANE_ONLY_MEMBERS = {"KittenTTSSynthesizer": 4}

#: The English/Turbo Chatterbox roles are Phase 9's AR-runner members. Until
#: 9-6 lands they stay honest `loop` entries; `test_ar_decode_members_are_pending_phase_9`
#: is the assertion 9-6 flips (move them into `EXECUTION_MODEL_CLASSES` as
#: `ar_decode` and delete the future-member expectation).
FUTURE_AR_DECODE_MEMBERS = frozenset({"ChatterboxTTS", "ChatterboxTurboTTS"})

#: Every registered model type → exactly one class. `padded_batch` is derived
#: from the handler's own set so the two cannot drift; the rest are explicit.
EXECUTION_MODEL_CLASSES: dict[str, str] = {
    **dict.fromkeys(BATCHED_INFERENCE_FAMILIES, "padded_batch"),
    # Per-request loops. hs1-synthesizer is a measured padding skip (8-4); the
    # Chatterbox family is product-deferred (KV-cache workstream). Phase 9 will
    # move the English/Turbo roles to `ar_decode`; embedding/VC stay loops.
    "HarmonySpeechSynthesizer": "loop",
    "ChatterboxTTS": "loop",
    "ChatterboxTurboTTS": "loop",
    "ChatterboxMultilingualTTS": "loop",
    "ChatterboxEmbedding": "loop",
    "ChatterboxVC": "loop",
    # Single-stream VAD: the number is queue depth only (8-5).
    "SileroVAD": "queue_depth",
    # Parallel lanes. Whisper adds opt-in cross-request window batches on top of
    # its M CT2 workers; KittenTTS is lanes-only (one request per lane).
    "FasterWhisper": "lanes_windows",
    "KittenTTSSynthesizer": "lanes_windows",
}


def _shipped_config(name: str):
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    with open(os.path.join(repo_root, name), encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _validate_padded_batch(entry: dict, config_name: str) -> None:
    name = entry["name"]
    assert entry["max_batch_size"] > 1, f"{config_name}: {name} has a padded-batch forward"
    assert entry.get("batched_inference", True) is True, f"{config_name}: {name} must enable its padded forward"
    budget = entry.get("batch_token_budget")
    if budget is not None:
        assert isinstance(budget, int) and budget >= 1, f"{config_name}: {name} batch_token_budget"


def _validate_loop(entry: dict, config_name: str) -> None:
    name = entry["name"]
    assert entry["max_batch_size"] == 1, f"{config_name}: {name} admits one request per step"
    assert entry.get("batched_inference") is not True, f"{config_name}: {name} has no batch path"


def _validate_queue_depth(entry: dict, config_name: str) -> None:
    name = entry["name"]
    assert entry["max_batch_size"] >= 1, f"{config_name}: {name} queue depth"
    assert entry.get("batched_inference") is not True, f"{config_name}: {name} batches nothing"


def _validate_lanes_windows(entry: dict, config_name: str) -> None:
    name = entry["name"]
    assert entry["max_batch_size"] >= 1, f"{config_name}: {name} admission/lane depth"
    assert entry.get("batched_inference") in (None, True, False), f"{config_name}: {name}"
    if entry["model_type"] == "FasterWhisper":
        workers = entry.get("num_workers", 1)
        assert isinstance(workers, int) and workers >= 1, f"{config_name}: {name} num_workers"
        # The window cap is a module constant (10-2 deviation 3); it must stay a
        # real positive bound because admission counts true windows.
        from harmonyspeech.task_handler.whisper_batching import WHISPER_WINDOW_BATCH_SIZE

        assert WHISPER_WINDOW_BATCH_SIZE >= 1
    else:
        # Lanes-only member: the value *is* the lane count, not a queue depth.
        assert entry["max_batch_size"] == LANE_ONLY_MEMBERS[entry["model_type"]], f"{config_name}: {name}"


def _validate_ar_decode(entry: dict, config_name: str) -> None:
    # Phase 9's class: the number is admission depth on the shared KV pool.
    assert entry["max_batch_size"] >= 1, f"{config_name}: {entry['name']} admission depth"


CLASS_VALIDATORS = {
    "padded_batch": _validate_padded_batch,
    "loop": _validate_loop,
    "queue_depth": _validate_queue_depth,
    "lanes_windows": _validate_lanes_windows,
    "ar_decode": _validate_ar_decode,
}


@pytest.mark.unit
def test_execution_model_class_table_is_complete_and_declarative():
    """Every registered type maps to exactly one of the five classes."""
    assert set(EXECUTION_MODEL_CLASSES) == set(_MODEL_CONFIGS)
    assert set(EXECUTION_MODEL_CLASSES.values()) <= set(EXECUTION_CLASS_SEMANTICS)
    assert set(EXECUTION_CLASS_SEMANTICS) == {
        "padded_batch",
        "loop",
        "queue_depth",
        "lanes_windows",
        "ar_decode",
    }
    assert set(CLASS_VALIDATORS) == set(EXECUTION_CLASS_SEMANTICS)
    # The padded-batch class is the handler's own set, so the two cannot drift.
    assert {model_type for model_type, cls in EXECUTION_MODEL_CLASSES.items() if cls == "padded_batch"} == set(
        BATCHED_INFERENCE_FAMILIES
    )


@pytest.mark.unit
def test_ar_decode_members_are_pending_phase_9():
    """The AR class has no shipped member until 9-6 moves en/Turbo into it.

    Phase 9-6 updates this test together with the class map. Until then the
    English/Turbo Chatterbox roles are honest per-request loops (size 1), which
    is exactly what the shipped configs must say.
    """
    assert "ar_decode" not in EXECUTION_MODEL_CLASSES.values(), "Phase 9-6 must update this test with the class move"
    for model_type in FUTURE_AR_DECODE_MEMBERS:
        assert EXECUTION_MODEL_CLASSES[model_type] == "loop"
    for config_name in ("config.gpu.yml", "config.yml"):
        for entry in _shipped_config(config_name)["model_configs"]:
            if entry["model_type"] in FUTURE_AR_DECODE_MEMBERS:
                assert entry["max_batch_size"] == 1, f"{config_name}: {entry['name']}"


@pytest.mark.unit
@pytest.mark.parametrize("config_name", ["config.gpu.yml", "config.yml"])
def test_shipped_config_batch_sizes_match_executor_reality(config_name):
    """`max_batch_size` is a capacity signal: it must equal what the executor does.

    The policy is the five-class execution-model table above (Phase 10-5):

    * `padded_batch` — one padded forward per length-similar sub-batch;
    * `loop` — per-request sequential, so exactly 1;
    * `queue_depth` — admission/backpressure only;
    * `lanes_windows` — M parallel lanes and (Whisper) opt-in window batches;
    * `ar_decode` — token-level AR decode on the shared KV pool (Phase 9).

    Every registered type must appear exactly once, and every shipped entry must
    satisfy its class's validator. Exact values are cross-checked against
    `docs/batching.md` by `tests/unit/test_batching_docs_matrix.py`.
    """
    entries = _shipped_config(config_name)["model_configs"]

    for entry in entries:
        model_type = entry["model_type"]
        assert model_type in EXECUTION_MODEL_CLASSES, (
            f"{config_name}: {entry['name']} has no execution-model class decision"
        )
        cls = EXECUTION_MODEL_CLASSES[model_type]
        CLASS_VALIDATORS[cls](entry, config_name)


@pytest.mark.unit
@pytest.mark.parametrize("config_name", ["config.gpu.yml", "config.yml"])
def test_whisper_is_lanes_windows_with_its_real_knobs(config_name):
    """Whisper's class is `lanes_windows`, and its knobs are the real ones."""
    from harmonyspeech.task_handler.whisper_batching import WHISPER_WINDOW_BATCH_SIZE

    entries = [
        entry
        for entry in _shipped_config(config_name)["model_configs"]
        if entry["model_type"] == "FasterWhisper"
    ]
    assert entries, config_name
    for entry in entries:
        assert EXECUTION_MODEL_CLASSES["FasterWhisper"] == "lanes_windows"
        assert entry["max_batch_size"] == 16, f"{config_name}: {entry['name']} admission depth"
        assert entry.get("num_workers", 1) >= 1
        # The shipped configs keep the window batcher opt-in until the B=1
        # real-model parity gate runs (10-5 dated decision, 2026-11-07).
        assert entry.get("batched_inference") is not True, f"{config_name}: {entry['name']} opt-in until parity"
    assert WHISPER_WINDOW_BATCH_SIZE == 8, "owner-approved initial window cap (2026-11-07)"


@pytest.mark.unit
@pytest.mark.parametrize("config_name", ["config.gpu.yml", "config.yml"])
def test_shipped_config_loads_with_batched_inference_flags(config_name):
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    config = EngineConfig.load_config_from_yaml(os.path.join(repo_root, config_name))

    batched = [cfg for cfg in config.model_configs if supports_batched_inference(cfg)]
    assert batched, config_name
    assert all(batched_inference_enabled(cfg) for cfg in batched)
    # Phase 8-6: every family with a padded-batch forward now publishes its
    # honest admission capacity, so the flag must be on and the capacity real.
    for cfg in batched:
        assert cfg.max_batch_size > 1, cfg.name

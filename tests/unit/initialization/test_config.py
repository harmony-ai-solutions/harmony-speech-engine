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


# ── Shipped config honesty (Phase 5-3, finalised in 8-6) ──────────────────────

#: Model types whose step is a plain per-request loop: one request per engine
#: step, so `max_batch_size` is exactly 1. The HarmonySpeech synthesizer's
#: vendored ForwardTacotron is padding-unsafe (Phase 8-4 audit: 0.16 peak drift);
#: the whole Chatterbox family is out of scope by product decision (KV-cache
#: workstream, see `docs/batching.md`).
SINGLE_REQUEST_MODEL_TYPES = frozenset(
    {
        "HarmonySpeechSynthesizer",
        "ChatterboxTTS",
        "ChatterboxTurboTTS",
        "ChatterboxMultilingualTTS",
        "ChatterboxEmbedding",
        "ChatterboxVC",
    }
)

#: Non-batched steps that still admit more than one request per step. Here the
#: number is an admission width, a queue depth or a lane count — never a padded
#: batch width — and each value is the one `docs/batching.md` justifies.
CONCURRENCY_ONLY_MODEL_TYPES = {
    # `BatchedInferencePipeline.transcribe()` batches the 30 s clips *inside one
    # audio*; 16 is the scheduler's admission width (Phase 8-5 audit).
    "FasterWhisper": 16,
    # Sequential loop kept: `get_speech_timestamps` is single-stream and the
    # shared wrapper is not thread-safe; 16 is queue depth (Phase 8-5 verdict).
    "SileroVAD": 16,
    # Static ONNX batch dimension of 1 → one request per lane; 4 is the lane
    # count over the one shared session (Phase 8-5).
    "KittenTTSSynthesizer": 4,
}


def _shipped_config(name: str):
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    with open(os.path.join(repo_root, name), encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@pytest.mark.unit
@pytest.mark.parametrize("config_name", ["config.gpu.yml", "config.yml"])
def test_shipped_config_batch_sizes_match_executor_reality(config_name):
    """`max_batch_size` is a capacity signal: it must equal what the executor does.

    The policy has exactly three classes (Phase 8-6 finalised the values):

    * a model type with a real padded-batch forward (`BATCHED_INFERENCE_FAMILIES`)
      admits more than one request per step and has batching enabled;
    * a model type whose step is a per-request loop admits exactly one
      (`SINGLE_REQUEST_MODEL_TYPES`);
    * a non-batched step whose number means admission width, queue depth or lane
      count keeps that documented value (`CONCURRENCY_ONLY_MODEL_TYPES`).

    The three sets together must cover the loader registry, so a new model type
    cannot ship without a deliberate capacity decision. The values themselves
    must also agree word for word with `docs/batching.md`
    (`tests/unit/test_batching_docs_matrix.py` cross-checks the matrix).
    """
    data = _shipped_config(config_name)
    entries = data["model_configs"]

    covered = BATCHED_INFERENCE_FAMILIES | SINGLE_REQUEST_MODEL_TYPES | set(CONCURRENCY_ONLY_MODEL_TYPES)
    assert covered == set(_MODEL_CONFIGS)

    for entry in entries:
        name = entry["name"]
        model_type = entry["model_type"]
        size = entry["max_batch_size"]
        if model_type in BATCHED_INFERENCE_FAMILIES:
            assert size > 1, f"{config_name}: {name} has a padded-batch forward"
            assert entry.get("batched_inference", True) is True, name
        elif model_type in SINGLE_REQUEST_MODEL_TYPES:
            assert size == 1, f"{config_name}: {name} admits one request per step"
        else:
            assert model_type in CONCURRENCY_ONLY_MODEL_TYPES, name
            assert size == CONCURRENCY_ONLY_MODEL_TYPES[model_type], f"{config_name}: {name}"


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

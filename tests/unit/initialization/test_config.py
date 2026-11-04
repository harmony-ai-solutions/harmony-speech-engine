"""Unit tests for harmonyspeech/common/config.py — DeviceConfig, ModelConfig, EngineConfig."""

import pytest
import torch

from harmonyspeech.common.config import DeviceConfig, EngineConfig, ModelConfig

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

# ── Shipped config honesty (Phase 5-3) ────────────────────────────────────────

import os as _os

import yaml as _yaml

from harmonyspeech.task_handler.batching import batched_inference_enabled, supports_batched_inference


def _shipped_config(name: str):
    repo_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
    with open(_os.path.join(repo_root, name), encoding="utf-8") as handle:
        return _yaml.safe_load(handle)


@pytest.mark.unit
@pytest.mark.parametrize("config_name", ["config.gpu.yml", "config.yml"])
def test_shipped_config_batch_sizes_match_executor_reality(config_name):
    """`max_batch_size` is a capacity signal: it must equal what the executor does.

    * Chatterbox is autoregressive and runs one request per step -> 1.
    * One-shot synthesizers with a padded-batch forward -> > 1 (and the flag on).
    * Every remaining handler runs a per-item loop -> 1.
    * Whisper (BatchedInferencePipeline) and the cheap CPU loops keep their values.
    """
    data = _shipped_config(config_name)
    entries = data["model_configs"]
    by_name = {entry["name"]: entry for entry in entries}

    chatterbox = [entry for entry in entries if entry["name"].startswith("chatterbox")]
    assert chatterbox, f"{config_name}: no Chatterbox entries found"
    for entry in chatterbox:
        assert entry["max_batch_size"] == 1, entry["name"]

    loop_only = {
        "hs1-encoder",
        "hs1-synthesizer",
        "hs1-vocoder",
        "ov1-tone-converter",
        "ov1-tone-converter-encoder",
        "ov2-tone-converter",
        "ov2-tone-converter-encoder",
        "voicefixer-restorer",
        "voicefixer-vocoder",
    }
    for name in loop_only:
        entry = by_name.get(name)
        if entry is None:
            continue
        assert entry["max_batch_size"] == 1, f"{name} runs a per-item loop"

    for name, entry in by_name.items():
        if name.startswith(("ov1-synthesizer-", "ov2-synthesizer-")):
            assert entry["max_batch_size"] > 1, name
            assert entry.get("batched_inference") is True, name

    whisper = [entry for entry in entries if entry["model_type"] == "FasterWhisper"]
    for entry in whisper:
        assert entry["max_batch_size"] == 16, entry["name"]

    kitten = [entry for entry in entries if entry["model_type"] == "KittenTTSSynthesizer"]
    for entry in kitten:
        assert entry["max_batch_size"] == 4, entry["name"]

    silero = [entry for entry in entries if entry["model_type"] == "SileroVAD"]
    for entry in silero:
        assert entry["max_batch_size"] == 16, entry["name"]


@pytest.mark.unit
@pytest.mark.parametrize("config_name", ["config.gpu.yml", "config.yml"])
def test_shipped_config_loads_with_batched_inference_flags(config_name):
    repo_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
    config = EngineConfig.load_config_from_yaml(_os.path.join(repo_root, config_name))

    synth = [cfg for cfg in config.model_configs if supports_batched_inference(cfg)]
    assert synth, config_name
    assert all(batched_inference_enabled(cfg) for cfg in synth)
    for cfg in synth:
        assert cfg.max_batch_size > 1

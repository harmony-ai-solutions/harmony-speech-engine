"""Unit tests for harmonyspeech/common/config.py — DeviceConfig, ModelConfig, EngineConfig."""
import pytest
import torch
from harmonyspeech.common.config import DeviceConfig, ModelConfig, EngineConfig


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
        name="t", model="m", model_type="MeloTTSSynthesizer",
        max_batch_size=1, device_config=cpu_device, dtype="float16",
    )
    assert cfg.dtype == torch.float16


def test_model_config_dtype_bfloat16(cpu_device):
    cfg = ModelConfig(
        name="t", model="m", model_type="MeloTTSSynthesizer",
        max_batch_size=1, device_config=cpu_device, dtype="bfloat16",
    )
    assert cfg.dtype == torch.bfloat16


def test_model_config_invalid_dtype_raises(cpu_device):
    with pytest.raises(ValueError, match="Unknown dtype"):
        ModelConfig(
            name="t", model="m", model_type="MeloTTSSynthesizer",
            max_batch_size=1, device_config=cpu_device, dtype="invalid_dtype",
        )


def test_model_config_invalid_load_format_raises(cpu_device):
    with pytest.raises(ValueError, match="Unknown load format"):
        ModelConfig(
            name="t", model="m", model_type="MeloTTSSynthesizer",
            max_batch_size=1, device_config=cpu_device,
            dtype="float32", load_format="garbage",
        )


def test_model_config_load_format_uppercase_normalized(cpu_device):
    """load_format is lowercased during validation."""
    cfg = ModelConfig(
        name="t", model="m", model_type="MeloTTSSynthesizer",
        max_batch_size=1, device_config=cpu_device,
        dtype="float32", load_format="AUTO",
    )
    assert cfg.load_format == "auto"


# ── EngineConfig ──────────────────────────────────────────────────────────────

def test_engine_config_to_dict(cpu_device):
    model_cfg = ModelConfig(
        name="t", model="m", model_type="MeloTTSSynthesizer",
        max_batch_size=1, device_config=cpu_device, dtype="float32",
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

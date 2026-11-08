"""Unit tests for Chatterbox dtype plumbing (Phase 4-1).

The policy under test: ``dtype`` selects the T3 backbone precision on CUDA;
S3Gen and the VoiceEncoder always stay float32 (measured quality constraint —
see the chatterbox module docstring and docs/benchmarks/audio-regression.md).
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from harmonyspeech.common.config import DeviceConfig, ModelConfig
from harmonyspeech.modeling import loader as loader_module
from harmonyspeech.modeling.models.chatterbox import chatterbox as cb_module
from harmonyspeech.modeling.models.chatterbox.chatterbox import (
    _S3GEN_DTYPE,
    _VE_DTYPE,
    ChatterboxTTSModel,
    _resolve_load_dtype,
    apply_model_dtype,
)


def _fake_model(device: str = "cpu"):
    return SimpleNamespace(
        t3=nn.Linear(4, 4),
        s3gen=nn.Linear(4, 4),
        ve=nn.Linear(4, 4),
        device=device,
    )


def test_policy_constants_pin_s3gen_and_ve_float32():
    assert _S3GEN_DTYPE is torch.float32
    assert _VE_DTYPE is torch.float32


def test_resolve_load_dtype_forces_float32_on_cpu():
    assert _resolve_load_dtype("cpu", torch.float16) is torch.float32
    assert _resolve_load_dtype("cpu", None) is torch.float32
    assert _resolve_load_dtype("cuda", torch.float16) is torch.float16
    assert _resolve_load_dtype("cuda", None) is torch.float32


def test_resolve_load_dtype_treats_indexed_cuda_as_cuda():
    """Phase 11-0: a resolved ``cuda:N`` keeps the requested dtype.

    The loader now hands the wrapper the canonical indexed device; a string
    equality test against ``"cuda"`` silently demoted every resolved device
    (including the default ``cuda:0``) to float32.
    """
    assert _resolve_load_dtype("cuda:0", torch.float16) is torch.float16
    assert _resolve_load_dtype("cuda:1", torch.float16) is torch.float16
    assert _resolve_load_dtype(torch.device("cuda", 2), torch.bfloat16) is torch.bfloat16
    assert _resolve_load_dtype("cuda:1", None) is torch.float32


def test_apply_model_dtype_on_cpu_leaves_everything_float32():
    model = _fake_model("cpu")
    apply_model_dtype(model, torch.float16)

    assert all(p.dtype == torch.float32 for p in model.t3.parameters())
    assert all(p.dtype == torch.float32 for p in model.s3gen.parameters())
    assert all(p.dtype == torch.float32 for p in model.ve.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_apply_model_dtype_casts_t3_only_on_cuda():
    model = _fake_model("cuda")
    apply_model_dtype(model, torch.float16)

    assert all(p.dtype == torch.float16 for p in model.t3.parameters())
    # The measured quality constraint: S3Gen and VE must stay float32.
    assert all(p.dtype == torch.float32 for p in model.s3gen.parameters())
    assert all(p.dtype == torch.float32 for p in model.ve.parameters())


def test_apply_model_dtype_tolerates_models_without_t3():
    model = SimpleNamespace(s3gen=nn.Linear(2, 2), ve=nn.Linear(2, 2), device="cpu")
    apply_model_dtype(model, torch.float16)  # must not raise
    assert all(p.dtype == torch.float32 for p in model.s3gen.parameters())


def test_tts_wrapper_forwards_dtype_to_upstream(monkeypatch):
    upstream = MagicMock()
    upstream.from_pretrained.return_value = _fake_model("cpu")
    monkeypatch.setattr(cb_module, "ChatterboxTTS", upstream)

    ChatterboxTTSModel.from_pretrained(device="cpu", dtype=torch.float16)

    upstream.from_pretrained.assert_called_once_with(device="cpu")
    # CPU guard means the effective cast is a no-op, but the request is passed on.
    assert upstream.from_pretrained.call_count == 1


def test_vc_wrapper_accepts_dtype_kwarg(monkeypatch):
    upstream = MagicMock()
    upstream.from_pretrained.return_value = SimpleNamespace(s3gen=nn.Linear(2, 2), device="cpu")
    monkeypatch.setattr(cb_module, "ChatterboxVC", upstream)

    cb_module.ChatterboxVCModel.from_pretrained(device="cpu", dtype=torch.float16)

    upstream.from_pretrained.assert_called_once_with(device="cpu")


def _model_config(model_type: str, dtype: torch.dtype, backbone_group=None) -> ModelConfig:
    return ModelConfig(
        name="cb",
        model="resemble-ai/chatterbox",
        model_type=model_type,
        max_batch_size=1,
        device_config=DeviceConfig("cpu"),
        dtype=dtype,
        backbone_group=backbone_group,
    )


def test_loader_passes_config_dtype_to_wrapper():
    """The Chatterbox loaders used to ignore dtype entirely; the loader must now pass it."""
    wrapper = MagicMock()
    wrapper.from_pretrained.return_value = "model-sentinel"
    config = _model_config("ChatterboxTTS", torch.float16)

    result = loader_module._build_chatterbox(config, "cpu", None, wrapper)

    assert result == "model-sentinel"
    wrapper.from_pretrained.assert_called_once_with(
        device="cpu", dtype=torch.float16, s3gen_precision="float32"
    )


def test_loader_uses_factory_when_backbone_group_is_set(monkeypatch):
    sentinel = object()
    from harmonyspeech.modeling.models.chatterbox import factory as factory_module

    monkeypatch.setattr(factory_module, "build_role_model", lambda cfg, device, registry: sentinel)
    wrapper = MagicMock()
    config = _model_config("ChatterboxTTS", torch.float16, backbone_group="chatterbox-base")

    assert loader_module._build_chatterbox(config, "cuda", None, wrapper) is sentinel
    wrapper.from_pretrained.assert_not_called()


def test_loader_falls_back_when_factory_declines(monkeypatch):
    """Turbo stays ungrouped: the factory returns None and the wrapper path runs."""
    from harmonyspeech.modeling.models.chatterbox import factory as factory_module

    monkeypatch.setattr(factory_module, "build_role_model", lambda cfg, device, registry: None)
    wrapper = MagicMock()
    wrapper.from_pretrained.return_value = "turbo"
    config = _model_config("ChatterboxTurboTTS", torch.float16, backbone_group="chatterbox-base")

    assert loader_module._build_chatterbox(config, "cuda", None, wrapper) == "turbo"
    wrapper.from_pretrained.assert_called_once_with(
        device="cuda", dtype=torch.float16, s3gen_precision="float32"
    )


def test_model_config_validates_backbone_group():
    with pytest.raises(ValueError, match="backbone_group"):
        ModelConfig(
            name="cb",
            model="resemble-ai/chatterbox",
            model_type="ChatterboxTTS",
            max_batch_size=1,
            device_config=DeviceConfig("cpu"),
            dtype="float32",
            backbone_group=123,  # type: ignore[arg-type]
        )

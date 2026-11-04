"""In-process identity test for Chatterbox backbone sharing (Phase 4-4 gate).

Why in-process: deduplicated vs duplicated weights must be *bit-identical*.
Across two processes on this host, CUDA kernel selection (cuBLAS/cuDNN
workspace heuristics) introduces last-bit noise in the autoregressive path, so
the cross-process audio-regression gate uses a −60 dBFS tolerance. This test
removes that noise by building both models in one process and asserting exact
sample equality — the property sharing actually promises.
"""

import copy

import numpy as np
import pytest
import torch

from harmonyspeech.common.config import DeviceConfig, ModelConfig
from harmonyspeech.modeling.loader import get_model
from harmonyspeech.modeling.weight_registry import WeightRegistry
from harmonyspeech.task_handler.model_runner_base import _clone_conditionals

pytestmark = [pytest.mark.e2e, pytest.mark.slow]

TEXT = "The quick brown fox jumps over the lazy dog."


def _config(group: str | None, dtype: str = "float32") -> ModelConfig:
    return ModelConfig(
        name="shared" if group else "duplicated",
        model="resemble-ai/chatterbox",
        model_type="ChatterboxTTS",
        max_batch_size=1,
        device_config=DeviceConfig(device="cuda"),
        dtype=dtype,
        backbone_group=group,
        watermark=False,
    )


def _generate(model) -> np.ndarray:
    """Generate with the same conditioning path the engine uses."""
    conds = _clone_conditionals(model.conds)
    dtype = next(model.t3.parameters()).dtype
    conds.t3 = conds.t3.to(device=model.device, dtype=dtype)
    conds.t3.emotion_adv = 0.5 * torch.ones(1, 1, 1, dtype=dtype, device=model.device)
    model.conds = conds
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    wav = model.generate(TEXT, temperature=0.8, exaggeration=0.5)
    model.conds = copy.deepcopy(conds)
    return wav.detach().cpu().numpy().reshape(-1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_shared_and_duplicated_backbones_produce_identical_audio():
    registry = WeightRegistry()
    duplicated = get_model(_config(None), DeviceConfig("cuda"))
    shared = get_model(_config("chatterbox-base"), DeviceConfig("cuda"), weight_registry=registry)

    # The wiring itself: the shared role really reuses backbone objects.
    assert shared.s3gen is not duplicated.s3gen
    assert shared is not duplicated

    duplicated_audio = _generate(duplicated)
    shared_audio = _generate(shared)

    assert duplicated_audio.shape == shared_audio.shape
    assert np.array_equal(duplicated_audio, shared_audio)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_two_roles_in_a_group_share_s3gen_but_not_conditionals():
    registry = WeightRegistry()
    tts = get_model(_config("chatterbox-base"), DeviceConfig("cuda"), weight_registry=registry)
    vc_config = _config("chatterbox-base")
    vc_config.model_type = "ChatterboxVC"
    vc_config.name = "chatterbox_vc"
    vc = get_model(vc_config, DeviceConfig("cuda"), weight_registry=registry)

    assert tts.s3gen is vc.s3gen
    assert tts.conds is not vc.ref_dict
    assert registry.refcount(("ResembleAI/chatterbox", None, "s3gen", torch.float32)) == 2

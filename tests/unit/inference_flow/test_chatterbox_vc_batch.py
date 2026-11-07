"""Chatterbox VC batched vocode hop (Phase 9-5).

The VC vocode step is the same audited S3Gen graph as the TTS post-T3 hop, so a
multi-request step can batch it. The shipped ``max_batch_size: 1`` keeps the
family a loop (it stays ``loop``-class until 9-6), so these gates drive the
handler directly with a real tiny S3Gen and a stub tokenizer.

Pinned here: one batched forward for a multi-request step, per-request pairing,
the baseline ``ref_dict`` restore, and the per-item loop fallback on a batch
failure.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from harmonyspeech.common.artifacts import AudioArtifact
from harmonyspeech.common.config import DeviceConfig, ModelConfig
from harmonyspeech.common.inputs import VoiceConversionRequestInput
from harmonyspeech.common.outputs import VoiceConversionRequestOutput
from harmonyspeech.common.request import EngineRequest
from harmonyspeech.task_handler.model_runner_base import ModelRunnerBase
from tests.unit.modeling.test_chatterbox_s3gen_batch import tiny_s3gen


def _ref_dict(prompt_len=4, seed=123):
    generator = torch.Generator().manual_seed(seed)
    return {
        "prompt_token": torch.randint(0, 100, (1, prompt_len), generator=generator),
        "prompt_token_len": torch.tensor([prompt_len]),
        "prompt_feat": torch.randn(1, 2 * prompt_len, 80, generator=generator),
        "prompt_feat_len": None,
        "embedding": torch.randn(1, 192, generator=generator),
    }


class _StubTokenizer:
    """Deterministic speech tokens from the source clip's length."""

    def __call__(self, audio):
        samples = int(audio.shape[-1])
        count = max(2, min(8, samples // 1600))
        return torch.arange(2, 2 + count, dtype=torch.long).unsqueeze(0), None


class _StubWatermarker:
    def apply_watermark(self, wav, sample_rate):
        return wav


class _StubVCModel:
    def __init__(self):
        self.s3gen = tiny_s3gen()
        self.s3gen.tokenizer = _StubTokenizer()
        self.ref_dict = None
        self.sr = 24000
        self.device = "cpu"
        self.watermarker = _StubWatermarker()
        self.generate_calls = []

    def set_target_voice(self, audio_buf):
        self.ref_dict = _ref_dict()

    def generate(self, audio=None, **kwargs):
        self.generate_calls.append(audio)
        return torch.zeros(1, 64)


def _runner(model, *, max_batch_size=2, batched_inference=None, batch_token_budget=128):
    runner = ModelRunnerBase.__new__(ModelRunnerBase)
    runner.device = "cpu"
    runner.model = model
    runner.model_config = ModelConfig(
        name="cb-vc",
        model="dummy/chatterbox",
        model_type="ChatterboxVC",
        max_batch_size=max_batch_size,
        device_config=DeviceConfig("cpu"),
        dtype="float32",
        batched_inference=batched_inference,
        batch_token_budget=batch_token_budget,
    )
    runner._baseline_state = None
    runner._capture_baseline_state()
    return runner


def _request(request_id):
    request_data = VoiceConversionRequestInput(
        request_id=request_id,
        requested_model="chatterbox_vc",
        model="cb-vc",
        source_audio="c291cmNl",
        target_audio="dGFyZ2V0",
        target_embedding=None,
        generation_options=None,
        output_options=None,
    )
    return EngineRequest(request_id=request_id, request_data=request_data, arrival_time=0.0)


def _source(samples, seed):
    generator = np.random.RandomState(seed)
    return AudioArtifact.from_numpy(
        (generator.randn(samples).astype(np.float32) * 0.1), 16000
    )


@pytest.mark.unit
def test_multi_request_vc_step_vocodes_in_one_batched_forward(monkeypatch):
    import harmonyspeech.modeling.models.chatterbox.s3gen_batch as s3gen_batch

    model = _StubVCModel()
    runner = _runner(model)
    requests = [_request("vc-a"), _request("vc-b")]
    inputs = [
        (_source(16000, 1), None, _source(8000, 2)),
        (_source(12000, 3), None, _source(8000, 4)),
    ]

    calls = []
    real = s3gen_batch.inference_batched

    def recording(s3gen, rows, **kwargs):
        calls.append(len(rows))
        return real(s3gen, rows, **kwargs)

    monkeypatch.setattr(s3gen_batch, "inference_batched", recording)
    results = runner._execute_chatterbox_vc(inputs, requests)

    assert [result.request_id for result in results] == ["vc-a", "vc-b"]
    for result in results:
        assert isinstance(result.result_data, VoiceConversionRequestOutput)
        assert result.result_data.finish_reason == "stop"
        assert result.result_data.audio_artifact is not None
    assert calls == [2], calls
    # The per-request loop never ran, and the baseline was restored.
    assert model.generate_calls == []
    assert model.ref_dict is None


@pytest.mark.unit
def test_vc_batch_failure_falls_back_to_the_per_item_loop(monkeypatch):
    import harmonyspeech.modeling.models.chatterbox.s3gen_batch as s3gen_batch

    model = _StubVCModel()
    runner = _runner(model)
    requests = [_request("vc-a"), _request("vc-b")]
    inputs = [
        (_source(16000, 1), None, _source(8000, 2)),
        (_source(12000, 3), None, _source(8000, 4)),
    ]

    def broken(s3gen, rows, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(s3gen_batch, "inference_batched", broken)
    results = runner._execute_chatterbox_vc(inputs, requests)

    assert [result.request_id for result in results] == ["vc-a", "vc-b"]
    assert all(result.result_data.finish_reason == "stop" for result in results)
    assert len(model.generate_calls) == 2, "the loop is the fallback"
    assert model.ref_dict is None


@pytest.mark.unit
def test_batched_inference_false_keeps_the_vc_loop():
    model = _StubVCModel()
    runner = _runner(model, batched_inference=False)
    requests = [_request("vc-a"), _request("vc-b")]
    inputs = [
        (_source(16000, 1), None, _source(8000, 2)),
        (_source(12000, 3), None, _source(8000, 4)),
    ]

    results = runner._execute_chatterbox_vc(inputs, requests)

    assert all(result.result_data.finish_reason == "stop" for result in results)
    assert len(model.generate_calls) == 2

"""Chatterbox VC batching (Phase 9-5 vocode + Phase 11-1 tokenize).

The VC step is the same audited S3Gen graph as the TTS post-T3 hop, and since
Phase 11-1 its source tokenizer is batched too. These gates drive the handler
directly with a real tiny S3Gen and a stub tokenizer that implements the
audited mel/quantize split.

Pinned here: one batched tokenize pass + one batched vocode forward for a
multi-request step, per-request pairing, the baseline ``ref_dict`` restore, the
per-item loop fallback on a batch failure, seeded loop-vs-batch equality,
poisoned-row isolation, a cache miss inside a batched step, and a 4-request
mixed-length engine-style step.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from harmonyspeech.common.artifacts import AudioArtifact
from harmonyspeech.common.config import DeviceConfig, ModelConfig
from harmonyspeech.common.inputs import TextToSpeechGenerationOptions, VoiceConversionRequestInput
from harmonyspeech.common.outputs import VoiceConversionRequestOutput
from harmonyspeech.common.request import EngineRequest
from harmonyspeech.processing.embedding_cache import EmbeddingCache
from harmonyspeech.task_handler.model_runner_base import ModelRunnerBase
from tests.unit.modeling.test_chatterbox_s3gen_batch import tiny_s3gen


def _ref_dict(prompt_len=4, seed=123):
    generator = torch.Generator().manual_seed(seed)
    return {
        "prompt_token": torch.randint(0, 100, (1, prompt_len), generator=generator),
        "prompt_token_len": torch.tensor([prompt_len]),
        "prompt_feat": torch.randn(1, 2 * prompt_len + 1, 80, generator=generator),  # real embed_ref shape
        "prompt_feat_len": None,
        "embedding": torch.randn(1, 192, generator=generator),
    }


class _StubTokenizer:
    """Deterministic tokenizer with the audited mel + quantize split.

    The token count is derived from the clip's sample count, identically for the
    single-clip ``__call__`` (the loop path) and the batched ``quantize`` (the
    Phase 11-1 path), so token equality is meaningful.
    """

    device = "cpu"

    @staticmethod
    def _count(samples: int) -> int:
        return max(2, min(8, samples // 1600))

    def log_mel_spectrogram(self, audio):
        # [1, T] -> [1, 1, T]: one "mel bin" whose frame count is the sample count.
        return audio[:, None, :]

    def quantize(self, mels, mel_lens):
        counts = [self._count(int(value)) for value in mel_lens]
        width = max(counts)
        tokens = torch.zeros(len(counts), width, dtype=torch.long)
        for row, count in enumerate(counts):
            tokens[row, :count] = torch.arange(2, 2 + count)
        return tokens, torch.tensor(counts, dtype=torch.long)

    def __call__(self, audio):
        count = self._count(int(audio.shape[-1]))
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


class _RealVocodeVCModel(_StubVCModel):
    """Loop path that mirrors upstream ``ChatterboxVC.generate`` (no watermark)."""

    def generate(self, audio=None, **kwargs):
        import librosa
        from chatterbox.models.s3tokenizer import S3_SR

        self.generate_calls.append(audio)
        audio_16, _ = librosa.load(audio, sr=S3_SR)
        audio_16 = torch.from_numpy(audio_16).float()[None]
        tokens, _ = self.s3gen.tokenizer(audio_16)
        wav, _ = self.s3gen.inference(speech_tokens=tokens, ref_dict=self.ref_dict)
        return wav


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


def _request(request_id, seed=None):
    options = (
        None
        if seed is None
        else TextToSpeechGenerationOptions(
            seed=seed, style=None, speed=None, pitch=None, energy=None
        )
    )
    request_data = VoiceConversionRequestInput(
        request_id=request_id,
        requested_model="chatterbox_vc",
        model="cb-vc",
        source_audio="c291cmNl",
        target_audio="dGFyZ2V0",
        target_embedding=None,
        generation_options=options,
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
def test_multi_request_vc_step_tokenizes_in_one_batched_pass(monkeypatch):
    import harmonyspeech.modeling.models.chatterbox.s3gen_batch as s3gen_batch

    model = _StubVCModel()
    runner = _runner(model)
    requests = [_request("vc-a"), _request("vc-b")]
    inputs = [
        (_source(16000, 1), None, _source(8000, 2)),
        (_source(12000, 3), None, _source(8000, 4)),
    ]

    calls = []
    real = s3gen_batch.tokenize_batched

    def recording(tokenizer, clips):
        calls.append(len(clips))
        return real(tokenizer, clips)

    monkeypatch.setattr(s3gen_batch, "tokenize_batched", recording)
    results = runner._execute_chatterbox_vc(inputs, requests)

    assert all(result.result_data.finish_reason == "stop" for result in results)
    assert calls == [2], calls


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
def test_tokenizer_failure_falls_back_to_the_per_item_loop(monkeypatch):
    import harmonyspeech.modeling.models.chatterbox.s3gen_batch as s3gen_batch

    model = _StubVCModel()
    runner = _runner(model)
    requests = [_request("vc-a"), _request("vc-b")]
    inputs = [
        (_source(16000, 1), None, _source(8000, 2)),
        (_source(12000, 3), None, _source(8000, 4)),
    ]

    def broken(tokenizer, clips):
        raise RuntimeError("tokenizer boom")

    monkeypatch.setattr(s3gen_batch, "tokenize_batched", broken)
    results = runner._execute_chatterbox_vc(inputs, requests)

    assert all(result.result_data.finish_reason == "stop" for result in results)
    assert len(model.generate_calls) == 2, "the loop is the fallback"


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


@pytest.mark.unit
def test_seeded_rows_are_identical_alone_or_batched():
    """A seeded VC request's audio is the same on the loop and the batched path."""
    requests = [_request("vc-a", seed=11), _request("vc-b", seed=22)]
    inputs = [
        (_source(16000, 1), None, _source(8000, 2)),
        (_source(12000, 3), None, _source(8000, 4)),
    ]

    loop_model = _RealVocodeVCModel()
    loop_runner = _runner(loop_model, max_batch_size=1, batched_inference=False)
    loop_results = loop_runner._execute_chatterbox_vc(inputs, requests)

    batch_model = _RealVocodeVCModel()
    batch_runner = _runner(batch_model, max_batch_size=2)
    batch_results = batch_runner._execute_chatterbox_vc(inputs, requests)

    for loop_result, batch_result in zip(loop_results, batch_results, strict=True):
        assert loop_result.request_id == batch_result.request_id
        loop_audio = loop_result.result_data.audio_artifact.numpy
        batch_audio = batch_result.result_data.audio_artifact.numpy
        assert loop_audio.shape == batch_audio.shape
        # Batched kernels accumulate slightly differently; the 9-5 floor is 2e-7.
        assert np.allclose(loop_audio, batch_audio, atol=1e-6), loop_result.request_id


@pytest.mark.unit
def test_poisoned_vc_row_errors_alone():
    """One bad target does not de-batch its neighbours (whole-step loop fallback)."""
    model = _StubVCModel()
    runner = _runner(model)
    requests = [_request("vc-good"), _request("vc-poisoned")]
    # The poisoned row has neither target audio nor embedding -> prep raises,
    # the whole step degrades to the loop, and only that request errors.
    inputs = [
        (_source(16000, 1), None, _source(8000, 2)),
        (_source(12000, 3), None, None),
    ]

    results = runner._execute_chatterbox_vc(inputs, requests)

    assert [result.request_id for result in results] == ["vc-good", "vc-poisoned"]
    by_id = {result.request_id: result for result in results}
    assert by_id["vc-good"].result_data.finish_reason == "stop"
    assert by_id["vc-poisoned"].result_data.finish_reason == "error"


@pytest.mark.unit
def test_cache_miss_inside_a_batched_step_computes_and_stores_the_target():
    model = _StubVCModel()
    runner = _runner(model)
    runner.model_identity = "cb-vc@1"
    runner.embedding_cache = EmbeddingCache()
    requests = [_request("vc-a"), _request("vc-b")]
    inputs = [
        (_source(16000, 1), None, _source(8000, 2)),
        (_source(12000, 3), None, _source(8000, 4)),
    ]

    results = runner._execute_chatterbox_vc(inputs, requests)

    assert all(result.result_data.finish_reason == "stop" for result in results)
    # The first batched step missed and computed both targets inside the batch.
    assert runner.embedding_cache.stores == 2
    assert model.generate_calls == []

    # A second identical step is served from the cache (no set_target_voice).
    model.set_target_voice_calls = 0
    real_set_target = model.set_target_voice

    def counting(audio_buf):
        model.set_target_voice_calls += 1
        return real_set_target(audio_buf)

    model.set_target_voice = counting
    second = runner._execute_chatterbox_vc(inputs, requests)
    assert all(result.result_data.finish_reason == "stop" for result in second)
    assert model.set_target_voice_calls == 0


@pytest.mark.unit
def test_mixed_length_engine_step_pairs_results_in_order():
    model = _StubVCModel()
    runner = _runner(model, max_batch_size=4)
    requests = [_request(f"vc-{index}") for index in range(4)]
    inputs = [
        (_source(samples, seed), None, _source(8000, 100 + seed))
        for samples, seed in ((16000, 1), (32000, 2), (9600, 3), (48000, 4))
    ]

    results = runner._execute_chatterbox_vc(inputs, requests)

    assert [result.request_id for result in results] == [f"vc-{index}" for index in range(4)]
    for result in results:
        assert result.result_data.finish_reason == "stop"
        assert result.result_data.audio_artifact is not None

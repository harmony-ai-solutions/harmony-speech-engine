"""Engine-level Chatterbox VC batching (Phase 11-1).

Runs the *real* engine (scheduler → executor → ``prepare_inputs`` → handler →
result assembly) with a stub VC model bound to a real tiny S3Gen and a stub
tokenizer that implements the audited mel/quantize split. Hermetic and CPU-only.

Pinned here: 4 mixed-length VC requests complete in one batched step with the
right request→audio pairing (the tiny HiFT's output length encodes the token
count), a poisoned source row errors alone while its neighbours finish, and a
cache miss inside a batched step is computed and then served from the cache.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
import torch

from harmonyspeech.common.config import DeviceConfig, ModelConfig
from harmonyspeech.common.inputs import VoiceConversionRequestInput
from harmonyspeech.common.outputs import VoiceConversionRequestOutput
from harmonyspeech.common.request import EngineRequest
from harmonyspeech.engine.harmonyspeech_engine import HarmonySpeechEngine
from harmonyspeech.processing.embedding_cache import EmbeddingCache
from harmonyspeech.task_handler.model_runner_base import ModelRunnerBase
from tests.audio_fixtures import tiny_wav_b64
from tests.unit.modeling.test_chatterbox_s3gen_batch import tiny_s3gen


class _StubTokenizer:
    """Deterministic tokenizer with the audited split; can poison one clip length."""

    device = "cpu"

    def __init__(self, poison_samples: int | None = None):
        self.poison_samples = poison_samples

    @staticmethod
    def _count(samples: int) -> int:
        return max(2, min(8, samples // 1600))

    def _check(self, samples: int) -> None:
        if self.poison_samples is not None and int(samples) == self.poison_samples:
            raise RuntimeError("poisoned source clip")

    def log_mel_spectrogram(self, audio):
        self._check(audio.shape[-1])
        return audio[:, None, :]

    def quantize(self, mels, mel_lens):
        counts = [self._count(int(value)) for value in mel_lens]
        width = max(counts)
        tokens = torch.zeros(len(counts), width, dtype=torch.long)
        for row, count in enumerate(counts):
            tokens[row, :count] = torch.arange(2, 2 + count)
        return tokens, torch.tensor(counts, dtype=torch.long)

    def __call__(self, audio):
        self._check(audio.shape[-1])
        count = self._count(int(audio.shape[-1]))
        return torch.arange(2, 2 + count, dtype=torch.long).unsqueeze(0), None


class _IdentityWatermarker:
    def apply_watermark(self, wav, sample_rate):
        return wav


def _ref_dict(prompt_len=4, seed=123):
    generator = torch.Generator().manual_seed(seed)
    return {
        "prompt_token": torch.randint(0, 100, (1, prompt_len), generator=generator),
        "prompt_token_len": torch.tensor([prompt_len]),
        "prompt_feat": torch.randn(1, 2 * prompt_len + 1, 80, generator=generator),  # real embed_ref shape
        "prompt_feat_len": None,
        "embedding": torch.randn(1, 192, generator=generator),
    }


class _VCStubModel:
    """Real tiny S3Gen + stub tokenizer; the loop path mirrors upstream VC."""

    sr = 24000

    def __init__(self, poison_samples: int | None = None):
        self.s3gen = tiny_s3gen()
        self.s3gen.tokenizer = _StubTokenizer(poison_samples)
        self.ref_dict = None
        self.device = "cpu"
        self.watermarker = _IdentityWatermarker()
        self.set_target_voice_calls = 0

    def set_target_voice(self, audio_buf):
        self.set_target_voice_calls += 1
        self.ref_dict = _ref_dict()

    def generate(self, audio=None, **kwargs):
        import librosa
        from chatterbox.models.s3tokenizer import S3_SR

        audio_16, _ = librosa.load(audio, sr=S3_SR)
        audio_16 = torch.from_numpy(audio_16).float()[None]
        tokens, _ = self.s3gen.tokenizer(audio_16)
        wav, _ = self.s3gen.inference(speech_tokens=tokens, ref_dict=self.ref_dict)
        return wav


class _StubExecutor:
    def __init__(self, runner):
        self.runner = runner

    def execute_model(self, requests_to_batch):
        return self.runner.execute_model(requests_to_batch)

    async def execute_model_async(self, requests_to_batch):
        return self.runner.execute_model(requests_to_batch)

    def check_health(self):
        return {"status": "ok"}


def _runner(model, *, max_batch_size=4, embedding_cache=None):
    runner = ModelRunnerBase.__new__(ModelRunnerBase)
    runner.device = "cpu"
    runner.model = model
    runner.model_config = ModelConfig(
        name="chatterbox_vc",
        model="dummy/chatterbox",
        model_type="ChatterboxVC",
        max_batch_size=max_batch_size,
        device_config=DeviceConfig("cpu"),
        dtype="float32",
        batched_inference=True,
        batch_token_budget=256,
    )
    runner._baseline_state = None
    runner._capture_baseline_state()
    if embedding_cache is not None:
        runner.model_identity = "chatterbox_vc@1"
        runner.embedding_cache = embedding_cache
    return runner


def _engine(runner):
    with patch("harmonyspeech.executor.cpu_executor.CPUExecutor._init_worker"):
        engine = HarmonySpeechEngine(model_configs=[runner.model_config], log_stats=False)
    engine.model_executors = {"chatterbox_vc": _StubExecutor(runner)}
    return engine


def _vc_request(request_id: str, source_b64: str, target_b64: str) -> EngineRequest:
    request_data = VoiceConversionRequestInput(
        request_id=request_id,
        requested_model="chatterbox_vc",
        model="chatterbox_vc",
        source_audio=source_b64,
        target_audio=target_b64,
        target_embedding=None,
        generation_options=None,
        output_options=None,
    )
    return EngineRequest(request_id=request_id, request_data=request_data, arrival_time=0.0)


def _drive(engine, max_steps: int = 8) -> dict[str, list]:
    collected: dict[str, list] = {}
    for _ in range(max_steps):
        if not engine.has_unfinished_requests():
            break
        outputs, _forwarded = engine.step()
        for output in outputs:
            collected.setdefault(output.request_id, []).append(output)
    return collected


def _expected_samples(source_seconds: float) -> int:
    """Output length for the tiny HiFT geometry with the real ref shape.

    The prompt mel is ``2 * prompt_len + 1`` frames (upstream ``embed_ref``
    shape), so a ``count``-token source leaves ``2 * count - 1`` generated mel
    frames; the tiny HiFT turns each frame into ``up_factor * hop = 8`` samples.
    """
    tokens = _StubTokenizer._count(int(16000 * source_seconds))
    return (2 * tokens - 1) * 8


@pytest.mark.integration
def test_four_mixed_length_vc_requests_batch_in_one_step_with_correct_pairing():
    model = _VCStubModel()
    runner = _runner(model, max_batch_size=4)
    engine = _engine(runner)

    # Distinct source lengths -> distinct stub token counts -> distinct output
    # lengths; that is the pairing marker (a mispaired row would carry its
    # neighbour's length).
    seconds = {"vc-a": 0.2, "vc-b": 0.4, "vc-c": 0.6, "vc-d": 0.8}
    target = tiny_wav_b64(16000, 0.5)
    for request_id, duration in seconds.items():
        request = _vc_request(request_id, tiny_wav_b64(16000, duration), target)
        engine.add_request(request.request_id, request.request_data)

    outputs = _drive(engine)

    assert sorted(outputs) == sorted(seconds)
    for request_id, duration in seconds.items():
        assert len(outputs[request_id]) == 1
        output = outputs[request_id][0]
        assert output.finish_reason == "stop"
        assert isinstance(output, VoiceConversionRequestOutput)
        assert output.audio_artifact.num_samples == _expected_samples(duration), request_id
    # The four requests reached the executor in one step (one batched hop).
    assert model.set_target_voice_calls == 4


@pytest.mark.integration
def test_poisoned_vc_row_errors_alone_while_its_neighbours_finish():
    poison_seconds = 0.8
    model = _VCStubModel(poison_samples=int(16000 * poison_seconds))
    runner = _runner(model, max_batch_size=4)
    engine = _engine(runner)

    seconds = {"vc-a": 0.2, "vc-b": 0.4, "vc-poisoned": poison_seconds, "vc-d": 0.6}
    target = tiny_wav_b64(16000, 0.5)
    for request_id, duration in seconds.items():
        request = _vc_request(request_id, tiny_wav_b64(16000, duration), target)
        engine.add_request(request.request_id, request.request_data)

    outputs = _drive(engine)

    assert sorted(outputs) == sorted(seconds)
    by_id = {request_id: entries[0] for request_id, entries in outputs.items()}
    assert by_id["vc-poisoned"].finish_reason == "error"
    for request_id in ("vc-a", "vc-b", "vc-d"):
        assert by_id[request_id].finish_reason == "stop"
        assert by_id[request_id].audio_artifact.num_samples == _expected_samples(
            seconds[request_id]
        )


@pytest.mark.integration
def test_cache_miss_inside_a_batched_step_is_computed_then_served_from_the_cache():
    model = _VCStubModel()
    cache = EmbeddingCache()
    runner = _runner(model, max_batch_size=4, embedding_cache=cache)
    engine = _engine(runner)

    target = tiny_wav_b64(16000, 0.5)
    first = {
        f"first-{index}": duration
        for index, duration in enumerate((0.2, 0.4, 0.6, 0.8))
    }
    for request_id, duration in first.items():
        request = _vc_request(request_id, tiny_wav_b64(16000, duration), target)
        engine.add_request(request.request_id, request.request_data)

    outputs = _drive(engine)
    assert all(entries[0].finish_reason == "stop" for entries in outputs.values())
    assert cache.stores == 4, "the four target misses were computed inside the batch"
    assert model.set_target_voice_calls == 4

    # A second step with the same target audio is a cache hit: no target compute.
    second = {f"second-{index}": duration for index, duration in enumerate((0.2, 0.4, 0.6, 0.8))}
    for request_id, duration in second.items():
        request = _vc_request(request_id, tiny_wav_b64(16000, duration), target)
        engine.add_request(request.request_id, request.request_data)

    second_outputs = _drive(engine)
    assert all(entries[0].finish_reason == "stop" for entries in second_outputs.values())
    assert model.set_target_voice_calls == 4, "cache hits skip set_target_voice"

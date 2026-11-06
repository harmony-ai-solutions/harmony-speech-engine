"""Unit tests for the batched tone-converter handler (Phase 8-3).

``_execute_openvoice_tone_converter`` now flattens a step into one row per
request (its own synthesizer output audio), runs the mel frontend per clip (the
frontend reflect-pads, so it cannot be batched bit-faithfully), pads the mel
frames, and forwards one ``voice_conversion_batched`` call per length-similar
sub-batch. What is proven here:

* four mixed-length requests share **one** batched forward;
* per-request outputs stay paired (deterministic stub model: batched == loop);
* a single request keeps the loop fast path, and ``batched_inference: false``
  never reaches the batched entry point;
* a poisoned request (missing embedding) errors alone, and a failing batched
  forward degrades to the loop.

Sub-batch grouping is the production rule (length similarity + per-step padded
area budget from :func:`default_token_budget`); the "one forward" assertions pass
a deliberately generous ``batch_token_budget`` so the expectation is about row
flattening, not budget arithmetic (which has its own tests).
"""

import types
from unittest.mock import patch

import numpy as np
import pytest
import torch

from harmonyspeech.common.artifacts import AudioArtifact, SpeakerArtifact
from harmonyspeech.common.config import DeviceConfig, ModelConfig
from harmonyspeech.common.inputs import TextToSpeechRequestInput
from harmonyspeech.common.request import EngineRequest
from harmonyspeech.modeling.models.openvoice.openvoice import SynthesizerTrn
from harmonyspeech.task_handler.model_runner_base import ModelRunnerBase

SAMPLE_RATE = 16000
FILTER_LENGTH = 16
HOP_LENGTH = 4
WIN_LENGTH = 16
GIN_CHANNELS = 4
#: prod(upsample_rates) of :func:`_tiny_tone_converter`.
SPF = 4
#: Big enough that length-similarity, not the padded-area budget, decides groups.
ROOMY_BUDGET = 1_000_000


def _fake_hf_config():
    return types.SimpleNamespace(
        data=types.SimpleNamespace(
            sampling_rate=SAMPLE_RATE,
            filter_length=FILTER_LENGTH,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
        )
    )


def _tiny_tone_converter() -> SynthesizerTrn:
    torch.manual_seed(0)
    return SynthesizerTrn(
        n_vocab=64,
        spec_channels=FILTER_LENGTH // 2 + 1,
        inter_channels=8,
        hidden_channels=8,
        filter_channels=8,
        n_heads=2,
        n_layers=1,
        kernel_size=3,
        p_dropout=0.0,
        resblock="1",
        resblock_kernel_sizes=[3],
        resblock_dilation_sizes=[[1, 3, 1]],
        upsample_rates=[2, 2],
        upsample_initial_channel=8,
        upsample_kernel_sizes=[4, 4],
        n_speakers=0,
        gin_channels=GIN_CHANNELS,
    ).eval()


def _runner(model, *, batched=True, **config) -> ModelRunnerBase:
    runner = ModelRunnerBase.__new__(ModelRunnerBase)
    runner.device = "cpu"
    runner.device_config = DeviceConfig("cpu")
    runner._baseline_state = None
    runner.model = model
    runner.embedding_cache = None
    runner.model_identity = None
    runner.model_config = ModelConfig(
        name="ov2-tone-converter",
        model="dummy/openvoice",
        model_type="OpenVoiceV2ToneConverter",
        max_batch_size=config.pop("max_batch_size", 8),
        device_config=DeviceConfig("cpu"),
        dtype="float32",
        batched_inference=batched,
        **config,
    )
    return runner


def _clip(samples: int, freq: float = 220.0) -> AudioArtifact:
    t = np.arange(samples) / SAMPLE_RATE
    return AudioArtifact.from_numpy((0.1 * np.sin(2 * np.pi * freq * t)).astype(np.float32), SAMPLE_RATE)


def _speaker(value: float) -> SpeakerArtifact:
    return SpeakerArtifact(embedding=torch.full((1, GIN_CHANNELS, 1), value), backend="openvoice")


def _triple(samples: int, freq: float, target: float, source: float):
    return (_clip(samples, freq), _speaker(target), _speaker(source))


def _request(request_id: str) -> EngineRequest:
    request_data = TextToSpeechRequestInput(
        request_id=request_id,
        requested_model="ov2-tone-converter",
        model="ov2-tone-converter",
        input_text="hello",
        mode="voice_cloning",
    )
    return EngineRequest(request_id=request_id, request_data=request_data, arrival_time=0.0)


def _run(runner, inputs, requests):
    with patch(
        "harmonyspeech.task_handler.model_runner_base.get_model_config", return_value=_fake_hf_config()
    ), patch("harmonyspeech.task_handler.model_runner_base.get_model_flavour", return_value="default"):
        return runner._execute_openvoice_tone_converter(inputs, requests)


def _mixed_batch():
    """Four distinct clips -> mel frames 500, 600, 700, 800 at this geometry."""
    clips = [
        _triple(2000, 220.0, 0.1, 0.2),
        _triple(2400, 330.0, 0.3, 0.4),
        _triple(2800, 440.0, 0.5, 0.6),
        _triple(3200, 550.0, 0.7, 0.8),
    ]
    requests = [_request(f"tc-{index}") for index in range(len(clips))]
    return clips, requests


class _RecordingToneConverter:
    """A real tiny converter whose two entry points are recorded (and can fail)."""

    def __init__(self, model: SynthesizerTrn, *, fail_batched: bool = False):
        self.dec = model.dec
        self.fail_batched = fail_batched
        self.batched_calls: list[list[int]] = []
        self.loop_calls: list[int] = []
        original_batched = model.voice_conversion_batched
        original_single = model.voice_conversion

        def _batched(y, y_lengths, sid_src, sid_tgt, tau=1.0):
            self.batched_calls.append([int(value) for value in y_lengths])
            if self.fail_batched:
                raise RuntimeError("simulated batched-forward failure")
            return original_batched(y, y_lengths, sid_src, sid_tgt, tau=tau)

        def _single(y, y_lengths, sid_src, sid_tgt, tau=1.0):
            self.loop_calls.append(int(y_lengths[0]))
            return original_single(y, y_lengths, sid_src, sid_tgt, tau=tau)

        self.voice_conversion_batched = _batched
        self.voice_conversion = _single


class _SignatureToneConverter:
    """Deterministic converter stub: output encodes the row's mel + source speaker.

    The real converter is stochastic (``tau=0.3``), so a deterministic stand-in is
    what makes an exact batched-vs-loop comparison meaningful at the handler
    level; the vendored math itself is covered with ``tau=0`` in
    ``tests/unit/modeling/test_batched_tone_converter.py``.
    """

    def __init__(self):
        self.dec = types.SimpleNamespace(upsample_rates=[2, 2])
        self.batched_calls: list[list[int]] = []
        self.loop_calls: list[int] = []

    @staticmethod
    def _signature(y, sid_src):
        return float(y.shape[-1]) * 1000.0 + float(y.mean()) * 10.0 + float(sid_src.mean()) * 0.1

    def voice_conversion(self, y, y_lengths, sid_src, sid_tgt, tau=1.0):
        frames = int(y_lengths[0])
        self.loop_calls.append(frames)
        return (
            torch.full((1, 1, frames * SPF), self._signature(y[0], sid_src[0])),
            torch.ones(1, 1, frames),
            None,
        )

    def voice_conversion_batched(self, y, y_lengths, sid_src, sid_tgt, tau=1.0):
        frames = [int(value) for value in y_lengths]
        self.batched_calls.append(frames)
        output = torch.zeros(len(frames), 1, max(frames) * SPF)
        for row, count in enumerate(frames):
            output[row, 0, : count * SPF] = self._signature(y[row, :, :count], sid_src[row])
        return output, torch.ones(len(frames), 1, max(frames)), None


# ---------------------------------------------------------------------------
# Batched path
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_tone_converter_forwards_a_batch_of_four_in_one_forward():
    model = _RecordingToneConverter(_tiny_tone_converter())
    runner = _runner(model, batch_token_budget=ROOMY_BUDGET)
    clips, requests = _mixed_batch()

    results = _run(runner, clips, requests)

    assert [result.request_id for result in results] == ["tc-0", "tc-1", "tc-2", "tc-3"]
    assert all(result.result_data.finish_reason == "stop" for result in results)
    # 500 + 600 + 700 + 800 mel frames -> exactly one padded forward.
    assert model.batched_calls == [[500, 600, 700, 800]]
    assert model.loop_calls == []
    # Frame-domain unpad: each output is the model's frame count * samples/frame.
    assert [result.result_data.audio_artifact.num_samples for result in results] == [
        500 * SPF,
        600 * SPF,
        700 * SPF,
        800 * SPF,
    ]
    for result in results:
        assert np.isfinite(result.result_data.audio_artifact.numpy).all()


@pytest.mark.unit
def test_tone_converter_batched_matches_the_per_item_loop():
    model = _SignatureToneConverter()
    batched_runner = _runner(model, batch_token_budget=ROOMY_BUDGET)
    loop_runner = _runner(model, batched=False)
    clips, requests = _mixed_batch()

    batched = _run(batched_runner, clips, requests)
    looped = _run(loop_runner, clips, requests)

    assert model.batched_calls == [[500, 600, 700, 800]]
    assert model.loop_calls == [500, 600, 700, 800]
    for index in range(len(clips)):
        batched_audio = batched[index].result_data.audio_artifact
        loop_audio = looped[index].result_data.audio_artifact
        assert batched_audio.num_samples == loop_audio.num_samples == (500 + 100 * index) * SPF
        assert np.array_equal(batched_audio.numpy, loop_audio.numpy)
    # Distinct source inputs must not all collapse to one output value.
    values = [float(result.result_data.audio_artifact.numpy[0]) for result in batched]
    assert len(set(values)) == len(values)


@pytest.mark.unit
def test_tone_converter_batched_outputs_stay_paired_with_their_requests():
    """Distinct clips give distinct outputs; request ids keep their own audio."""
    model = _SignatureToneConverter()
    runner = _runner(model, batch_token_budget=ROOMY_BUDGET)
    clips, requests = _mixed_batch()

    forward = _run(runner, clips, requests)
    reversed_inputs = list(reversed(clips))
    reversed_requests = list(reversed(requests))
    backward = _run(runner, reversed_inputs, reversed_requests)

    by_id = {result.request_id: result.result_data.audio_artifact.numpy for result in forward}
    for result in backward:
        assert np.array_equal(result.result_data.audio_artifact.numpy, by_id[result.request_id])


# ---------------------------------------------------------------------------
# Loop paths and isolation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_tone_converter_single_request_keeps_the_loop_path():
    model = _RecordingToneConverter(_tiny_tone_converter())
    runner = _runner(model)
    clips, requests = _mixed_batch()

    results = _run(runner, clips[:1], requests[:1])

    assert model.batched_calls == []
    assert model.loop_calls == [500]
    assert results[0].result_data.finish_reason == "stop"


@pytest.mark.unit
def test_tone_converter_batched_inference_disabled_uses_the_loop():
    model = _SignatureToneConverter()
    runner = _runner(model, batched=False)
    clips, requests = _mixed_batch()

    results = _run(runner, clips, requests)

    assert model.batched_calls == []
    assert model.loop_calls == [500, 600, 700, 800]
    assert all(result.result_data.finish_reason == "stop" for result in results)


@pytest.mark.unit
def test_tone_converter_batched_failure_falls_back_to_the_loop():
    model = _RecordingToneConverter(_tiny_tone_converter(), fail_batched=True)
    batched_runner = _runner(model, batch_token_budget=ROOMY_BUDGET)
    loop_runner = _runner(_RecordingToneConverter(_tiny_tone_converter()), batched=False)
    clips, requests = _mixed_batch()

    with torch.no_grad():
        results = _run(batched_runner, clips, requests)
        reference = _run(loop_runner, clips, requests)

    assert model.batched_calls == [[500, 600, 700, 800]]
    assert model.loop_calls == [500, 600, 700, 800]
    assert all(result.result_data.finish_reason == "stop" for result in results)
    for index in range(len(clips)):
        assert (
            results[index].result_data.audio_artifact.num_samples
            == reference[index].result_data.audio_artifact.num_samples
        )


@pytest.mark.unit
def test_tone_converter_poisoned_request_errors_alone():
    """A missing target embedding fails that request only; the rest still finish."""
    model = _RecordingToneConverter(_tiny_tone_converter())
    runner = _runner(model, batch_token_budget=ROOMY_BUDGET)
    clips, requests = _mixed_batch()
    clips[1] = (clips[1][0], None, clips[1][2])

    with torch.no_grad():
        results = _run(runner, clips, requests)

    assert [result.result_data.finish_reason for result in results] == [
        "stop",
        "error",
        "stop",
        "stop",
    ]
    assert "missing embedding tensor" in results[1].result_data.error
    # The poisoned row failed inside the sub-batch, so that sub-batch degraded to
    # the per-item loop; the healthy requests were computed there.
    assert model.loop_calls == [500, 700, 800]

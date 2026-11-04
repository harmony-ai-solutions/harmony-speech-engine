"""Engine-level mixed-length batched synthesis (Phase 5-2).

Four requests with very different text lengths go through the real engine,
scheduler and OpenVoice-family handler in as few engine steps as the batch
configuration allows; the assertions cover the end-to-end contract of the padded
path: every request gets exactly one result, each result carries its own
(audio length consistent) waveform, and a pathological length mix is sub-batched
instead of padding everything to the longest text.
"""

from unittest.mock import patch

import pytest
import torch

from harmonyspeech.common.config import DeviceConfig, ModelConfig
from harmonyspeech.common.inputs import TextToSpeechRequestInput
from harmonyspeech.common.outputs import SpeechSynthesisRequestOutput
from harmonyspeech.common.request import EngineRequest
from harmonyspeech.engine.harmonyspeech_engine import HarmonySpeechEngine
from harmonyspeech.task_handler.model_runner_base import ModelRunnerBase

HOP = 8
SAMPLE_RATE = 200
GAP = int(SAMPLE_RATE * 0.05)
SAMPLING = {"noise_scale": 0.667, "noise_scale_w": 0.6, "sdp_ratio": 0.2}
FRAMES_PER_TOKEN = 2


class LengthStubModel(torch.nn.Module):
    """One-shot synthesizer stand-in: frames are a function of the input length."""

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Linear(1, 1)
        self.batched_calls: list[list[int]] = []
        self.loop_calls: list[int] = []

    def infer(self, x, x_lengths, sid=None, **kwargs):
        tokens = int(x_lengths[0])
        self.loop_calls.append(tokens)
        return (torch.zeros(1, 1, tokens * FRAMES_PER_TOKEN * HOP), None, None, None)

    def infer_batched(self, x, x_lengths, sid=None, **kwargs):
        lengths = [int(value) for value in x_lengths]
        self.batched_calls.append(lengths)
        y_lengths = x_lengths * FRAMES_PER_TOKEN
        output = torch.zeros(len(lengths), 1, max(lengths) * FRAMES_PER_TOKEN * HOP)
        return output, y_lengths, None, None, None


class StubExecutor:
    def __init__(self, runner):
        self.runner = runner

    def execute_model(self, requests_to_batch):
        return self.runner.execute_model(requests_to_batch)

    async def execute_model_async(self, requests_to_batch):
        return self.runner.execute_model(requests_to_batch)

    def check_health(self):
        return {"status": "ok"}


def _fake_hf_config():
    import types

    return types.SimpleNamespace(
        data=types.SimpleNamespace(sampling_rate=SAMPLE_RATE, hop_length=HOP),
        speakers={"default": 0},
    )


def _request(request_id: str, text: str) -> EngineRequest:
    request_data = TextToSpeechRequestInput(
        request_id=request_id,
        requested_model="ov1",
        model="ov1",
        input_text=text,
        mode="single_speaker_tts",
    )
    return EngineRequest(request_id=request_id, request_data=request_data, arrival_time=0.0)


def _prepare_for_lengths(model_config, requests_to_batch):
    """Prepare stand-in inputs whose token count is the request's text length."""

    def tokens(text: str) -> int:
        return len(text) * 5

    return [
        ([torch.arange(1, tokens(request.request_data.input_text) + 1, dtype=torch.long)], 0, 1.0, dict(SAMPLING))
        for request in requests_to_batch
    ]


@pytest.mark.integration
def test_four_mixed_length_requests_produce_four_correct_length_outputs():
    model = LengthStubModel()
    runner = ModelRunnerBase.__new__(ModelRunnerBase)
    runner.model = model
    runner.device = "cpu"
    runner._baseline_state = None
    runner.model_config = ModelConfig(
        name="ov1",
        model="dummy/model",
        model_type="OpenVoiceV1Synthesizer",
        max_batch_size=4,
        device_config=DeviceConfig("cpu"),
        dtype="float32",
    )

    with patch("harmonyspeech.executor.cpu_executor.CPUExecutor._init_worker"):
        engine = HarmonySpeechEngine(model_configs=[runner.model_config], log_stats=False)
    engine.model_executors = {"ov1": StubExecutor(runner)}

    texts = {
        "mix-long": "a" * 40,  # 200 tokens
        "mix-medium": "b" * 20,  # 100 tokens
        "mix-short": "c" * 8,  # 40 tokens
        "mix-tiny": "d" * 4,  # 20 tokens
    }
    for request_id, text in texts.items():
        engine.add_request(request_id, _request(request_id, text).request_data)

    with patch(
        "harmonyspeech.task_handler.model_runner_base.prepare_inputs", side_effect=_prepare_for_lengths
    ), patch(
        "harmonyspeech.task_handler.model_runner_base.get_model_config", return_value=_fake_hf_config()
    ), patch("harmonyspeech.task_handler.model_runner_base.get_model_flavour", return_value="default"):
        collected: dict[str, list] = {}
        for _ in range(8):
            if not engine.has_unfinished_requests():
                break
            outputs, _forwarded = engine.step()
            for output in outputs:
                collected.setdefault(output.request_id, []).append(output)

    assert sorted(collected) == sorted(texts)
    for request_id, text in texts.items():
        assert len(collected[request_id]) == 1
        result = collected[request_id][0]
        assert isinstance(result, SpeechSynthesisRequestOutput)
        assert result.finish_reason == "stop"
        expected_samples = len(text) * 5 * FRAMES_PER_TOKEN * HOP + GAP
        assert result.audio_artifact.waveform.shape[-1] == expected_samples

    # The long text cannot share a padded batch with the tiny one
    # (T_max / T_min = 10 > 4), so the step is split into length-similar groups;
    # every request still went through the batched forward exactly once.
    assert sum(len(call) for call in model.batched_calls) == 4
    assert len(model.batched_calls) >= 2
    for call in model.batched_calls:
        assert max(call) / min(call) <= 4
    assert model.loop_calls == []


@pytest.mark.integration
def test_two_similar_length_requests_share_one_padded_forward():
    model = LengthStubModel()
    runner = ModelRunnerBase.__new__(ModelRunnerBase)
    runner.model = model
    runner.device = "cpu"
    runner._baseline_state = None
    runner.model_config = ModelConfig(
        name="ov1",
        model="dummy/model",
        model_type="OpenVoiceV1Synthesizer",
        max_batch_size=4,
        device_config=DeviceConfig("cpu"),
        dtype="float32",
    )

    with patch("harmonyspeech.executor.cpu_executor.CPUExecutor._init_worker"):
        engine = HarmonySpeechEngine(model_configs=[runner.model_config], log_stats=False)
    engine.model_executors = {"ov1": StubExecutor(runner)}

    for request_id, text in (("pa", "a" * 10), ("pb", "b" * 11)):
        engine.add_request(request_id, _request(request_id, text).request_data)

    with patch(
        "harmonyspeech.task_handler.model_runner_base.prepare_inputs", side_effect=_prepare_for_lengths
    ), patch(
        "harmonyspeech.task_handler.model_runner_base.get_model_config", return_value=_fake_hf_config()
    ), patch("harmonyspeech.task_handler.model_runner_base.get_model_flavour", return_value="default"):
        outputs = []
        for _ in range(4):
            if not engine.has_unfinished_requests():
                break
            step_outputs, _ = engine.step()
            outputs.extend(step_outputs)

    assert len(outputs) == 2
    assert model.batched_calls == [[50, 55]]

"""Unit tests for the batched synthesis handlers (Phase 5-2) and batch invariants.

The handlers are exercised with stub models that record whether the batched
forward or the per-item loop ran, and that derive their output length from the
per-sample ``x_lengths`` — so the tests prove the *pairing* (request ↔ padded
row ↔ unpadded slice) rather than any particular audio content.
"""

from unittest.mock import patch

import pytest
import torch

from harmonyspeech.common.config import DeviceConfig, ModelConfig
from harmonyspeech.common.errors import EngineBatchError
from harmonyspeech.common.inputs import TextToSpeechRequestInput
from harmonyspeech.common.outputs import SpeechSynthesisRequestOutput
from harmonyspeech.common.request import EngineRequest
from harmonyspeech.task_handler.model_runner_base import ModelRunnerBase, assert_batch_aligned

HOP = 4
SAMPLE_RATE = 100
#: `_assemble` inserts 50 ms of silence per sentence at the model sample rate.
GAP = int(SAMPLE_RATE * 0.05)
SAMPLING = {"noise_scale": 0.667, "noise_scale_w": 0.6, "sdp_ratio": 0.2}


def _fake_hf_config():
    import types

    return types.SimpleNamespace(
        data=types.SimpleNamespace(sampling_rate=SAMPLE_RATE, hop_length=HOP),
        speakers={"default": 0},
    )


class StubOpenVoiceModel(torch.nn.Module):
    """OpenVoice V1 synthesizer stand-in: frames = tokens * frames_per_token."""

    def __init__(self, frames_per_token: int = 2):
        super().__init__()
        self.anchor = torch.nn.Linear(1, 1)
        self.frames_per_token = frames_per_token
        self.batched_calls: list[dict] = []
        self.loop_calls: list[int] = []

    def infer(self, x, x_lengths, sid=None, **kwargs):
        tokens = int(x_lengths[0])
        self.loop_calls.append(tokens)
        frames = tokens * self.frames_per_token
        return (torch.zeros(1, 1, frames * HOP), None, None, None)

    def infer_batched(self, x, x_lengths, sid=None, **kwargs):
        self.batched_calls.append(
            {
                "batch": int(x.shape[0]),
                "x_lengths": [int(value) for value in x_lengths],
                "sid": [int(value) for value in sid] if sid is not None else None,
                **kwargs,
            }
        )
        y_lengths = x_lengths * self.frames_per_token
        output = torch.zeros(int(x.shape[0]), 1, int(x_lengths.max()) * self.frames_per_token * HOP)
        return output, y_lengths, None, None, None


class StubMeloModel(torch.nn.Module):
    """MeloTTS synthesizer stand-in with the same frame contract."""

    def __init__(self, frames_per_token: int = 2):
        super().__init__()
        self.anchor = torch.nn.Linear(1, 1)
        self.frames_per_token = frames_per_token
        self.batched_calls: list[dict] = []
        self.loop_calls: list[int] = []

    def infer(self, x, x_lengths, sid, tone, language, bert, ja_bert, **kwargs):
        tokens = int(x_lengths[0])
        self.loop_calls.append(tokens)
        return (torch.zeros(1, 1, tokens * self.frames_per_token * HOP), None, None, None)

    def infer_batched(self, x, x_lengths, sid, tone, language, bert, ja_bert, **kwargs):
        self.batched_calls.append(
            {
                "batch": int(x.shape[0]),
                "x_lengths": [int(value) for value in x_lengths],
                "bert": tuple(bert.shape),
                "ja_bert": tuple(ja_bert.shape),
                "tone": tuple(tone.shape),
                "language": tuple(language.shape),
                **kwargs,
            }
        )
        y_lengths = x_lengths * self.frames_per_token
        output = torch.zeros(int(x.shape[0]), 1, int(x_lengths.max()) * self.frames_per_token * HOP)
        return output, y_lengths, None, None, None


def _runner(model, model_type: str, **config) -> ModelRunnerBase:
    runner = ModelRunnerBase.__new__(ModelRunnerBase)
    runner.device = "cpu"
    runner.device_config = DeviceConfig("cpu")
    runner._baseline_state = None
    runner.model = model
    runner.model_config = ModelConfig(
        name="stub",
        model="dummy/model",
        model_type=model_type,
        max_batch_size=config.pop("max_batch_size", 4),
        device_config=DeviceConfig("cpu"),
        dtype="float32",
        **config,
    )
    return runner


def _request(request_id: str, text: str = "hello") -> EngineRequest:
    request_data = TextToSpeechRequestInput(
        request_id=request_id,
        requested_model="stub",
        model="stub",
        input_text=text,
        mode="single_speaker_tts",
    )
    return EngineRequest(request_id=request_id, request_data=request_data, arrival_time=0.0)


def _openvoice_inputs(lengths, speaker_id=0, speed=1.0):
    return [
        ([torch.arange(1, length + 1, dtype=torch.long)], speaker_id, speed, dict(SAMPLING))
        for length in lengths
    ]


def _melo_inputs(lengths, speaker_id=0, speed=1.0):
    inputs = []
    for length in lengths:
        item = (
            torch.zeros(1024, length),
            torch.zeros(768, length),
            torch.arange(1, length + 1, dtype=torch.long),
            torch.zeros(length, dtype=torch.long),
            torch.zeros(length, dtype=torch.long),
        )
        inputs.append(([item], speaker_id, speed, dict(SAMPLING)))
    return inputs


def _run_openvoice(runner, lengths):
    requests = [_request(f"ov-{index}") for index in range(len(lengths))]
    with patch(
        "harmonyspeech.task_handler.model_runner_base.get_model_config", return_value=_fake_hf_config()
    ), patch("harmonyspeech.task_handler.model_runner_base.get_model_flavour", return_value="default"):
        return runner._execute_openvoice_synthesizer(_openvoice_inputs(lengths), requests)


def _run_melo(runner, lengths):
    requests = [_request(f"melo-{index}") for index in range(len(lengths))]
    with patch(
        "harmonyspeech.task_handler.model_runner_base.get_model_config", return_value=_fake_hf_config()
    ), patch("harmonyspeech.task_handler.model_runner_base.get_model_flavour", return_value="default"):
        return runner._execute_melotts_synthesizer(_melo_inputs(lengths), requests)


# ---------------------------------------------------------------------------
# OpenVoice V1 batching
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_openvoice_four_requests_use_one_batched_forward():
    model = StubOpenVoiceModel()
    runner = _runner(model, "OpenVoiceV1Synthesizer")

    results = _run_openvoice(runner, [4, 4, 4, 4])

    assert len(model.batched_calls) == 1
    assert model.loop_calls == []
    call = model.batched_calls[0]
    assert call["batch"] == 4
    assert call["x_lengths"] == [4, 4, 4, 4]

    assert [result.request_id for result in results] == ["ov-0", "ov-1", "ov-2", "ov-3"]
    assert all(isinstance(result.result_data, SpeechSynthesisRequestOutput) for result in results)
    for result in results:
        assert result.result_data.finish_reason == "stop"
        # frames = tokens * 2, samples = frames * HOP, plus the 50 ms gap.
        assert result.result_data.audio_artifact.waveform.shape[-1] == 4 * 2 * HOP + GAP


@pytest.mark.unit
def test_openvoice_single_request_keeps_the_loop_fast_path():
    model = StubOpenVoiceModel()
    runner = _runner(model, "OpenVoiceV1Synthesizer")

    results = _run_openvoice(runner, [5])

    assert model.batched_calls == []
    assert model.loop_calls == [5]
    assert results[0].result_data.audio_artifact.waveform.shape[-1] == 5 * 2 * HOP + GAP


@pytest.mark.unit
def test_openvoice_batched_inference_false_uses_the_loop():
    model = StubOpenVoiceModel()
    runner = _runner(model, "OpenVoiceV1Synthesizer", batched_inference=False)

    results = _run_openvoice(runner, [3, 4, 5])

    assert model.batched_calls == []
    assert model.loop_calls == [3, 4, 5]
    assert len(results) == 3


@pytest.mark.unit
def test_openvoice_mixed_lengths_are_sub_batched():
    model = StubOpenVoiceModel()
    runner = _runner(model, "OpenVoiceV1Synthesizer", max_batch_size=4)

    results = _run_openvoice(runner, [40, 5, 6, 5])

    # The 40-token text cannot share a padded batch with the 5-token ones
    # (T_max / T_min > 4), so it is forwarded on its own.
    assert len(model.batched_calls) >= 2
    assert sorted(length for call in model.batched_calls for length in call["x_lengths"]) == [5, 5, 6, 40]
    assert len(results) == 4
    lengths = [result.result_data.audio_artifact.waveform.shape[-1] for result in results]
    assert lengths == [tokens * 2 * HOP + GAP for tokens in (40, 5, 6, 5)]


@pytest.mark.unit
def test_openvoice_token_budget_caps_a_sub_batch():
    model = StubOpenVoiceModel()
    runner = _runner(model, "OpenVoiceV1Synthesizer", max_batch_size=4, batch_token_budget=20)

    _run_openvoice(runner, [10, 10, 10, 10])

    assert all(call["batch"] * max(call["x_lengths"]) <= 20 for call in model.batched_calls)
    assert sum(call["batch"] for call in model.batched_calls) == 4


@pytest.mark.unit
def test_openvoice_batched_failure_falls_back_to_the_loop(loguru_records):
    class FailingModel(StubOpenVoiceModel):
        def infer_batched(self, *args, **kwargs):
            raise RuntimeError("boom")

    model = FailingModel()
    runner = _runner(model, "OpenVoiceV1Synthesizer")

    results = _run_openvoice(runner, [3, 4])

    assert model.loop_calls == [3, 4]
    assert [result.input_data.model for result in results] == ["stub", "stub"]
    assert all(result.result_data.finish_reason == "stop" for result in results)
    assert any("falling back to the per-request loop" in message for message in loguru_records)


@pytest.mark.unit
def test_openvoice_multiple_sentences_per_request_are_flattened_and_reassembled():
    model = StubOpenVoiceModel()
    runner = _runner(model, "OpenVoiceV1Synthesizer")
    inputs = [
        ([torch.tensor([1, 2]), torch.tensor([3])], 0, 1.0, dict(SAMPLING)),
        ([torch.tensor([4, 5, 6])], 0, 1.0, dict(SAMPLING)),
    ]
    requests = [_request("ov-a"), _request("ov-b")]

    with patch(
        "harmonyspeech.task_handler.model_runner_base.get_model_config", return_value=_fake_hf_config()
    ), patch("harmonyspeech.task_handler.model_runner_base.get_model_flavour", return_value="default"):
        results = runner._execute_openvoice_synthesizer(inputs, requests)

    assert sum(call["batch"] for call in model.batched_calls) == 3
    # Request A: two sentences (2 and 1 tokens) -> 2*2*4 + gap + 1*2*4 + gap.
    assert results[0].result_data.audio_artifact.waveform.shape[-1] == (2 * 2 * HOP + GAP) + (1 * 2 * HOP + GAP)
    # Request B: one sentence of 3 tokens.
    assert results[1].result_data.audio_artifact.waveform.shape[-1] == 3 * 2 * HOP + GAP


@pytest.mark.unit
def test_openvoice_speed_parameter_is_applied_per_sample():
    model = StubOpenVoiceModel()
    runner = _runner(model, "OpenVoiceV1Synthesizer")
    inputs = _openvoice_inputs([4, 4])
    inputs[1] = (inputs[1][0], inputs[1][1], 2.0, inputs[1][3])
    requests = [_request("ov-slow"), _request("ov-fast")]

    with patch(
        "harmonyspeech.task_handler.model_runner_base.get_model_config", return_value=_fake_hf_config()
    ), patch("harmonyspeech.task_handler.model_runner_base.get_model_flavour", return_value="default"):
        runner._execute_openvoice_synthesizer(inputs, requests)

    call = model.batched_calls[0]
    assert call["length_scale"] == [1.0, 0.5]


# ---------------------------------------------------------------------------
# MeloTTS batching
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_melo_four_requests_use_one_batched_forward():
    model = StubMeloModel()
    runner = _runner(model, "MeloTTSSynthesizer")

    results = _run_melo(runner, [4, 3, 4, 4])

    assert len(model.batched_calls) == 1
    assert model.loop_calls == []
    call = model.batched_calls[0]
    assert call["batch"] == 4
    # Features are padded to [B, D, T_max]; ids to [B, T_max].
    assert call["bert"] == (4, 1024, 4)
    assert call["ja_bert"] == (4, 768, 4)
    assert call["tone"] == (4, 4)
    assert call["language"] == (4, 4)

    lengths = [result.result_data.audio_artifact.waveform.shape[-1] for result in results]
    assert lengths == [tokens * 2 * HOP + GAP for tokens in (4, 3, 4, 4)]


@pytest.mark.unit
def test_melo_single_request_keeps_the_loop_fast_path():
    model = StubMeloModel()
    runner = _runner(model, "MeloTTSSynthesizer")

    _run_melo(runner, [6])

    assert model.batched_calls == []
    assert model.loop_calls == [6]


# ---------------------------------------------------------------------------
# Invariants: alignment + one result per request id per step
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_assert_batch_aligned_raises_engine_batch_error():
    with pytest.raises(EngineBatchError, match="3 inputs vs 2 requests"):
        assert_batch_aligned("handler", [1, 2, 3], [1, 2])


@pytest.mark.unit
def test_engine_batch_error_is_an_assertion_error():
    """Backwards compatibility: Phase 0-3 callers asserted with AssertionError."""
    assert issubclass(EngineBatchError, AssertionError)


@pytest.mark.unit
def test_build_result_rejects_a_duplicate_request_id_within_one_step():
    runner = _runner(StubOpenVoiceModel(), "OpenVoiceV1Synthesizer")
    runner._step_result_ids = set()
    request = _request("dup-1")

    runner._build_result(request, "", SpeechSynthesisRequestOutput)

    with pytest.raises(EngineBatchError, match="duplicate result for request dup-1"):
        runner._build_result(request, "", SpeechSynthesisRequestOutput)


@pytest.mark.unit
def test_error_result_registers_the_id_without_raising():
    runner = _runner(StubOpenVoiceModel(), "OpenVoiceV1Synthesizer")
    runner._step_result_ids = set()
    request = _request("err-1")

    first = runner._build_error_result(request, ValueError("bad"))
    second = runner._build_error_result(request, ValueError("bad again"))

    assert first.result_data.finish_reason == "error"
    assert second.result_data.finish_reason == "error"
    assert runner._step_result_ids == {"err-1"}


@pytest.mark.unit
def test_no_step_scope_means_no_duplicate_guard():
    """Handlers called directly (no engine step) do not track result ids."""
    runner = _runner(StubOpenVoiceModel(), "OpenVoiceV1Synthesizer")
    request = _request("direct-1")

    runner._build_result(request, "", SpeechSynthesisRequestOutput)
    runner._build_result(request, "", SpeechSynthesisRequestOutput)

    assert getattr(runner, "_step_result_ids", None) is None


@pytest.mark.unit
def test_execute_model_resets_the_result_scope_between_steps():
    model = StubOpenVoiceModel()
    runner = _runner(model, "OpenVoiceV1Synthesizer")
    request = _request("repeat-1")

    # The same request object executed in two separate steps must not trip the
    # duplicate guard (its scope is one step).
    with patch(
        "harmonyspeech.task_handler.model_runner_base.prepare_inputs",
        return_value=_openvoice_inputs([3]),
    ), patch(
        "harmonyspeech.task_handler.model_runner_base.get_model_config", return_value=_fake_hf_config()
    ), patch("harmonyspeech.task_handler.model_runner_base.get_model_flavour", return_value="default"):
        first = runner.execute_model([request])
        second = runner.execute_model([request])

    assert first[0].result_data.finish_reason == "stop"
    assert second[0].result_data.finish_reason == "stop"


@pytest.mark.unit
def test_execute_partition_converts_handler_batch_error_to_per_request_errors():
    model = StubOpenVoiceModel()
    runner = _runner(model, "OpenVoiceV1Synthesizer")
    requests = [_request("be-1"), _request("be-2")]

    with patch(
        "harmonyspeech.task_handler.model_runner_base.prepare_inputs",
        return_value=_openvoice_inputs([3, 3]),
    ), patch.object(
        runner, "_dispatch_handler", side_effect=EngineBatchError("pairing broke")
    ):
        results = runner.execute_model(requests)

    assert [result.request_id for result in results] == ["be-1", "be-2"]
    for result in results:
        assert result.result_data.finish_reason == "error"
        assert "pairing broke" in result.result_data.error

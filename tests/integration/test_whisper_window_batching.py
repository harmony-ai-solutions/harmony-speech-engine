"""Cross-request Whisper window batching — integration gates (Phase 10-2).

Drives the real handler (and, for the second test, the real engine → scheduler →
executor → handler seam) with four mixed-length requests:

* two short STT requests (single 30 s window each),
* one >60 s STT request (three 30 s windows),
* one pipeline-VAD request (``DetectVoiceActivityRequestInput``) routed to the
  same whisper handler.

The CT2 binding, Silero VAD and the feature extractor are stubbed (no model
weights are downloaded); window formation, prompt building, timestamp splitting
and stitching run through the real faster-whisper helpers.

Real-model parity (B=1 vs ``pipeline.transcribe``) and throughput are the
benchmark-host gate recorded as ``_tbd_`` in the phase file; this suite is
hermetic and CPU-only.
"""

import json
from unittest.mock import patch

import pytest

from harmonyspeech.common.config import DeviceConfig, ModelConfig
from harmonyspeech.common.inputs import DetectVoiceActivityRequestInput, SpeechTranscribeRequestInput
from harmonyspeech.common.request import EngineRequest, RequestStatus
from harmonyspeech.engine.harmonyspeech_engine import HarmonySpeechEngine
from harmonyspeech.task_handler import model_runner_base, whisper_batching
from harmonyspeech.task_handler.model_runner_base import ModelRunnerBase
from tests.audio_fixtures import tiny_wav_b64
from tests.unit.inference_flow.test_whisper_window_batching import (
    SAMPLE_RATE,
    _clips_for,
    _FakeCT2,
    _FakeHFTokenizer,
    _FakeWhisperModel,
    _SequentialInfo,
    _SequentialSegment,
)

# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


class _StubExecutor:
    def __init__(self, runner):
        self.runner = runner

    def execute_model(self, requests_to_batch):
        return self.runner.execute_model(requests_to_batch)

    async def execute_model_async(self, requests_to_batch):
        return self.runner.execute_model(requests_to_batch)

    def check_health(self):
        return {"status": "ok"}


def _config(name="whisper", *, max_batch_size=16, num_workers=2, batched_inference=True):
    return ModelConfig(
        name=name,
        model="dummy/whisper",
        model_type="FasterWhisper",
        max_batch_size=max_batch_size,
        num_workers=num_workers,
        batched_inference=batched_inference,
        device_config=DeviceConfig("cpu"),
        dtype="float32",
    )


def _runner(model, config):
    runner = ModelRunnerBase.__new__(ModelRunnerBase)
    runner.model = model
    runner.device = "cpu"
    runner.model_config = config
    runner._baseline_state = None
    return runner


def _engine(config, runner):
    with patch("harmonyspeech.executor.cpu_executor.CPUExecutor._init_worker"):
        engine = HarmonySpeechEngine(model_configs=[config], log_stats=False)
    engine.model_executors = {config.name: _StubExecutor(runner)}
    return engine


def _drive(engine, max_steps=12):
    collected = {}
    for _ in range(max_steps):
        if not engine.has_unfinished_requests():
            break
        outputs, _forwarded = engine.step()
        for output in outputs:
            collected.setdefault(output.request_id, []).append(output)
    return collected


def _artifact(seconds):
    import numpy as np

    from harmonyspeech.common.artifacts import AudioArtifact

    return AudioArtifact.from_numpy(np.zeros(int(SAMPLE_RATE * seconds), dtype=np.float32), SAMPLE_RATE)


def _handler_request(request_id, seconds, *, request_cls=SpeechTranscribeRequestInput):
    return EngineRequest(
        request_id=request_id,
        request_data=request_cls(request_id=request_id, requested_model="whisper", model="whisper", input_audio=None),
        arrival_time=0.0,
        artifacts={"input_audio": _artifact(seconds)},
    )


def _engine_request(request_id, seconds, *, vad=False):
    data = (
        DetectVoiceActivityRequestInput(
            request_id=request_id,
            requested_model="whisper",
            model="whisper",
            input_audio=tiny_wav_b64(SAMPLE_RATE, seconds),
        )
        if vad
        else SpeechTranscribeRequestInput(
            request_id=request_id,
            requested_model="whisper",
            model="whisper",
            input_audio=tiny_wav_b64(SAMPLE_RATE, seconds),
        )
    )
    return EngineRequest(request_id=request_id, request_data=data, arrival_time=0.0)


def _run_handler(runner, requests):
    inputs = [request.artifacts["input_audio"] for request in requests]
    with patch.object(whisper_batching, "get_speech_timestamps", _clips_for(65.0)), patch.object(
        whisper_batching, "get_suppressed_tokens", lambda tokenizer, values: (1, 2, 3)
    ):
        return runner._execute_faster_whisper(inputs, requests)


# ---------------------------------------------------------------------------
# 1. Four mixed-length requests in one formed window batch
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_four_mixed_requests_pair_and_stitch_offsets_in_one_window_batch():
    model = _FakeWhisperModel()
    runner = _runner(model, _config())
    requests = [
        _handler_request("short-a", 0.2),
        _handler_request("short-b", 0.3),
        _handler_request("long", 65.0),
        _handler_request("vad", 0.25, request_cls=DetectVoiceActivityRequestInput),
    ]

    outputs = _run_handler(runner, requests)

    assert [output.request_id for output in outputs] == ["short-a", "short-b", "long", "vad"]
    payloads = {output.request_id: json.loads(output.result_data.output) for output in outputs}

    assert payloads["short-a"]["text"] == "r200"
    assert payloads["short-b"]["text"] == "r300"
    assert payloads["vad"]["text"] == "r250"
    # The >60 s request is three windows stitched in order, with per-window seeks
    # and offsets (0 / 30 / 60 s) — not one 30 s window.
    assert payloads["long"]["text"] == "r30000r30000r5000"
    assert len(payloads["long"]["segments"]) == 3
    assert [segment["seek"] for segment in payloads["long"]["segments"]] == [0, 3000, 6000]
    assert [segment["start"] for segment in payloads["long"]["segments"]] == pytest.approx([0.0, 30.0, 60.0])
    assert [segment["end"] for segment in payloads["long"]["segments"]] == pytest.approx([1.2, 31.2, 61.2])

    # Every request produced a VAD artifact with the same payload shape.
    for output in outputs:
        assert output.result_data.finish_reason == "stop"
        assert output.result_data.vad_artifact is not None
        assert output.result_data.vad_artifact.source == "whisper"

    # One encode + one generate for the whole step (5 windows from 4 requests).
    assert len(model.ct2.generate_calls) == 1
    assert model.ct2.generate_calls[0]["markers"] == [200.0, 300.0, 30000.0, 30000.0, 5000.0, 250.0]
    assert model.encode_calls == 1


# ---------------------------------------------------------------------------
# 2. Abort mid-batch errors only the aborted request
# ---------------------------------------------------------------------------


class _AbortingCT2(_FakeCT2):
    def __init__(self, abort_request, abort_marker, **kwargs):
        super().__init__(**kwargs)
        self.abort_request = abort_request
        self.abort_marker = int(abort_marker)

    def generate(self, encoder_output, prompts, **kwargs):
        results = super().generate(encoder_output, prompts, **kwargs)
        if self.abort_marker in [int(marker) for marker in encoder_output.markers]:
            # The engine aborts this request while CT2 is decoding the batch.
            self.abort_request.status = RequestStatus.FINISHED_ABORTED
        return results


@pytest.mark.integration
def test_abort_mid_batch_errors_only_the_aborted_request():
    model = _FakeWhisperModel()
    requests = [
        _handler_request("short-a", 0.2),
        _handler_request("aborted", 0.3),
        _handler_request("long", 65.0),
    ]
    ct2 = _AbortingCT2(abort_request=requests[1], abort_marker=300)
    ct2.is_multilingual = True
    model.ct2 = ct2
    model.model = ct2
    model.hf_tokenizer = _FakeHFTokenizer(token_text=ct2._token_text)
    runner = _runner(model, _config())

    outputs = _run_handler(runner, requests)
    by_id = {output.request_id: output for output in outputs}

    assert by_id["short-a"].result_data.finish_reason == "stop"
    assert json.loads(by_id["short-a"].result_data.output)["text"] == "r200"
    assert by_id["aborted"].result_data.finish_reason == "error"
    assert "aborted" in by_id["aborted"].result_data.error
    assert by_id["long"].result_data.finish_reason == "stop"
    assert len(json.loads(by_id["long"].result_data.output)["segments"]) == 3


# ---------------------------------------------------------------------------
# 3. Engine seam: four mixed requests, outputs paired by request id
# ---------------------------------------------------------------------------


class _SequentialPipeline:
    def __init__(self, model=None):
        self.model = model
        self.last_speech_timestamp = 0.0

    def transcribe(self, audio, batch_size=16, **kwargs):
        return [_SequentialSegment("sequential")], _SequentialInfo()


@pytest.mark.integration
def test_engine_routes_four_mixed_requests_and_pairs_outputs_by_id():
    model = _FakeWhisperModel()
    config = _config()
    runner = _runner(model, config)
    engine = _engine(config, runner)

    specs = [
        ("short-a", 0.20, False),
        ("short-b", 0.30, False),
        ("long", 65.0, False),
        ("vad", 0.25, True),
    ]
    with patch.object(whisper_batching, "get_speech_timestamps", _clips_for(65.0)), patch.object(
        whisper_batching, "get_suppressed_tokens", lambda tokenizer, values: (1, 2, 3)
    ), patch.object(model_runner_base, "BatchedInferencePipeline", _SequentialPipeline):
        for request_id, seconds, vad in specs:
            request = _engine_request(request_id, seconds, vad=vad)
            engine.add_request(request_id, request.request_data)
        outputs = _drive(engine)

    assert sorted(outputs) == sorted(request_id for request_id, _seconds, _vad in specs)
    for request_id in ("short-a", "short-b", "long"):
        assert len(outputs[request_id]) == 1
        assert outputs[request_id][0].finish_reason == "stop"
    # The three STT requests share one batched handler call; the VAD request is
    # partitioned by request type and takes the len==1 sequential path (the same
    # handler), which is the documented engine behaviour.
    long_payload = json.loads(outputs["long"][0].output)
    assert long_payload["text"] == "r30000r30000r5000"
    assert len(long_payload["segments"]) == 3
    assert outputs["vad"][0].vad_artifact.segments[0]["text"] == "sequential"

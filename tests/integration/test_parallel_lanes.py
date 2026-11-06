"""Parallel-lane execution and the whisper audit (Phase 8-5).

KittenTTS cannot batch (every cached graph has a static batch dimension of 1), so
a step runs one request per lane — ``max_batch_size`` lanes over one shared ONNX
session. These tests drive the real engine → scheduler → executor → handler seam
with stub models and pin what the lanes promise: outputs paired by request id, a
poisoned request erroring alone, and real concurrent wall-clock progress.

The second half pins the Phase 8-5 whisper audit: ``BatchedInferencePipeline`` is
a single-audio API, so the handler builds one pipeline per step and transcribes
each request once — that is the documented answer, not a missing batch.

Real-model numbers (KittenTTS lane/`intra_op` matrix, Silero lane verdict) are
recorded in ``docs/batching.md``; the suite stays hermetic and CPU-only.
"""

import json
import os
import sys
import threading
import time
from dataclasses import dataclass
from unittest.mock import patch

import numpy as np
import pytest

from harmonyspeech.common.config import DeviceConfig, ModelConfig
from harmonyspeech.common.inputs import (
    DetectVoiceActivityRequestInput,
    SpeechTranscribeRequestInput,
    TextToSpeechRequestInput,
)
from harmonyspeech.common.outputs import SpeechTranscriptionRequestOutput, TextToSpeechRequestOutput
from harmonyspeech.common.request import EngineRequest
from harmonyspeech.engine.harmonyspeech_engine import HarmonySpeechEngine
from harmonyspeech.task_handler import model_runner_base
from harmonyspeech.task_handler.model_runner_base import ModelRunnerBase
from tests.audio_fixtures import tiny_wav_b64

# ---------------------------------------------------------------------------
# Harness: real engine, stub model executors (same pattern as 5-1 invariants)
# ---------------------------------------------------------------------------


class StubExecutor:
    def __init__(self, runner):
        self.runner = runner

    def execute_model(self, requests_to_batch):
        return self.runner.execute_model(requests_to_batch)

    async def execute_model_async(self, requests_to_batch):
        return self.runner.execute_model(requests_to_batch)

    def check_health(self):
        return {"status": "ok"}


def _model_config(name: str, model_type: str, max_batch_size: int) -> ModelConfig:
    return ModelConfig(
        name=name,
        model="dummy/model",
        model_type=model_type,
        max_batch_size=max_batch_size,
        device_config=DeviceConfig("cpu"),
        dtype="float32",
    )


def _runner(model, model_type: str, max_batch_size: int, name: str):
    runner = ModelRunnerBase.__new__(ModelRunnerBase)
    runner.model = model
    runner.device = "cpu"
    runner.model_config = _model_config(name, model_type, max_batch_size)
    runner._baseline_state = None
    return runner


def _build_engine(config, runner):
    with patch("harmonyspeech.executor.cpu_executor.CPUExecutor._init_worker"):
        engine = HarmonySpeechEngine(model_configs=[config], log_stats=False)
    engine.model_executors = {config.name: StubExecutor(runner)}
    return engine


def _tts_request(request_id: str, model: str, text: str) -> EngineRequest:
    return EngineRequest(
        request_id=request_id,
        request_data=TextToSpeechRequestInput(
            request_id=request_id,
            requested_model=model,
            model=model,
            input_text=text,
            mode="single_speaker_tts",
        ),
        arrival_time=0.0,
    )


def _drive(engine, max_steps: int = 12) -> dict[str, list]:
    collected: dict[str, list] = {}
    for _ in range(max_steps):
        if not engine.has_unfinished_requests():
            break
        outputs, _forwarded = engine.step()
        for output in outputs:
            collected.setdefault(output.request_id, []).append(output)
    return collected


class LaneKittenStub:
    """KittenTTS stand-in that records call concurrency and encodes the text.

    ``time.sleep`` releases the GIL, so this measures the lane *dispatch*
    (requests really overlap), not ONNX compute — the real-model speedups are in
    ``docs/batching.md``.
    """

    sample_rate = 24000

    def __init__(self, delay: float = 0.03):
        self.delay = delay
        self._lock = threading.Lock()
        self.active = 0
        self.peak = 0
        self.calls = []

    def generate(self, text=None, voice=None, speed=1.0, clean_text=True):
        with self._lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.calls.append(text)
        try:
            time.sleep(self.delay)
        finally:
            with self._lock:
                self.active -= 1
        value = (sum(ord(char) for char in text) % 997) / 997.0
        return np.full(1200, value, dtype=np.float32)


def _expected_value(text: str) -> float:
    return (sum(ord(char) for char in text) % 997) / 997.0


# ---------------------------------------------------------------------------
# KittenTTS lanes: pairing, isolation, real concurrency
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_eight_concurrent_kitten_requests_pair_outputs_by_request_id():
    model = LaneKittenStub()
    config = _model_config("kitten", "KittenTTSSynthesizer", max_batch_size=8)
    runner = _runner(model, "KittenTTSSynthesizer", 8, "kitten")
    engine = _build_engine(config, runner)

    texts = {f"lane-{index}": f"sentence number {index}" for index in range(8)}
    for request_id, text in texts.items():
        engine.add_request(request_id, _tts_request(request_id, "kitten", text).request_data)

    outputs = _drive(engine)

    assert sorted(outputs) == sorted(texts)
    for request_id, text in texts.items():
        assert len(outputs[request_id]) == 1
        output = outputs[request_id][0]
        assert output.finish_reason == "stop"
        assert isinstance(output, TextToSpeechRequestOutput)
        assert float(output.audio_artifact.waveform[0, 0]) == pytest.approx(
            _expected_value(text), abs=1e-6
        )
    # One batch of eight reached the model, and the lanes really overlapped.
    assert sorted(model.calls) == sorted(texts.values())
    assert model.peak > 1, "lanes did not run concurrently"


@pytest.mark.integration
def test_poisoned_kitten_request_errors_only_itself():
    class PoisonedKitten(LaneKittenStub):
        def generate(self, text=None, voice=None, speed=1.0, clean_text=True):
            if text == "explode":
                raise RuntimeError("this lane's synthesis failed")
            return super().generate(text=text, voice=voice, speed=speed, clean_text=clean_text)

    model = PoisonedKitten()
    config = _model_config("kitten", "KittenTTSSynthesizer", max_batch_size=8)
    runner = _runner(model, "KittenTTSSynthesizer", 8, "kitten")
    engine = _build_engine(config, runner)

    texts = {"ok-1": "alpha", "ok-2": "beta", "poison": "explode", "ok-3": "gamma"}
    for request_id, text in texts.items():
        engine.add_request(request_id, _tts_request(request_id, "kitten", text).request_data)

    outputs = _drive(engine)

    assert sorted(outputs) == sorted(texts)
    assert outputs["poison"][0].finish_reason == "error"
    assert "this lane's synthesis failed" in outputs["poison"][0].error
    for request_id in ("ok-1", "ok-2", "ok-3"):
        assert len(outputs[request_id]) == 1
        assert outputs[request_id][0].finish_reason == "stop"
        assert float(outputs[request_id][0].audio_artifact.waveform[0, 0]) == pytest.approx(
            _expected_value(texts[request_id]), abs=1e-6
        )


@pytest.mark.integration
@pytest.mark.skipif(os.cpu_count() < 4, reason="lane speedup needs >= 4 cores")
def test_kitten_lanes_beat_a_sequential_loop():
    """Eight lanes must finish the same work at least twice as fast as the loop.

    The stub sleeps (GIL released) for the length of a real mini-variant
    inference, so this asserts the mechanism: requests that used to be processed
    one after another now overlap. The measured real-model speedup (1.71x at the
    shipped 4 lanes, 1.91x at 8 lanes, mini on a 12-core host) is recorded in
    ``docs/batching.md``.
    """
    delay = 0.12
    inputs = [(f"sentence number {index}", "Jasper", 1.0) for index in range(8)]
    requests = [_tts_request(f"speed-{index}", "kitten", text) for index, (text, _v, _s) in enumerate(inputs)]

    sequential_runner = _runner(LaneKittenStub(delay=delay), "KittenTTSSynthesizer", 1, "kitten")
    sequential_runner._step_result_ids = None  # no engine step scope: guard off
    started = time.perf_counter()
    sequential_results = sequential_runner._execute_kittentts_synthesizer(inputs, requests)
    sequential_wall = time.perf_counter() - started

    lane_runner = _runner(LaneKittenStub(delay=delay), "KittenTTSSynthesizer", 8, "kitten")
    lane_runner._step_result_ids = None
    started = time.perf_counter()
    lane_results = lane_runner._execute_kittentts_synthesizer(inputs, requests)
    lane_wall = time.perf_counter() - started

    assert [result.request_id for result in lane_results] == [request.request_id for request in requests]
    assert [result.request_id for result in sequential_results] == [request.request_id for request in requests]
    assert all(result.result_data.finish_reason == "stop" for result in lane_results)
    # Same answers through both paths (pairing survives the lane shuffle).
    for lane_result, sequential_result in zip(lane_results, sequential_results, strict=True):
        assert float(lane_result.result_data.audio_artifact.waveform[0, 0]) == pytest.approx(
            float(sequential_result.result_data.audio_artifact.waveform[0, 0]), abs=1e-6
        )
    assert sequential_wall / lane_wall >= 2.0, (
        f"lanes were only {sequential_wall / lane_wall:.2f}x faster "
        f"({sequential_wall:.2f}s -> {lane_wall:.2f}s)"
    )


# ---------------------------------------------------------------------------
# Whisper audit: one pipeline per step, one transcribe call per request
# ---------------------------------------------------------------------------


@dataclass
class _FakeSegment:
    text: str


@dataclass
class _FakeInfo:
    language: str = "en"
    language_probability: float = 0.99
    duration: float = 0.3


class _CountingPipeline:
    """Stands in for faster-whisper's single-audio ``BatchedInferencePipeline``."""

    constructed = 0
    instances: list = []

    def __init__(self, model=None):
        type(self).constructed += 1
        self.model = model
        self.last_speech_timestamp = 0.0
        self.calls: list[dict] = []
        type(self).instances.append(self)

    def transcribe(self, audio, batch_size=16):
        self.calls.append(
            {
                "seconds": len(audio) / 16000.0,
                "batch_size": batch_size,
                "cursor_at_entry": self.last_speech_timestamp,
            }
        )
        # faster-whisper advances this cursor for word timestamps; the handler
        # must reset it per request.
        self.last_speech_timestamp = 9.9
        key = "x" if len(audio) / 16000.0 < 0.25 else "y"
        return [_FakeSegment(TRANSCRIPTS[key])], _FakeInfo()


TRANSCRIPTS = {"x": "first transcript", "y": "second transcript"}


def _transcribe_request(request_id: str, seconds: float) -> EngineRequest:
    return EngineRequest(
        request_id=request_id,
        request_data=SpeechTranscribeRequestInput(
            request_id=request_id,
            requested_model="whisper",
            model="whisper",
            input_audio=tiny_wav_b64(16000, seconds),
        ),
        arrival_time=0.0,
    )


def _reset_pipeline_spy() -> None:
    _CountingPipeline.constructed = 0
    _CountingPipeline.instances = []


@pytest.mark.integration
def test_whisper_step_builds_one_pipeline_and_transcribes_each_request_once():
    """The STT step loops (single-audio API) but shares one pipeline per step."""
    _reset_pipeline_spy()
    config = _model_config("whisper", "FasterWhisper", max_batch_size=4)
    runner = _runner(object(), "FasterWhisper", 4, "whisper")
    engine = _build_engine(config, runner)

    requests = [("stt-x", 0.10), ("stt-y", 0.40), ("stt-x2", 0.12), ("stt-y2", 0.35)]
    with patch.object(model_runner_base, "BatchedInferencePipeline", _CountingPipeline):
        for request_id, seconds in requests:
            request = _transcribe_request(request_id, seconds)
            engine.add_request(request_id, request.request_data)
        outputs = _drive(engine)

    assert sorted(outputs) == sorted(request_id for request_id, _seconds in requests)
    assert all(len(entries) == 1 for entries in outputs.values())
    assert all(isinstance(entries[0], SpeechTranscriptionRequestOutput) for entries in outputs.values())
    # One transcript per request, paired by id (not by position in the step).
    assert json.loads(outputs["stt-x"][0].output)["text"] == TRANSCRIPTS["x"]
    assert json.loads(outputs["stt-y"][0].output)["text"] == TRANSCRIPTS["y"]
    assert json.loads(outputs["stt-x2"][0].output)["text"] == TRANSCRIPTS["x"]
    assert json.loads(outputs["stt-y2"][0].output)["text"] == TRANSCRIPTS["y"]

    assert _CountingPipeline.constructed == 1, "one pipeline per step, not one per request"
    pipeline = _CountingPipeline.instances[0]
    # One transcribe call per request, each on that request's own audio.
    assert len(pipeline.calls) == 4
    assert [round(call["seconds"], 2) for call in pipeline.calls] == [0.1, 0.4, 0.12, 0.35]
    assert all(call["batch_size"] == 16 for call in pipeline.calls)
    # The pipeline's word-timestamp cursor is reset before every request.
    assert all(call["cursor_at_entry"] == 0.0 for call in pipeline.calls)


@pytest.mark.integration
def test_whisper_vad_step_takes_the_same_per_request_path():
    """A pipeline VAD request routed to whisper uses the same loop and pipeline."""
    _reset_pipeline_spy()
    config = _model_config("whisper", "FasterWhisper", max_batch_size=4)
    runner = _runner(object(), "FasterWhisper", 4, "whisper")
    engine = _build_engine(config, runner)

    with patch.object(model_runner_base, "BatchedInferencePipeline", _CountingPipeline):
        for request_id, seconds in (("vad-a", 0.10), ("vad-b", 0.40)):
            request_data = DetectVoiceActivityRequestInput(
                request_id=request_id,
                requested_model="whisper",
                model="whisper",
                input_audio=tiny_wav_b64(16000, seconds),
            )
            engine.add_request(request_id, request_data)
        outputs = _drive(engine)

    assert sorted(outputs) == ["vad-a", "vad-b"]
    assert all(entries[0].finish_reason == "stop" for entries in outputs.values())
    # A VAD request routed here carries whisper segments, like the STT path.
    assert [segment["text"] for segment in outputs["vad-a"][0].vad_artifact.segments] == [
        TRANSCRIPTS["x"]
    ]
    assert [segment["text"] for segment in outputs["vad-b"][0].vad_artifact.segments] == [
        TRANSCRIPTS["y"]
    ]

    assert _CountingPipeline.constructed == 1
    pipeline = _CountingPipeline.instances[0]
    # One transcribe call per request, on that request's own audio (0.10 s and
    # 0.40 s of the fixture, so the pairing is visible in the calls).
    assert len(pipeline.calls) == 2
    assert [round(call["seconds"], 1) for call in pipeline.calls] == [0.1, 0.4]
    assert all(call["batch_size"] == 16 for call in pipeline.calls)
    assert all(call["cursor_at_entry"] == 0.0 for call in pipeline.calls)


if __name__ == "__main__":  # pragma: no cover - debugging aid
    sys.exit(pytest.main([__file__, "-q"]))


if __name__ == "__main__":  # pragma: no cover - debugging aid
    sys.exit(pytest.main([__file__, "-q"]))

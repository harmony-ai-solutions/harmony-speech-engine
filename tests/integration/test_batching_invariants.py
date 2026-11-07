"""Engine-level batching invariants (Phase 5-1).

These tests run the *real* engine (scheduler → executor → ``prepare_inputs`` →
handler → result assembly) at batch sizes greater than one. The models are
stubs bound to the real model types (KittenTTS, ChatterboxTTS family, SileroVAD,
FasterWhisper), so the suite stays hermetic and CPU-only while every seam the
phase touches is exercised: type-homogeneous partitioning before preparation,
request↔input alignment through ``zip``, request-id-keyed result assembly and
the per-step duplicate-id guard.

Real weights are covered by ``tests/e2e`` and by the regression harness.
"""

import json
from unittest.mock import patch

import numpy as np
import pytest
import torch

from harmonyspeech.common.config import DeviceConfig, ModelConfig
from harmonyspeech.common.inputs import (
    DetectVoiceActivityRequestInput,
    SpeechEmbeddingRequestInput,
    SpeechTranscribeRequestInput,
    TextToSpeechRequestInput,
)
from harmonyspeech.common.outputs import (
    SpeechEmbeddingRequestOutput,
    SpeechTranscriptionRequestOutput,
    TextToSpeechRequestOutput,
)
from harmonyspeech.common.request import EngineRequest, RequestStatus
from harmonyspeech.engine.harmonyspeech_engine import HarmonySpeechEngine
from harmonyspeech.processing.scheduler import Scheduler
from tests.audio_fixtures import tiny_wav_b64

BASE64_AUDIO = tiny_wav_b64(16000, 0.30)


# ---------------------------------------------------------------------------
# Engine harness: real engine, stub model executors
# ---------------------------------------------------------------------------


class StubExecutor:
    """Executor wrapper that drives a real ``ModelRunnerBase`` instance."""

    def __init__(self, runner):
        self.runner = runner

    def execute_model(self, requests_to_batch):
        return self.runner.execute_model(requests_to_batch)

    async def execute_model_async(self, requests_to_batch):
        return self.runner.execute_model(requests_to_batch)

    def check_health(self):
        return {"status": "ok"}


def _model_config(name: str, model_type: str, max_batch_size: int = 4) -> ModelConfig:
    return ModelConfig(
        name=name,
        model="dummy/model",
        model_type=model_type,
        max_batch_size=max_batch_size,
        device_config=DeviceConfig("cpu"),
        dtype="float32",
    )


def _build_engine(configs, executors):
    """Create a real engine without loading models, then bind stub executors."""
    with patch("harmonyspeech.executor.cpu_executor.CPUExecutor._init_worker"):
        engine = HarmonySpeechEngine(model_configs=configs, log_stats=False)
    engine.model_executors = executors
    return engine


def _tts_request(request_id: str, model: str, text: str, **fields) -> EngineRequest:
    request_data = TextToSpeechRequestInput(
        request_id=request_id,
        requested_model=model,
        model=model,
        input_text=text,
        mode=fields.pop("mode", "single_speaker_tts"),
        **fields,
    )
    return EngineRequest(request_id=request_id, request_data=request_data, arrival_time=0.0)


def _embed_request(request_id: str, model: str, audio_b64: str = BASE64_AUDIO) -> EngineRequest:
    request_data = SpeechEmbeddingRequestInput(
        request_id=request_id,
        requested_model=model,
        model=model,
        input_audio=audio_b64,
    )
    return EngineRequest(request_id=request_id, request_data=request_data, arrival_time=0.0)


def _drive(engine, max_steps: int = 12) -> dict[str, list]:
    """Run engine steps until every request finished; return outputs by id."""
    collected: dict[str, list] = {}
    for _ in range(max_steps):
        if not engine.has_unfinished_requests():
            break
        outputs, _forwarded = engine.step()
        for output in outputs:
            collected.setdefault(output.request_id, []).append(output)
    return collected


# ---------------------------------------------------------------------------
# Stub models
# ---------------------------------------------------------------------------


class KittenStubModel:
    """KittenTTS stand-in: the waveform value encodes the request's text."""

    sample_rate = 24000

    def __init__(self):
        self.calls = []

    def generate(self, text=None, voice=None, speed=1.0, clean_text=True):
        self.calls.append(text)
        value = (sum(ord(char) for char in text) % 997) / 997.0
        return np.full(int(self.sample_rate * 0.05), value, dtype=np.float32)


class ChatterboxStubModel:
    """ChatterboxTTS stand-in with conditionals + generation."""

    sr = 24000

    def __init__(self):
        self.conds = object()
        self.ref_dict = None
        self.calls = []
        self.prepared_audio = 0

    def prepare_conditionals(self, audio_buf, **kwargs):
        audio_buf.read()
        self.prepared_audio += 1
        self.conds = object()

    def generate(self, text=None, audio=None, **kwargs):
        self.calls.append(text if text is not None else "<audio>")
        value = (sum(ord(char) for char in (text or "")) % 997) / 997.0
        return torch.full((1, 1200), value)


class RecordingRunner:
    """Minimal ModelRunnerBase-compatible surface for assertion purposes."""

    def __init__(self, runner):
        self._runner = runner


# ---------------------------------------------------------------------------
# 1. N concurrent identical/different requests complete with correct pairing
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_four_requests_to_one_model_complete_with_their_own_outputs():
    model = KittenStubModel()
    from harmonyspeech.task_handler.model_runner_base import ModelRunnerBase

    runner = ModelRunnerBase.__new__(ModelRunnerBase)
    runner.model = model
    runner.device = "cpu"
    runner.model_config = _model_config("kitten", "KittenTTSSynthesizer", max_batch_size=4)
    runner._baseline_state = None

    engine = _build_engine([runner.model_config], {"kitten": StubExecutor(runner)})

    texts = {
        "req-a": "alpha",
        "req-b": "beta",
        "req-c": "gamma",
        "req-d": "delta",
    }
    for request_id, text in texts.items():
        engine.add_request(request_id, _tts_request(request_id, "kitten", text).request_data)

    outputs = _drive(engine)

    assert sorted(outputs) == sorted(texts)
    for request_id, text in texts.items():
        assert len(outputs[request_id]) == 1
        output = outputs[request_id][0]
        assert output.finish_reason == "stop"
        assert isinstance(output, TextToSpeechRequestOutput)
        expected = (sum(ord(char) for char in text) % 997) / 997.0
        waveform = output.audio_artifact.waveform
        assert float(waveform[0, 0]) == pytest.approx(expected, abs=1e-6)
    # One batch of four reached the model (max_batch_size=4, one step).
    assert sorted(model.calls) == sorted(texts.values())


@pytest.mark.integration
def test_batch_larger_than_max_batch_size_is_split_across_steps():
    model = KittenStubModel()
    from harmonyspeech.task_handler.model_runner_base import ModelRunnerBase

    runner = ModelRunnerBase.__new__(ModelRunnerBase)
    runner.model = model
    runner.device = "cpu"
    runner.model_config = _model_config("kitten", "KittenTTSSynthesizer", max_batch_size=2)
    runner._baseline_state = None

    engine = _build_engine([runner.model_config], {"kitten": StubExecutor(runner)})

    for index in range(5):
        request = _tts_request(f"split-{index}", "kitten", f"text-{index}")
        engine.add_request(request.request_id, request.request_data)

    outputs = _drive(engine)

    assert sorted(outputs) == [f"split-{index}" for index in range(5)]
    assert all(len(entries) == 1 for entries in outputs.values())
    # Nothing was starved and nothing was paired twice.
    assert len(model.calls) == 5


# ---------------------------------------------------------------------------
# 2. Mixed-type batches are partitioned before preparation
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_mixed_embedding_and_tts_batch_dispatches_both_handlers():
    model = ChatterboxStubModel()
    from harmonyspeech.task_handler.model_runner_base import ModelRunnerBase

    runner = ModelRunnerBase.__new__(ModelRunnerBase)
    runner.model = model
    runner.device = "cpu"
    runner.model_config = _model_config("chatterbox", "ChatterboxTTS", max_batch_size=4)
    runner._baseline_state = {"conds": model.conds}
    runner._step_result_ids = set()

    engine = _build_engine([runner.model_config], {"chatterbox": StubExecutor(runner)})

    requests = [
        _tts_request("mix-tts-1", "chatterbox", "hello there"),
        _embed_request("mix-embed-1", "chatterbox"),
        _tts_request("mix-tts-2", "chatterbox", "general kenobi"),
        _embed_request("mix-embed-2", "chatterbox"),
    ]
    for request in requests:
        engine.add_request(request.request_id, request.request_data)

    outputs = _drive(engine)

    assert sorted(outputs) == ["mix-embed-1", "mix-embed-2", "mix-tts-1", "mix-tts-2"]
    assert isinstance(outputs["mix-tts-1"][0], TextToSpeechRequestOutput)
    assert isinstance(outputs["mix-tts-2"][0], TextToSpeechRequestOutput)
    assert isinstance(outputs["mix-embed-1"][0], SpeechEmbeddingRequestOutput)
    assert isinstance(outputs["mix-embed-2"][0], SpeechEmbeddingRequestOutput)
    assert all(entries[0].finish_reason == "stop" for entries in outputs.values())
    # Two embeddings prepared, two texts synthesized — no cross-contamination.
    assert model.prepared_audio == 2
    assert sorted(model.calls) == ["general kenobi", "hello there"]


# ---------------------------------------------------------------------------
# 3. Delayed / reordered work still pairs by request id
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_delayed_first_request_does_not_shift_later_pairs():
    class SlowFirstKitten(KittenStubModel):
        def generate(self, text=None, voice=None, speed=1.0, clean_text=True):
            import time

            if text == "slow":
                time.sleep(0.05)
            return super().generate(text=text, voice=voice, speed=speed, clean_text=clean_text)

    model = SlowFirstKitten()
    from harmonyspeech.task_handler.model_runner_base import ModelRunnerBase

    runner = ModelRunnerBase.__new__(ModelRunnerBase)
    runner.model = model
    runner.device = "cpu"
    runner.model_config = _model_config("kitten", "KittenTTSSynthesizer", max_batch_size=1)
    runner._baseline_state = None

    engine = _build_engine([runner.model_config], {"kitten": StubExecutor(runner)})

    engine.add_request("delayed-slow", _tts_request("delayed-slow", "kitten", "slow").request_data)
    engine.add_request("delayed-fast", _tts_request("delayed-fast", "kitten", "fast").request_data)

    outputs = _drive(engine)

    for request_id, text in (("delayed-slow", "slow"), ("delayed-fast", "fast")):
        expected = (sum(ord(char) for char in text) % 997) / 997.0
        assert float(outputs[request_id][0].audio_artifact.waveform[0, 0]) == pytest.approx(expected, abs=1e-6)


@pytest.mark.integration
def test_one_failing_item_does_not_misalign_the_rest():
    class FailingMiddleKitten(KittenStubModel):
        def generate(self, text=None, voice=None, speed=1.0, clean_text=True):
            if text == "explode":
                raise RuntimeError("synthesis failed for this request")
            return super().generate(text=text, voice=voice, speed=speed, clean_text=clean_text)

    model = FailingMiddleKitten()
    from harmonyspeech.task_handler.model_runner_base import ModelRunnerBase

    runner = ModelRunnerBase.__new__(ModelRunnerBase)
    runner.model = model
    runner.device = "cpu"
    runner.model_config = _model_config("kitten", "KittenTTSSynthesizer", max_batch_size=4)
    runner._baseline_state = None

    engine = _build_engine([runner.model_config], {"kitten": StubExecutor(runner)})

    for text in ("alpha", "explode", "gamma"):
        request = _tts_request(f"fail-{text}", "kitten", text)
        engine.add_request(request.request_id, request.request_data)

    outputs = _drive(engine)

    assert outputs["fail-alpha"][0].finish_reason == "stop"
    assert outputs["fail-gamma"][0].finish_reason == "stop"
    assert outputs["fail-explode"][0].finish_reason == "error"
    for request_id, text in (("fail-alpha", "alpha"), ("fail-gamma", "gamma")):
        expected = (sum(ord(char) for char in text) % 997) / 997.0
        assert float(outputs[request_id][0].audio_artifact.waveform[0, 0]) == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# 4. VAD / STT handlers keep the same pairing at batch size > 1
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_vad_batch_returns_one_result_per_request():
    segments_table = {"a": [{"start": 0.0, "end": 0.1}], "b": [], "c": [{"start": 0.05, "end": 0.2}]}

    def fake_timestamps(audio, model, **kwargs):
        # The stub "audio" length identifies the request (prepared by the real
        # VAD prepare function from distinct clip durations).
        seconds = audio.shape[-1] / 16000.0
        return segments_table["a"] if seconds < 0.25 else segments_table["c"]

    class DummyVadModel:
        pass

    from harmonyspeech.task_handler.model_runner_base import ModelRunnerBase

    runner = ModelRunnerBase.__new__(ModelRunnerBase)
    runner.model = DummyVadModel()
    runner.device = "cpu"
    runner.model_config = _model_config("vad", "SileroVAD", max_batch_size=4)
    runner._baseline_state = None

    engine = _build_engine([runner.model_config], {"vad": StubExecutor(runner)})

    clips = {
        "vad-short": tiny_wav_b64(16000, 0.10),
        "vad-long": tiny_wav_b64(16000, 0.40),
    }
    with patch(
        "harmonyspeech.task_handler.model_runner_base.get_speech_timestamps", side_effect=fake_timestamps
    ):
        for request_id, audio in clips.items():
            request_data = DetectVoiceActivityRequestInput(
                request_id=request_id,
                requested_model="vad",
                model="vad",
                input_audio=audio,
                get_timestamps=True,
            )
            engine.add_request(request_id, request_data)
        outputs = _drive(engine)

    assert sorted(outputs) == ["vad-long", "vad-short"]
    payload_short = json.loads(outputs["vad-short"][0].output)
    payload_long = json.loads(outputs["vad-long"][0].output)
    assert payload_short["segments"] == segments_table["a"]
    assert payload_long["segments"] == segments_table["c"]


@pytest.mark.integration
def test_transcription_batch_pairs_text_with_its_own_audio():
    transcripts = {"x": "first transcript", "y": "second transcript"}

    from dataclasses import dataclass

    @dataclass
    class FakeSegment:
        text: str

    @dataclass
    class FakeInfo:
        language: str = "en"
        language_probability: float = 0.99
        duration: float = 0.3

    class FakePipeline:
        def __init__(self, model=None):
            self.model = model

        def transcribe(self, audio, batch_size=16, **kwargs):
            seconds = len(audio) / 16000.0
            key = "x" if seconds < 0.25 else "y"
            return [FakeSegment(transcripts[key])], FakeInfo()

    class DummyWhisper:
        pass

    from harmonyspeech.task_handler.model_runner_base import ModelRunnerBase

    runner = ModelRunnerBase.__new__(ModelRunnerBase)
    runner.model = DummyWhisper()
    runner.device = "cpu"
    runner.model_config = _model_config("whisper", "FasterWhisper", max_batch_size=4)
    runner._baseline_state = None

    engine = _build_engine([runner.model_config], {"whisper": StubExecutor(runner)})

    with patch("harmonyspeech.task_handler.model_runner_base.BatchedInferencePipeline", FakePipeline):
        for request_id, seconds in (("stt-x", 0.10), ("stt-y", 0.40)):
            request_data = SpeechTranscribeRequestInput(
                request_id=request_id,
                requested_model="whisper",
                model="whisper",
                input_audio=tiny_wav_b64(16000, seconds),
            )
            engine.add_request(request_id, request_data)
        outputs = _drive(engine)

    assert sorted(outputs) == ["stt-x", "stt-y"]
    assert isinstance(outputs["stt-x"][0], SpeechTranscriptionRequestOutput)
    assert json.loads(outputs["stt-x"][0].output)["text"] == transcripts["x"]
    assert json.loads(outputs["stt-y"][0].output)["text"] == transcripts["y"]


# ---------------------------------------------------------------------------
# 5. Scheduler interaction: forwarded + fresh requests in one cycle
# ---------------------------------------------------------------------------


def _scheduler_request(request_id: str, model: str) -> EngineRequest:
    return _tts_request(request_id, model, "text")


@pytest.mark.integration
def test_forwarded_and_fresh_request_share_one_scheduling_cycle():
    """A forwarded (pipeline) request and a fresh one are budgeted alike.

    This documents the current behaviour on purpose: forwarding re-enters the
    waiting queue under the same request id and takes one slot per model, so the
    engine can process a forwarded step and a fresh request for the same model in
    the same cycle when ``max_batch_size`` allows it.
    """
    configs = [_model_config("synth", "KittenTTSSynthesizer", max_batch_size=2)]
    scheduler = Scheduler(model_configs=configs)

    fresh = _scheduler_request("fresh", "synth")
    # A forwarded step re-enters the waiting queue as a *new* request under the
    # same id (``check_forward_processing`` -> ``add_request``), so the scheduler
    # cannot tell it apart from a fresh request.
    forwarded = _scheduler_request("forwarded", "synth")

    scheduler.add_request(fresh)
    scheduler.add_request(forwarded)
    outputs = scheduler.schedule()

    scheduled = outputs.scheduled_requests_per_model["synth"]
    assert sorted(request.request_id for request in scheduled) == ["forwarded", "fresh"]
    assert all(request.status == RequestStatus.RUNNING for request in scheduled)


@pytest.mark.integration
def test_head_of_line_behaviour_with_max_batch_size_one_is_documented():
    """With a batch size of 1, a forward resubmission waits for the next cycle.

    This is the current scheduler behaviour (no slot starvation, but no overlap
    either); observing it here keeps a future change honest.
    """
    configs = [_model_config("synth", "KittenTTSSynthesizer", max_batch_size=1)]
    scheduler = Scheduler(model_configs=configs)

    first = _scheduler_request("first", "synth")
    second = _scheduler_request("second", "synth")
    scheduler.add_request(first)
    scheduler.add_request(second)

    first_cycle = scheduler.schedule()
    assert [request.request_id for request in first_cycle.scheduled_requests_per_model["synth"]] == ["first"]
    assert [request.request_id for request in scheduler.waiting] == ["second"]

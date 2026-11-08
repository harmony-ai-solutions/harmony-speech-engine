"""Whisper exhaustion semantics on CPU (Phase 10-5).

These are the local proof of the behaviour the owner asked about, before the
benchmark-host run: with deliberately tight caps (M=2 workers, window batch=4,
admission/queue depth=16) a heterogeneous flood of 100 requests must complete
(or time out cleanly), lose nothing and never pair one request's transcript with
another's audio. A memory-budget trip must shrink admission, degrade latency
gracefully and never crash. An abort storm must error only the aborted requests.

Everything runs through the **real** engine loop, scheduler, lane admission and
handler; only the model surfaces are stubbed (CT2 binding, VAD, feature
extractor, pipeline). No weights are downloaded and no CUDA is required.
"""

import json
import time
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from harmonyspeech.common.artifacts import AudioArtifact
from harmonyspeech.common.config import DeviceConfig, ModelConfig
from harmonyspeech.common.inputs import SpeechTranscribeRequestInput
from harmonyspeech.common.request import EngineRequest, RequestStatus
from harmonyspeech.engine.harmonyspeech_engine import HarmonySpeechEngine
from harmonyspeech.processing.scheduler import Scheduler
from harmonyspeech.task_handler import model_runner_base, whisper_batching
from harmonyspeech.task_handler.model_runner_base import ModelRunnerBase
from scripts.bench_throughput import whisper_flood_scenario
from tests.unit.inference_flow.test_whisper_window_batching import (
    SAMPLE_RATE,
    _clips_for,
    _FakeCT2,
    _FakeWhisperModel,
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


class _TightBatcher(whisper_batching.WhisperWindowBatcher):
    """Window batcher for the tight-cap exhaustion scenario (cap 4).

    The cap travels through the shipped ``whisper_window_batch_size`` config key
    since Phase 11-0, so the double simply honours the value the runner passes
    (the test config sets 4).
    """

    def __init__(self, model, window_batch_size=4):
        super().__init__(model, window_batch_size=window_batch_size)


class _BudgetExecutor(_StubExecutor):
    """Stub executor that also carries the metadata ``get_vram_stats`` reads."""

    def __init__(self, runner, config):
        super().__init__(runner)
        self.model_config = config
        self.driver_worker = SimpleNamespace(model_runner=runner)


@dataclass
class _SequentialSegment:
    text: str


@dataclass
class _SequentialInfo:
    language: str = "en"
    language_probability: float = 0.99
    duration: float = 1.0


class _SequentialPipeline:
    """Stand-in for ``BatchedInferencePipeline`` (single-request fallback path)."""

    def __init__(self, model=None):
        self.model = model
        self.last_speech_timestamp = 0.0

    def transcribe(self, audio, batch_size=16, **kwargs):
        seconds = len(audio) / SAMPLE_RATE
        return [_SequentialSegment(f"seq{int(round(seconds * 1000))}")], _SequentialInfo(duration=seconds)


def _config(*, max_batch_size=16, num_workers=2, batched_inference=True, whisper_window_batch_size=4):
    return ModelConfig(
        name="whisper",
        model="dummy/whisper",
        model_type="FasterWhisper",
        max_batch_size=max_batch_size,
        num_workers=num_workers,
        batched_inference=batched_inference,
        # Tight cap of the exhaustion scenario, now a real per-model knob.
        whisper_window_batch_size=whisper_window_batch_size,
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


def _engine(config, runner, *, request_timeout_s=600.0):
    with patch("harmonyspeech.executor.cpu_executor.CPUExecutor._init_worker"):
        engine = HarmonySpeechEngine(
            model_configs=[config], log_stats=False, request_timeout_s=request_timeout_s
        )
    engine.model_executors = {config.name: _StubExecutor(runner)}
    return engine


def _artifact(seconds):
    import numpy as np

    return AudioArtifact.from_numpy(np.zeros(int(SAMPLE_RATE * seconds), dtype=np.float32), SAMPLE_RATE)


def _request(request_id, seconds, *, task="transcribe", language=None):
    return EngineRequest(
        request_id=request_id,
        request_data=SpeechTranscribeRequestInput(
            request_id=request_id,
            requested_model="whisper",
            model="whisper",
            input_audio=None,
            task=task,
            language=language,
        ),
        arrival_time=0.0,
        artifacts={"input_audio": _artifact(seconds)},
    )


def _drive(engine, max_steps=400):
    collected = {}
    for _ in range(max_steps):
        if not engine.has_unfinished_requests():
            break
        outputs, _forwarded = engine.step()
        for output in outputs:
            collected.setdefault(output.request_id, []).append(output)
    return collected


def _expected_text(seconds: float) -> str:
    """The fake CT2's transcript for one request (marker = window duration ms)."""
    full = int(seconds // 30)
    remainder = seconds - full * 30
    text = "r30000" * full
    if remainder > 0.0005:
        text += f"r{int(round(remainder * 1000))}"
    return text


def _patches():
    return (
        patch.object(whisper_batching, "get_speech_timestamps", _clips_for(0)),
        patch.object(whisper_batching, "get_suppressed_tokens", lambda tokenizer, values: (1, 2, 3)),
        patch.object(model_runner_base, "WhisperWindowBatcher", _TightBatcher),
        patch.object(model_runner_base, "BatchedInferencePipeline", _SequentialPipeline),
    )


def _start(patches):
    for active in patches:
        active.start()


def _stop(patches):
    for active in patches:
        active.stop()


# ---------------------------------------------------------------------------
# 1. 100-request heterogeneous flood against tight caps
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_tight_cap_100_request_flood_completes_with_zero_loss_and_zero_pairing_errors():
    config = _config(max_batch_size=16, num_workers=2, batched_inference=True)
    model = _FakeWhisperModel()
    runner = _runner(model, config)
    engine = _engine(config, runner)

    specs = whisper_flood_scenario(100, max_seconds=480.0)
    patches = _patches()
    _start(patches)
    try:
        for spec in specs:
            request_id = f"flood-{spec['index']}"
            request = _request(
                request_id,
                spec["duration_s"],
                task=spec["task"],
                language=spec["language"],
            )
            engine.add_request(request_id, request.request_data, artifacts=request.artifacts)
        outputs = _drive(engine)
    finally:
        _stop(patches)

    # Zero request loss and zero duplicate pairing: every submitted id has
    # exactly one output, and no id is missing.
    assert len(outputs) == len(specs)
    assert set(outputs) == {f"flood-{spec['index']}" for spec in specs}

    for spec in specs:
        request_id = f"flood-{spec['index']}"
        results = outputs[request_id]
        assert len(results) == 1, f"{request_id} produced {len(results)} outputs"
        result = results[0]
        assert result.finish_reason == "stop", (request_id, result.error)
        payload = json.loads(result.output)
        # Pairing proof: the transcript is built from *this* request's own
        # window markers (duration -> ceil(seconds/30) windows).
        assert payload["text"] == _expected_text(spec["duration_s"]), request_id
        assert len(payload["segments"]) == max(1, int(spec["duration_s"] // 30) + (spec["duration_s"] % 30 > 0))

    # No batch-level failure fell back; the window batcher did all the work.
    assert runner.whisper_lane_stats()["batch_fallbacks"] == 0


# ---------------------------------------------------------------------------
# 2. Timeout storm: every request resolves, none is lost
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_timeout_storm_resolves_every_request_cleanly():
    config = _config(max_batch_size=16, num_workers=2, batched_inference=True)
    model = _FakeWhisperModel()
    runner = _runner(model, config)
    engine = _engine(config, runner, request_timeout_s=0.001)

    request_ids = [f"timeout-{index}" for index in range(30)]
    for index, request_id in enumerate(request_ids):
        request = _request(request_id, 2.0 + index * 0.1)
        engine.add_request(request_id, request.request_data, artifacts=request.artifacts)

    # Make every deadline unambiguously overdue (Windows' monotonic clock has a
    # coarse resolution, so sleeping a millisecond is not deterministic).
    for request in engine.scheduler.waiting:
        request.deadline = time.monotonic() - 1.0
    outputs = _drive(engine)

    assert set(outputs) == set(request_ids)
    for request_id in request_ids:
        results = outputs[request_id]
        assert len(results) == 1
        assert results[0].finish_reason == "error"
        assert "timeout" in results[0].error.lower()


# ---------------------------------------------------------------------------
# 3. Memory-budget trip: admission shrinks, latency degrades, no crash
# ---------------------------------------------------------------------------


def _patch_cuda(monkeypatch, free: dict, total: int = 1000):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda index=None: 0)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda index=None: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda index=None: 0)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda index=None: SimpleNamespace(total_memory=total)
    )
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index=None: "fake-gpu")
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda index=None: (free["value"], total))


def _budget_engine(monkeypatch, *, budget=1000, free_state=None):
    free_state = free_state if free_state is not None else {"value": 600}
    config = _config(max_batch_size=16, num_workers=2, batched_inference=True)
    model = _FakeWhisperModel()
    # `get_vram_stats` only counts Whisper as VRAM when its native device is CUDA.
    model.ct2.device = "cuda"
    runner = _runner(model, config)
    runner.whisper_probe = {
        "peak_activation_bytes": 800,
        "per_window_bytes": 100,
        "measured": True,
        "windows_probed": 8,
    }
    engine = _engine(config, runner)
    engine.model_executors = {"whisper": _BudgetExecutor(runner, config)}
    engine.whisper_vram_budget_bytes = budget
    _patch_cuda(monkeypatch, free_state)
    return engine, config, runner


@pytest.mark.integration
def test_budget_trip_shrinks_admission_observably(monkeypatch):
    free = {"value": 600}
    engine, config, _runner_obj = _budget_engine(monkeypatch, budget=1000, free_state=free)

    assert engine.get_whisper_admission()["allowed_windows"] == 6
    free["value"] = 200
    assert engine.get_whisper_admission()["allowed_windows"] == 2
    free["value"] = 0
    # Over budget: the default draining floor keeps one window alive; an
    # explicit hard stop is available for exhaustion tests.
    assert engine.get_whisper_admission()["allowed_windows"] == 1
    assert engine.get_whisper_admission(min_windows=0)["allowed_windows"] == 0

    # The shrink is observable where it matters: how many requests the real
    # scheduler admits into one step.
    free["value"] = 200
    scheduler = Scheduler([config], whisper_admission=engine.get_whisper_admission)
    for index in range(10):
        scheduler.add_request(_request(f"budget-{index}", 10.0))
    admitted = scheduler.schedule().scheduled_requests_per_model["whisper"]
    assert len(admitted) == 2, "allowed_windows=2 must bound the step"
    assert len(scheduler.waiting) == 8


@pytest.mark.integration
def test_budget_trip_flood_completes_under_the_dragging_floor(monkeypatch):
    """Over budget, admission runs one window at a time but nothing is lost."""
    free = {"value": 0}  # over budget -> allowed_windows == 1 (draining floor)
    engine, config, runner = _budget_engine(monkeypatch, budget=1000, free_state=free)

    patches = _patches()
    _start(patches)
    try:
        request_ids = [f"trip-{index}" for index in range(24)]
        for index, request_id in enumerate(request_ids):
            request = _request(request_id, 4.0 + index * 0.05)
            engine.add_request(request_id, request.request_data, artifacts=request.artifacts)
        outputs = _drive(engine)
    finally:
        _stop(patches)

    assert set(outputs) == set(request_ids)
    for request_id in request_ids:
        assert len(outputs[request_id]) == 1
        assert outputs[request_id][0].finish_reason == "stop"
    assert engine.get_whisper_admission()["allowed_windows"] == 1


# ---------------------------------------------------------------------------
# 4. Abort storm: waiting aborts vanish cleanly, in-flight aborts error alone
# ---------------------------------------------------------------------------


class _AbortStormCT2(_FakeCT2):
    def __init__(self, abort_by_marker, **kwargs):
        super().__init__(**kwargs)
        self.abort_by_marker = dict(abort_by_marker)

    def generate(self, encoder_output, prompts, **kwargs):
        results = super().generate(encoder_output, prompts, **kwargs)
        for marker in encoder_output.markers:
            request = self.abort_by_marker.get(int(marker))
            if request is not None:
                request.status = RequestStatus.FINISHED_ABORTED
        return results


@pytest.mark.integration
def test_abort_storm_loses_no_unrelated_request_and_errors_only_aborted_ones():
    config = _config(max_batch_size=16, num_workers=2, batched_inference=True)
    model = _FakeWhisperModel()
    runner = _runner(model, config)
    engine = _engine(config, runner)

    # Unique short durations -> one unique window marker per request.
    durations = {f"abort-{index}": 1.0 + index * 0.1 for index in range(40)}
    requests = {
        request_id: _request(request_id, seconds) for request_id, seconds in durations.items()
    }

    # 10 requests are aborted while still waiting; 10 more are aborted while the
    # batcher is decoding their window; 20 must complete untouched.
    waiting_aborts = {f"abort-{index}" for index in range(0, 40, 4)}  # 10 ids
    in_flight_aborts = {f"abort-{index}" for index in range(2, 40, 4) if f"abort-{index}" not in waiting_aborts}
    in_flight_aborts = set(sorted(in_flight_aborts)[:10])
    healthy = set(durations) - waiting_aborts - in_flight_aborts

    abort_by_marker = {}
    ct2 = _AbortStormCT2(abort_by_marker)
    ct2.is_multilingual = True
    model.ct2 = ct2
    model.model = ct2
    from tests.unit.inference_flow.test_whisper_window_batching import _FakeHFTokenizer

    model.hf_tokenizer = _FakeHFTokenizer(token_text=ct2._token_text)

    patches = _patches()
    _start(patches)
    try:
        for request_id, request in requests.items():
            engine.add_request(request_id, request.request_data, artifacts=request.artifacts)
        # The engine builds its own EngineRequest objects; the abort markers must
        # point at *those* instances, not the caller-side stand-ins.
        engine_requests = {request.request_id: request for request in engine.scheduler.waiting}
        ct2.abort_by_marker = {
            int(round(durations[request_id] * 1000)): engine_requests[request_id]
            for request_id in in_flight_aborts
        }
        for request_id in waiting_aborts:
            engine.abort_request(request_id)
        outputs = _drive(engine)
    finally:
        _stop(patches)

    # Waiting aborts are removed cleanly (no output, no hang, no reschedule).
    assert set(outputs) == healthy | in_flight_aborts
    assert not (waiting_aborts & set(outputs))
    # No duplicate pairing anywhere.
    for request_id, results in outputs.items():
        assert len(results) == 1, f"{request_id} produced {len(results)} outputs"

    for request_id in healthy:
        result = outputs[request_id][0]
        assert result.finish_reason == "stop", (request_id, result.error)
        assert json.loads(result.output)["text"] == _expected_text(durations[request_id])
    for request_id in in_flight_aborts:
        result = outputs[request_id][0]
        assert result.finish_reason == "error"
        assert "aborted" in result.error.lower()


# ---------------------------------------------------------------------------
# 5. Fallback counter: a batch-level failure degrades, never loses a request
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_batcher_failure_falls_back_and_counts_once():
    config = _config(max_batch_size=16, num_workers=2, batched_inference=True)
    model = _FakeWhisperModel()
    runner = _runner(model, config)
    engine = _engine(config, runner)

    class _ExplodingBatcher(whisper_batching.WhisperWindowBatcher):
        def __init__(self, model):
            super().__init__(model)

        def transcribe_batch(self, audio_artifacts, requests):
            raise RuntimeError("simulated batch-level CT2 failure")

    request_ids = [f"fallback-{index}" for index in range(5)]
    patches = (
        patch.object(model_runner_base, "WhisperWindowBatcher", _ExplodingBatcher),
        patch.object(model_runner_base, "BatchedInferencePipeline", _SequentialPipeline),
    )
    _start(patches)
    try:
        for index, request_id in enumerate(request_ids):
            request = _request(request_id, 2.0 + index * 0.2)
            engine.add_request(request_id, request.request_data, artifacts=request.artifacts)
        outputs = _drive(engine)
    finally:
        _stop(patches)

    assert set(outputs) == set(request_ids)
    for request_id in request_ids:
        assert len(outputs[request_id]) == 1
        assert outputs[request_id][0].finish_reason == "stop"
        assert json.loads(outputs[request_id][0].output)["text"].startswith("seq")

    stats = runner.whisper_lane_stats()
    assert stats["batch_fallbacks"] == 1

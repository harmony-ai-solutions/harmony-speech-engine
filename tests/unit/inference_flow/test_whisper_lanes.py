"""Whisper execution lanes — CT2 workers, fairness and cancellation (Phase 10-1).

The suite pins the four promises of the phase:

* ``num_workers`` is plumbed config → loader → ``WhisperModel`` and defaults to
  ``1`` (exactly the pre-Phase-10 behaviour);
* the handler dispatches a step to an **M-bound** pool (M = ``num_workers``) and
  each worker consumes its transcribe generator to the end;
* admission selects between the ``<=30 s`` and ``>30 s`` lanes at a 3:1 credit
  ratio, composes with the Phase 10-3 memory governor, and leaves every other
  model type on its boolean rule;
* an abort between segments errors only the aborted request.

No model weights are downloaded: faster-whisper's pipeline is stubbed, the way
the Phase 8-5 handler tests already do it.
"""

import json
import threading
import time
from dataclasses import dataclass
from unittest.mock import patch

import numpy as np
import pytest

from harmonyspeech.common.artifacts import AudioArtifact
from harmonyspeech.common.config import DeviceConfig, EngineConfig, ModelConfig
from harmonyspeech.common.inputs import SpeechTranscribeRequestInput
from harmonyspeech.common.request import EngineRequest, RequestStatus
from harmonyspeech.engine.whisper_lane_metrics import render_whisper_lane_metrics
from harmonyspeech.processing.scheduler import (
    LaneCredits,
    Scheduler,
    plan_lane_admission,
    whisper_lane,
)
from harmonyspeech.task_handler import model_runner_base
from harmonyspeech.task_handler.model_runner_base import (
    ModelRunnerBase,
    WhisperLanePool,
    WhisperRequestAbortedError,
    _ct2_queue_state,
)

SAMPLE_RATE = 16000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _whisper_config(name: str = "whisper", *, max_batch_size: int = 16, num_workers: int | None = None):
    kwargs = {}
    if num_workers is not None:
        kwargs["num_workers"] = num_workers
    return ModelConfig(
        name=name,
        model="dummy/whisper",
        model_type="FasterWhisper",
        max_batch_size=max_batch_size,
        device_config=DeviceConfig("cpu"),
        dtype="float32",
        **kwargs,
    )


def _audio(seconds: float) -> AudioArtifact:
    return AudioArtifact.from_numpy(np.zeros(int(SAMPLE_RATE * seconds), dtype=np.float32), SAMPLE_RATE)


def _request(request_id: str, seconds: float, *, model: str = "whisper") -> EngineRequest:
    return EngineRequest(
        request_id=request_id,
        request_data=SpeechTranscribeRequestInput(
            request_id=request_id, requested_model=model, model=model, input_audio=None
        ),
        arrival_time=0.0,
        artifacts={"input_audio": _audio(seconds)},
    )


@dataclass
class _FakeSegment:
    text: str


@dataclass
class _FakeInfo:
    language: str = "en"
    language_probability: float = 0.99
    duration: float = 1.0


class _ConcurrencyProbe:
    """Shared gauge: how many fake transcribes overlap across pool threads."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.active = 0
        self.peak = 0
        self.calls = 0

    def enter(self) -> None:
        with self.lock:
            self.active += 1
            self.calls += 1
            self.peak = max(self.peak, self.active)

    def leave(self) -> None:
        with self.lock:
            self.active -= 1


def _pipeline_class(probe: _ConcurrencyProbe, *, segments=("alpha", "beta", "gamma"), delay=0.01):
    """Build a ``BatchedInferencePipeline`` stand-in sharing one probe."""

    class _Pipeline:
        constructed = 0

        def __init__(self, model=None):
            type(self).constructed += 1
            self.model = model
            self.last_speech_timestamp = 0.0

        def transcribe(self, audio, batch_size=16, **kwargs):
            probe.enter()

            def generator():
                try:
                    for index, text in enumerate(segments):
                        if index:
                            time.sleep(delay)
                        yield _FakeSegment(text)
                finally:
                    probe.leave()

            return generator(), _FakeInfo()

    return _Pipeline


# ---------------------------------------------------------------------------
# 1. Config plumbing: default 1, per-model override, loader pass-through
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_num_workers_defaults_to_one():
    """An unconfigured Whisper model keeps today's behaviour (CT2 serialises)."""
    assert _whisper_config().num_workers == 1


@pytest.mark.unit
@pytest.mark.parametrize("value", [2, 4])
def test_num_workers_override(value):
    assert _whisper_config(num_workers=value).num_workers == value


@pytest.mark.unit
@pytest.mark.parametrize("value", [0, -1, True, "4"])
def test_num_workers_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="num_workers"):
        _whisper_config(num_workers=value)


@pytest.mark.unit
def test_yaml_num_workers_is_per_model(tmp_path):
    yaml_file = tmp_path / "config.yml"
    yaml_file.write_text(
        """
model_configs:
  - name: turbo
    model: large-v3-turbo
    model_type: FasterWhisper
    max_batch_size: 16
    num_workers: 2
    dtype: float32
    device_config:
      device: cpu
  - name: tiny
    model: tiny
    model_type: FasterWhisper
    max_batch_size: 16
    num_workers: 4
    dtype: float32
    device_config:
      device: cpu
  - name: medium
    model: medium
    model_type: FasterWhisper
    max_batch_size: 16
    dtype: float32
    device_config:
      device: cpu
""",
        encoding="utf-8",
    )

    config = EngineConfig.load_config_from_yaml(str(yaml_file))
    by_name = {entry.name: entry for entry in config.model_configs}
    assert by_name["turbo"].num_workers == 2
    assert by_name["tiny"].num_workers == 4
    assert by_name["medium"].num_workers == 1  # unconfigured -> default


@pytest.mark.unit
@patch("harmonyspeech.modeling.loader.WhisperModel")
def test_loader_passes_num_workers_to_whisper_model(mock_whisper_cls):
    from harmonyspeech.modeling.loader import get_model

    get_model(_whisper_config(num_workers=4), DeviceConfig("cpu"))

    mock_whisper_cls.assert_called_once()
    _, kwargs = mock_whisper_cls.call_args
    assert kwargs["num_workers"] == 4


@pytest.mark.unit
def test_shipped_configs_carry_the_approved_initial_worker_counts():
    """Owner-approved 2026-11-07 initial values, pending the 10-5 benchmark."""
    import os

    import yaml

    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )

    def workers(config_name: str) -> dict[str, int]:
        with open(os.path.join(repo_root, config_name), encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        return {
            entry["name"]: entry.get("num_workers", 1)
            for entry in data["model_configs"]
            if entry["model_type"] == "FasterWhisper"
        }

    gpu = workers("config.gpu.yml")
    assert gpu["faster-whisper-large-v3-turbo"] == 2
    assert gpu["faster-whisper-tiny"] == 4

    cpu = workers("config.yml")
    assert cpu["faster-whisper-tiny"] == 4
    assert cpu["faster-whisper-medium"] == 1


# ---------------------------------------------------------------------------
# 2. Lane selection from the admission-time artifact
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "seconds,expected",
    [
        (0.1, "short"),
        (10.0, "short"),
        (30.0, "short"),  # boundary is inclusive
        (30.01, "long"),
        (120.0, "long"),
    ],
)
def test_whisper_lane_uses_artifact_duration(seconds, expected):
    assert whisper_lane(_request("r", seconds)) == expected


@pytest.mark.unit
def test_whisper_lane_treats_unknown_duration_as_long():
    request = EngineRequest(
        request_id="r",
        request_data=SpeechTranscribeRequestInput(
            request_id="r", requested_model="whisper", model="whisper"
        ),
        arrival_time=0.0,
    )
    assert whisper_lane(request) == "long"


# ---------------------------------------------------------------------------
# 3. Credit ratio and lane planning
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_lane_credits_drain_three_short_then_one_long():
    credits = LaneCredits()
    picks = [credits.pick(True, True) for _ in range(8)]
    assert picks == ["short", "short", "short", "long", "short", "short", "short", "long"]


@pytest.mark.unit
def test_lane_credits_serve_the_only_available_lane():
    credits = LaneCredits()
    assert [credits.pick(True, False) for _ in range(5)] == ["short"] * 5
    # A long request arriving now is not starved: only the remaining short
    # credit precedes it.
    assert credits.pick(True, True) == "short"
    assert credits.pick(True, True) == "long"


@pytest.mark.unit
def test_long_first_queue_does_not_starve_the_short_lane():
    """A long request queued first cannot fill the step ahead of shorts."""
    credits = LaneCredits()
    candidates = [(_request("long", 60.0), "long")] + [
        (_request(f"short-{index}", 5.0), "short") for index in range(6)
    ]

    selected, leftover = plan_lane_admission(candidates, 3, credits)

    assert [request.request_id for request in selected] == ["short-0", "short-1", "short-2"]
    assert [request.request_id for request in leftover] == ["long", "short-3", "short-4", "short-5"]

    # The next batch reaches the long request after exactly the short credit.
    selected_2, _leftover_2 = plan_lane_admission(
        [(request, whisper_lane(request)) for request in leftover], 3, credits
    )
    assert selected_2[0].request_id == "long"


@pytest.mark.unit
def test_plan_lane_admission_preserves_fifo_within_a_lane():
    credits = LaneCredits()
    candidates = [
        (_request("s1", 1.0), "short"),
        (_request("l1", 90.0), "long"),
        (_request("s2", 2.0), "short"),
        (_request("l2", 91.0), "long"),
    ]
    selected, leftover = plan_lane_admission(candidates, 3, credits)
    assert [request.request_id for request in selected] == ["s1", "s2", "l1"]
    assert [request.request_id for request in leftover] == ["l2"]


# ---------------------------------------------------------------------------
# 4. M-bound dispatch: semaphore, full generator consumption, abort checks
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_pool_never_exceeds_m_concurrent_transcribes():
    probe = _ConcurrencyProbe()
    pipeline_cls = _pipeline_class(probe)
    pool = WhisperLanePool(object(), num_workers=2)
    requests = [_request(f"r{index}", 1.0) for index in range(8)]
    inputs = [request.artifacts["input_audio"] for request in requests]

    with patch.object(model_runner_base, "BatchedInferencePipeline", pipeline_cls):
        pairs = pool.run(inputs, requests, pool.transcribe)

    assert all(error is None for _payload, error in pairs)
    assert probe.calls == 8
    assert probe.peak == 2, f"expected the M=2 bound, observed {probe.peak}"
    assert pool.stats()["in_flight"] == 0
    assert pool.stats()["workers"] == 2


@pytest.mark.unit
def test_single_request_takes_the_direct_path_without_a_pool():
    probe = _ConcurrencyProbe()
    pipeline_cls = _pipeline_class(probe)
    pool = WhisperLanePool(object(), num_workers=4)
    request = _request("solo", 1.0)

    with patch.object(model_runner_base, "BatchedInferencePipeline", pipeline_cls):
        pairs = pool.run([request.artifacts["input_audio"]], [request], pool.transcribe)

    payload, error = pairs[0]
    assert error is None
    assert json.loads(payload[0])["text"] == "alphabetagamma"
    # The direct path must not create the worker pool.
    assert pool._pool is None
    assert probe.peak == 1


@pytest.mark.unit
def test_worker_consumes_the_generator_fully_inside_the_worker():
    consumed = threading.Event()
    seen_threads = []

    class _Pipeline:
        def __init__(self, model=None):
            self.last_speech_timestamp = 0.0

        def transcribe(self, audio, batch_size=16, **kwargs):
            seen_threads.append(threading.current_thread().name)

            def generator():
                try:
                    yield _FakeSegment("one ")
                    yield _FakeSegment("two")
                finally:
                    consumed.set()

            return generator(), _FakeInfo()

    pool = WhisperLanePool(object(), num_workers=2)
    request = _request("r", 1.0)
    with patch.object(model_runner_base, "BatchedInferencePipeline", _Pipeline):
        pairs = pool.run(
            [request.artifacts["input_audio"]],
            [request],
            pool.transcribe,
        )

    payload, error = pairs[0]
    assert error is None
    assert json.loads(payload[0])["text"] == "one two"
    assert consumed.is_set(), "the generator must be exhausted inside the worker"
    assert seen_threads, "transcribe must run in a worker"


@pytest.mark.unit
def test_lane_pool_passes_task_language_and_prompt_to_the_pipeline():
    """Phase 10-4: the sequential path applies the request's task/prompt."""
    captured = {}

    class _Pipeline:
        def __init__(self, model=None):
            self.last_speech_timestamp = 0.0

        def transcribe(self, audio, batch_size=16, **kwargs):
            captured.update(kwargs)
            return [_FakeSegment("translated")], _FakeInfo()

    pool = WhisperLanePool(object(), num_workers=1)
    request = _request("r", 1.0)
    request.request_data.task = "translate"
    request.request_data.language = "de"
    request.request_data.prompt = "Harmony"

    with patch.object(model_runner_base, "BatchedInferencePipeline", _Pipeline):
        payload, error = pool.run([request.artifacts["input_audio"]], [request], pool.transcribe)[0]

    assert error is None
    assert json.loads(payload[0])["text"] == "translated"
    assert captured == {"task": "translate", "language": "de", "initial_prompt": "Harmony"}


@pytest.mark.unit
def test_abort_between_segments_raises_and_stops_consuming():
    request = _request("aborted", 1.0)
    yielded = []

    class _Pipeline:
        def __init__(self, model=None):
            self.last_speech_timestamp = 0.0

        def transcribe(self, audio, batch_size=16, **kwargs):
            def generator():
                yielded.append("one")
                yield _FakeSegment("one")
                # The engine aborts while the worker is between segments.
                request.status = RequestStatus.FINISHED_ABORTED
                yielded.append("two")
                yield _FakeSegment("two")
                yielded.append("three")
                yield _FakeSegment("three")

            return generator(), _FakeInfo()

    pool = WhisperLanePool(object(), num_workers=1)
    with patch.object(model_runner_base, "BatchedInferencePipeline", _Pipeline):
        _payload, error = pool.run(
            [request.artifacts["input_audio"]], [request], pool.transcribe
        )[0]

    assert isinstance(error, WhisperRequestAbortedError)
    # The second segment was pulled to learn the request was aborted; the third
    # never was.
    assert yielded == ["one", "two"]


@pytest.mark.unit
def test_pool_records_queue_wait_histogram():
    probe = _ConcurrencyProbe()
    pipeline_cls = _pipeline_class(probe, segments=("only",))
    pool = WhisperLanePool(object(), num_workers=1)
    request = _request("r", 1.0)
    request.metrics.time_in_queue = 0.03

    with patch.object(model_runner_base, "BatchedInferencePipeline", pipeline_cls):
        pool.run([request.artifacts["input_audio"]], [request], pool.transcribe)

    queue_wait = pool.stats()["queue_wait"]
    assert queue_wait["count"] == 1
    assert queue_wait["sum"] == pytest.approx(0.03)
    assert queue_wait["max"] == pytest.approx(0.03)
    assert sum(queue_wait["buckets"]) == 1


# ---------------------------------------------------------------------------
# 5. Handler: aborting one request leaves its neighbours intact
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_handler_abort_errors_only_the_aborted_request():
    short_request = _request("short", 0.1)
    long_request = _request("long", 0.4)

    class _Pipeline:
        def __init__(self, model=None):
            self.last_speech_timestamp = 0.0

        def transcribe(self, audio, batch_size=16, **kwargs):
            seconds = len(audio) / SAMPLE_RATE

            def generator():
                for index, text in enumerate(("one", "two", "three")):
                    if index == 1 and seconds > 0.2:
                        # Simulate the engine aborting this request mid-flight.
                        long_request.status = RequestStatus.FINISHED_ABORTED
                    yield _FakeSegment(text)

            return generator(), _FakeInfo()

    runner = ModelRunnerBase.__new__(ModelRunnerBase)
    runner.model = object()
    runner.device = "cpu"
    runner.model_config = _whisper_config(num_workers=2)
    runner._baseline_state = None

    with patch.object(model_runner_base, "BatchedInferencePipeline", _Pipeline):
        results = runner._execute_faster_whisper(
            [short_request.artifacts["input_audio"], long_request.artifacts["input_audio"]],
            [short_request, long_request],
        )

    by_id = {result.request_id: result for result in results}
    assert by_id["short"].result_data.finish_reason == "stop"
    assert json.loads(by_id["short"].result_data.output)["text"] == "onetwothree"
    assert by_id["long"].result_data.finish_reason == "error"
    assert "aborted" in by_id["long"].result_data.error


# ---------------------------------------------------------------------------
# 6. Scheduler admission: lanes × capacity × memory governor
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_scheduler_whisper_admission_composes_with_memory_governor():
    config = _whisper_config(max_batch_size=16)
    scheduler = Scheduler(
        [config],
        whisper_admission=lambda active_windows=0: {"enabled": True, "allowed_windows": 2},
    )
    for index in range(5):
        scheduler.add_request(_request(f"r{index}", 1.0))

    scheduled = scheduler.schedule().scheduled_requests_per_model["whisper"]
    assert len(scheduled) == 2, "the governor's allowed_windows must bound the step"
    assert len(scheduler.waiting) == 3


@pytest.mark.unit
def test_scheduler_memory_governor_subtracts_running_windows():
    config = _whisper_config(max_batch_size=16)
    scheduler = Scheduler(
        [config],
        whisper_admission=lambda active_windows=0: {"enabled": True, "allowed_windows": 1},
    )
    running = _request("running", 1.0)
    running.set_running()
    scheduler.running.append(running)
    scheduler.add_request(_request("waiting", 1.0))

    outputs = scheduler.schedule()
    assert outputs.scheduled_requests_per_model == {}
    assert [request.request_id for request in scheduler.waiting] == ["waiting"]


@pytest.mark.unit
def test_scheduler_disabled_governor_falls_back_to_capacity():
    config = _whisper_config(max_batch_size=3)
    scheduler = Scheduler(
        [config],
        whisper_admission=lambda active_windows=0: {"enabled": False, "allowed_windows": 0},
    )
    for index in range(5):
        scheduler.add_request(_request(f"r{index}", 1.0))

    scheduled = scheduler.schedule().scheduled_requests_per_model["whisper"]
    assert len(scheduled) == 3


@pytest.mark.unit
def test_scheduler_keeps_boolean_admission_for_non_whisper_models():
    config = ModelConfig(
        name="synth",
        model="dummy/model",
        model_type="KittenTTSSynthesizer",
        max_batch_size=4,
        device_config=DeviceConfig("cpu"),
        dtype="float32",
    )
    scheduler = Scheduler([config])
    running = EngineRequest(
        request_id="running",
        request_data=SpeechTranscribeRequestInput(
            request_id="running", requested_model="synth", model="synth"
        ),
        arrival_time=0.0,
    )
    running.set_running()
    scheduler.running.append(running)
    waiting = EngineRequest(
        request_id="waiting",
        request_data=SpeechTranscribeRequestInput(
            request_id="waiting", requested_model="synth", model="synth"
        ),
        arrival_time=0.0,
    )
    scheduler.add_request(waiting)

    outputs = scheduler.schedule()
    assert outputs.scheduled_requests_per_model == {}
    assert [request.request_id for request in scheduler.waiting] == ["waiting"]


@pytest.mark.unit
def test_scheduler_whisper_lane_depths():
    config = _whisper_config(max_batch_size=1)
    scheduler = Scheduler([config])
    for request_id, seconds in (("s1", 1.0), ("l1", 60.0), ("s2", 2.0)):
        scheduler.add_request(_request(request_id, seconds))
    # max_batch_size 1 schedules one (a short one) and leaves the rest queued.
    scheduler.schedule()

    depths = scheduler.whisper_lane_depths()
    assert depths["whisper"] == {"short": 1, "long": 1}


# ---------------------------------------------------------------------------
# 7. Metrics surface
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_ct2_queue_state_reads_native_counters():
    class _Native:
        num_queued_batches = 3
        num_active_batches = 2

    class _Model:
        model = _Native()

    assert _ct2_queue_state(_Model()) == {"available": True, "queued": 3, "active": 2}
    assert _ct2_queue_state(object()) == {"available": False, "queued": 0, "active": 0}


@pytest.mark.unit
def test_whisper_lane_stats_defaults_without_a_pool():
    runner = ModelRunnerBase.__new__(ModelRunnerBase)
    runner.model = object()
    runner.model_config = _whisper_config(num_workers=3)

    stats = runner.whisper_lane_stats()
    assert stats["workers"] == 3
    assert stats["in_flight"] == 0
    assert stats["ct2"]["available"] is False


@pytest.mark.unit
def test_engine_get_whisper_lane_stats_aggregates_models():
    from types import SimpleNamespace

    from harmonyspeech.engine.harmonyspeech_engine import HarmonySpeechEngine

    runner = SimpleNamespace(
        whisper_lane_stats=lambda: {
            "workers": 2,
            "in_flight": 1,
            "queue_wait": {"sum": 0.1, "count": 1, "max": 0.1, "buckets": []},
            "ct2": {"available": True, "queued": 3, "active": 1},
        }
    )
    engine = HarmonySpeechEngine.__new__(HarmonySpeechEngine)
    engine.model_executors = {
        "whisper": SimpleNamespace(
            model_config=_whisper_config(num_workers=2),
            driver_worker=SimpleNamespace(model_runner=runner),
        ),
        "synth": SimpleNamespace(
            model_config=ModelConfig(
                name="synth",
                model="dummy/model",
                model_type="MeloTTSSynthesizer",
                max_batch_size=8,
                device_config=DeviceConfig("cpu"),
                dtype="float32",
            ),
            driver_worker=SimpleNamespace(model_runner=SimpleNamespace()),
        ),
    }
    engine.scheduler = Scheduler([_whisper_config(num_workers=2)])
    engine.scheduler.add_request(_request("s", 1.0))
    engine.scheduler.add_request(_request("l", 60.0))

    stats = engine.get_whisper_lane_stats()

    assert stats["enabled"] is True
    assert set(stats["models"]) == {"whisper"}, "non-Whisper executors stay out of the lane gauges"
    entry = stats["models"]["whisper"]
    assert entry["workers"] == 2
    assert entry["in_flight"] == 1
    assert entry["lane_short_depth"] == 1
    assert entry["lane_long_depth"] == 1
    assert entry["ct2"]["queued"] == 3


@pytest.mark.unit
def test_async_wrapper_proxies_whisper_lane_stats():
    """The serving layer holds the async wrapper (Phase 10-1 carry-over fix)."""
    from types import SimpleNamespace

    from harmonyspeech.engine.async_harmonyspeech import AsyncHarmonySpeech

    expected = {"enabled": True, "models": {"whisper": {"workers": 2}}}
    wrapper = AsyncHarmonySpeech.__new__(AsyncHarmonySpeech)
    wrapper.engine = SimpleNamespace(get_whisper_lane_stats=lambda: expected)

    assert wrapper.get_whisper_lane_stats() == expected


@pytest.mark.unit
def test_render_whisper_lane_metrics_emits_every_family():
    stats = {
        "enabled": True,
        "models": {
            "whisper": {
                "workers": 2,
                "in_flight": 1,
                "lane_short_depth": 4,
                "lane_long_depth": 1,
                "queue_wait": {"sum": 0.5, "count": 3, "max": 0.3, "buckets": [0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0]},
                "ct2": {"available": True, "queued": 5, "active": 2},
            }
        },
    }
    text = render_whisper_lane_metrics(stats)

    assert 'hse_whisper_workers{model="whisper"} 2' in text
    assert 'hse_whisper_in_flight{model="whisper"} 1' in text
    assert 'hse_whisper_lane_short_depth{model="whisper"} 4' in text
    assert 'hse_whisper_lane_long_depth{model="whisper"} 1' in text
    assert 'hse_whisper_ct2_queued_batches{model="whisper"} 5' in text
    assert 'hse_whisper_ct2_active_batches{model="whisper"} 2' in text
    assert 'hse_whisper_queue_wait_seconds_count{model="whisper"} 3' in text
    assert 'hse_whisper_queue_wait_seconds_bucket{model="whisper",le="+Inf"} 3' in text
    assert "# TYPE hse_whisper_queue_wait_seconds histogram" in text


@pytest.mark.unit
def test_render_whisper_lane_metrics_emits_idle_shape():
    text = render_whisper_lane_metrics(None)
    assert 'hse_whisper_workers{model="none"} 0' in text
    assert 'hse_whisper_lane_short_depth{model="none"} 0' in text
    assert 'hse_whisper_queue_wait_seconds_count{model="none"} 0' in text

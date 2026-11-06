"""Unit tests for parallel lanes and the KittenTTS lane wiring (Phase 8-5).

Covers the shared lane mechanism in ``harmonyspeech.task_handler.batching``
(pool sizing, input order, per-item error isolation), the KittenTTS side of it
(serialised espeak calls, fixed ORT session options, load-time warmup), and the
Silero VAD verdict (the step stays on the calling thread).
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from harmonyspeech.common.artifacts import AudioArtifact
from harmonyspeech.common.config import DeviceConfig, ModelConfig
from harmonyspeech.common.inputs import DetectVoiceActivityRequestInput
from harmonyspeech.common.request import EngineRequest
from harmonyspeech.modeling.models.kittentts.kittentts import KittenTTSSynthesizer
from harmonyspeech.modeling.models.kittentts.onnx_model import (
    KITTEN_TTS_INTRA_OP_NUM_THREADS,
    KittenTTS_1_Onnx,
    build_session_options,
)
from harmonyspeech.task_handler import model_runner_base
from harmonyspeech.task_handler.batching import lane_count, run_in_lanes

# ---------------------------------------------------------------------------
# Shared lane mechanism
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    ("requests", "max_batch_size", "expected"),
    [
        (1, 4, 1),
        (3, 4, 3),
        (4, 4, 4),
        (8, 4, 4),
        (8, 8, 8),
        (5, 1, 1),
        (0, 4, 1),
        (5, 0, 1),
        (5, -3, 1),
    ],
)
def test_lane_count_is_min_of_work_width_and_admission(requests, max_batch_size, expected):
    """``min(len(inputs), max_batch_size)``, never below one lane."""
    assert lane_count(requests, max_batch_size) == expected


@pytest.mark.unit
def test_run_in_lanes_preserves_input_order_with_variable_latency():
    """A slow early item must not shift a later item's result."""
    def work(item):
        time.sleep(0.05 if item == 0 else 0.005)
        return item * 10

    results = run_in_lanes(work, list(range(6)), lanes=3)

    assert [value for value, error in results] == [0, 10, 20, 30, 40, 50]
    assert all(error is None for _value, error in results)


@pytest.mark.unit
def test_run_in_lanes_isolates_a_failing_item():
    """One lane's exception becomes that item's error only."""
    def work(item):
        if item == "poison":
            raise RuntimeError("lane exploded")
        return item.upper()

    items = ["a", "poison", "c"]
    results = run_in_lanes(work, items, lanes=3)

    assert results[0] == ("A", None)
    assert results[2] == ("C", None)
    value, error = results[1]
    assert value is None
    assert isinstance(error, RuntimeError)
    assert "lane exploded" in str(error)


@pytest.mark.unit
def test_run_in_lanes_runs_inline_for_a_single_lane():
    """``lanes <= 1`` is the single-request fast path: same thread, no pool."""
    caller = threading.get_ident()
    threads = []

    def work(item):
        threads.append(threading.get_ident())
        return item

    assert run_in_lanes(work, [1, 2], lanes=1) == [(1, None), (2, None)]
    assert threads == [caller, caller]


@pytest.mark.unit
def test_run_in_lanes_never_exceeds_the_lane_count():
    """The pool is bounded by ``lanes`` even when the step is much larger."""
    lock = threading.Lock()
    state = {"active": 0, "peak": 0}

    def work(item):
        with lock:
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
        time.sleep(0.03)
        with lock:
            state["active"] -= 1
        return item

    results = run_in_lanes(work, list(range(12)), lanes=3)

    assert [value for value, _error in results] == list(range(12))
    assert state["peak"] <= 3


# ---------------------------------------------------------------------------
# KittenTTS: phonemizer lock, session options, warmup
# ---------------------------------------------------------------------------


class _RecordingPhonemizer:
    """Stand-in for ``EspeakBackend`` that records overlapping calls."""

    def __init__(self, delay=0.02):
        self.delay = delay
        self._lock = threading.Lock()
        self.active = 0
        self.peak = 0
        self.calls = 0

    def phonemize(self, texts):
        with self._lock:
            self.active += 1
            self.calls += 1
            self.peak = max(self.peak, self.active)
        time.sleep(self.delay)
        with self._lock:
            self.active -= 1
        return ["h ɛ l oʊ" for _ in texts]


def _bare_kitten(phonemizer_backend) -> KittenTTS_1_Onnx:
    """A ``KittenTTS_1_Onnx`` without the ONNX session, for pure-python paths."""
    model = KittenTTS_1_Onnx.__new__(KittenTTS_1_Onnx)
    model.voices = {"expr-voice-5-m": np.zeros((64, 256), dtype=np.float32)}
    model.available_voices = ["expr-voice-5-m"]
    model.voice_aliases = {}
    model.speed_priors = {}
    model.text_cleaner = lambda text: [1, 2, 3]
    model.phonemizer = phonemizer_backend
    return model


@pytest.mark.unit
def test_kitten_phonemizer_calls_are_serialised_by_the_module_lock():
    """Concurrent lanes must never enter the espeak backend together.

    ``phonemizer``'s ``EspeakBackend`` keeps mutable post-processing state; the
    unprotected version of this test fails with "number of lines in input and
    output must be equal" (reproduced against the real backend).
    """
    backend = _RecordingPhonemizer()
    model = _bare_kitten(backend)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda index: model._prepare_inputs(f"text {index}", "expr-voice-5-m"), range(8)))

    assert backend.calls == 8
    assert backend.peak == 1, "phonemize calls overlapped; the module lock is not held"


@pytest.mark.unit
def test_kitten_session_options_are_fixed_and_single_inter_op_thread():
    """Both thread counts are baked in, not inherited or configured (8-5)."""
    options = build_session_options()

    assert options.intra_op_num_threads == KITTEN_TTS_INTRA_OP_NUM_THREADS
    assert options.intra_op_num_threads == 0  # ORT default: one thread per core
    assert options.inter_op_num_threads == 1


@pytest.mark.unit
def test_kitten_warmup_runs_once_at_load(monkeypatch):
    """The wrapper runs exactly one warmup generate while loading."""
    loaded = []

    class FakeModel:
        def __init__(self):
            self.warmups = 0

        def warmup(self):
            self.warmups += 1

    fake = FakeModel()

    def fake_load(self, repo_id, cache_dir):
        loaded.append(repo_id)
        return fake

    monkeypatch.setattr(KittenTTSSynthesizer, "_load_model", fake_load)

    synthesizer = KittenTTSSynthesizer("KittenML/kitten-tts-mini-0.8")

    assert loaded == ["KittenML/kitten-tts-mini-0.8"]
    assert fake.warmups == 1
    assert synthesizer._model is fake


@pytest.mark.unit
def test_kitten_warmup_failure_does_not_fail_the_load(monkeypatch):
    """A broken warmup is logged and ignored; the model still loads."""
    class BrokenModel:
        def warmup(self):
            raise RuntimeError("warmup probe failed")

    monkeypatch.setattr(
        KittenTTSSynthesizer, "_load_model", lambda self, repo_id, cache_dir: BrokenModel()
    )

    synthesizer = KittenTTSSynthesizer("KittenML/kitten-tts-mini-0.8")

    assert isinstance(synthesizer._model, BrokenModel)


@pytest.mark.unit
def test_kitten_onnx_wrapper_has_a_tiny_warmup():
    """The warmup input is the shortest chunk the pipeline can produce."""
    calls = []

    def fake_generate(text, **kwargs):
        calls.append((text, kwargs))
        return np.zeros(8, dtype=np.float32)

    model = KittenTTS_1_Onnx.__new__(KittenTTS_1_Onnx)
    model.generate = fake_generate

    model.warmup()

    assert len(calls) == 1
    assert calls[0][0] == "Hi"
    assert calls[0][1]["voice"] == "expr-voice-5-m"


# ---------------------------------------------------------------------------
# Silero VAD verdict: sequential, on the calling thread
# ---------------------------------------------------------------------------


def _vad_runner(max_batch_size: int):
    runner = model_runner_base.ModelRunnerBase.__new__(model_runner_base.ModelRunnerBase)
    runner.model = object()
    runner.device = "cpu"
    runner.model_config = ModelConfig(
        name="vad",
        model="silero-vad",
        model_type="SileroVAD",
        max_batch_size=max_batch_size,
        device_config=DeviceConfig("cpu"),
        dtype="float32",
    )
    runner._baseline_state = None
    runner._step_result_ids = set()
    return runner


@pytest.mark.unit
def test_silero_vad_step_stays_on_the_calling_thread(monkeypatch):
    """Measured verdict (8-5): lanes are worse than useless, so no pool runs.

    The shared ``silero_vad`` wrapper carries the streaming state on the
    instance, so concurrency would corrupt results, and per-lane wrappers
    measured 0.95x at 8 lanes. The handler therefore calls the VAD inline.
    """
    threads = []

    def fake_timestamps(audio, model, **kwargs):
        threads.append(threading.get_ident())
        return [{"start": 0, "end": 512}]

    monkeypatch.setattr(model_runner_base, "get_speech_timestamps", fake_timestamps)

    runner = _vad_runner(max_batch_size=8)
    inputs = [
        (AudioArtifact.from_numpy(np.zeros(3200, dtype=np.float32), 16000), {
            "threshold": 0.5,
            "min_speech_duration_ms": 250,
            "min_silence_duration_ms": 100,
            "speech_pad_ms": 30,
            "return_seconds": False,
            "get_timestamps": True,
        })
        for _ in range(4)
    ]
    requests = [
        EngineRequest(
            request_id=f"vad-{index}",
            request_data=DetectVoiceActivityRequestInput(
                request_id=f"vad-{index}", requested_model="vad", model="vad"
            ),
            arrival_time=0.0,
        )
        for index in range(4)
    ]

    results = runner._execute_silero_vad(inputs, requests)

    assert [result.request_id for result in results] == [f"vad-{index}" for index in range(4)]
    assert all(result.result_data.finish_reason == "stop" for result in results)
    assert threads == [threading.get_ident()] * 4

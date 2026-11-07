"""Engine-level multi-step AR scheduling (Phase 9-4).

These tests run the *real* engine loop and scheduler with a stub AR executor
that behaves like 9-3's runner will: it acquires a real 9-2 ``ARCacheSlot`` on
its first decode step, returns an :class:`UnfinishedRequestOutput` for every
partial step, and releases the slot on completion/error. No model weights are
involved, so the suite is hermetic and CPU-only.

What is pinned:

* an unfinished request survives many engine steps and yields exactly one final
  client response;
* a second request is admitted while the first decodes (capacity-based, not the
  old "model has running work -> admit nothing" boolean);
* abort/disconnect mid-decode frees the row and its cache slot, affecting only
  that request;
* a per-request deadline still bounds a multi-step decode and errors only the
  expired request (the Phase 0-5 invariant: the loop lives);
* a poisoned decode row errors alone;
* non-AR families keep the boolean rule byte-for-byte even when AR capacity is
  injected, and an AR model without a capacity signal falls back to it.
"""

import time
from unittest.mock import patch

import pytest
import torch

from harmonyspeech.common.config import DeviceConfig, ModelConfig
from harmonyspeech.common.inputs import TextToSpeechRequestInput
from harmonyspeech.common.outputs import RequestOutput, TextToSpeechRequestOutput, UnfinishedRequestOutput
from harmonyspeech.common.request import EngineRequest, ExecutorResult
from harmonyspeech.engine.ar_cache import ARCacheDescriptor, ARCachePool
from harmonyspeech.engine.ar_decode_metrics import render_ar_decode_metrics
from harmonyspeech.engine.async_harmonyspeech import AsyncHarmonySpeech, RequestTracker, _AsyncHarmonySpeech
from harmonyspeech.engine.harmonyspeech_engine import HarmonySpeechEngine
from harmonyspeech.processing.scheduler import Scheduler

AR_MODEL = "ar"
AR_TYPE = "ChatterboxTTS"
ARENA_BUDGET = 64 * 1024


# ---------------------------------------------------------------------------
# Harness: real engine/scheduler, stub AR executor, real 9-2 slot lifecycle
# ---------------------------------------------------------------------------


def _model_config(name: str = AR_MODEL, model_type: str = AR_TYPE, max_batch_size: int = 4) -> ModelConfig:
    return ModelConfig(
        name=name,
        model="dummy/model",
        model_type=model_type,
        max_batch_size=max_batch_size,
        device_config=DeviceConfig("cpu"),
        dtype="float32",
    )


def _tts_request(request_id: str, model: str = AR_MODEL) -> TextToSpeechRequestInput:
    return TextToSpeechRequestInput(
        request_id=request_id,
        requested_model=model,
        model=model,
        input_text="hello",
        mode="single_speaker_tts",
    )


def _engine_request(request_id: str, model: str = AR_MODEL) -> EngineRequest:
    return EngineRequest(request_id=request_id, request_data=_tts_request(request_id, model), arrival_time=0.0)


class StubARExecutor:
    """AR-runner stand-in: one token per step, real pool slots, per-row isolation."""

    def __init__(self, pool: ARCachePool, *, steps: int = 3, poison_at: str | None = None) -> None:
        self.pool = pool
        self.steps = steps
        self.poison_at = poison_at
        self.slots: dict[str, object] = {}
        self.tokens: dict[str, int] = {}
        #: Request ids per step, in scheduling order (the admission witness).
        self.calls: list[list[str]] = []
        #: Request ids released through the engine's abort/expiry hook.
        self.released: list[str] = []

    def _acquire(self, request):
        request_id = request.request_id
        if request_id not in self.slots:
            slot = self.pool.acquire(AR_MODEL)
            if slot is None:
                return None
            self.slots[request_id] = slot
            self.tokens[request_id] = 0
        return self.slots[request_id]

    def execute_model(self, requests_to_batch):
        self.calls.append([request.request_id for request in requests_to_batch])
        results = []
        for request in requests_to_batch:
            request_id = request.request_id
            slot = self._acquire(request)
            if slot is None:
                results.append(self._error_result(request, RuntimeError("no AR slot")))
                continue
            self.tokens[request_id] += 1
            tokens = self.tokens[request_id]
            if self.poison_at == request_id and tokens >= 2:
                # A poisoned row errors alone and still releases its slot.
                self.slots.pop(request_id).release()
                self.tokens.pop(request_id, None)
                results.append(self._error_result(request, RuntimeError("poisoned decode row")))
                continue
            if tokens >= self.steps:
                self.slots.pop(request_id).release()
                output = TextToSpeechRequestOutput(
                    request_id=request_id, text="hello", output="", finish_reason="stop"
                )
                # Optional final progress so the counters include the last token.
                output.progress = {"tokens": tokens, "step": tokens}
                self.tokens.pop(request_id, None)
                results.append(ExecutorResult(request_id, request.request_data, output))
            else:
                output = UnfinishedRequestOutput(
                    request_id=request_id, progress={"tokens": tokens, "step": tokens}
                )
                results.append(ExecutorResult(request_id, request.request_data, output))
        return results

    async def execute_model_async(self, requests_to_batch):
        return self.execute_model(requests_to_batch)

    def check_health(self):
        return {"status": "ok"}

    def release_request(self, request_id: str) -> None:
        """Engine abort/expiry hook: drop the row and free its cache slot."""
        slot = self.slots.pop(request_id, None)
        self.tokens.pop(request_id, None)
        self.released.append(request_id)
        if slot is not None and not slot.released:
            slot.release()

    @staticmethod
    def _error_result(request, error: Exception) -> ExecutorResult:
        output = RequestOutput(request_id=request.request_id, finish_reason="error", error=str(error))
        return ExecutorResult(request.request_id, request.request_data, output)


def _install_arena(engine: HarmonySpeechEngine, config: ModelConfig, slots: int = 2) -> ARCachePool:
    """Replace the engine pool with a deterministic CPU arena for ``config``."""
    pool = ARCachePool(budget_bytes=ARENA_BUDGET, device="cpu")
    pool.register(
        ARCacheDescriptor(
            model_key=config.name,
            layers=1,
            kv_heads=1,
            head_dim=1,
            dtype=torch.float32,
            cap_tokens=8,
            slots=slots,
        )
    )
    engine.ar_cache_pool = pool
    return pool


def _build_engine(*, steps: int = 3, poison_at: str | None = None, max_batch_size: int = 4, slots: int = 2):
    config = _model_config(max_batch_size=max_batch_size)
    with patch("harmonyspeech.executor.cpu_executor.CPUExecutor._init_worker"):
        engine = HarmonySpeechEngine(
            model_configs=[config],
            log_stats=False,
            ar_cache_enabled=True,
            ar_cache_budget_bytes=ARENA_BUDGET,
        )
    pool = _install_arena(engine, config, slots=slots)
    executor = StubARExecutor(pool, steps=steps, poison_at=poison_at)
    engine.model_executors = {config.name: executor}
    return engine, executor


def _drive(engine: HarmonySpeechEngine, max_steps: int = 20) -> dict[str, list]:
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
# Scheduler: capacity admission composition + non-AR byte-for-byte boolean
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_ar_capacity_admission_composes_slots_and_max_batch_size():
    scheduler = Scheduler(
        [_model_config(max_batch_size=4)],
        ar_admission=lambda name: {
            "enabled": True,
            "slots": 2,
            "slots_active": 1,
            "slots_free": 1,
            "active_rows": 1,
        },
    )
    running = _engine_request("running")
    running.set_running()
    running.set_progress({"tokens": 1})
    scheduler.running.append(running)
    scheduler.add_request(_engine_request("waiting-1"))
    scheduler.add_request(_engine_request("waiting-2"))

    outputs = scheduler.schedule()

    # The running row continues; exactly one new row fits the one free slot
    # (the per-step width would have allowed three more).
    assert [r.request_id for r in outputs.scheduled_requests_per_model["ar"]] == ["running", "waiting-1"]
    assert [r.request_id for r in scheduler.waiting] == ["waiting-2"]


@pytest.mark.integration
def test_non_ar_model_keeps_boolean_admission_with_ar_capacity_injected():
    scheduler = Scheduler(
        [_model_config(name="synth", model_type="KittenTTSSynthesizer", max_batch_size=4)],
        ar_admission=lambda name: {
            "enabled": True,
            "slots": 8,
            "slots_active": 1,
            "slots_free": 7,
            "active_rows": 1,
        },
    )
    running = _engine_request("running", model="synth")
    running.set_running()
    scheduler.running.append(running)
    scheduler.add_request(_engine_request("waiting", model="synth"))

    outputs = scheduler.schedule()

    # The historical rule: a non-AR model with running work admits nothing this
    # step, no matter what the AR capacity hook would allow.
    assert outputs.scheduled_requests_per_model == {}
    assert [r.request_id for r in scheduler.waiting] == ["waiting"]


@pytest.mark.integration
def test_ar_model_without_capacity_signal_keeps_boolean_admission():
    def no_arena(_name):
        raise KeyError("no arena registered yet")

    scheduler = Scheduler([_model_config(max_batch_size=4)], ar_admission=no_arena)
    running = _engine_request("running")
    running.set_running()
    running.set_progress({"tokens": 1})
    scheduler.running.append(running)
    scheduler.add_request(_engine_request("waiting"))

    outputs = scheduler.schedule()

    # The row continues, but no new admission while the capacity signal is
    # missing (the pre-9-3 / --no-ar-cache fallback).
    assert [r.request_id for r in outputs.scheduled_requests_per_model["ar"]] == ["running"]
    assert [r.request_id for r in scheduler.waiting] == ["waiting"]


@pytest.mark.integration
def test_unfinished_ar_request_is_rescheduled_from_running():
    scheduler = Scheduler([_model_config()])
    request = _engine_request("row")
    request.set_running()
    request.set_progress({"tokens": 5})
    scheduler.running.append(request)

    outputs = scheduler.schedule()

    assert [r.request_id for r in outputs.scheduled_requests_per_model["ar"]] == ["row"]
    assert not scheduler.waiting
    assert len(scheduler.running) == 1, "a continuing row must not be duplicated in running"


# ---------------------------------------------------------------------------
# Engine: unfinished survives steps, one final response
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_unfinished_request_survives_steps_and_finishes():
    engine, executor = _build_engine(steps=3)

    engine.add_request("solo", _tts_request("solo"))
    outputs = _drive(engine)

    assert list(outputs) == ["solo"]
    assert len(outputs["solo"]) == 1, "the client sees exactly one final response"
    assert outputs["solo"][0].finish_reason == "stop"
    # The same request was scheduled on all three decode steps.
    assert executor.calls == [["solo"], ["solo"], ["solo"]]
    assert engine.ar_cache_pool.capacity("ar")["slots_active"] == 0
    assert engine.get_num_unfinished_requests() == 0
    # Progress counters saw every step (the final result carried the last token).
    stats = engine.get_ar_decode_stats()["models"]["ar"]
    assert stats["tokens_total"] == 3
    assert stats["steps_total"] == 3
    assert stats["tokens_per_step"] == pytest.approx(1.0)


@pytest.mark.integration
def test_second_request_admitted_while_first_decodes():
    engine, executor = _build_engine(steps=3)

    engine.add_request("first", _tts_request("first"))
    engine.step()  # first starts decoding and holds one of two slots
    assert engine.ar_cache_pool.capacity("ar")["slots_active"] == 1

    # Arrives after the first row is already RUNNING and unfinished. The old
    # boolean rule would have left it waiting for the whole decode.
    engine.add_request("second", _tts_request("second"))
    engine.step()

    assert executor.calls[1] == ["first", "second"], "capacity admission let the second row in"
    assert engine.ar_cache_pool.capacity("ar")["slots_active"] == 2

    outputs = _drive(engine)
    assert outputs["first"][-1].finish_reason == "stop"
    assert outputs["second"][-1].finish_reason == "stop"
    assert engine.ar_cache_pool.capacity("ar")["slots_active"] == 0


# ---------------------------------------------------------------------------
# Engine: abort / deadline / poison isolation
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_abort_mid_decode_frees_row_and_slot():
    engine, executor = _build_engine(steps=10)

    engine.add_request("keep", _tts_request("keep"))
    engine.add_request("abort-me", _tts_request("abort-me"))
    engine.step()
    assert engine.ar_cache_pool.capacity("ar")["slots_active"] == 2

    engine.abort_request("abort-me")

    assert [r.request_id for r in engine.scheduler.running] == ["keep"]
    assert engine.ar_cache_pool.capacity("ar")["slots_active"] == 1
    assert "abort-me" in executor.released

    outputs = _drive(engine)
    assert outputs["keep"][-1].finish_reason == "stop"
    assert "abort-me" not in outputs, "an aborted request produces no client output"
    assert engine.ar_cache_pool.capacity("ar")["slots_active"] == 0


@pytest.mark.integration
def test_deadline_expiry_errors_request_alone_and_loop_lives():
    engine, executor = _build_engine(steps=10)

    engine.add_request("slow", _tts_request("slow"))
    engine.add_request("healthy", _tts_request("healthy"))
    engine.step()
    assert engine.ar_cache_pool.capacity("ar")["slots_active"] == 2

    # Deadlines are re-checked every step: make the first row overdue mid-decode.
    for request in engine.scheduler.running:
        if request.request_id == "slow":
            request.deadline = time.monotonic() - 1

    outputs, forwarded = engine.step()

    assert forwarded == []
    assert [output.request_id for output in outputs] == ["slow"]
    assert outputs[0].finish_reason == "error"
    assert "exceeded timeout" in outputs[0].error
    # Only the expired request is gone; the other row keeps decoding.
    assert [r.request_id for r in engine.scheduler.running] == ["healthy"]
    assert engine.ar_cache_pool.capacity("ar")["slots_active"] == 1
    assert "slow" in executor.released

    collected = {"slow": outputs}
    collected.update(_drive(engine))
    assert collected["healthy"][-1].finish_reason == "stop"

    # The engine loop lives: a fresh request still completes after the timeout.
    engine.add_request("after", _tts_request("after"))
    after = _drive(engine)
    assert after["after"][-1].finish_reason == "stop"


@pytest.mark.integration
def test_poisoned_decoding_request_errors_alone():
    engine, executor = _build_engine(steps=4, poison_at="poison")

    engine.add_request("poison", _tts_request("poison"))
    engine.add_request("survivor", _tts_request("survivor"))
    engine.step()
    assert engine.ar_cache_pool.capacity("ar")["slots_active"] == 2

    outputs, _forwarded = engine.step()  # second decode step poisons the first row

    assert [output.request_id for output in outputs] == ["poison"]
    assert outputs[0].finish_reason == "error"
    assert "poisoned" in outputs[0].error
    assert [r.request_id for r in engine.scheduler.running] == ["survivor"]
    assert engine.ar_cache_pool.capacity("ar")["slots_active"] == 1

    collected = {"poison": outputs}
    collected.update(_drive(engine))
    assert collected["survivor"][-1].finish_reason == "stop"
    assert engine.ar_cache_pool.capacity("ar")["slots_active"] == 0


# ---------------------------------------------------------------------------
# Engine: progress metrics surface
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_ar_decode_progress_metrics_exposed():
    engine, _executor = _build_engine(steps=2)
    engine.add_request("metrics", _tts_request("metrics"))
    engine.step()

    stats = engine.get_ar_decode_stats()
    assert stats["enabled"] is True
    model = stats["models"]["ar"]
    assert model["active_rows"] == 1
    assert model["slots"] == 2
    assert model["slots_active"] == 1
    assert model["slots_free"] == 1
    assert model["tokens_total"] == 1
    assert model["steps_total"] == 1
    assert model["tokens_per_step"] == pytest.approx(1.0)

    # The /health payload carries the same block.
    assert engine.check_health()["ar_decode"] == stats

    text = render_ar_decode_metrics(stats)
    assert 'hse_ar_decode_active_rows{model="ar"} 1' in text
    assert 'hse_ar_decode_slots_free{model="ar"} 1' in text
    assert 'hse_ar_decode_tokens_total{model="ar"} 1' in text
    assert 'hse_ar_decode_steps_total{model="ar"} 1' in text
    assert "hse_ar_decode_tokens_per_step" in text

    _drive(engine)
    idle = render_ar_decode_metrics(None)
    for series in (
        "hse_ar_decode_active_rows",
        "hse_ar_decode_slots",
        "hse_ar_decode_tokens_total",
        "hse_ar_decode_steps_total",
        "hse_ar_decode_tokens_per_step",
    ):
        assert series in idle, "idle scrapes must still emit every series"


# ---------------------------------------------------------------------------
# Async loop: a progress-only step keeps the background loop stepping
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_engine_step_keeps_looping_on_unfinished_progress():
    config = _model_config()
    with patch("harmonyspeech.executor.cpu_executor.CPUExecutor._init_worker"):
        engine = _AsyncHarmonySpeech(
            model_configs=[config],
            log_stats=False,
            ar_cache_enabled=True,
            ar_cache_budget_bytes=ARENA_BUDGET,
        )
    pool = _install_arena(engine, config)
    executor = StubARExecutor(pool, steps=2)
    engine.model_executors = {config.name: executor}

    wrapper = AsyncHarmonySpeech.__new__(AsyncHarmonySpeech)
    wrapper.log_requests = False
    wrapper._errored_with = None
    wrapper.engine = engine
    wrapper._request_tracker = RequestTracker()

    stream = wrapper._request_tracker.add_request("async-solo", request_data=_tts_request("async-solo"))

    # Step 1 advances one token and yields no client output, but must report
    # "still in progress" so run_engine_loop keeps stepping instead of blocking
    # on wait_for_new_requests.
    assert await wrapper.engine_step() is True
    assert stream._queue.empty()

    # Step 2 completes the request and yields exactly one final output.
    assert await wrapper.engine_step() is True
    items = []
    while not stream._queue.empty():
        items.append(stream._queue.get_nowait())
    finals = [item for item in items if isinstance(item, RequestOutput)]
    assert len(finals) == 1
    assert finals[0].request_id == "async-solo"
    assert finals[0].finish_reason == "stop"
    assert not engine.has_unfinished_requests()

"""Unit tests for per-request deadlines, the loop watchdog and health (Phase 0-5)."""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from harmonyspeech.common.config import DeviceConfig, ModelConfig
from harmonyspeech.common.inputs import TextToSpeechRequestInput
from harmonyspeech.engine import async_harmonyspeech as async_engine_module
from harmonyspeech.engine.async_harmonyspeech import (
    AsyncHarmonySpeech,
    RequestTracker,
    _AsyncHarmonySpeech,
)
from harmonyspeech.engine.harmonyspeech_engine import HarmonySpeechEngine
from harmonyspeech.processing.scheduler import Scheduler


def _cfg(name: str = "kitten", model_type: str = "KittenTTSSynthesizer") -> ModelConfig:
    return ModelConfig(
        name=name,
        model=f"dummy/{name}",
        model_type=model_type,
        max_batch_size=1,
        device_config=DeviceConfig("cpu"),
        dtype="float32",
    )


def _engine_shell(timeout_s: float | None = 120.0) -> HarmonySpeechEngine:
    config = _cfg()
    engine = HarmonySpeechEngine.__new__(HarmonySpeechEngine)
    engine.model_configs = [config]
    engine.model_executors = {config.name: MagicMock()}
    engine.scheduler = Scheduler(model_configs=[config])
    engine.request_timeout_s = timeout_s
    engine._pending_error_outputs = []
    engine.log_stats = False
    engine.available_families = []
    return engine


def _kitten_request(request_id: str) -> TextToSpeechRequestInput:
    return TextToSpeechRequestInput(
        request_id=request_id,
        requested_model="kitten",
        model="kitten",
        input_text="hello",
        mode="single_speaker_tts",
    )


# ---------------------------------------------------------------------------
# Deadlines
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_add_request_stamps_deadline_from_configured_timeout():
    engine = _engine_shell(timeout_s=120.0)
    before = time.monotonic()

    engine.add_request("deadline-1", _kitten_request("deadline-1"))

    queued = engine.scheduler.waiting[0]
    assert queued.deadline is not None
    assert before + 119.0 < queued.deadline <= time.monotonic() + 120.0


@pytest.mark.unit
def test_request_timeout_zero_disables_deadline():
    engine = _engine_shell(timeout_s=0)

    engine.add_request("deadline-2", _kitten_request("deadline-2"))

    assert engine.scheduler.waiting[0].deadline is None


@pytest.mark.unit
def test_expired_waiting_request_resolves_as_timeout_error():
    engine = _engine_shell(timeout_s=120.0)
    engine.add_request("timeout-1", _kitten_request("timeout-1"))
    engine.scheduler.waiting[0].deadline = time.monotonic() - 1

    request_outputs, forwarded = engine.step()

    assert forwarded == []
    assert len(request_outputs) == 1
    output = request_outputs[0]
    assert output.request_id == "timeout-1"
    assert output.finish_reason == "error"
    assert "exceeded timeout of 120s" in output.error
    assert engine.get_num_unfinished_requests() == 0
    assert not engine.scheduler.waiting
    assert not engine.scheduler.running


@pytest.mark.unit
def test_expired_running_request_is_dropped_from_running():
    engine = _engine_shell(timeout_s=120.0)
    engine.add_request("timeout-2", _kitten_request("timeout-2"))
    queued = engine.scheduler.waiting.popleft()
    queued.set_running()
    engine.scheduler.running.append(queued)
    queued.deadline = time.monotonic() - 1

    expired = engine._expire_overdue_requests()

    assert [output.request_id for output in expired] == ["timeout-2"]
    assert not engine.scheduler.running
    assert engine.get_num_unfinished_requests() == 0


@pytest.mark.unit
def test_unexpired_request_is_left_alone():
    engine = _engine_shell(timeout_s=120.0)
    engine.add_request("alive-1", _kitten_request("alive-1"))

    assert engine._expire_overdue_requests() == []
    assert [request.request_id for request in engine.scheduler.waiting] == ["alive-1"]


# ---------------------------------------------------------------------------
# check_health
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_check_health_returns_structured_payload_for_all_executors():
    engine = _engine_shell()
    healthy_executor = MagicMock()
    healthy_executor.model_config = SimpleNamespace(model_type="KittenTTSSynthesizer")
    failing_executor = MagicMock()
    failing_executor.model_config = SimpleNamespace(model_type="MeloTTSSynthesizer")
    failing_executor.check_health.side_effect = RuntimeError("CUDA context lost")
    engine.model_executors = {"kitten": healthy_executor, "melo": failing_executor}

    payload = engine.check_health()

    assert payload["healthy"] is False
    assert payload["models"]["kitten"]["healthy"] is True
    assert payload["models"]["kitten"]["model_type"] == "KittenTTSSynthesizer"
    assert payload["models"]["melo"]["healthy"] is False
    assert "CUDA context lost" in payload["models"]["melo"]["error"]
    # Both executors were probed (no iteration over dict keys).
    healthy_executor.check_health.assert_called_once()
    failing_executor.check_health.assert_called_once()


@pytest.mark.unit
def test_check_health_does_not_raise_on_failing_executor():
    engine = _engine_shell()
    executor = MagicMock()
    executor.model_config = SimpleNamespace(model_type="KittenTTSSynthesizer")
    executor.check_health.side_effect = AttributeError("'str' object has no attribute 'check_health'")
    engine.model_executors = {"kitten": executor}

    payload = engine.check_health()

    assert payload["healthy"] is False


@pytest.mark.unit
async def test_check_health_async_returns_payload():
    engine = _AsyncHarmonySpeech.__new__(_AsyncHarmonySpeech)
    executor = MagicMock()
    executor.model_config = SimpleNamespace(model_type="KittenTTSSynthesizer")
    engine.model_executors = {"kitten": executor}

    payload = await engine.check_health_async()

    assert payload["healthy"] is True
    assert payload["models"]["kitten"]["model_type"] == "KittenTTSSynthesizer"
    assert payload["models"]["kitten"]["healthy"] is True
    # Phase 4-5: VRAM accounting rides along with the health payload.
    assert "vram" in payload
    assert payload["vram"]["resident_models"] == 1


# ---------------------------------------------------------------------------
# Loop watchdog
# ---------------------------------------------------------------------------


def _wrapper_with_tracker() -> tuple[AsyncHarmonySpeech, RequestTracker]:
    wrapper = AsyncHarmonySpeech.__new__(AsyncHarmonySpeech)
    wrapper.log_requests = False
    wrapper._errored_with = None
    tracker = RequestTracker()
    wrapper._request_tracker = tracker
    wrapper.engine = MagicMock()
    wrapper.engine.scheduler.running = []
    wrapper.engine.scheduler.free_finished_requests = MagicMock()
    wrapper.engine.abort_request = MagicMock()
    return wrapper, tracker


@pytest.mark.unit
async def test_run_engine_loop_survives_iteration_timeout(monkeypatch):
    """A step that exceeds the watchdog must not kill the loop (Phase 0-5)."""
    monkeypatch.setattr(async_engine_module, "ENGINE_ITERATION_TIMEOUT_S", 0.05)

    wrapper, tracker = _wrapper_with_tracker()
    stream = tracker.add_request("slow-001", request_data=MagicMock())
    tracker.get_new_and_finished_requests()  # move the stream into _request_streams

    in_flight = MagicMock()
    in_flight.request_id = "slow-001"
    wrapper.engine.scheduler.running = [in_flight]

    calls = {"n": 0}
    first_iteration_started = asyncio.Event()

    async def fake_engine_step():
        calls["n"] += 1
        # Keep the loop's "wait for new requests" from blocking.
        tracker.add_request(f"keepalive-{calls['n']}", request_data=MagicMock())
        if calls["n"] == 1:
            first_iteration_started.set()
            await asyncio.sleep(1.0)  # exceeds the watchdog
            return True
        return True

    wrapper.engine_step = fake_engine_step

    loop_task = asyncio.create_task(wrapper.run_engine_loop())
    await asyncio.wait_for(first_iteration_started.wait(), timeout=2)
    await asyncio.sleep(0.3)  # let the watchdog trip and further iterations run
    loop_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await loop_task

    # The loop kept iterating after the timeout instead of dying.
    assert calls["n"] >= 2
    assert wrapper.errored is False

    # The in-flight request was reported as errored, not left hanging.
    received = []
    while not stream._queue.empty():
        received.append(stream._queue.get_nowait())
    assert any(isinstance(item, TimeoutError) for item in received)
    wrapper.engine.abort_request.assert_called_with(["slow-001"])


@pytest.mark.unit
async def test_run_engine_loop_escalates_after_consecutive_timeouts(monkeypatch):
    """Repeated consecutive watchdog trips mark the engine dead (last resort)."""
    monkeypatch.setattr(async_engine_module, "ENGINE_ITERATION_TIMEOUT_S", 0.05)
    monkeypatch.setattr(async_engine_module, "ENGINE_MAX_CONSECUTIVE_ITERATION_TIMEOUTS", 2)

    wrapper, tracker = _wrapper_with_tracker()
    tracker.add_request("initial", request_data=MagicMock())

    calls = {"n": 0}

    async def always_slow():
        calls["n"] += 1
        tracker.add_request(f"slow-{calls['n']}", request_data=MagicMock())
        await asyncio.sleep(1.0)
        return True

    wrapper.engine_step = always_slow

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(wrapper.run_engine_loop(), timeout=5)

    assert calls["n"] == 2
    assert wrapper.errored is True

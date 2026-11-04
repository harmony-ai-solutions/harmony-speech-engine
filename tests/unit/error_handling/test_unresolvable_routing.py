"""Unit tests for unresolvable routing / no-hang guarantees (Phase 0-4).

A request whose model family is not fully configured must resolve as a typed
error within one engine step instead of waiting in the scheduler forever, and
the scheduler itself must never requeue a request for an unknown model.
"""

import base64
from types import SimpleNamespace

import pytest

from harmonyspeech.common.config import DeviceConfig, ModelConfig
from harmonyspeech.common.inputs import TextToSpeechRequestInput
from harmonyspeech.common.request import EngineRequest
from harmonyspeech.engine.harmonyspeech_engine import HarmonySpeechEngine
from harmonyspeech.processing.scheduler import Scheduler


def _cfg(name: str, model_type: str, max_batch_size: int = 1) -> ModelConfig:
    return ModelConfig(
        name=name,
        model=f"dummy/{name}",
        model_type=model_type,
        max_batch_size=max_batch_size,
        device_config=DeviceConfig("cpu"),
        dtype="float32",
    )


def _engine_shell(*configs: ModelConfig) -> HarmonySpeechEngine:
    engine = HarmonySpeechEngine.__new__(HarmonySpeechEngine)
    engine.model_configs = list(configs)
    engine.model_executors = {config.name: object() for config in configs}
    engine.scheduler = Scheduler(model_configs=list(configs))
    engine.request_timeout_s = 120.0
    engine._pending_error_outputs = []
    engine.log_stats = False
    engine.available_families = []
    return engine


def _incomplete_openvoice_request(request_id: str = "ghost-1") -> TextToSpeechRequestInput:
    return TextToSpeechRequestInput(
        request_id=request_id,
        requested_model="openvoice_v2",
        model="openvoice_v2",
        input_text="hello",
        mode="voice_cloning",
        input_audio=base64.b64encode(b"reference audio").decode(),
        input_vad_data=None,
        input_embedding=None,
    )


# ---------------------------------------------------------------------------
# Engine: unresolvable family resolves as an error in one step
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_unresolvable_family_resolves_as_error_within_one_step():
    # openvoice_v2 is missing Whisper / encoder / tone converter here.
    engine = _engine_shell(_cfg("ov2-synthesizer-en", "MeloTTSSynthesizer"))

    engine.add_request("ghost-1", _incomplete_openvoice_request("ghost-1"))
    request_outputs, forwarded = engine.step()

    assert forwarded == []
    assert len(request_outputs) == 1
    output = request_outputs[0]
    assert output.request_id == "ghost-1"
    assert output.finish_reason == "error"
    assert "openvoice_v2" in output.error
    assert "missing one of" in output.error
    assert "OpenVoiceV2ToneConverter" in output.error
    # The request never entered the scheduler, so nothing can be requeued.
    assert not engine.scheduler.has_unfinished_requests()
    assert not engine.scheduler.waiting
    assert not engine.scheduler.running


@pytest.mark.unit
def test_unresolvable_request_is_not_requeued_on_later_steps():
    engine = _engine_shell(_cfg("ov2-synthesizer-en", "MeloTTSSynthesizer"))
    engine.add_request("ghost-2", _incomplete_openvoice_request("ghost-2"))

    first_outputs, _ = engine.step()
    assert [o.request_id for o in first_outputs] == ["ghost-2"]

    # Manual stepping must drain, never loop.
    for _ in range(5):
        later_outputs, later_forwarded = engine.step()
        assert later_outputs == []
        assert later_forwarded == []
    assert engine.get_num_unfinished_requests() == 0


@pytest.mark.unit
def test_unknown_model_name_resolves_as_error():
    engine = _engine_shell(_cfg("kitten", "KittenTTSSynthesizer"))

    request_data = TextToSpeechRequestInput(
        request_id="ghost-3",
        requested_model="not_a_family",
        model="not_A_configured_model",
        input_text="hello",
        mode="single_speaker_tts",
    )
    engine.add_request("ghost-3", request_data)
    outputs, _ = engine.step()

    assert len(outputs) == 1
    assert outputs[0].finish_reason == "error"
    assert "not_A_configured_model" in outputs[0].error


@pytest.mark.unit
def test_routable_request_still_schedules():
    """Regression guard: a resolvable request is still placed on the queue."""
    engine = _engine_shell(_cfg("kitten", "KittenTTSSynthesizer"))

    request_data = TextToSpeechRequestInput(
        request_id="ok-1",
        requested_model="kitten",
        model="kitten",
        input_text="hello",
        mode="single_speaker_tts",
    )
    engine.add_request("ok-1", request_data)

    assert [request.request_id for request in engine.scheduler.waiting] == ["ok-1"]
    queued = engine.scheduler.waiting[0]
    assert queued.deadline is not None


# ---------------------------------------------------------------------------
# Scheduler: unknown model safety net
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_scheduler_routes_unknown_model_to_ignored():
    scheduler = Scheduler(model_configs=[_cfg("kitten", "KittenTTSSynthesizer")])
    request = EngineRequest(
        request_id="sched-ghost",
        request_data=SimpleNamespace(model="never-configured"),
        arrival_time=0.0,
    )
    scheduler.add_request(request)

    outputs = scheduler.schedule()

    assert outputs.scheduled_requests_per_model == {}
    assert [r.request_id for r in outputs.ignored_requests] == ["sched-ghost"]
    assert not scheduler.waiting
    assert not scheduler.running


@pytest.mark.unit
def test_ignored_unknown_model_becomes_error_output():
    engine = _engine_shell(_cfg("kitten", "KittenTTSSynthesizer"))
    request = EngineRequest(
        request_id="ignored-1",
        request_data=SimpleNamespace(model="never-configured"),
        arrival_time=0.0,
    )

    outputs, _ = engine._process_model_outputs(outputs=[], ignored_requests=[request])

    assert len(outputs) == 1
    assert outputs[0].finish_reason == "error"
    assert "never-configured" in outputs[0].error


@pytest.mark.unit
def test_ignored_known_model_keeps_skipped_reason():
    engine = _engine_shell(_cfg("kitten", "KittenTTSSynthesizer"))
    request = EngineRequest(
        request_id="ignored-2",
        request_data=SimpleNamespace(model="kitten"),
        arrival_time=0.0,
    )
    # Ignored requests normally come from the waiting queue.
    engine.scheduler.add_request(request)

    outputs, _ = engine._process_model_outputs(outputs=[], ignored_requests=[request])

    assert len(outputs) == 1
    assert outputs[0].finish_reason == "skipped"

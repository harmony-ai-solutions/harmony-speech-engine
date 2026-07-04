"""Unit tests for engine error resilience.

Covers four fixes for two production issues:

Issue 1 — "request ID ... is not of type SpeechEmbeddingRequestInput"
    When a voice-cloning TTS request (model=chatterbox, input_audio set,
    input_embedding None) is routed to the ChatterboxEmbedding executor, the
    request stays as TextToSpeechRequestInput. prepare_inputs for
    ChatterboxEmbedding must accept that type (not just SpeechEmbeddingRequestInput).

Issue 2 — "Background loop has errored already"
    A single bad request killing the entire engine. Three layers of defence:
    a) model_runner_base.execute_model wraps prepare_inputs so failures become
       per-request error results instead of unhandled exceptions.
    b) check_forward_processing recognises error results and does NOT forward
       them (and marks them finished so the scheduler cleans up).
    c) engine_step catches any remaining exception from step_async and
       propagates it to in-flight requests without killing the background loop.
"""

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from harmonyspeech.common.config import DeviceConfig, ModelConfig
from harmonyspeech.common.inputs import (
    SpeechEmbeddingRequestInput,
    TextToSpeechRequestInput,
)
from harmonyspeech.common.outputs import RequestOutput, SpeechEmbeddingRequestOutput
from harmonyspeech.common.request import EngineRequest, ExecutorResult, RequestStatus
from harmonyspeech.engine.harmonyspeech_engine import HarmonySpeechEngine

# ===========================================================================
# Issue 1: ChatterboxEmbedding prepare_inputs accepts TextToSpeechRequestInput
# ===========================================================================


def _make_chatterbox_embedding_config() -> ModelConfig:
    return ModelConfig(
        name="cb-embed",
        model="dummy/chatterbox",
        model_type="ChatterboxEmbedding",
        max_batch_size=4,
        device_config=DeviceConfig("cpu"),
        dtype="float32",
    )


def _make_engine_request(request_data) -> EngineRequest:
    return EngineRequest(request_id="tts-test-001", request_data=request_data, arrival_time=0.0)


@pytest.mark.unit
def test_prepare_inputs_chatterbox_embedding_accepts_tts_request():
    """prepare_inputs for ChatterboxEmbedding must NOT reject TextToSpeechRequestInput.

    This is the voice-cloning pipeline path: the TTS request (with input_audio)
    is routed to the embedding executor first, so prepare_inputs must accept it.
    """
    from harmonyspeech.task_handler.inputs import prepare_inputs

    config = _make_chatterbox_embedding_config()
    tts_input = TextToSpeechRequestInput(
        request_id="tts-test-001",
        requested_model="chatterbox",
        model="cb-embed",
        input_text="Hello world",
        mode="voice_cloning",
        input_audio=base64.b64encode(b"fake_wav_bytes").decode(),
    )
    engine_req = _make_engine_request(tts_input)

    # Must not raise ValueError
    inputs = prepare_inputs(config, [engine_req])
    assert len(inputs) == 1
    # prepare_chatterbox_embedding_inputs returns raw decoded audio bytes
    assert inputs[0] == b"fake_wav_bytes"


@pytest.mark.unit
def test_prepare_inputs_chatterbox_embedding_accepts_embed_request():
    """prepare_inputs for ChatterboxEmbedding must still accept SpeechEmbeddingRequestInput."""
    from harmonyspeech.task_handler.inputs import prepare_inputs

    config = _make_chatterbox_embedding_config()
    embed_input = SpeechEmbeddingRequestInput(
        request_id="embed-test-001",
        requested_model="chatterbox",
        model="cb-embed",
        input_audio=base64.b64encode(b"embed_audio").decode(),
    )
    engine_req = _make_engine_request(embed_input)

    inputs = prepare_inputs(config, [engine_req])
    assert len(inputs) == 1
    assert inputs[0] == b"embed_audio"


@pytest.mark.unit
def test_prepare_inputs_chatterbox_embedding_rejects_unknown_type():
    """prepare_inputs for ChatterboxEmbedding must still reject truly unsupported types."""
    from harmonyspeech.task_handler.inputs import prepare_inputs

    config = _make_chatterbox_embedding_config()
    bogus = MagicMock()  # Not a recognised request input type
    engine_req = _make_engine_request(bogus)

    with pytest.raises(ValueError, match="not of type"):
        prepare_inputs(config, [engine_req])


# ===========================================================================
# Issue 2a: execute_model catches prepare_inputs failures
# ===========================================================================


@pytest.mark.unit
def test_execute_model_returns_error_results_on_prepare_failure():
    """When prepare_inputs raises, execute_model must return per-request error results
    instead of propagating the exception (which would kill the engine)."""
    from harmonyspeech.task_handler.model_runner_base import ModelRunnerBase

    config = _make_chatterbox_embedding_config()
    runner = ModelRunnerBase.__new__(ModelRunnerBase)
    runner.model_config = config
    runner.device = "cpu"

    tts_input = TextToSpeechRequestInput(
        request_id="tts-fail-001",
        requested_model="chatterbox",
        model="cb-embed",
        input_text="Hello",
        mode="voice_cloning",
    )
    engine_req = _make_engine_request(tts_input)

    bad_input_msg = "simulated prepare failure"

    with patch("harmonyspeech.task_handler.model_runner_base.prepare_inputs", side_effect=ValueError(bad_input_msg)):
        results = runner.execute_model([engine_req])

    assert len(results) == 1
    result = results[0]
    assert isinstance(result, ExecutorResult)
    assert result.result_data.finish_reason == "error"
    assert bad_input_msg in result.result_data.error


# ===========================================================================
# Issue 2b: check_forward_processing handles error results
# ===========================================================================


def _make_engine_shell(*model_configs) -> HarmonySpeechEngine:
    """Create a HarmonySpeechEngine shell without running __init__."""
    engine = HarmonySpeechEngine.__new__(HarmonySpeechEngine)
    engine.model_configs = list(model_configs)
    engine.scheduler = MagicMock()
    return engine


def _make_error_result(request_id="err-001", requested_model="chatterbox") -> ExecutorResult:
    """Build an ExecutorResult whose result_data is an error RequestOutput."""
    tts_input = TextToSpeechRequestInput(
        request_id=request_id,
        requested_model=requested_model,
        model="cb-embed",
        input_text="Hello",
        mode="voice_cloning",
    )
    error_output = RequestOutput(request_id=request_id, finish_reason="error", error="boom")
    return ExecutorResult(request_id=request_id, input_data=tts_input, result_data=error_output)


@pytest.mark.unit
def test_check_forward_processing_error_result_not_forwarded():
    """An error result must NOT be forwarded through the pipeline."""
    embed_cfg = MagicMock()
    embed_cfg.name = "cb-embed"
    embed_cfg.model_type = "ChatterboxEmbedding"
    tts_cfg = MagicMock()
    tts_cfg.name = "cb-tts"
    tts_cfg.model_type = "ChatterboxTTS"
    engine = _make_engine_shell(tts_cfg, embed_cfg)

    result = _make_error_result()
    new_status, forwarding_request = engine.check_forward_processing(result)

    # Must be FINISHED_STOPPED (not FORWARDED) and no forwarding_request
    assert new_status == RequestStatus.FINISHED_STOPPED
    assert forwarding_request is None


@pytest.mark.unit
def test_check_forward_processing_error_result_updates_scheduler():
    """An error result must mark the request as FINISHED_STOPPED in the scheduler
    so it gets cleaned up (not leaked in the running queue)."""
    engine = _make_engine_shell()
    result = _make_error_result(request_id="cleanup-test-001")

    engine.check_forward_processing(result)

    engine.scheduler.update_request_status.assert_called_once_with(
        "cleanup-test-001", RequestStatus.FINISHED_STOPPED
    )


@pytest.mark.unit
def test_check_forward_processing_normal_embedding_still_forwards():
    """Regression guard: a successful embedding result must still be forwarded."""
    embed_cfg = MagicMock()
    embed_cfg.name = "cb-embed"
    embed_cfg.model_type = "ChatterboxEmbedding"
    tts_cfg = MagicMock()
    tts_cfg.name = "cb-tts"
    tts_cfg.model_type = "ChatterboxTTS"
    engine = _make_engine_shell(tts_cfg, embed_cfg)

    tts_input = TextToSpeechRequestInput(
        request_id="fwd-001",
        requested_model="chatterbox",
        model="cb-embed",
        input_text="Hello",
        mode="voice_cloning",
        input_audio="base64audio==",
    )
    embed_output = SpeechEmbeddingRequestOutput(request_id="fwd-001", output="base64conds==")
    result = ExecutorResult(request_id="fwd-001", input_data=tts_input, result_data=embed_output)

    with patch.object(engine, "add_request") as mock_add:
        new_status, forwarding_request = engine.check_forward_processing(result)

    assert new_status == RequestStatus.FINISHED_FORWARDED
    assert forwarding_request is not None
    assert forwarding_request.input_embedding == "base64conds=="
    assert forwarding_request.input_audio is None
    mock_add.assert_called_once()


# ===========================================================================
# Issue 2c: engine_step catches step_async exceptions
# ===========================================================================


@pytest.mark.unit
async def test_engine_step_survives_step_async_exception():
    """engine_step must catch exceptions from step_async and propagate them to
    in-flight request streams WITHOUT raising (which would kill the background loop)."""
    from harmonyspeech.engine.async_harmonyspeech import AsyncHarmonySpeech, RequestTracker

    wrapper = AsyncHarmonySpeech.__new__(AsyncHarmonySpeech)
    wrapper.log_requests = False

    # Mock the underlying engine
    mock_engine = MagicMock()
    mock_engine.add_request_async = AsyncMock()
    boom = RuntimeError("GPU OOM or similar catastrophic model error")
    mock_engine.step_async = AsyncMock(side_effect=boom)

    # Simulate one in-flight request in the scheduler's running queue
    in_flight_req = MagicMock()
    in_flight_req.request_id = "tts-inflight-001"
    mock_engine.scheduler.running = [in_flight_req]
    wrapper.engine = mock_engine

    # Set up the request tracker with the in-flight request
    wrapper._request_tracker = RequestTracker()
    stream = wrapper._request_tracker.add_request("tts-inflight-001", request_data=MagicMock())

    # engine_step must NOT raise
    result = await wrapper.engine_step()
    assert result is False  # no requests in progress after failure

    # The in-flight request's stream must have received the exception
    received = []
    while not stream._queue.empty():
        received.append(stream._queue.get_nowait())
    assert any(isinstance(item, RuntimeError) and "GPU OOM" in str(item) for item in received)


@pytest.mark.unit
async def test_engine_step_aborts_failed_requests_in_scheduler():
    """After catching a step_async exception, engine_step must abort the failed
    requests so they don't block the scheduler's batch slot."""
    from harmonyspeech.engine.async_harmonyspeech import AsyncHarmonySpeech, RequestTracker

    wrapper = AsyncHarmonySpeech.__new__(AsyncHarmonySpeech)
    wrapper.log_requests = False

    mock_engine = MagicMock()
    mock_engine.add_request_async = AsyncMock()
    mock_engine.step_async = AsyncMock(side_effect=RuntimeError("crash"))

    in_flight_req = MagicMock()
    in_flight_req.request_id = "tts-crash-001"
    mock_engine.scheduler.running = [in_flight_req]
    mock_engine.abort_request = MagicMock()
    wrapper.engine = mock_engine

    wrapper._request_tracker = RequestTracker()
    wrapper._request_tracker.add_request("tts-crash-001", request_data=MagicMock())

    await wrapper.engine_step()

    # abort_request must have been called with the failed request ID
    mock_engine.abort_request.assert_called_once()
    call_args = mock_engine.abort_request.call_args[0][0]
    assert "tts-crash-001" in list(call_args)


@pytest.mark.unit
async def test_engine_step_happy_path_unchanged():
    """Regression guard: the normal (no-error) path through engine_step still works."""
    from harmonyspeech.engine.async_harmonyspeech import AsyncHarmonySpeech, RequestTracker

    wrapper = AsyncHarmonySpeech.__new__(AsyncHarmonySpeech)
    wrapper.log_requests = False

    good_output = RequestOutput(request_id="tts-ok-001", finish_reason="stop")
    mock_engine = MagicMock()
    mock_engine.add_request_async = AsyncMock()
    mock_engine.step_async = AsyncMock(return_value=([good_output], []))
    mock_engine.scheduler.running = []
    wrapper.engine = mock_engine

    wrapper._request_tracker = RequestTracker()
    wrapper._request_tracker.add_request("tts-ok-001", request_data=MagicMock())

    result = await wrapper.engine_step()
    assert result is True

"""Unit tests for Chatterbox family routing via the declarative pipeline registry.

These replace the pre-Phase-3 tests of ``reroute_request_chatterbox()`` /
``reroute_request_chatterbox_vc()``. The behavioural assertions are unchanged —
the same request shapes must reach the same executors — but routing now goes
through ``resolve_entry`` + ``next_step`` on the ``chatterbox`` / ``chatterbox_vc``
specs, and forwarding through ``advance``.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from harmonyspeech.common.artifacts import SpeakerArtifact
from harmonyspeech.common.inputs import (
    SpeechEmbeddingRequestInput,
    TextToSpeechRequestInput,
    VoiceConversionRequestInput,
)
from harmonyspeech.common.outputs import SpeechEmbeddingRequestOutput
from harmonyspeech.processing.pipeline_registry import PIPELINE_SPECS, field_present, next_step, resolve_entry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model_config(name: str, model_type: str) -> SimpleNamespace:
    """Minimal ModelConfig stand-in for routing tests."""
    return SimpleNamespace(name=name, model_type=model_type, language=None)


def _route(request, *model_configs, artifacts=None):
    """Resolve ``request.model`` the way admission does (no engine instance)."""
    spec = PIPELINE_SPECS[request.requested_model]
    state = resolve_entry(spec, request, artifacts or {}, list(model_configs))
    assert state is not None, f"no entry rule matched {type(request).__name__}"
    config = next_step(spec, state, request, list(model_configs), artifacts or {})
    request.model = config.name if config is not None else None
    return state, config


def _make_tts_request(
    requested_model: str = "chatterbox", input_audio=None, input_embedding=None, mode="single_speaker_tts"
) -> TextToSpeechRequestInput:
    return TextToSpeechRequestInput(
        request_id="req-1",
        requested_model=requested_model,
        model=requested_model,
        input_text="hello",
        mode=mode,
        language_id=None,
        input_audio=input_audio,
        input_embedding=input_embedding,
    )


def _make_embed_request(requested_model: str = "chatterbox", input_audio=None) -> SpeechEmbeddingRequestInput:
    return SpeechEmbeddingRequestInput(
        request_id="req-2", requested_model=requested_model, model=requested_model, input_audio=input_audio
    )


def _make_vc_request(requested_model: str = "chatterbox_vc") -> VoiceConversionRequestInput:
    return VoiceConversionRequestInput(
        request_id="req-3",
        requested_model=requested_model,
        model=requested_model,
        source_audio="base64audio==",
        target_audio=None,
        target_embedding=None,
        generation_options=None,
        output_options=None,
    )


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_tts_without_reference_routes_to_synthesize():
    """TTS without audio/embedding (single speaker) routes straight to ChatterboxTTS."""
    tts_cfg = _make_model_config("cb-tts", "ChatterboxTTS")
    req = _make_tts_request(mode="single_speaker_tts")

    state, config = _route(req, tts_cfg)

    assert config.name == "cb-tts"
    assert PIPELINE_SPECS["chatterbox"].steps[state.step_index].name == "synthesize"


@pytest.mark.unit
def test_tts_with_precomputed_embedding_routes_to_synthesize():
    """TTS with input_embedding (no audio) skips the embed step."""
    tts_cfg = _make_model_config("cb-tts", "ChatterboxTTS")
    embed_cfg = _make_model_config("cb-embed", "ChatterboxEmbedding")
    req = _make_tts_request(input_audio=None, input_embedding="base64encodedembedding==")

    state, config = _route(req, tts_cfg, embed_cfg)

    assert config.name == "cb-tts"
    assert PIPELINE_SPECS["chatterbox"].steps[state.step_index].name == "synthesize"


@pytest.mark.unit
def test_tts_with_input_audio_routes_to_embed_step():
    """TTS with input_audio and no embedding routes to ChatterboxEmbedding first."""
    tts_cfg = _make_model_config("cb-tts", "ChatterboxTTS")
    embed_cfg = _make_model_config("cb-embed", "ChatterboxEmbedding")
    req = _make_tts_request(input_audio="base64audio==", input_embedding=None)

    state, config = _route(req, tts_cfg, embed_cfg)

    assert config.name == "cb-embed"
    assert PIPELINE_SPECS["chatterbox"].steps[state.step_index].name == "embed"


@pytest.mark.unit
def test_tts_with_input_audio_without_embed_executor_goes_straight_to_synthesis():
    """Without a ChatterboxEmbedding executor the TTS model embeds inline."""
    tts_cfg = _make_model_config("cb-tts", "ChatterboxTTS")
    req = _make_tts_request(input_audio="base64audio==", input_embedding=None)

    state, config = _route(req, tts_cfg)

    assert config.name == "cb-tts"
    assert PIPELINE_SPECS["chatterbox"].steps[state.step_index].name == "synthesize"


@pytest.mark.unit
def test_embed_request_routes_to_embedding_executor():
    """SpeechEmbeddingRequestInput routes to ChatterboxEmbedding and stops there."""
    embed_cfg = _make_model_config("cb-embed", "ChatterboxEmbedding")
    req = _make_embed_request(input_audio="base64audio==")

    state, config = _route(req, embed_cfg)

    assert config.name == "cb-embed"
    assert state.stop_after == "embed"


@pytest.mark.unit
def test_embed_request_falls_back_to_tts_executor():
    """SpeechEmbeddingRequestInput without ChatterboxEmbedding uses the TTS model."""
    tts_cfg = _make_model_config("cb-tts", "ChatterboxTTS")
    req = _make_embed_request(input_audio="base64audio==")

    _state, config = _route(req, tts_cfg)

    assert config.name == "cb-tts"


@pytest.mark.unit
def test_vc_request_routes_to_chatterbox_vc():
    """VoiceConversionRequestInput routes to the ChatterboxVC executor."""
    vc_cfg = _make_model_config("cb-vc", "ChatterboxVC")
    req = _make_vc_request()

    state, config = _route(req, vc_cfg)

    assert config.name == "cb-vc"
    assert PIPELINE_SPECS["chatterbox_vc"].steps[state.step_index].name == "convert"


@pytest.mark.unit
def test_turbo_tts_routes_to_turbo_executor():
    """TTS on chatterbox_turbo routes to the ChatterboxTurboTTS executor."""
    turbo_cfg = _make_model_config("cb-turbo", "ChatterboxTurboTTS")
    req = _make_tts_request(requested_model="chatterbox_turbo")

    _state, config = _route(req, turbo_cfg)

    assert config.name == "cb-turbo"


@pytest.mark.unit
def test_multilingual_tts_routes_to_multilingual_executor():
    """TTS on chatterbox_multilingual routes to the ChatterboxMultilingualTTS executor."""
    multi_cfg = _make_model_config("cb-multi", "ChatterboxMultilingualTTS")
    req = _make_tts_request(requested_model="chatterbox_multilingual")

    _state, config = _route(req, multi_cfg)

    assert config.name == "cb-multi"


@pytest.mark.unit
def test_embed_step_requires_reference_audio():
    """The embed step refuses to run without reference audio (typed error)."""
    from harmonyspeech.processing.pipeline_registry import MissingStepInputError

    embed_cfg = _make_model_config("cb-embed", "ChatterboxEmbedding")
    req = _make_tts_request(input_audio=None, input_embedding=None)
    spec = PIPELINE_SPECS["chatterbox"]
    state = resolve_entry(spec, req, {})
    state.step_index = spec.step_index("embed")

    with pytest.raises(MissingStepInputError, match="input_audio"):
        next_step(spec, state, req, [embed_cfg], {})


@pytest.mark.unit
def test_field_present_prefers_artifact_over_wire_field():
    request = _make_tts_request(input_audio=None)
    artifacts = {"input_audio": object()}

    assert field_present(request, artifacts, "input_audio") is True
    assert field_present(request, {}, "input_audio") is False


# ---------------------------------------------------------------------------
# Forwarding
# ---------------------------------------------------------------------------


@pytest.mark.unit
@patch("harmonyspeech.engine.harmonyspeech_engine.HarmonySpeechEngine.add_request")
def test_forward_processing_transfers_embedding_artifact(mock_add_request):
    """After the embed step, the speaker artifact is forwarded and audio dropped."""
    from harmonyspeech.common.request import EngineRequest, ExecutorResult
    from harmonyspeech.engine.harmonyspeech_engine import HarmonySpeechEngine

    tts_cfg = MagicMock()
    tts_cfg.name = "cb-tts"
    tts_cfg.model_type = "ChatterboxTTS"
    embed_cfg = MagicMock()
    embed_cfg.name = "cb-embed"
    embed_cfg.model_type = "ChatterboxEmbedding"

    engine = HarmonySpeechEngine.__new__(HarmonySpeechEngine)
    engine.model_configs = [tts_cfg, embed_cfg]
    engine.scheduler = MagicMock()

    conditionals = SimpleNamespace(gen={})
    speaker = SpeakerArtifact(conditionals=conditionals, backend="chatterbox")
    original_req = _make_tts_request(input_audio="base64audio==", input_embedding=None, mode="voice_cloning")
    original_req.model = "cb-embed"  # admission routed it to the embedding executor
    engine_request = EngineRequest(
        request_id="test-req-001", request_data=original_req, arrival_time=0.0, artifacts={"input_audio": object()}
    )
    engine._find_engine_request = lambda request_id: engine_request

    result = ExecutorResult(
        request_id="test-req-001",
        input_data=original_req,
        result_data=SpeechEmbeddingRequestOutput(request_id="test-req-001", output="", embedding_artifact=speaker),
    )

    new_status, forwarding_req = engine.check_forward_processing(result)

    assert new_status.name == "FINISHED_FORWARDED"
    assert forwarding_req is original_req
    mock_add_request.assert_called_once()
    kwargs = mock_add_request.call_args.kwargs
    assert kwargs["artifacts"]["input_embedding"] is speaker
    assert "input_audio" not in kwargs["artifacts"]
    assert kwargs["pipeline_state"].step_index == 1  # synthesize is next
    # No legacy wire field is written on the internal hop.
    assert original_req.input_embedding is None
    assert original_req.input_audio is None

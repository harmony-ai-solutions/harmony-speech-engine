"""Unit tests for Chatterbox routing logic in HarmonySpeechEngine.

Tests reroute_request_chatterbox(), reroute_request_chatterbox_vc(),
check_reroute_request_to_model() dispatch, and check_forward_processing()
multi-step embed→TTS chain — all without loading any real models.
"""
import pytest
from unittest.mock import MagicMock, patch

from harmonyspeech.engine.harmonyspeech_engine import HarmonySpeechEngine
from harmonyspeech.common.config import ModelConfig, DeviceConfig
from harmonyspeech.common.inputs import (
    TextToSpeechRequestInput,
    SpeechEmbeddingRequestInput,
    VoiceConversionRequestInput,
)
from harmonyspeech.common.outputs import SpeechEmbeddingRequestOutput


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model_config(name: str, model_type: str) -> ModelConfig:
    """Minimal ModelConfig for routing tests."""
    cfg = MagicMock(spec=ModelConfig)
    cfg.name = name
    cfg.model_type = model_type
    return cfg


def _make_engine(*model_configs) -> HarmonySpeechEngine:
    """Create a HarmonySpeechEngine shell with given ModelConfig list (no real models)."""
    engine = HarmonySpeechEngine.__new__(HarmonySpeechEngine)
    engine.model_configs = list(model_configs)
    engine.scheduler = MagicMock()
    return engine


def _make_tts_request(
    requested_model: str = "chatterbox",
    input_audio=None,
    input_embedding=None,
) -> TextToSpeechRequestInput:
    req = MagicMock(spec=TextToSpeechRequestInput)
    req.requested_model = requested_model
    req.input_audio = input_audio
    req.input_embedding = input_embedding
    req.model = requested_model
    return req


def _make_embed_request() -> SpeechEmbeddingRequestInput:
    req = MagicMock(spec=SpeechEmbeddingRequestInput)
    req.requested_model = "chatterbox"
    req.model = "chatterbox"
    return req


def _make_vc_request() -> VoiceConversionRequestInput:
    req = MagicMock(spec=VoiceConversionRequestInput)
    req.requested_model = "chatterbox_vc"
    req.model = "chatterbox_vc"
    return req


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_reroute_tts_no_cloning():
    """TTS without audio/embedding routes directly to ChatterboxTTS executor."""
    tts_cfg = _make_model_config("cb-tts", "ChatterboxTTS")
    engine = _make_engine(tts_cfg)
    req = _make_tts_request(requested_model="chatterbox")

    engine.reroute_request_chatterbox(req)

    assert req.model == "cb-tts"


@pytest.mark.unit
def test_reroute_tts_with_precomputed_embedding():
    """TTS with input_embedding (no audio) routes directly to ChatterboxTTS (no embed step)."""
    tts_cfg = _make_model_config("cb-tts", "ChatterboxTTS")
    embed_cfg = _make_model_config("cb-embed", "ChatterboxEmbedding")
    engine = _make_engine(tts_cfg, embed_cfg)
    req = _make_tts_request(
        requested_model="chatterbox",
        input_audio=None,
        input_embedding="base64encodedembedding==",
    )

    engine.reroute_request_chatterbox(req)

    # Must route to TTS directly — not to embedding
    assert req.model == "cb-tts"


@pytest.mark.unit
def test_reroute_tts_with_input_audio_to_embed_step():
    """TTS with input_audio + no embedding routes to ChatterboxEmbedding first."""
    tts_cfg = _make_model_config("cb-tts", "ChatterboxTTS")
    embed_cfg = _make_model_config("cb-embed", "ChatterboxEmbedding")
    engine = _make_engine(tts_cfg, embed_cfg)
    req = _make_tts_request(
        requested_model="chatterbox",
        input_audio="base64audio==",
        input_embedding=None,
    )

    engine.reroute_request_chatterbox(req)

    assert req.model == "cb-embed"


@pytest.mark.unit
def test_reroute_tts_embedding_fallback_to_tts():
    """TTS with input_audio but no ChatterboxEmbedding in config falls back to ChatterboxTTS."""
    tts_cfg = _make_model_config("cb-tts", "ChatterboxTTS")
    engine = _make_engine(tts_cfg)  # No embed config
    req = _make_tts_request(
        requested_model="chatterbox",
        input_audio="base64audio==",
        input_embedding=None,
    )
    # model stays at sentinel when no embed config found; reroute then checks TTS path
    # Set model to initial state (not yet routed)
    req.model = "chatterbox"

    engine.reroute_request_chatterbox(req)

    # Without ChatterboxEmbedding, the embed branch loop finishes without setting model;
    # fallback: request.model remains the sentinel and should be re-routed on next call
    # OR: the routing function should fall through to TTS.
    # Per CONTEXT.md, the fallback routes to ChatterboxTTS executor.
    # If implementation uses a direct fallback: model == "cb-tts"
    assert req.model == "cb-tts"


@pytest.mark.unit
def test_reroute_embed_request():
    """SpeechEmbeddingRequestInput routes to ChatterboxEmbedding executor."""
    embed_cfg = _make_model_config("cb-embed", "ChatterboxEmbedding")
    engine = _make_engine(embed_cfg)
    req = _make_embed_request()

    engine.reroute_request_chatterbox(req)

    assert req.model == "cb-embed"


@pytest.mark.unit
def test_reroute_embed_fallback_to_tts():
    """SpeechEmbeddingRequestInput without ChatterboxEmbedding in config falls back to ChatterboxTTS."""
    tts_cfg = _make_model_config("cb-tts", "ChatterboxTTS")
    engine = _make_engine(tts_cfg)  # No embed config
    req = _make_embed_request()
    req.model = "chatterbox"

    engine.reroute_request_chatterbox(req)

    # Fallback: routes to ChatterboxTTS when ChatterboxEmbedding unavailable
    assert req.model == "cb-tts"


@pytest.mark.unit
def test_reroute_vc_request():
    """VoiceConversionRequestInput routes to ChatterboxVC executor."""
    vc_cfg = _make_model_config("cb-vc", "ChatterboxVC")
    engine = _make_engine(vc_cfg)
    req = _make_vc_request()

    engine.reroute_request_chatterbox_vc(req)

    assert req.model == "cb-vc"


@pytest.mark.unit
def test_reroute_turbo_tts():
    """TTS on chatterbox_turbo model routes to ChatterboxTurboTTS executor."""
    turbo_cfg = _make_model_config("cb-turbo", "ChatterboxTurboTTS")
    engine = _make_engine(turbo_cfg)
    req = _make_tts_request(requested_model="chatterbox_turbo")

    engine.reroute_request_chatterbox(req)

    assert req.model == "cb-turbo"


@pytest.mark.unit
def test_reroute_multilingual_tts():
    """TTS on chatterbox_multilingual model routes to ChatterboxMultilingualTTS executor."""
    multi_cfg = _make_model_config("cb-multi", "ChatterboxMultilingualTTS")
    engine = _make_engine(multi_cfg)
    req = _make_tts_request(requested_model="chatterbox_multilingual")

    engine.reroute_request_chatterbox(req)

    assert req.model == "cb-multi"


@pytest.mark.unit
@patch.object(HarmonySpeechEngine, 'add_request')
def test_forward_processing_transfers_embedding(mock_add_request):
    """check_forward_processing() sets input_embedding and clears input_audio after embed step."""
    from harmonyspeech.common.request import ExecutorResult
    
    tts_cfg = _make_model_config("cb-tts", "ChatterboxTTS")
    embed_cfg = _make_model_config("cb-embed", "ChatterboxEmbedding")
    engine = _make_engine(tts_cfg, embed_cfg)

    # Build a fake completed embed result
    fake_embedding_output = "base64serializedconditionals=="
    embed_output = SpeechEmbeddingRequestOutput(
        request_id="test-req-001",
        output=fake_embedding_output
    )

    # Original TTS request (was routed to embed step first)
    original_req = _make_tts_request(
        requested_model="chatterbox",
        input_audio="base64audio==",
        input_embedding=None,
    )

    # Create ExecutorResult with embed output and original request as input
    result = ExecutorResult(
        request_id="test-req-001",
        input_data=original_req,
        result_data=embed_output,
    )

    # Call check_forward_processing - it returns (new_status, forwarding_request)
    new_status, forwarding_req = engine.check_forward_processing(result)

    # Verify add_request was called with embedding set and audio cleared
    mock_add_request.assert_called_once()
    call_args = mock_add_request.call_args
    # First arg is request_id, second is request_data
    resubmitted = call_args[0][1]
    assert resubmitted.input_embedding == fake_embedding_output
    assert resubmitted.input_audio is None
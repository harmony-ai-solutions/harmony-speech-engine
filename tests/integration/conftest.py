"""Integration test conftest.py — app and async client fixtures."""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from fastapi.testclient import TestClient

from harmonyspeech.endpoints.openai.protocol import (
    ModelList,
    ModelCard,
    TextToSpeechResponse,
    SpeechToTextResponse,
    EmbedSpeakerResponse,
    VoiceConversionResponse,
    DetectVoiceActivityResponse,
    AudioConversionResponse,
)


@pytest.fixture(scope="module")
def test_app():
    """
    Provides a FastAPI TestClient for integration tests.
    
    Note: Full app startup (model loading) is NOT performed here —
    integration tests in Phase 3 will configure this fixture with
    mock models. This fixture is a placeholder that will be expanded.
    """
    # Import deferred to avoid triggering model loading at collection time
    from fastapi.testclient import TestClient
    from harmonyspeech.endpoints.openai.api_server import app
    with TestClient(app, raise_server_exceptions=True) as client:
        yield client


@pytest.fixture(scope="module")
def mock_engine_app():
    """
    Provides a FastAPI TestClient with all serving module globals mocked.
    
    This fixture patches all 7 serving module globals in harmonyspeech.endpoints.openai.api_server:
    - openai_serving_tts
    - openai_serving_stt
    - openai_serving_vc
    - openai_serving_embedding
    - openai_serving_vad
    - openai_serving_ac
    - engine
    
    Each serving mock has async generate/detect/embed methods returning appropriate response fixtures.
    The engine mock has check_health that returns None (healthy).
    """
    # Create mock serving instances
    mock_tts = MagicMock()
    mock_tts.generate = AsyncMock(return_value=TextToSpeechResponse(data="dGVzdA=="))
    mock_tts.get_model_list = AsyncMock(return_value=ModelList(data=[ModelCard(id="test-tts-model")]))
    mock_tts.show_available_models = AsyncMock(return_value=ModelList(data=[ModelCard(id="test-tts-model")]))
    mock_tts.create_text_to_speech = AsyncMock(return_value=TextToSpeechResponse(data="dGVzdA=="))
    # Mock engine for health check
    mock_tts.engine = MagicMock()
    mock_tts.engine.check_health = AsyncMock(return_value=None)

    mock_stt = MagicMock()
    mock_stt.generate = AsyncMock(return_value=SpeechToTextResponse(text="test transcription"))
    mock_stt.get_model_list = AsyncMock(return_value=ModelList(data=[ModelCard(id="test-stt-model")]))
    mock_stt.show_available_models = AsyncMock(return_value=ModelList(data=[ModelCard(id="test-stt-model")]))
    mock_stt.create_transcription = AsyncMock(return_value=SpeechToTextResponse(text="test transcription"))
    mock_stt.engine = MagicMock()
    mock_stt.engine.check_health = AsyncMock(return_value=None)

    mock_vc = MagicMock()
    mock_vc.generate = AsyncMock(return_value=VoiceConversionResponse(data="dGVzdA=="))
    mock_vc.get_model_list = AsyncMock(return_value=ModelList(data=[ModelCard(id="test-vc-model")]))
    mock_vc.show_available_models = AsyncMock(return_value=ModelList(data=[ModelCard(id="test-vc-model")]))
    mock_vc.create_voice_conversion = AsyncMock(return_value=VoiceConversionResponse(data="dGVzdA=="))
    mock_vc.engine = MagicMock()
    mock_vc.engine.check_health = AsyncMock(return_value=None)

    mock_embedding = MagicMock()
    mock_embedding.generate = AsyncMock(return_value=EmbedSpeakerResponse(data="dGVzdA=="))
    mock_embedding.get_model_list = AsyncMock(return_value=ModelList(data=[ModelCard(id="test-embed-model")]))
    mock_embedding.show_available_models = AsyncMock(return_value=ModelList(data=[ModelCard(id="test-embed-model")]))
    mock_embedding.create_speaker_embedding = AsyncMock(return_value=EmbedSpeakerResponse(data="dGVzdA=="))
    mock_embedding.engine = MagicMock()
    mock_embedding.engine.check_health = AsyncMock(return_value=None)

    mock_vad = MagicMock()
    mock_vad.generate = AsyncMock(return_value=DetectVoiceActivityResponse(speech_activity=True))
    mock_vad.get_model_list = AsyncMock(return_value=ModelList(data=[ModelCard(id="test-vad-model")]))
    mock_vad.show_available_models = AsyncMock(return_value=ModelList(data=[ModelCard(id="test-vad-model")]))
    mock_vad.detect_voice_activity = AsyncMock(return_value=DetectVoiceActivityResponse(speech_activity=True))
    mock_vad.engine = MagicMock()
    mock_vad.engine.check_health = AsyncMock(return_value=None)

    mock_ac = MagicMock()
    mock_ac.generate = AsyncMock(return_value=AudioConversionResponse(data="dGVzdA=="))
    mock_ac.get_model_list = AsyncMock(return_value=ModelList(data=[ModelCard(id="test-ac-model")]))
    mock_ac.show_available_models = AsyncMock(return_value=ModelList(data=[ModelCard(id="test-ac-model")]))
    mock_ac.create_audio_conversion = AsyncMock(return_value=AudioConversionResponse(data="dGVzdA=="))
    mock_ac.engine = MagicMock()
    mock_ac.engine.check_health = AsyncMock(return_value=None)

    mock_engine = MagicMock()
    mock_engine.check_health = AsyncMock(return_value=None)

    # Create a mock engine_args with the needed attribute
    mock_engine_args = MagicMock()
    mock_engine_args.disable_log_stats = True  # Disable stats logging in tests

    # Patch all the module globals
    patches = [
        patch("harmonyspeech.endpoints.openai.api_server.openai_serving_tts", mock_tts),
        patch("harmonyspeech.endpoints.openai.api_server.openai_serving_stt", mock_stt),
        patch("harmonyspeech.endpoints.openai.api_server.openai_serving_vc", mock_vc),
        patch("harmonyspeech.endpoints.openai.api_server.openai_serving_embedding", mock_embedding),
        patch("harmonyspeech.endpoints.openai.api_server.openai_serving_vad", mock_vad),
        patch("harmonyspeech.endpoints.openai.api_server.openai_serving_ac", mock_ac),
        patch("harmonyspeech.endpoints.openai.api_server.engine", mock_engine),
        patch("harmonyspeech.endpoints.openai.api_server.engine_args", mock_engine_args),
    ]

    # Apply all patches
    for p in patches:
        p.start()

    # Import and create the TestClient using build_app
    from harmonyspeech.endpoints.openai.api_server import build_app
    from harmonyspeech.endpoints.openai.args import make_arg_parser
    
    # Create a proper args object using the arg parser
    parser = make_arg_parser()
    args = parser.parse_args([])
    
    app = build_app(args)
    with TestClient(app, raise_server_exceptions=True) as client:
        yield client

    # Stop all patches
    for p in patches:
        p.stop()

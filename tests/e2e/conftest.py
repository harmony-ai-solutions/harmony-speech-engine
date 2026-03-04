"""E2E test conftest.py — marks all e2e tests and provides model download fixtures."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from harmonyspeech.common.config import EngineConfig, ModelConfig, DeviceConfig
from harmonyspeech.engine.args_tools import AsyncEngineArgs
from harmonyspeech.engine.async_harmonyspeech import AsyncHarmonySpeech
from harmonyspeech.endpoints.openai.serving_text_to_speech import OpenAIServingTextToSpeech


# KittenTTS voices available across all variants
KITTENTTS_VOICES = ["Bella", "Jasper", "Luna", "Bruno", "Rosie", "Hugo", "Kiki", "Leo"]


def _create_kittentts_engine_fixture(name: str, model: str):
    """Factory to create a KittenTTS engine fixture for a specific variant."""
    def fixture(models_cache_dir):
        # Build ModelConfig for the KittenTTS variant
        model_config = ModelConfig(
            name=name,
            model=model,
            model_type="KittenTTSSynthesizer",
            max_batch_size=1,
            dtype="float32",
            device_config=DeviceConfig(device="cpu"),
            voices=KITTENTTS_VOICES,
        )
        
        # Build EngineConfig with the single model
        engine_config = EngineConfig(model_configs=[model_config])
        
        # Build AsyncEngineArgs
        engine_args = AsyncEngineArgs(
            disable_log_stats=True,
            disable_log_requests=True,
        )
        
        # Create the async engine
        engine = AsyncHarmonySpeech.from_engine_args_and_config(
            engine_args,
            engine_config,
            start_engine_loop=True
        )
        
        # Create the serving handler
        serving_tts = OpenAIServingTextToSpeech(
            engine,
            OpenAIServingTextToSpeech.models_from_config(engine_config.model_configs)
        )
        
        return (engine, serving_tts)
    
    fixture.__name__ = f"kittentts_{name}_engine"
    return pytest.fixture(scope="session")(fixture)


# Create the 4 KittenTTS engine fixtures
kittentts_mini_engine = _create_kittentts_engine_fixture("kitten-tts-mini", "KittenML/kitten-tts-mini-0.8")
kittentts_micro_engine = _create_kittentts_engine_fixture("kitten-tts-micro", "KittenML/kitten-tts-micro-0.8")
kittentts_nano_engine = _create_kittentts_engine_fixture("kitten-tts-nano", "KittenML/kitten-tts-nano-0.8-fp32")
kittentts_nano_int8_engine = _create_kittentts_engine_fixture("kitten-tts-nano-int8", "KittenML/kitten-tts-nano-0.8-int8")


@pytest.fixture(scope="session")
def mock_raw_request():
    """Session-scoped mock FastAPI Request for serving handler calls."""
    mock = MagicMock()
    mock.is_disconnected = AsyncMock(return_value=False)
    return mock


def pytest_collection_modifyitems(items):
    """Automatically mark all tests in tests/e2e/ with @pytest.mark.e2e."""
    for item in items:
        if "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)


@pytest.fixture(scope="session")
def models_cache_dir(tmp_path_factory):
    """
    Session-scoped temp directory for model weight caching during E2E tests.
    Models downloaded here are shared across all e2e tests in one session.
    """
    return tmp_path_factory.mktemp("models_cache")

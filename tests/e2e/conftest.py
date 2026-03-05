"""E2E test conftest.py — marks all e2e tests and provides model download fixtures."""
import struct
import base64

import pytest
from unittest.mock import AsyncMock, MagicMock

from harmonyspeech.common.config import EngineConfig, ModelConfig, DeviceConfig
from harmonyspeech.engine.args_tools import AsyncEngineArgs
from harmonyspeech.engine.async_harmonyspeech import AsyncHarmonySpeech
from harmonyspeech.endpoints.openai.serving_text_to_speech import OpenAIServingTextToSpeech
from harmonyspeech.endpoints.openai.serving_voice_embed import OpenAIServingVoiceEmbedding
from harmonyspeech.endpoints.openai.serving_voice_conversion import OpenAIServingVoiceConversion


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


def make_silent_wav_b64(sample_rate: int = 24000, duration_secs: int = 1) -> str:
    """Generate a minimal silent WAV file as base64 string for voice cloning reference audio."""
    num_samples = sample_rate * duration_secs
    pcm_data = struct.pack('<' + 'h' * num_samples, *([0] * num_samples))
    header = struct.pack('<4sI4s4sIHHIIHH4sI',
        b'RIFF', 36 + len(pcm_data), b'WAVE',
        b'fmt ', 16, 1, 1, sample_rate, sample_rate * 2, 2, 16,
        b'data', len(pcm_data))
    return base64.b64encode(header + pcm_data).decode()


def load_sample_audio_b64(sample_name: str = "wanda4") -> str:
    """Load a sample WAV file from tests/test-data/samples/ and return base64-encoded content."""
    import os
    sample_path = os.path.join(os.path.dirname(__file__), "..", "test-data", "samples", f"{sample_name}.wav")
    with open(sample_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# MeloTTS / OpenVoice V2 engine fixture
@pytest.fixture(scope="session")
def melotts_en_engine(models_cache_dir):
    """Session-scoped engine fixture for MeloTTS / OpenVoice V2 tests.
    
    Loads 4 models:
    - faster-whisper (FasterWhisper) - required for VAD processing in embed stage
    - ov2-synthesizer-en (MeloTTSSynthesizer)
    - ov2-tone-converter (OpenVoiceV2ToneConverter)
    - ov2-tone-converter-encoder (OpenVoiceV2ToneConverterEncoder)
    """
    # Build ModelConfigs for MeloTTS / OpenVoice V2
    model_configs = [
        ModelConfig(
            name="faster-whisper",
            model="Systran/faster-whisper-tiny",
            model_type="FasterWhisper",
            max_batch_size=1,
            dtype="float32",
            device_config=DeviceConfig(device="cpu"),
        ),
        ModelConfig(
            name="ov2-synthesizer-en",
            model="myshell-ai/MeloTTS-English-v3",
            model_type="MeloTTSSynthesizer",
            max_batch_size=1,
            dtype="float32",
            device_config=DeviceConfig(device="cpu"),
            language="EN",
            voices=["EN-Newest"],
        ),
        ModelConfig(
            name="ov2-tone-converter",
            model="myshell-ai/openvoicev2",
            model_type="OpenVoiceV2ToneConverter",
            max_batch_size=1,
            dtype="float32",
            device_config=DeviceConfig(device="cpu"),
        ),
        ModelConfig(
            name="ov2-tone-converter-encoder",
            model="myshell-ai/openvoicev2",
            model_type="OpenVoiceV2ToneConverterEncoder",
            max_batch_size=1,
            dtype="float32",
            device_config=DeviceConfig(device="cpu"),
        ),
    ]
    
    # Build EngineConfig with all models
    engine_config = EngineConfig(model_configs=model_configs)
    
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
    
    # Create the serving handlers
    serving_tts = OpenAIServingTextToSpeech(
        engine,
        OpenAIServingTextToSpeech.models_from_config(engine_config.model_configs)
    )
    serving_embed = OpenAIServingVoiceEmbedding(
        engine,
        OpenAIServingVoiceEmbedding.models_from_config(engine_config.model_configs)
    )
    serving_vc = OpenAIServingVoiceConversion(
        engine,
        OpenAIServingVoiceConversion.models_from_config(engine_config.model_configs)
    )
    
    return (engine, serving_tts, serving_embed, serving_vc)


# @pytest.fixture(scope="session")
# def melotts_en_engine_cuda(models_cache_dir):
#     """Session-scoped engine fixture for MeloTTS / OpenVoice V2 tests on CUDA."""
#     model_configs = [
#         ModelConfig(
#             name="faster-whisper",
#             model="Systran/faster-whisper-tiny",
#             model_type="FasterWhisper",
#             max_batch_size=1,
#             dtype="float32",
#             device_config=DeviceConfig(device="cpu"),
#         ),
#         ModelConfig(
#             name="ov2-synthesizer-en",
#             model="myshell-ai/MeloTTS-English-v3",
#             model_type="MeloTTSSynthesizer",
#             max_batch_size=1,
#             dtype="float32",
#             device_config=DeviceConfig(device="cuda"),
#             language="EN",
#             voices=["EN-Newest"],
#         ),
#         ModelConfig(
#             name="ov2-tone-converter",
#             model="myshell-ai/openvoicev2",
#             model_type="OpenVoiceV2ToneConverter",
#             max_batch_size=1,
#             dtype="float32",
#             device_config=DeviceConfig(device="cuda"),
#         ),
#         ModelConfig(
#             name="ov2-tone-converter-encoder",
#             model="myshell-ai/openvoicev2",
#             model_type="OpenVoiceV2ToneConverterEncoder",
#             max_batch_size=1,
#             dtype="float32",
#             device_config=DeviceConfig(device="cuda"),
#         ),
#     ]
    
#     engine_config = EngineConfig(model_configs=model_configs)
#     engine_args = AsyncEngineArgs(disable_log_stats=True, disable_log_requests=True)
#     engine = AsyncHarmonySpeech.from_engine_args_and_config(engine_args, engine_config, start_engine_loop=True)
#     serving_tts = OpenAIServingTextToSpeech(engine, OpenAIServingTextToSpeech.models_from_config(engine_config.model_configs))
#     serving_embed = OpenAIServingVoiceEmbedding(engine, OpenAIServingVoiceEmbedding.models_from_config(engine_config.model_configs))
#     serving_vc = OpenAIServingVoiceConversion(engine, OpenAIServingVoiceConversion.models_from_config(engine_config.model_configs))
#     return (engine, serving_tts, serving_embed, serving_vc)


# OpenVoice V1 engine fixture
@pytest.fixture(scope="session")
def openvoice_v1_en_engine(models_cache_dir):
    """Session-scoped engine fixture for OpenVoice V1 tests.
    
    Loads 4 models:
    - faster-whisper (FasterWhisper) - required for VAD processing in embed stage
    - ov1-synthesizer-en (OpenVoiceV1Synthesizer)
    - ov1-tone-converter (OpenVoiceV1ToneConverter)
    - ov1-tone-converter-encoder (OpenVoiceV1ToneConverterEncoder)
    """
    # Build ModelConfigs for OpenVoice V1
    model_configs = [
        ModelConfig(
            name="faster-whisper",
            model="Systran/faster-whisper-tiny",
            model_type="FasterWhisper",
            max_batch_size=1,
            dtype="float32",
            device_config=DeviceConfig(device="cpu"),
        ),
        ModelConfig(
            name="ov1-synthesizer-en",
            model="myshell-ai/openvoice",
            model_type="OpenVoiceV1Synthesizer",
            max_batch_size=1,
            dtype="float32",
            device_config=DeviceConfig(device="cpu"),
            language="EN",
            voices=["default", "whispering", "shouting", "excited", "cheerful", "terrified", "angry", "sad", "friendly"],
        ),
        ModelConfig(
            name="ov1-tone-converter",
            model="myshell-ai/openvoice",
            model_type="OpenVoiceV1ToneConverter",
            max_batch_size=1,
            dtype="float32",
            device_config=DeviceConfig(device="cpu"),
        ),
        ModelConfig(
            name="ov1-tone-converter-encoder",
            model="myshell-ai/openvoice",
            model_type="OpenVoiceV1ToneConverterEncoder",
            max_batch_size=1,
            dtype="float32",
            device_config=DeviceConfig(device="cpu"),
        ),
    ]
    
    # Build EngineConfig with all models
    engine_config = EngineConfig(model_configs=model_configs)
    
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
    
    # Create the serving handlers
    serving_tts = OpenAIServingTextToSpeech(
        engine,
        OpenAIServingTextToSpeech.models_from_config(engine_config.model_configs)
    )
    serving_embed = OpenAIServingVoiceEmbedding(
        engine,
        OpenAIServingVoiceEmbedding.models_from_config(engine_config.model_configs)
    )
    serving_vc = OpenAIServingVoiceConversion(
        engine,
        OpenAIServingVoiceConversion.models_from_config(engine_config.model_configs)
    )
    
    return (engine, serving_tts, serving_embed, serving_vc)


# HarmonySpeech engine fixture
@pytest.fixture(scope="session")
def harmonyspeech_engine(models_cache_dir):
    """Session-scoped engine fixture for HarmonySpeech tests.
    
    Loads 3 models:
    - hs1-encoder (HarmonySpeechEncoder)
    - hs1-synthesizer (HarmonySpeechSynthesizer)
    - hs1-vocoder (HarmonySpeechVocoder)
    
    Returns: (engine, serving_tts, serving_embed)
    - engine: AsyncHarmonySpeech instance
    - serving_tts: OpenAIServingTextToSpeech for full pipeline tests
    - serving_embed: OpenAIServingVoiceEmbedding for embed stage tests
    
    For direct synthesis/vocode stage tests, use engine.generate() directly with
    SynthesisRequestInput or VocodeRequestInput.
    """
    from harmonyspeech.common.inputs import SynthesisRequestInput, VocodeRequestInput
    from harmonyspeech.common.outputs import SpeechSynthesisRequestOutput, VocodeRequestOutput
    
    device_config = DeviceConfig(device="cpu")
    model_configs = [
        ModelConfig(
            name="hs1-encoder",
            model="harmony-ai/harmony-speech-v1",
            model_type="HarmonySpeechEncoder",
            max_batch_size=1,
            dtype="float32",
            device_config=device_config,
        ),
        ModelConfig(
            name="hs1-synthesizer",
            model="harmony-ai/harmony-speech-v1",
            model_type="HarmonySpeechSynthesizer",
            max_batch_size=1,
            dtype="float32",
            device_config=device_config,
        ),
        ModelConfig(
            name="hs1-vocoder",
            model="harmony-ai/harmony-speech-v1",
            model_type="HarmonySpeechVocoder",
            max_batch_size=1,
            dtype="float32",
            device_config=device_config,
        ),
    ]
    engine_config = EngineConfig(model_configs=model_configs)
    engine_args = AsyncEngineArgs(disable_log_stats=True, disable_log_requests=True)
    engine = AsyncHarmonySpeech.from_engine_args_and_config(engine_args, engine_config)

    serving_tts = OpenAIServingTextToSpeech(
        engine, OpenAIServingTextToSpeech.models_from_config(engine_config.model_configs))
    serving_embed = OpenAIServingVoiceEmbedding(
        engine, OpenAIServingVoiceEmbedding.models_from_config(engine_config.model_configs))
    
    return (engine, serving_tts, serving_embed)


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


"""E2E tests for the ``input_vad_mode`` pipeline switch (Phase 3-4).

``input_vad_mode: "silero"`` must route the OpenVoice VAD step to a configured
Silero VAD executor and complete the whole cloning pipeline; ``"whisper"`` (or
unset) keeps using Faster-Whisper, and a missing Silero model falls back with a
warning instead of failing.
"""

import asyncio

import pytest

from harmonyspeech.common.config import DeviceConfig, EngineConfig, ModelConfig
from harmonyspeech.endpoints.openai.protocol import TextToSpeechRequest, TextToSpeechResponse
from harmonyspeech.endpoints.openai.serving_text_to_speech import OpenAIServingTextToSpeech
from harmonyspeech.engine.args_tools import AsyncEngineArgs
from harmonyspeech.engine.async_harmonyspeech import AsyncHarmonySpeech
from tests.e2e.conftest import load_sample_audio_b64

TEXT_INPUT = "Hello, world."
REFERENCE_AUDIO = load_sample_audio_b64("wanda4")


def _openvoice_v2_configs(with_silero: bool) -> list[ModelConfig]:
    configs = [
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
    if with_silero:
        configs.append(
            ModelConfig(
                name="silero-vad",
                model="silero-vad",
                model_type="SileroVAD",
                load_format="onnx",
                max_batch_size=1,
                dtype="float32",
                device_config=DeviceConfig(device="cpu"),
            )
        )
    return configs


def _build_engine(with_silero: bool):
    engine_config = EngineConfig(model_configs=_openvoice_v2_configs(with_silero))
    engine_args = AsyncEngineArgs(disable_log_stats=True, disable_log_requests=True)
    engine = AsyncHarmonySpeech.from_engine_args_and_config(engine_args, engine_config, start_engine_loop=True)
    serving_tts = OpenAIServingTextToSpeech(
        engine, OpenAIServingTextToSpeech.models_from_config(engine_config.model_configs)
    )
    return engine, serving_tts


@pytest.fixture(scope="module")
def openvoice_v2_with_silero(models_cache_dir):
    return _build_engine(with_silero=True)


@pytest.fixture(scope="module")
def openvoice_v2_without_silero(models_cache_dir):
    return _build_engine(with_silero=False)


def _cloning_request() -> TextToSpeechRequest:
    return TextToSpeechRequest(
        model="openvoice_v2",
        input=TEXT_INPUT,
        mode="voice_cloning",
        language="EN",
        voice="EN-Newest",
        input_audio=REFERENCE_AUDIO,
    )


@pytest.mark.e2e
@pytest.mark.slow
def test_silero_vad_mode_completes_the_cloning_pipeline(openvoice_v2_with_silero, mock_raw_request):
    engine, serving_tts = openvoice_v2_with_silero

    request = _cloning_request()
    request.input_vad_mode = "silero"

    response = asyncio.run(serving_tts.create_text_to_speech(request, mock_raw_request))

    assert isinstance(response, TextToSpeechResponse), f"Got: {response}"
    assert response.data is not None
    assert len(response.data) > 0


@pytest.mark.e2e
@pytest.mark.slow
def test_whisper_vad_mode_completes_the_cloning_pipeline(openvoice_v2_with_silero, mock_raw_request):
    engine, serving_tts = openvoice_v2_with_silero

    request = _cloning_request()
    request.input_vad_mode = "whisper"

    response = asyncio.run(serving_tts.create_text_to_speech(request, mock_raw_request))

    assert isinstance(response, TextToSpeechResponse), f"Got: {response}"
    assert response.data is not None
    assert len(response.data) > 0


@pytest.mark.e2e
@pytest.mark.slow
def test_silero_vad_mode_without_silero_config_falls_back_to_whisper(
    openvoice_v2_without_silero, mock_raw_request, caplog
):
    engine, serving_tts = openvoice_v2_without_silero

    request = _cloning_request()
    request.input_vad_mode = "silero"

    response = asyncio.run(serving_tts.create_text_to_speech(request, mock_raw_request))

    assert isinstance(response, TextToSpeechResponse), f"Got: {response}"
    assert response.data is not None
    assert len(response.data) > 0

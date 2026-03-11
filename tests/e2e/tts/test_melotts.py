"""E2E tests for MeloTTS / OpenVoice V2.

These tests verify the full stack path from protocol request through serving handler
to the engine and back, covering:
- Single-speaker TTS (direct synthesizer model)
- Voice cloning (full 3-stage toolchain: embed -> synthesize -> tone transfer)
- Individual stage access: synthesize-only, tone-transfer-only, embed-only
"""
import asyncio

import pytest

from harmonyspeech.endpoints.openai.protocol import (
    TextToSpeechRequest,
    TextToSpeechResponse,
    VoiceConversionRequest,
    VoiceConversionResponse,
    EmbedSpeakerRequest,
    EmbedSpeakerResponse,
    ErrorResponse,
)
from tests.e2e.conftest import load_sample_audio_b64


TEXT_INPUT = "Hello, world."
REFERENCE_AUDIO = load_sample_audio_b64("wanda4")


@pytest.mark.e2e
@pytest.mark.slow
def test_melotts_en_single_speaker(melotts_en_engine, mock_raw_request):
    """MeloTTS EN: single-speaker TTS through full engine stack."""
    engine, serving_tts, serving_embed, serving_vc = melotts_en_engine
    request = TextToSpeechRequest(
        model="ov2-synthesizer-en",
        input=TEXT_INPUT,
        mode="single_speaker_tts",
        language="EN",
        voice="EN-Newest",
    )
    response = asyncio.run(serving_tts.create_text_to_speech(request, mock_raw_request))
    assert isinstance(response, TextToSpeechResponse), f"Got: {response}"
    assert response.data is not None
    assert len(response.data) > 0


@pytest.mark.e2e
@pytest.mark.slow
def test_melotts_en_voice_cloning(melotts_en_engine, mock_raw_request):
    """MeloTTS EN voice cloning: embed -> synthesize -> tone transfer toolchain."""
    engine, serving_tts, serving_embed, serving_vc = melotts_en_engine
    request = TextToSpeechRequest(
        model="openvoice_v2",
        input=TEXT_INPUT,
        mode="voice_cloning",
        language="EN",
        voice="EN-Newest",
        input_audio=REFERENCE_AUDIO,
    )
    response = asyncio.run(serving_tts.create_text_to_speech(request, mock_raw_request))
    assert isinstance(response, TextToSpeechResponse), f"Got: {response}"
    assert response.data is not None
    assert len(response.data) > 0


@pytest.mark.e2e
@pytest.mark.slow
def test_melotts_synthesize_stage(melotts_en_engine, mock_raw_request):
    """MeloTTS synthesize stage only: text -> audio via direct synthesizer model."""
    engine, serving_tts, serving_embed, serving_vc = melotts_en_engine
    request = TextToSpeechRequest(
        model="ov2-synthesizer-en",
        input=TEXT_INPUT,
        mode="single_speaker_tts",
        language="EN",
        voice="EN-Newest",
    )
    response = asyncio.run(serving_tts.create_text_to_speech(request, mock_raw_request))
    assert isinstance(response, TextToSpeechResponse), f"Got: {response}"
    assert response.data is not None
    assert len(response.data) > 0


@pytest.mark.e2e
@pytest.mark.slow
def test_openvoice_v2_tone_transfer_stage(melotts_en_engine, mock_raw_request):
    """OpenVoice V2 tone transfer stage: source_audio + target_embedding -> converted audio.
    
    Uses the openvoice_v2 toolchain name with the voice conversion endpoint.
    """
    engine, serving_tts, serving_embed, serving_vc = melotts_en_engine
    
    # Step 1: get embedding using toolchain name with embed endpoint
    embed_request = EmbedSpeakerRequest(model="openvoice_v2", input_audio=REFERENCE_AUDIO)
    embed_response = asyncio.run(serving_embed.create_voice_embedding(embed_request, mock_raw_request))
    assert isinstance(embed_response, EmbedSpeakerResponse), f"Embed failed: {embed_response}"
    assert embed_response.data is not None
    assert len(embed_response.data) > 0
    
    # Step 2: synthesize some audio to use as source
    tts_request = TextToSpeechRequest(
        model="ov2-synthesizer-en",
        input=TEXT_INPUT,
        mode="single_speaker_tts",
        language="EN",
        voice="EN-Newest"
    )
    tts_response = asyncio.run(serving_tts.create_text_to_speech(tts_request, mock_raw_request))
    assert isinstance(tts_response, TextToSpeechResponse), f"TTS failed: {tts_response}"
    assert tts_response.data is not None
    
    # Step 3: tone transfer using toolchain name with VC endpoint
    vc_request = VoiceConversionRequest(
        model="openvoice_v2",
        source_audio=tts_response.data,
        target_embedding=embed_response.data,
    )
    vc_response = asyncio.run(serving_vc.convert_voice(vc_request, mock_raw_request))
    assert isinstance(vc_response, VoiceConversionResponse), f"Got: {vc_response}"
    assert vc_response.data is not None
    assert len(vc_response.data) > 0


@pytest.mark.e2e
@pytest.mark.slow
def test_openvoice_v2_embed_stage(melotts_en_engine, mock_raw_request):
    """OpenVoice V2 speaker embedding stage: audio -> speaker embedding.
    
    Uses the openvoice_v2 toolchain name with the embedding endpoint.
    """
    engine, serving_tts, serving_embed, serving_vc = melotts_en_engine
    # Use toolchain name "openvoice_v2" with embed endpoint
    request = EmbedSpeakerRequest(model="openvoice_v2", input_audio=REFERENCE_AUDIO)
    response = asyncio.run(serving_embed.create_voice_embedding(request, mock_raw_request))
    assert isinstance(response, EmbedSpeakerResponse), f"Got: {response}"
    assert response.data is not None
    assert len(response.data) > 0


# @pytest.mark.e2e
# @pytest.mark.slow
# @pytest.mark.cuda
# def test_openvoice_v2_tone_transfer_stage_cuda(melotts_en_engine_cuda, mock_raw_request):
#     """OpenVoice V2 tone transfer stage on CUDA: synthesized audio + embedding -> voice-converted audio.
    
#     Uses the openvoice_v2 toolchain name with the voice conversion endpoint.
#     This test requires CUDA because the tone converter is too slow on CPU.
#     """
#     engine, serving_tts, serving_embed, serving_vc = melotts_en_engine_cuda
    
#     # Step 1: get speaker embedding from reference audio
#     embed_request = EmbedSpeakerRequest(model="openvoice_v2", input_audio=REFERENCE_AUDIO)
#     embed_response = asyncio.run(serving_embed.create_voice_embedding(embed_request, mock_raw_request))
#     assert isinstance(embed_response, EmbedSpeakerResponse), f"Embed failed: {embed_response}"
#     assert embed_response.data is not None
#     assert len(embed_response.data) > 0
    
#     # Step 2: synthesize some audio to use as source
#     tts_request = TextToSpeechRequest(
#         model="ov2-synthesizer-en",
#         input=TEXT_INPUT,
#         mode="single_speaker_tts",
#         language="EN",
#         voice="EN-Newest"
#     )
#     tts_response = asyncio.run(serving_tts.create_text_to_speech(tts_request, mock_raw_request))
#     assert isinstance(tts_response, TextToSpeechResponse), f"TTS failed: {tts_response}"
#     assert tts_response.data is not None
    
#     # Step 3: tone transfer using toolchain name with VC endpoint
#     vc_request = VoiceConversionRequest(
#         model="openvoice_v2",
#         source_audio=tts_response.data,
#         target_embedding=embed_response.data,
#     )
#     vc_response = asyncio.run(serving_vc.convert_voice(vc_request, mock_raw_request))
#     assert isinstance(vc_response, VoiceConversionResponse), f"Got: {vc_response}"
#     assert vc_response.data is not None
#     assert len(vc_response.data) > 0

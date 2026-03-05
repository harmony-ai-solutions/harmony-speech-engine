"""HarmonySpeech E2E tests - full 3-stage toolchain and individual stage access."""
import asyncio
import pytest

from harmonyspeech.common.inputs import SynthesisRequestInput, VocodeRequestInput
from harmonyspeech.common.outputs import SpeechSynthesisRequestOutput, VocodeRequestOutput
from harmonyspeech.endpoints.openai.protocol import (
    TextToSpeechRequest,
    TextToSpeechResponse,
    EmbedSpeakerRequest,
    EmbedSpeakerResponse,
    SynthesizeAudioRequest,
    SynthesizeAudioResponse,
    VocodeAudioRequest,
    VocodeAudioResponse,
)
from tests.e2e.conftest import load_sample_audio_b64


TEXT_INPUT = "Hello, world."
REFERENCE_AUDIO = load_sample_audio_b64("wanda4")


async def _get_embedding(serving_embed, embed_req, mock_raw_request):
    """Helper to get speaker embedding from reference audio."""
    embed_resp = await serving_embed.create_voice_embedding(embed_req, mock_raw_request)
    if not isinstance(embed_resp, EmbedSpeakerResponse):
        raise TypeError(f"Expected EmbedSpeakerResponse, got {type(embed_resp).__name__}")
    if embed_resp.data is None or len(embed_resp.data) == 0:
        raise ValueError("Embedding response data is empty")
    return embed_resp.data


async def _get_synthesis(engine, synth_req):
    """Helper to run synthesis stage directly through engine."""
    request_id = f"synth-{id(synth_req)}"
    result_generator = engine.generate(
        request_id=request_id,
        request_data=SynthesisRequestInput.from_openai(request_id, synth_req),
    )
    
    final_res = None
    async for res in result_generator:
        final_res = res
    
    if final_res is None:
        raise ValueError("No result from synthesis")
    
    if not isinstance(final_res, SpeechSynthesisRequestOutput):
        raise TypeError(f"Expected SpeechSynthesisRequestOutput, got {type(final_res).__name__}")
    
    if final_res.output is None or len(final_res.output) == 0:
        raise ValueError("Synthesis output is empty")
    
    return final_res.output


async def _get_vocode(engine, vocode_req):
    """Helper to run vocode stage directly through engine."""
    request_id = f"vocode-{id(vocode_req)}"
    result_generator = engine.generate(
        request_id=request_id,
        request_data=VocodeRequestInput.from_openai(request_id, vocode_req),
    )
    
    final_res = None
    async for res in result_generator:
        final_res = res
    
    if final_res is None:
        raise ValueError("No result from vocode")
    
    if not isinstance(final_res, VocodeRequestOutput):
        raise TypeError(f"Expected VocodeRequestOutput, got {type(final_res).__name__}")
    
    if final_res.output is None or len(final_res.output) == 0:
        raise ValueError("Vocode output is empty")
    
    return final_res.output


@pytest.mark.e2e
@pytest.mark.slow
def test_harmonyspeech_voice_cloning(harmonyspeech_engine, mock_raw_request):
    """HarmonySpeech voice cloning: embed -> synthesize -> vocode full toolchain.
    
    This test sends a TextToSpeechRequest with reference audio through the full pipeline:
    - TextToSpeechRequest with model="harmonyspeech", mode="voice_cloning", input_audio
    - Engine routes to HarmonySpeechEncoder first (embed reference speaker)
    - Encoder output (embedding) is forwarded to HarmonySpeechSynthesizer
    - Synthesizer output (mel) is forwarded to HarmonySpeechVocoder
    - Final audio is returned in TextToSpeechResponse
    """
    engine, serving_tts, serving_embed = harmonyspeech_engine
    
    request = TextToSpeechRequest(
        model="harmonyspeech",
        input=TEXT_INPUT,
        mode="voice_cloning",
        input_audio=REFERENCE_AUDIO,
    )
    
    response = asyncio.run(serving_tts.create_text_to_speech(request, mock_raw_request))
    
    assert isinstance(response, TextToSpeechResponse), f"Got: {response}"
    assert response.data is not None, "Response data is None"
    assert len(response.data) > 0, "Response data is empty"


@pytest.mark.e2e
@pytest.mark.slow
def test_harmonyspeech_embed_stage(harmonyspeech_engine, mock_raw_request):
    """HarmonySpeech encoder stage: reference audio -> speaker embedding.
    
    This test directly invokes the encoder stage:
    - EmbedSpeakerRequest with model="hs1-encoder", input_audio
    - OpenAIServingVoiceEmbedding handles the request
    - Returns EmbedSpeakerResponse with base64-encoded speaker embedding
    """
    engine, serving_tts, serving_embed = harmonyspeech_engine
    
    request = EmbedSpeakerRequest(
        model="hs1-encoder",
        input_audio=REFERENCE_AUDIO
    )
    
    response = asyncio.run(serving_embed.create_voice_embedding(request, mock_raw_request))
    
    assert isinstance(response, EmbedSpeakerResponse), f"Got: {response}"
    assert response.data is not None, "Response data is None"
    assert len(response.data) > 0, "Response data is empty"


@pytest.mark.e2e
@pytest.mark.slow
def test_harmonyspeech_synthesize_stage(harmonyspeech_engine, mock_raw_request):
    """HarmonySpeech synthesizer stage: text + embedding -> mel spectrogram.
    
    This test directly invokes the synthesis stage:
    1. First get an embedding from the encoder stage
    2. Then run synthesis with SynthesizeAudioRequest via engine.generate()
    3. Returns base64-encoded mel spectrogram (or audio depending on model output)
    """
    engine, serving_tts, serving_embed = harmonyspeech_engine
    
    # First get an embedding
    embed_req = EmbedSpeakerRequest(model="hs1-encoder", input_audio=REFERENCE_AUDIO)
    embedding = asyncio.run(_get_embedding(serving_embed, embed_req, mock_raw_request))
    
    # Then synthesize using the embedding
    synth_req = SynthesizeAudioRequest(
        model="hs1-synthesizer",
        input=TEXT_INPUT,
        input_embedding=embedding,
    )
    
    synth_output = asyncio.run(_get_synthesis(engine, synth_req))
    
    assert synth_output is not None, "Synthesis output is None"
    assert len(synth_output) > 0, "Synthesis output is empty"


@pytest.mark.e2e
@pytest.mark.slow
def test_harmonyspeech_vocode_stage(harmonyspeech_engine, mock_raw_request):
    """HarmonySpeech vocoder stage: mel spectrogram -> final audio.
    
    This test directly invokes the vocoder stage:
    1. Get an embedding from the encoder stage
    2. Run synthesis to get mel spectrogram
    3. Run vocode with VocodeAudioRequest via engine.generate()
    4. Returns base64-encoded final audio
    """
    engine, serving_tts, serving_embed = harmonyspeech_engine
    
    # Get an embedding
    embed_req = EmbedSpeakerRequest(model="hs1-encoder", input_audio=REFERENCE_AUDIO)
    embedding = asyncio.run(_get_embedding(serving_embed, embed_req, mock_raw_request))
    
    # Synthesize to get mel spectrogram
    synth_req = SynthesizeAudioRequest(
        model="hs1-synthesizer",
        input=TEXT_INPUT,
        input_embedding=embedding,
    )
    synth_output = asyncio.run(_get_synthesis(engine, synth_req))
    
    # Vocode the mel spectrogram
    vocode_req = VocodeAudioRequest(
        model="hs1-vocoder",
        input_audio=synth_output,
    )
    
    vocode_output = asyncio.run(_get_vocode(engine, vocode_req))
    
    assert vocode_output is not None, "Vocode output is None"
    assert len(vocode_output) > 0, "Vocode output is empty"

import base64
import io
import json
from concurrent.futures import ThreadPoolExecutor

import torch
import librosa
import numpy as np
from loguru import logger

from harmonyspeech.common.config import ModelConfig
from harmonyspeech.common.inputs import *
from harmonyspeech.common.request import EngineRequest
from harmonyspeech.modeling.loader import get_model_flavour, get_model_config
from harmonyspeech.modeling.models.harmonyspeech.common import preprocess_wav
from harmonyspeech.modeling.models.harmonyspeech.encoder.inputs import get_input_frames
from harmonyspeech.modeling.models.harmonyspeech.synthesizer.inputs import prepare_synthesis_inputs


def prepare_inputs(model_config: ModelConfig, requests_to_batch: List[EngineRequest]):
    """
    Prepares the request data depending on the model type this runner is executing.
    Throws a NotImplementedError if the model type is unknown
    :param model_config:
    :param requests_to_batch:
    :return:
    """
    inputs = []
    if model_config.model_type == "HarmonySpeechEncoder":
        for r in requests_to_batch:
            if (
                isinstance(r.request_data, TextToSpeechRequestInput) or
                isinstance(r.request_data, SpeechEmbeddingRequestInput)
            ):
                inputs.append(r.request_data)
            else:
                raise ValueError(
                    f"request ID {r.request_id} is not of type TextToSpeechRequestInput or "
                    f"SpeechEmbeddingRequestInput")
        return prepare_harmonyspeech_encoder_inputs(inputs)
    elif model_config.model_type == "HarmonySpeechSynthesizer":
        for r in requests_to_batch:
            if (
                isinstance(r.request_data, TextToSpeechRequestInput) or
                isinstance(r.request_data, SynthesisRequestInput)
            ):
                inputs.append(r.request_data)
            else:
                raise ValueError(f"request ID {r.request_id} is not of type TextToSpeechRequestInput")
        return prepare_harmonyspeech_synthesizer_inputs(inputs)
    elif model_config.model_type == "HarmonySpeechVocoder":
        for r in requests_to_batch:
            if (
                isinstance(r.request_data, TextToSpeechRequestInput) or
                isinstance(r.request_data, VocodeRequestInput)
            ):
                inputs.append(r.request_data)
            else:
                raise ValueError(
                    f"request ID {r.request_id} is not of type TextToSpeechRequestInput or "
                    f"VocodeAudioRequestInput")
        return prepare_harmonyspeech_vocoder_inputs(inputs)
    elif model_config.model_type in ["OpenVoiceV1ToneConverter", "OpenVoiceV2ToneConverter"]:
        for r in requests_to_batch:
            if (
                isinstance(r.request_data, TextToSpeechRequestInput) or
                isinstance(r.request_data, VoiceConversionRequestInput)
            ):
                inputs.append(r.request_data)
            else:
                raise ValueError(
                    f"request ID {r.request_id} is not of type TextToSpeechRequestInput or "
                    f"VoiceConversionRequestInput or SpeechEmbeddingRequestInput")
        return prepare_openvoice_tone_converter_inputs(model_config, inputs)
    elif model_config.model_type in ["OpenVoiceV1ToneConverterEncoder", "OpenVoiceV2ToneConverter"]:
        for r in requests_to_batch:
            if (
                isinstance(r.request_data, TextToSpeechRequestInput) or
                isinstance(r.request_data, VoiceConversionRequestInput)
            ):
                inputs.append(r.request_data)
            else:
                raise ValueError(
                    f"request ID {r.request_id} is not of type TextToSpeechRequestInput or "
                    f"VoiceConversionRequestInput or SpeechEmbeddingRequestInput")
        return prepare_openvoice_tone_converter_inputs(inputs)
    else:
        raise NotImplementedError(f"Cannot provide Inputs for model {model_config.model_type}")


def prepare_harmonyspeech_encoder_inputs(requests_to_batch: List[Union[
    TextToSpeechRequestInput,
    SpeechEmbeddingRequestInput
]]):
    # We're expecting audio in waveform format in the requests
    def prepare(request):
        # Make sure Audio is decoded from Base64
        input_audio = base64.b64decode(request.input_audio)
        preprocessed_audio = preprocess_wav(input_audio)
        input_frames = get_input_frames(preprocessed_audio)
        # input_frames_tensors = torch.from_numpy(input_frames).to(self.device)
        return input_frames

    with ThreadPoolExecutor() as executor:
        inputs = list(executor.map(prepare, requests_to_batch))

    return inputs


def prepare_harmonyspeech_synthesizer_inputs(requests_to_batch: List[Union[
    TextToSpeechRequestInput,
    SynthesisRequestInput
]]):
    # We're recieving a text, a voice embedding and voice modifiers
    def prepare(request):
        input_text, input_embedding = prepare_synthesis_inputs(request.input_text, request.input_embedding)

        # Base Input modifiers for HS Forward Tacotron
        speed_function = 1.0
        pitch_function = lambda x: x
        energy_function = lambda x: x

        if request.generation_options:
            if request.generation_options.speed:
                speed_function = float(request.generation_options.speed)
            if request.generation_options.pitch:
                pitch_function = lambda x: x * float(request.generation_options.pitch)
            if request.generation_options.energy:
                energy_function = lambda x: x * float(request.generation_options.energy)

        return input_text, input_embedding, speed_function, pitch_function, energy_function

    with ThreadPoolExecutor() as executor:
        inputs = list(executor.map(prepare, requests_to_batch))

    return inputs


def prepare_harmonyspeech_vocoder_inputs(requests_to_batch: List[Union[
    TextToSpeechRequestInput,
    VocodeRequestInput
]]):
    # We're expecting a synthesized mel spectogram and breaks
    def prepare(request):
        # TODO: Adapt this after optimizing synthesis step encoding
        synthesis_input = json.loads(request.input_audio)
        syn_mel = synthesis_input["mel"] if "mel" in synthesis_input else None
        syn_breaks = synthesis_input["breaks"] if "breaks" in synthesis_input else None

        # Decode input from base64
        try:
            # Theoretically, vocoder works without breaks
            if syn_breaks is not None:
                syn_breaks = base64.b64decode(syn_breaks.encode('utf-8'))
                syn_breaks = json.loads(syn_breaks)

            syn_mel = base64.b64decode(syn_mel.encode('utf-8'))
            syn_mel = json.loads(syn_mel)
            syn_mel = np.array(syn_mel, dtype=np.float32)
        except Exception as e:
            logger.error(str(e))

        # Normalize mel for decoding
        input_mel = syn_mel / 4.0  # Fixme: make this configurable
        return input_mel, syn_breaks

    with ThreadPoolExecutor() as executor:
        inputs = list(executor.map(prepare, requests_to_batch))

    return inputs


def prepare_openvoice_tone_converter_encoder_inputs(model_config: ModelConfig, requests_to_batch: List[Union[
    SpeechEmbeddingRequestInput,
    TextToSpeechRequestInput
]]):
    # Get model flavour if applicable
    flavour = get_model_flavour(model_config)
    # Load config
    hf_config = get_model_config(
        model_config.model,
        model_config.model_type,
        model_config.revision,
        flavour
    )

    # We're expecting audio in waveform format in the requests
    def prepare(request):
        # Make sure Audio is decoded from Base64
        # REMARK: Embedding is only set IF we want to convert the speaker voice in the audio to the embedded voice
        #         Otherwise (if only audio provided), we want to get the embedding instead.
        input_audio = base64.b64decode(request.input_audio)
        input_embedding = base64.b64decode(request.input_embedding.encode('utf-8')) if request.input_embedding else None
        bytes_pointer = io.BytesIO(input_audio)
        audio_data, _ = librosa.load(bytes_pointer, sr=hf_config.data.sampling_rate)

        # Split into different parts using VAD module
        # TODO: This could run through Whisper as well, see OpenVoice repo
        audio_data_tensor = torch.Tensor(audio_data)
        vad_segments = split_audio_vad(audio_data_tensor, hf_config.data.sampling_rate)



        return audio_ref, input_embedding

    with ThreadPoolExecutor() as executor:
        inputs = list(executor.map(prepare, requests_to_batch))

    return inputs


def prepare_openvoice_tone_converter_inputs(model_config: ModelConfig, requests_to_batch: List[Union[
    SpeechEmbeddingRequestInput,
    TextToSpeechRequestInput,
    VoiceConversionRequestInput
]]):
    # Get model flavour if applicable
    flavour = get_model_flavour(model_config)
    # Load config
    hf_config = get_model_config(
        model_config.model,
        model_config.model_type,
        model_config.revision,
        flavour
    )

    # We're expecting audio in waveform format in the requests
    def prepare(request):
        # Make sure Audio is decoded from Base64
        # REMARK: Embedding is only set IF we want to convert the speaker voice in the audio to the embedded voice
        #         Otherwise (if only audio provided), we want to get the embedding instead.
        input_audio = base64.b64decode(request.input_audio)
        input_embedding = base64.b64decode(request.input_embedding.encode('utf-8')) if request.input_embedding else None
        bytes_pointer = io.BytesIO(input_audio)
        audio_ref, _ = librosa.load(bytes_pointer, sr=hf_config.data.sampling_rate)
        return audio_ref, input_embedding

    with ThreadPoolExecutor() as executor:
        inputs = list(executor.map(prepare, requests_to_batch))

    return inputs

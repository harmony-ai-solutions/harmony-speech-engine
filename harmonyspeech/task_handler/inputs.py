import base64
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor
from tempfile import NamedTemporaryFile

import torch
import librosa
import numpy as np
from loguru import logger
from pydub import AudioSegment

from harmonyspeech.common.config import ModelConfig
from harmonyspeech.common.inputs import *
from harmonyspeech.common.request import EngineRequest
from harmonyspeech.modeling.loader import get_model_flavour, get_model_config, get_model_speaker
from harmonyspeech.modeling.models.harmonyspeech.common import preprocess_wav
from harmonyspeech.modeling.models.harmonyspeech.encoder.inputs import get_input_frames
from harmonyspeech.modeling.models.harmonyspeech.synthesizer.inputs import prepare_synthesis_inputs
from harmonyspeech.modeling.models.openvoice.inputs import normalize_text_inputs


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
                raise ValueError(f"request ID {r.request_id} is not of type TextToSpeechRequestInput or "
                                 f"SynthesisRequestInput")
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
                    f"VoiceConversionRequestInput")
        return prepare_openvoice_tone_converter_inputs(model_config, inputs)
    elif model_config.model_type in ["OpenVoiceV1ToneConverterEncoder", "OpenVoiceV2ToneConverterEncoder"]:
        for r in requests_to_batch:
            if (
                isinstance(r.request_data, SpeechEmbeddingRequestInput) or
                isinstance(r.request_data, TextToSpeechRequestInput) or
                isinstance(r.request_data, VoiceConversionRequestInput)
            ):
                inputs.append(r.request_data)
            else:
                raise ValueError(
                    f"request ID {r.request_id} is not of type TextToSpeechRequestInput or "
                    f"VoiceConversionRequestInput or SpeechEmbeddingRequestInput")
        return prepare_openvoice_tone_converter_encoder_inputs(model_config, inputs)
    elif model_config.model_type == "FasterWhisper":
        for r in requests_to_batch:
            if (
                isinstance(r.request_data, TextToSpeechRequestInput) or
                isinstance(r.request_data, SpeechTranscribeRequestInput)
            ):
                inputs.append(r.request_data)
            else:
                raise ValueError(
                    f"request ID {r.request_id} is not of type TextToSpeechRequestInput or "
                    f"SpeechTranscribeRequestInput")
        return prepare_faster_whisper_inputs(inputs)
    elif model_config.model_type == "OpenVoiceV1Synthesizer":
        for r in requests_to_batch:
            if (
                isinstance(r.request_data, TextToSpeechRequestInput) or
                isinstance(r.request_data, SynthesisRequestInput)
            ):
                inputs.append(r.request_data)
            else:
                raise ValueError(f"request ID {r.request_id} is not of type TextToSpeechRequestInput or "
                                 f"SynthesisRequestInput")
        return prepare_openvoice_synthesizer_inputs(model_config, inputs)
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

    # Based on VAD Data we're processing the Audio file provided
    # TODO: This expects Whisper VAD input, needs rework to be compatible with Silero VAD
    def prepare(request):
        # Make sure Audio is decoded from Base64
        input_audio = base64.b64decode(request.input_audio)
        input_vad_data = json.loads(request.input_vad_data)
        segments = input_vad_data["segments"] if "segments" in input_vad_data else []
        info = input_vad_data["info"] if "info" in input_vad_data else None

        # Decode Object Data
        if len(segments) > 0:
            segments = base64.b64decode(segments.encode('utf-8'))
            segments = json.loads(segments)
        if len(info) > 0:
            info = base64.b64decode(info.encode('utf-8'))
            info = json.loads(info)

        # Load Audio from BytesIO
        bytes_pointer = io.BytesIO(input_audio)
        audio_data, _ = librosa.load(bytes_pointer, sr=hf_config.data.sampling_rate)
        # convert from float to uint16
        # https://stackoverflow.com/questions/58810035/converting-audio-files-between-pydub-and-librosa
        audio_data = np.array(audio_data * (1 << 15), dtype=np.int16)
        audio = AudioSegment(
            audio_data.tobytes(),
            frame_rate=hf_config.data.sampling_rate,
            sample_width=audio_data.dtype.itemsize,
            channels=1
        )
        max_len = len(audio)

        # Iterate over VAD segments to build a list of audio segments exactly matching VAD parts
        # see OpenVoice - se_extractor.py:split_audio_whisper
        vad_audio_segments = []
        s_ind = 0
        start_time = None

        for k, segment in enumerate(segments):
            # process with the time
            if k == 0:
                start_time = max(0, segment["start"])
            end_time = segment["end"]

            # clean text
            text = segment["text"].replace('...', '')

            # left 0.08s for each audio
            audio_seg = audio[int(start_time * 1000): min(max_len, int(end_time * 1000) + 80)]

            # filter out the segment if shorter than 1.5s and longer than 20s
            save = 1.5 < audio_seg.duration_seconds < 20. and 2 <= len(text) < 200
            if save:
                # https://github.com/PyFilesystem/pyfilesystem2/issues/402#issuecomment-638750112
                # https://stackoverflow.com/questions/71765778/how-to-process-files-in-fastapi-from-multiple-clients-without-saving-the-files-t
                tmp_file = NamedTemporaryFile(delete=False, suffix=".ove")
                try:
                    audio_seg.export(tmp_file, format='wav')
                    audio_seg_bytes, _ = librosa.load(tmp_file.name, sr=hf_config.data.sampling_rate)
                    vad_audio_segments.append(audio_seg_bytes)
                except Exception as e:
                    logger.error(str(e))
                finally:
                    tmp_file.close()
                    os.unlink(tmp_file.name)

            if k < len(segments) - 1:
                start_time = max(0, segments[k + 1].start - 0.08)

            s_ind = s_ind + 1

        # Audio segments returned here will be used for inference
        # see reference implementation in OpenVoice - ToneColorConverter:extract_se
        return vad_audio_segments

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
        # Make sure Audio and Embedding are decoded from Base64
        input_audio = base64.b64decode(request.input_audio.encode('utf-8'))
        input_embedding = base64.b64decode(request.input_embedding.encode('utf-8'))
        input_audio_ref = io.BytesIO(input_audio)
        input_embedding_ref = io.BytesIO(input_embedding)
        audio_ref, _ = librosa.load(input_audio_ref, sr=hf_config.data.sampling_rate)

        # Get Base Speaker Embedding from Repo
        source_speaker_embedding_file = get_model_speaker(
            model_config.model,
            model_config.model_type,
            model_config.revision,
            request.language_id,
            request.voice_id
        )
        source_embedding_ref = io.BytesIO(source_speaker_embedding_file)

        return audio_ref, input_embedding_ref, source_embedding_ref

    with ThreadPoolExecutor() as executor:
        inputs = list(executor.map(prepare, requests_to_batch))

    return inputs


def prepare_faster_whisper_inputs(requests_to_batch: List[Union[
    TextToSpeechRequestInput,
    SpeechTranscribeRequestInput
]]):
    def prepare(request):
        # Make sure Audio data is decoded from Base64
        input_audio = base64.b64decode(request.input_audio)
        input_audio_ref = io.BytesIO(input_audio)
        audio_ref, _ = librosa.load(input_audio_ref, sr=16000)
        # input_frames_tensors = torch.from_numpy(input_frames).to(self.device)
        return audio_ref

    with ThreadPoolExecutor() as executor:
        inputs = list(executor.map(prepare, requests_to_batch))

    return inputs


def prepare_openvoice_synthesizer_inputs(model_config: ModelConfig, requests_to_batch: List[Union[
    TextToSpeechRequestInput,
    SynthesisRequestInput
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

    # We're recieving a text, a voice embedding and voice modifiers
    def prepare(request):
        input_text = request.input_text
        speaker_id = hf_config.speakers[request.voice_id]

        # Base Input modifiers for OpenVoice V1
        speed_modifier = 1.0
        if request.generation_options:
            if request.generation_options.speed:
                speed_modifier = float(request.generation_options.speed)

        text_normalized = normalize_text_inputs(input_text, hf_config, model_config.language)
        return text_normalized, speaker_id, speed_modifier

    with ThreadPoolExecutor() as executor:
        inputs = list(executor.map(prepare, requests_to_batch))

    return inputs

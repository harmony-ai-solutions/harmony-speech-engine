import base64
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor
from tempfile import NamedTemporaryFile

import librosa
import numpy as np
import torch
from chatterbox.tts import Conditionals
from loguru import logger
from pydub import AudioSegment

from harmonyspeech.common.config import ModelConfig
from harmonyspeech.common.inputs import (
    AudioConversionRequestInput,
    DetectVoiceActivityRequestInput,
    SpeechEmbeddingRequestInput,
    SpeechTranscribeRequestInput,
    SynthesisRequestInput,
    TextToSpeechRequestInput,
    VocodeRequestInput,
    VoiceConversionRequestInput,
)
from harmonyspeech.common.request import EngineRequest
from harmonyspeech.modeling.loader import get_model_config, get_model_flavour, get_model_speaker
from harmonyspeech.modeling.models.harmonyspeech.common import preprocess_wav
from harmonyspeech.modeling.models.harmonyspeech.encoder.inputs import get_input_frames
from harmonyspeech.modeling.models.harmonyspeech.synthesizer.inputs import prepare_synthesis_inputs
from harmonyspeech.modeling.models.melo.inputs import normalize_text_inputs as melo_normalize_inputs
from harmonyspeech.modeling.models.openvoice.inputs import normalize_text_inputs as ov1_normalize_inputs


def prepare_inputs(model_config: ModelConfig, requests_to_batch: list[EngineRequest]):
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
            if isinstance(r.request_data, TextToSpeechRequestInput) or isinstance(
                r.request_data, SpeechEmbeddingRequestInput
            ):
                inputs.append(r.request_data)
            else:
                raise ValueError(
                    f"request ID {r.request_id} is not of type TextToSpeechRequestInput or SpeechEmbeddingRequestInput"
                )
        return prepare_harmonyspeech_encoder_inputs(inputs)
    elif model_config.model_type == "HarmonySpeechSynthesizer":
        for r in requests_to_batch:
            if isinstance(r.request_data, TextToSpeechRequestInput) or isinstance(
                r.request_data, SynthesisRequestInput
            ):
                inputs.append(r.request_data)
            else:
                raise ValueError(
                    f"request ID {r.request_id} is not of type TextToSpeechRequestInput or SynthesisRequestInput"
                )
        return prepare_harmonyspeech_synthesizer_inputs(inputs)
    elif model_config.model_type == "HarmonySpeechVocoder":
        for r in requests_to_batch:
            if isinstance(r.request_data, TextToSpeechRequestInput) or isinstance(r.request_data, VocodeRequestInput):
                inputs.append(r.request_data)
            else:
                raise ValueError(
                    f"request ID {r.request_id} is not of type TextToSpeechRequestInput or VocodeAudioRequestInput"
                )
        return prepare_harmonyspeech_vocoder_inputs(inputs)
    elif model_config.model_type in ["OpenVoiceV1ToneConverter", "OpenVoiceV2ToneConverter"]:
        for r in requests_to_batch:
            if isinstance(r.request_data, TextToSpeechRequestInput) or isinstance(
                r.request_data, VoiceConversionRequestInput
            ):
                inputs.append(r.request_data)
            else:
                raise ValueError(
                    f"request ID {r.request_id} is not of type TextToSpeechRequestInput or VoiceConversionRequestInput"
                )
        return prepare_openvoice_tone_converter_inputs(model_config, inputs)
    elif model_config.model_type in ["OpenVoiceV1ToneConverterEncoder", "OpenVoiceV2ToneConverterEncoder"]:
        for r in requests_to_batch:
            if (
                isinstance(r.request_data, SpeechEmbeddingRequestInput)
                or isinstance(r.request_data, TextToSpeechRequestInput)
                or isinstance(r.request_data, VoiceConversionRequestInput)
            ):
                inputs.append(r.request_data)
            else:
                raise ValueError(
                    f"request ID {r.request_id} is not of type TextToSpeechRequestInput or VoiceConversionRequestInput or SpeechEmbeddingRequestInput"
                )
        return prepare_openvoice_tone_converter_encoder_inputs(model_config, inputs)
    elif model_config.model_type == "FasterWhisper":
        for r in requests_to_batch:
            if (
                isinstance(r.request_data, SpeechEmbeddingRequestInput)
                or isinstance(r.request_data, TextToSpeechRequestInput)
                or isinstance(r.request_data, SpeechTranscribeRequestInput)
                or isinstance(r.request_data, DetectVoiceActivityRequestInput)
            ):
                inputs.append(r.request_data)
            else:
                raise ValueError(
                    f"request ID {r.request_id} is not of type TextToSpeechRequestInput or SpeechTranscribeRequestInput or SpeechEmbeddingRequestInput orDetectVoiceActivityRequestInput"
                )
        return prepare_faster_whisper_inputs(inputs)
    elif model_config.model_type == "OpenVoiceV1Synthesizer":
        for r in requests_to_batch:
            if isinstance(r.request_data, TextToSpeechRequestInput) or isinstance(
                r.request_data, SynthesisRequestInput
            ):
                inputs.append(r.request_data)
            else:
                raise ValueError(
                    f"request ID {r.request_id} is not of type TextToSpeechRequestInput or SynthesisRequestInput"
                )
        return prepare_openvoice_synthesizer_inputs(model_config, inputs)
    elif model_config.model_type == "MeloTTSSynthesizer":
        for r in requests_to_batch:
            if isinstance(r.request_data, TextToSpeechRequestInput) or isinstance(
                r.request_data, SynthesisRequestInput
            ):
                inputs.append(r.request_data)
            else:
                raise ValueError(
                    f"request ID {r.request_id} is not of type TextToSpeechRequestInput or SynthesisRequestInput"
                )
        return prepare_melotts_synthesizer_inputs(model_config, inputs)
    elif model_config.model_type == "VoiceFixerRestorer":
        for r in requests_to_batch:
            if isinstance(r.request_data, AudioConversionRequestInput):
                inputs.append(r.request_data)
            else:
                raise ValueError(f"request ID {r.request_id} is not of type AudioConversionRequestInput")
        return prepare_voicefixer_restorer_inputs(inputs)
    elif model_config.model_type == "VoiceFixerVocoder":
        for r in requests_to_batch:
            if isinstance(r.request_data, AudioConversionRequestInput):
                inputs.append(r.request_data)
            else:
                raise ValueError(f"request ID {r.request_id} is not of type AudioConversionRequestInput")
        return prepare_voicefixer_vocoder_inputs(inputs)
    elif model_config.model_type == "SileroVAD":
        for r in requests_to_batch:
            if (
                isinstance(r.request_data, SpeechEmbeddingRequestInput)
                or isinstance(r.request_data, TextToSpeechRequestInput)
                or isinstance(r.request_data, SpeechTranscribeRequestInput)
                or isinstance(r.request_data, DetectVoiceActivityRequestInput)
            ):
                inputs.append(r.request_data)
            else:
                raise ValueError(
                    f"request ID {r.request_id} is not of type TextToSpeechRequestInput or SpeechTranscribeRequestInput or SpeechEmbeddingRequestInput orDetectVoiceActivityRequestInput"
                )
        return prepare_silero_vad_inputs(inputs)
    elif model_config.model_type == "KittenTTSSynthesizer":
        for r in requests_to_batch:
            if isinstance(r.request_data, TextToSpeechRequestInput) or isinstance(
                r.request_data, SynthesisRequestInput
            ):
                inputs.append(r.request_data)
            else:
                raise ValueError(
                    f"request ID {r.request_id} is not of type TextToSpeechRequestInput or SynthesisRequestInput"
                )
        return prepare_kittentts_synthesizer_inputs(inputs)
    elif model_config.model_type == "ChatterboxTTS":
        tts_inputs = []
        embed_inputs = []
        for r in requests_to_batch:
            if isinstance(r.request_data, SpeechEmbeddingRequestInput):
                # Embedding request routed to TTS model (no dedicated ChatterboxEmbedding)
                embed_inputs.append(r.request_data)
            elif isinstance(r.request_data, (TextToSpeechRequestInput, SynthesisRequestInput)):
                tts_inputs.append(r.request_data)
            else:
                raise ValueError(
                    f"ChatterboxTTS prepare_inputs: request ID {r.request_id} is not TextToSpeechRequestInput, SynthesisRequestInput, or SpeechEmbeddingRequestInput"
                )
        if embed_inputs:
            return prepare_chatterbox_embedding_inputs(embed_inputs)
        return prepare_chatterbox_tts_inputs(tts_inputs)
    elif model_config.model_type == "ChatterboxTurboTTS":
        tts_inputs = []
        embed_inputs = []
        for r in requests_to_batch:
            if isinstance(r.request_data, SpeechEmbeddingRequestInput):
                embed_inputs.append(r.request_data)
            elif isinstance(r.request_data, (TextToSpeechRequestInput, SynthesisRequestInput)):
                tts_inputs.append(r.request_data)
            else:
                raise ValueError(
                    f"ChatterboxTurboTTS prepare_inputs: request ID {r.request_id} is not TextToSpeechRequestInput, SynthesisRequestInput, or SpeechEmbeddingRequestInput"
                )
        if embed_inputs:
            return prepare_chatterbox_embedding_inputs(embed_inputs)
        return prepare_chatterbox_turbo_tts_inputs(tts_inputs)
    elif model_config.model_type == "ChatterboxMultilingualTTS":
        tts_inputs = []
        embed_inputs = []
        for r in requests_to_batch:
            if isinstance(r.request_data, SpeechEmbeddingRequestInput):
                embed_inputs.append(r.request_data)
            elif isinstance(r.request_data, (TextToSpeechRequestInput, SynthesisRequestInput)):
                tts_inputs.append(r.request_data)
            else:
                raise ValueError(
                    f"ChatterboxMultilingualTTS prepare_inputs: request ID {r.request_id} is not TextToSpeechRequestInput, SynthesisRequestInput, or SpeechEmbeddingRequestInput"
                )
        if embed_inputs:
            return prepare_chatterbox_embedding_inputs(embed_inputs)
        return prepare_chatterbox_multilingual_tts_inputs(tts_inputs)
    elif model_config.model_type == "ChatterboxVC":
        for r in requests_to_batch:
            if isinstance(r.request_data, VoiceConversionRequestInput):
                inputs.append(r.request_data)
            else:
                raise ValueError(f"request ID {r.request_id} is not of type VoiceConversionRequestInput")
        return prepare_chatterbox_vc_inputs(inputs)
    elif model_config.model_type == "ChatterboxEmbedding":
        # Accept both SpeechEmbeddingRequestInput (dedicated embed endpoint) and
        # TextToSpeechRequestInput (voice-cloning pipeline routes the TTS request
        # here for the embedding step before forwarding to the TTS executor).
        # prepare_chatterbox_embedding_inputs only needs input_audio, which both
        # request types carry.
        for r in requests_to_batch:
            if isinstance(r.request_data, (SpeechEmbeddingRequestInput, TextToSpeechRequestInput)):
                inputs.append(r.request_data)
            else:
                raise ValueError(
                    f"request ID {r.request_id} is not of type SpeechEmbeddingRequestInput or TextToSpeechRequestInput"
                )
        return prepare_chatterbox_embedding_inputs(inputs)
    else:
        raise NotImplementedError(f"Cannot provide Inputs for model {model_config.model_type}")


def prepare_harmonyspeech_encoder_inputs(
    requests_to_batch: list[TextToSpeechRequestInput | SpeechEmbeddingRequestInput],
):
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


def prepare_voicefixer_restorer_inputs(requests_to_batch: list[AudioConversionRequestInput]):
    """
    Prepare inputs for VoiceFixerRestorer model.
    Expects audio data in base64 format and converts to tensor format.
    """

    def prepare(request):
        # Make sure Audio is decoded from Base64
        input_audio = base64.b64decode(request.source_audio)
        input_audio_ref = io.BytesIO(input_audio)

        # Load audio using librosa at VoiceFixer's expected sample rate (44100 Hz)
        audio_data, _ = librosa.load(input_audio_ref, sr=44100)

        # Convert to tensor format expected by VoiceFixerRestorer
        # The model expects [batch, channels, samples] format
        # Add channel dimension if mono audio, then add batch dimension
        if audio_data.ndim == 1:
            # [samples] -> [channels, samples] -> [batch, channels, samples]
            audio_tensor = torch.FloatTensor(audio_data).unsqueeze(0).unsqueeze(0)
        else:
            # [channels, samples] -> [batch, channels, samples]
            audio_tensor = torch.FloatTensor(audio_data).unsqueeze(0)

        return audio_tensor

    with ThreadPoolExecutor() as executor:
        inputs = list(executor.map(prepare, requests_to_batch))

    return inputs


def prepare_voicefixer_vocoder_inputs(requests_to_batch: list[AudioConversionRequestInput]):
    """
    Prepare inputs for VoiceFixerVocoder model.
    Expects mel spectrogram data in base64 format and converts to tensor format.
    """

    def prepare(request):
        # Decode mel spectrogram from base64
        mel_data = base64.b64decode(request.input_mel_spectrogram)
        mel_json = json.loads(mel_data.decode("utf-8"))

        # Convert to numpy array and then to tensor
        mel_array = np.array(mel_json, dtype=np.float32)

        # Convert to tensor format expected by VoiceFixerVocoder
        # The model expects [batch, mel_bins, time] format
        if mel_array.ndim == 2:
            mel_tensor = torch.FloatTensor(mel_array).unsqueeze(0)  # Add batch dimension
        else:
            mel_tensor = torch.FloatTensor(mel_array)

        return mel_tensor

    with ThreadPoolExecutor() as executor:
        inputs = list(executor.map(prepare, requests_to_batch))

    return inputs


def prepare_harmonyspeech_synthesizer_inputs(requests_to_batch: list[TextToSpeechRequestInput | SynthesisRequestInput]):
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


def prepare_harmonyspeech_vocoder_inputs(requests_to_batch: list[TextToSpeechRequestInput | VocodeRequestInput]):
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
                syn_breaks = base64.b64decode(syn_breaks.encode("utf-8"))
                syn_breaks = json.loads(syn_breaks)

            syn_mel = base64.b64decode(syn_mel.encode("utf-8"))
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


def prepare_openvoice_tone_converter_encoder_inputs(
    model_config: ModelConfig, requests_to_batch: list[SpeechEmbeddingRequestInput | TextToSpeechRequestInput]
):
    # Get model flavour if applicable
    flavour = get_model_flavour(model_config)
    # Load config
    hf_config = get_model_config(model_config.model, model_config.model_type, model_config.revision, flavour)

    # Based on VAD Data we're processing the Audio file provided
    # TODO: This expects Whisper VAD input, needs rework to be compatible with Silero VAD
    def prepare(request):
        # Make sure Audio is decoded from Base64
        input_audio = base64.b64decode(request.input_audio)
        input_vad_data = json.loads(request.input_vad_data)
        segments = input_vad_data["segments"] if "segments" in input_vad_data else []
        info = input_vad_data["info"] if "info" in input_vad_data else None

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
            channels=1,
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
            text = segment["text"].replace("...", "")

            # left 0.08s for each audio
            audio_seg = audio[int(start_time * 1000) : min(max_len, int(end_time * 1000) + 80)]

            # filter out the segment if shorter than 1.5s and longer than 20s
            save = 1.5 < audio_seg.duration_seconds < 20.0 and 2 <= len(text) < 200
            if save:
                # https://github.com/PyFilesystem/pyfilesystem2/issues/402#issuecomment-638750112
                # https://stackoverflow.com/questions/71765778/how-to-process-files-in-fastapi-from-multiple-clients-without-saving-the-files-t
                tmp_file = NamedTemporaryFile(delete=False, suffix=".ove")
                try:
                    audio_seg.export(tmp_file, format="wav")
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


def prepare_openvoice_tone_converter_inputs(
    model_config: ModelConfig,
    requests_to_batch: list[SpeechEmbeddingRequestInput | TextToSpeechRequestInput | VoiceConversionRequestInput],
):
    # Get model flavour if applicable
    flavour = get_model_flavour(model_config)
    # Load config
    hf_config = get_model_config(model_config.model, model_config.model_type, model_config.revision, flavour)

    # We're expecting audio in waveform format in the requests
    def prepare(request):
        # Handle both TextToSpeechRequestInput (for voice cloning) and VoiceConversionRequestInput
        # For VoiceConversionRequestInput: source_audio, target_embedding
        # For TextToSpeechRequestInput: input_audio, input_embedding
        if hasattr(request, "source_audio"):
            # VoiceConversionRequestInput
            audio_data = request.source_audio
            embedding_data = request.target_embedding
        else:
            # TextToSpeechRequestInput
            audio_data = request.input_audio
            embedding_data = request.input_embedding

        # Make sure Audio and Embedding are decoded from Base64
        input_audio = base64.b64decode(audio_data.encode("utf-8"))
        input_embedding = base64.b64decode(embedding_data.encode("utf-8"))
        input_audio_ref = io.BytesIO(input_audio)
        input_embedding_ref = io.BytesIO(input_embedding)
        audio_ref, _ = librosa.load(input_audio_ref, sr=hf_config.data.sampling_rate)

        # For voice conversion, use target embedding directly
        # For voice cloning via TTS, get source embedding from repo
        if hasattr(request, "language_id") and hasattr(request, "voice_id"):
            # TextToSpeechRequestInput - get source embedding from repo
            source_speaker_embedding_file = get_model_speaker(
                model_config.model,
                model_config.model_type,
                model_config.revision,
                request.language_id,
                request.voice_id,
            )
            source_embedding_ref = io.BytesIO(source_speaker_embedding_file)
        else:
            # VoiceConversionRequestInput - use target embedding as source
            # Create a fresh BytesIO since input_embedding_ref may have been read
            source_embedding_ref = io.BytesIO(input_embedding)

        return audio_ref, input_embedding_ref, source_embedding_ref

    with ThreadPoolExecutor() as executor:
        inputs = list(executor.map(prepare, requests_to_batch))

    return inputs


def prepare_faster_whisper_inputs(
    requests_to_batch: list[TextToSpeechRequestInput | SpeechTranscribeRequestInput | DetectVoiceActivityRequestInput],
):
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


def prepare_openvoice_synthesizer_inputs(
    model_config: ModelConfig, requests_to_batch: list[TextToSpeechRequestInput | SynthesisRequestInput]
):
    # Get model flavour if applicable
    flavour = get_model_flavour(model_config)
    # Load config
    hf_config = get_model_config(model_config.model, model_config.model_type, model_config.revision, flavour)

    # We're recieving a text, a voice embedding and voice modifiers
    def prepare(request):
        input_text = request.input_text
        speaker_id = hf_config.speakers[request.voice_id]

        # Base Input modifiers for OpenVoice V1
        speed_modifier = 1.0
        if request.generation_options:
            if request.generation_options.speed:
                speed_modifier = float(request.generation_options.speed)

        text_normalized = ov1_normalize_inputs(input_text, hf_config, model_config.language)
        return text_normalized, speaker_id, speed_modifier

    with ThreadPoolExecutor() as executor:
        inputs = list(executor.map(prepare, requests_to_batch))

    return inputs


def prepare_melotts_synthesizer_inputs(
    model_config: ModelConfig, requests_to_batch: list[TextToSpeechRequestInput | SynthesisRequestInput]
):
    # Get model flavour if applicable
    flavour = get_model_flavour(model_config)
    # Load config
    hf_config = get_model_config(model_config.model, model_config.model_type, model_config.revision, flavour)

    # We're recieving a text, a voice embedding and voice modifiers
    def prepare(request):
        input_text = request.input_text
        speaker_id = hf_config.data.spk2id[request.voice_id]

        # Base Input modifiers for MeloTTS
        speed_modifier = 1.0
        if request.generation_options:
            if request.generation_options.speed:
                speed_modifier = float(request.generation_options.speed)

        inference_items = melo_normalize_inputs(input_text, hf_config, model_config.language)
        return inference_items, speaker_id, speed_modifier

    with ThreadPoolExecutor() as executor:
        inputs = list(executor.map(prepare, requests_to_batch))

    return inputs


def prepare_silero_vad_inputs(requests_to_batch: list[DetectVoiceActivityRequestInput]):
    """
    Prepare inputs for Silero VAD model.
    Expects audio data in base64 format and converts to tensor format.
    Includes VAD parameters as part of the input object.
    """

    def prepare(request):
        # Decode base64 audio
        input_audio = base64.b64decode(request.input_audio)
        input_audio_ref = io.BytesIO(input_audio)

        # Load audio at 16kHz (Silero's expected sample rate)
        audio_ref, _ = librosa.load(input_audio_ref, sr=16000)

        # Convert to tensor format expected by Silero VAD
        audio_tensor = torch.FloatTensor(audio_ref)

        # Extract VAD parameters from request
        vad_params = {
            "threshold": getattr(request, "threshold", 0.5),
            "min_speech_duration_ms": getattr(request, "min_speech_duration_ms", 250),
            "min_silence_duration_ms": getattr(request, "min_silence_duration_ms", 100),
            "speech_pad_ms": getattr(request, "speech_pad_ms", 30),
            "return_seconds": getattr(request, "return_seconds", False),
            "get_timestamps": getattr(request, "get_timestamps", False),
        }

        return (audio_tensor, vad_params)

    with ThreadPoolExecutor() as executor:
        inputs = list(executor.map(prepare, requests_to_batch))

    return inputs


def prepare_kittentts_synthesizer_inputs(requests_to_batch: list[TextToSpeechRequestInput | SynthesisRequestInput]):
    """
    Prepare inputs for KittenTTSSynthesizer model.
    Extracts text, voice name, and speed from the request.
    No preprocessing or BERT tokenization needed — KittenTTS handles this internally.
    """

    def prepare(request):
        input_text = request.input_text

        # Voice selection: use voice_id from request or fall back to default
        voice = request.voice_id if request.voice_id else "Jasper"

        # Speed modifier
        speed_modifier = 1.0
        if request.generation_options:
            if request.generation_options.speed:
                speed_modifier = float(request.generation_options.speed)

        return input_text, voice, speed_modifier

    with ThreadPoolExecutor() as executor:
        inputs = list(executor.map(prepare, requests_to_batch))

    return inputs


def prepare_chatterbox_tts_inputs(requests_to_batch: list[TextToSpeechRequestInput | SynthesisRequestInput]):
    """
    Prepare inputs for ChatterboxTTS model.

    Returns list of tuples:
        (input_text, conditionals_or_None, exaggeration, cfg_weight, temperature,
         repetition_penalty, top_p, min_p)

    Raises ValueError for:
    - Non-None top_k or norm_loudness (Turbo-only params)
    - Both input_audio AND input_embedding provided (conflict)
    """

    def prepare(request):
        opts = request.generation_options

        # Validate unsupported params — ChatterboxTTS rejects Turbo-only fields
        if opts is not None:
            if opts.top_k is not None:
                raise ValueError("top_k is not supported by ChatterboxTTS")
            if opts.norm_loudness is not None:
                raise ValueError("norm_loudness is not supported by ChatterboxTTS")

        # Conflict check: cannot have both audio AND embedding
        if getattr(request, "input_audio", None) is not None and getattr(request, "input_embedding", None) is not None:
            raise ValueError("Provide either input_audio or input_embedding, not both.")

        # Deserialize pre-computed Conditionals if provided
        conditionals = None
        if getattr(request, "input_embedding", None) is not None:
            embedding_bytes = base64.b64decode(request.input_embedding)
            embedding_buf = io.BytesIO(embedding_bytes)
            conditionals = Conditionals.load(embedding_buf, map_location="cpu")

        # Apply model-specific defaults for ChatterboxTTS
        exaggeration = opts.exaggeration if opts is not None and opts.exaggeration is not None else 0.5
        cfg_weight = opts.cfg_weight if opts is not None and opts.cfg_weight is not None else 0.5
        temperature = opts.temperature if opts is not None and opts.temperature is not None else 0.8
        repetition_penalty = (
            opts.repetition_penalty if opts is not None and opts.repetition_penalty is not None else 1.2
        )
        top_p = opts.top_p if opts is not None and opts.top_p is not None else 1.0
        min_p = opts.min_p if opts is not None and opts.min_p is not None else 0.05

        return request.input_text, conditionals, exaggeration, cfg_weight, temperature, repetition_penalty, top_p, min_p

    with ThreadPoolExecutor() as executor:
        inputs = list(executor.map(prepare, requests_to_batch))
    return inputs


def prepare_chatterbox_turbo_tts_inputs(requests_to_batch: list[TextToSpeechRequestInput | SynthesisRequestInput]):
    """
    Prepare inputs for ChatterboxTurboTTS model.

    Returns list of tuples:
        (input_text, conditionals_or_None, temperature, repetition_penalty, top_p, top_k, norm_loudness)

    Raises ValueError for:
    - Non-None exaggeration, cfg_weight, or min_p (base-TTS-only params)
    - Both input_audio AND input_embedding provided (conflict)
    """

    def prepare(request):
        opts = request.generation_options

        # Validate unsupported params — ChatterboxTurboTTS rejects base-TTS-only fields
        if opts is not None:
            if opts.exaggeration is not None:
                raise ValueError("exaggeration is not supported by ChatterboxTurboTTS")
            if opts.cfg_weight is not None:
                raise ValueError("cfg_weight is not supported by ChatterboxTurboTTS")
            if opts.min_p is not None:
                raise ValueError("min_p is not supported by ChatterboxTurboTTS")

        # Conflict check: cannot have both audio AND embedding
        if getattr(request, "input_audio", None) is not None and getattr(request, "input_embedding", None) is not None:
            raise ValueError("Provide either input_audio or input_embedding, not both.")

        # Deserialize pre-computed Conditionals if provided
        conditionals = None
        if getattr(request, "input_embedding", None) is not None:
            embedding_bytes = base64.b64decode(request.input_embedding)
            embedding_buf = io.BytesIO(embedding_bytes)
            conditionals = Conditionals.load(embedding_buf, map_location="cpu")

        # Apply Turbo-specific defaults
        temperature = opts.temperature if opts is not None and opts.temperature is not None else 0.8
        repetition_penalty = (
            opts.repetition_penalty if opts is not None and opts.repetition_penalty is not None else 1.2
        )
        top_p = opts.top_p if opts is not None and opts.top_p is not None else 0.95
        top_k = opts.top_k if opts is not None and opts.top_k is not None else 1000
        norm_loudness = opts.norm_loudness if opts is not None and opts.norm_loudness is not None else True

        return request.input_text, conditionals, temperature, repetition_penalty, top_p, top_k, norm_loudness

    with ThreadPoolExecutor() as executor:
        inputs = list(executor.map(prepare, requests_to_batch))
    return inputs


def prepare_chatterbox_multilingual_tts_inputs(
    requests_to_batch: list[TextToSpeechRequestInput | SynthesisRequestInput],
):
    """
    Prepare inputs for ChatterboxMultilingualTTS model.

    Language validation is handled upstream by the serving engine.
    This function only defaults language_id to 'en' when absent/None.

    Returns list of tuples:
        (input_text, language_id, conditionals_or_None, exaggeration, cfg_weight, temperature,
         repetition_penalty, top_p, min_p)

    Raises ValueError for:
    - Non-None top_k or norm_loudness (Turbo-only params)
    - Both input_audio AND input_embedding provided (conflict)
    """

    def prepare(request):
        opts = request.generation_options

        # Validate unsupported params — ChatterboxMultilingualTTS rejects Turbo-only fields
        if opts is not None:
            if opts.top_k is not None:
                raise ValueError("top_k is not supported by ChatterboxMultilingualTTS")
            if opts.norm_loudness is not None:
                raise ValueError("norm_loudness is not supported by ChatterboxMultilingualTTS")

        # Conflict check: cannot have both audio AND embedding
        if getattr(request, "input_audio", None) is not None and getattr(request, "input_embedding", None) is not None:
            raise ValueError("Provide either input_audio or input_embedding, not both.")

        # Deserialize pre-computed Conditionals if provided
        conditionals = None
        if getattr(request, "input_embedding", None) is not None:
            embedding_bytes = base64.b64decode(request.input_embedding)
            embedding_buf = io.BytesIO(embedding_bytes)
            conditionals = Conditionals.load(embedding_buf, map_location="cpu")

        # Default language_id to 'en' if not provided
        language_id = request.language_id if request.language_id is not None else "en"

        # Apply model-specific defaults for ChatterboxMultilingualTTS
        exaggeration = opts.exaggeration if opts is not None and opts.exaggeration is not None else 0.5
        cfg_weight = opts.cfg_weight if opts is not None and opts.cfg_weight is not None else 0.5
        temperature = opts.temperature if opts is not None and opts.temperature is not None else 0.8
        repetition_penalty = (
            opts.repetition_penalty if opts is not None and opts.repetition_penalty is not None else 1.2
        )
        top_p = opts.top_p if opts is not None and opts.top_p is not None else 1.0
        min_p = opts.min_p if opts is not None and opts.min_p is not None else 0.05

        return (
            request.input_text,
            language_id,
            conditionals,
            exaggeration,
            cfg_weight,
            temperature,
            repetition_penalty,
            top_p,
            min_p,
        )

    with ThreadPoolExecutor() as executor:
        inputs = list(executor.map(prepare, requests_to_batch))
    return inputs


def prepare_chatterbox_embedding_inputs(requests_to_batch: list[SpeechEmbeddingRequestInput]):
    """
    Prepare inputs for ChatterboxEmbedding model.

    Returns list of raw audio bytes (base64-decoded, no filesystem I/O).
    """

    def prepare(request):
        # Decode base64 audio to raw bytes
        audio_bytes = base64.b64decode(request.input_audio)
        return audio_bytes

    with ThreadPoolExecutor() as executor:
        inputs = list(executor.map(prepare, requests_to_batch))
    return inputs


def prepare_chatterbox_vc_inputs(requests_to_batch: list[VoiceConversionRequestInput]):
    """
    Prepare inputs for ChatterboxVC model.

    Returns list of tuples:
        (source_audio_bytes, target_conditionals_or_None, target_audio_bytes_or_None)

    Raises ValueError for:
    - Both target_audio AND target_embedding provided
    - Neither target_audio NOR target_embedding provided
    """

    def prepare(request):
        # Validate: must provide exactly one of target_audio or target_embedding
        target_audio = getattr(request, "target_audio", None)
        target_embedding = getattr(request, "target_embedding", None)

        if target_audio is not None and target_embedding is not None:
            raise ValueError("Provide either target_audio or target_embedding, not both.")
        if target_audio is None and target_embedding is None:
            raise ValueError("ChatterboxVC requires either target_audio or target_embedding.")

        # Decode source audio
        source_bytes = base64.b64decode(request.source_audio)

        # Process target
        target_conditionals = None
        target_audio_bytes = None

        if target_embedding is not None:
            # Deserialize pre-computed Conditionals
            embedding_bytes = base64.b64decode(target_embedding)
            embedding_buf = io.BytesIO(embedding_bytes)
            target_conditionals = Conditionals.load(embedding_buf, map_location="cpu")
        else:
            # Decode target audio bytes
            target_audio_bytes = base64.b64decode(target_audio)

        return source_bytes, target_conditionals, target_audio_bytes

    with ThreadPoolExecutor() as executor:
        inputs = list(executor.map(prepare, requests_to_batch))
    return inputs

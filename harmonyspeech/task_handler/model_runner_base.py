import base64
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Union
from loguru import logger

import soundfile as sf
import numpy as np
import torch

from harmonyspeech.common.config import DeviceConfig, ModelConfig
from harmonyspeech.common.inputs import TextToSpeechRequestInput, SpeechEmbeddingRequestInput, VocodeRequestInput, \
    SynthesisRequestInput
from harmonyspeech.common.outputs import SpeechEmbeddingRequestOutput, SpeechSynthesisRequestOutput, VocodeRequestOutput
from harmonyspeech.common.request import EngineRequest, ExecutorResult
from harmonyspeech.modeling.loader import get_model
from harmonyspeech.modeling.models.harmonyspeech.common import preprocess_wav
from harmonyspeech.modeling.models.harmonyspeech.encoder.inputs import get_input_frames
from harmonyspeech.modeling.models.harmonyspeech.synthesizer.inputs import prepare_synthesis_inputs


class ModelRunnerBase:

    def __init__(
        self,
        model_config: ModelConfig,
        device_config: DeviceConfig,
        is_driver_worker: bool = False,
        *args,
        **kwargs,
    ):
        self.model_config = model_config
        self.is_driver_worker = is_driver_worker

        self.device_config = (device_config if device_config is not None else DeviceConfig())
        self.device = self.device_config.device

        self.model = None
        self.model_memory_usage = 0

    def _load_model(self):
        return get_model(
                self.model_config,
                self.device_config,
            )

    @torch.inference_mode()
    def execute_model(
        self,
        requests_to_batch: List[EngineRequest]
    ) -> List[ExecutorResult]:
        """
        Executes a group of batched requests against the model which is loaded
        :param requests_to_batch:
        :return:
        """
        inputs = self.prepare_inputs(requests_to_batch)
        outputs = []

        model_type = getattr(self.model_config, 'model_type', None)
        if model_type == "HarmonySpeechEncoder":
            outputs = self._execute_harmonyspeech_encoder(inputs, requests_to_batch)
        elif model_type == "HarmonySpeechSynthesizer":
            outputs = self._execute_harmonyspeech_synthesizer(inputs, requests_to_batch)
        elif model_type == "HarmonySpeechVocoder":
            outputs = self._execute_harmonyspeech_vocoder(inputs, requests_to_batch)
        else:
            raise NotImplementedError(f"Model {model_type} is not supported")

        return outputs

    def _execute_harmonyspeech_encoder(self, inputs, requests_to_batch):
        # FIXME: This is not properly batched
        def embed_utterance(utterances):
            utterances_tensor = torch.from_numpy(utterances).to(self.device)
            kwargs = {
                "utterances": utterances_tensor
            }
            partial_embeds = self.model(**kwargs).detach().cpu().numpy()
            # Compute the utterance embedding from the partial embeddings
            raw_embed = np.mean(partial_embeds, axis=0)
            embed = raw_embed / np.linalg.norm(raw_embed, 2)
            embed = base64.b64encode(embed).decode('UTF-8')
            return embed

        outputs = []
        for i, x in enumerate(inputs):
            initial_request = requests_to_batch[i]
            request_id = initial_request.request_id
            metrics = initial_request.metrics
            metrics.finished_time = time.time()

            result = ExecutorResult(
                request_id=request_id,
                input_data=initial_request.request_data,
                result_data=SpeechEmbeddingRequestOutput(
                    request_id=request_id,
                    output=embed_utterance(x),
                    finish_reason="stop",
                    metrics=metrics
                )
            )
            outputs.append(result)
        return outputs

    def _execute_harmonyspeech_synthesizer(self, inputs, requests_to_batch):
        # FIXME: This is not properly batched
        def synthesize_text(input_params):
            # Get inputs
            chars, speaker_embeddings, speed_modifier, pitch_function, energy_function = input_params

            # Convert to tensor
            chars = torch.tensor(chars).long().to(self.device)
            speaker_embeddings = torch.tensor(speaker_embeddings).float().to(self.device)

            kwargs = {
                "x": chars,
                "spk_emb": speaker_embeddings,
                "alpha": speed_modifier,
                "pitch_function": pitch_function,
                "energy_function": energy_function
            }
            _, mels, _, _, _  = self.model.generate(**kwargs)
            mels = mels.detach().cpu().numpy()
            # Combine Spectograms from generation
            specs = []
            for m in mels:
                specs.append(m)
            breaks = [subspec.shape[1] for subspec in specs]
            full_spectogram = np.concatenate(specs, axis=1)

            # Build response - TODO: This whole encoding process can be optimized later I think
            breaks_json = json.dumps(breaks)
            full_spectogram_json = full_spectogram.copy(order='C')  # Make C-Contigous to allow encoding
            full_spectogram_json = json.dumps(full_spectogram_json.tolist())

            response = {
                "mel": base64.b64encode(full_spectogram_json.encode('utf-8')).decode('utf-8'),
                "breaks": base64.b64encode(breaks_json.encode('utf-8')).decode('utf-8')
            }
            return json.dumps(response)

        outputs = []
        for i, x in enumerate(inputs):
            initial_request = requests_to_batch[i]
            request_id = initial_request.request_id
            metrics = initial_request.metrics
            metrics.finished_time = time.time()

            result = ExecutorResult(
                request_id=request_id,
                input_data=initial_request.request_data,
                result_data=SpeechSynthesisRequestOutput(
                    request_id=request_id,
                    output=synthesize_text(x),
                    finish_reason="stop",
                    metrics=metrics
                )
            )
            outputs.append(result)
        return outputs

    def _execute_harmonyspeech_vocoder(self, inputs, requests_to_batch):
        # FIXME: This is not properly batched
        def vocode_mel(input_params):
            mel, breaks = input_params
            kwargs = {
                "c": mel.T
            }
            wav = self.model.inference(**kwargs).detach().cpu().numpy()
            wav = wav.squeeze(1)

            # Add breaks if defined
            if breaks is not None:
                # Add breaks - FIXME: Make this parameterized
                b_ends = np.cumsum(np.array(breaks) * 200)  # hop_size 200
                b_starts = np.concatenate(([0], b_ends[:-1]))
                wavs = [wav[start:end] for start, end, in zip(b_starts, b_ends)]
                syn_breaks = [np.zeros(int(0.15 * 16000))] * len(breaks)  # Sample Rate 16000
                wav = np.concatenate([i for w, b in zip(wavs, syn_breaks) for i in (w, b)])

            # Normalize
            wav = wav / np.abs(wav).max() * 0.97

            # Encode as WAV and return base64
            with io.BytesIO() as handle:
                sf.write(handle, wav, samplerate=16000, format='wav')
                wav_string = handle.getvalue()
            encoded_wav = base64.b64encode(wav_string).decode('UTF-8')
            return encoded_wav

        outputs = []
        for i, x in enumerate(inputs):
            initial_request = requests_to_batch[i]
            request_id = initial_request.request_id
            metrics = initial_request.metrics
            metrics.finished_time = time.time()

            result = ExecutorResult(
                request_id=request_id,
                input_data=initial_request.request_data,
                result_data=VocodeRequestOutput(
                    request_id=request_id,
                    output=vocode_mel(x),
                    finish_reason="stop",
                    metrics=metrics
                )
            )
            outputs.append(result)
        return outputs

    def prepare_inputs(self, requests_to_batch: List[EngineRequest]):
        """
        Prepares the request data depending on the model type this runner is executing.
        Throws a NotImplementedError if the model type is unknown
        :param requests_to_batch:
        :return:
        """
        inputs = []
        model_type = getattr(self.model_config, 'model_type', None)
        if model_type == "HarmonySpeechEncoder":
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
            return self._prepare_harmonyspeech_encoder_inputs(inputs)
        elif model_type == "HarmonySpeechSynthesizer":
            for r in requests_to_batch:
                if (
                    isinstance(r.request_data, TextToSpeechRequestInput) or
                    isinstance(r.request_data, SynthesisRequestInput)
                ):
                    inputs.append(r.request_data)
                else:
                    raise ValueError(f"request ID {r.request_id} is not of type TextToSpeechRequestInput")
            return self._prepare_harmonyspeech_synthesizer_inputs(inputs)
        elif model_type == "HarmonySpeechVocoder":
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
            return self._prepare_harmonyspeech_vocoder_inputs(inputs)
        else:
            raise NotImplementedError(f"Cannot provide Inputs for model {model_type}")

    def _prepare_harmonyspeech_encoder_inputs(self, requests_to_batch: List[Union[
        TextToSpeechRequestInput,
        SpeechEmbeddingRequestInput
    ]]):
        # We're expecting audio in waveform format in the requests
        def prepare(request):
            # FIXME: This is not properly batched
            # Make sure Audio is decoded from Base64
            input_audio = base64.b64decode(request.input_audio)
            preprocessed_audio = preprocess_wav(input_audio)
            input_frames = get_input_frames(preprocessed_audio)
            # input_frames_tensors = torch.from_numpy(input_frames).to(self.device)
            return input_frames

        with ThreadPoolExecutor() as executor:
            inputs = list(executor.map(prepare, requests_to_batch))

        return inputs

    def _prepare_harmonyspeech_synthesizer_inputs(self, requests_to_batch: List[Union[
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

    def _prepare_harmonyspeech_vocoder_inputs(self, requests_to_batch: List[Union[
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

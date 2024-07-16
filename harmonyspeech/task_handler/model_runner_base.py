import base64
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Union

import numpy as np
import torch

from harmonyspeech.common.config import DeviceConfig, ModelConfig
from harmonyspeech.common.inputs import TextToSpeechRequestInput, SpeechEmbeddingRequestInput, VocodeRequestInput, \
    SynthesisRequestInput
from harmonyspeech.common.outputs import SpeechEmbeddingRequestOutput
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
        # We're expecting audio in waveform format in the requests
        def prepare(request):
            return prepare_synthesis_inputs(request.input_text, request.target_embedding)

        with ThreadPoolExecutor() as executor:
            inputs = list(executor.map(prepare, requests_to_batch))

        return inputs

    def _prepare_harmonyspeech_vocoder_inputs(self, requests_to_batch: List[Union[
        TextToSpeechRequestInput,
        VocodeRequestInput
    ]]):
        # We're expecting audio in waveform format in the requests
        def prepare(request):
            preprocessed_audio = preprocess_wav(request.input_audio)
            input_frames = get_input_frames(preprocessed_audio)
            # input_frames_tensors = torch.from_numpy(input_frames).to(self.device)
            return input_frames

        with ThreadPoolExecutor() as executor:
            inputs = list(executor.map(prepare, requests_to_batch))

        return inputs

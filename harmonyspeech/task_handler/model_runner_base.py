import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Union

import numpy as np
import torch

from harmonyspeech.common.config import DeviceConfig, ModelConfig
from harmonyspeech.common.inputs import TextToSpeechRequestInput, SpeechEmbeddingRequestInput, VocodeAudioRequestInput
from harmonyspeech.common.outputs import SpeechEmbeddingRequestOutput
from harmonyspeech.common.request import EngineRequest, ExecutorResult
from harmonyspeech.modeling.loader import get_model
from harmonyspeech.modeling.models.harmonyspeech.common import preprocess_wav
from harmonyspeech.modeling.models.harmonyspeech.encoder.inputs import get_input_frames


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
        model_executable = self.model
        inputs = self.prepare_inputs(requests_to_batch)
        outputs = []

        model_name = getattr(self.model_config, 'model_name', None)
        if model_name == "HarmonySpeechEncoder":
            # FIXME: This is not properly batched
            def embed_utterance(utterances):
                utterances_tensor = torch.from_numpy(utterances).to(self.device)
                kwargs = {
                    "utterances": utterances_tensor
                }
                partial_embeds = model_executable(**kwargs).detach().cpu().numpy()
                # Compute the utterance embedding from the partial embeddings
                raw_embed = np.mean(partial_embeds, axis=0)
                embed = raw_embed / np.linalg.norm(raw_embed, 2)
                return embed

            for i, x in enumerate(inputs):
                initial_request = requests_to_batch[i]
                request_id = initial_request.request_id
                metrics = initial_request.metrics
                metrics.finished_time = time.time()

                result = ExecutorResult(
                    request_id=request_id,
                    result_data=SpeechEmbeddingRequestOutput(
                        request_id=request_id,
                        output=embed_utterance(x),
                        finish_reason="stop",
                        metrics=metrics
                    )
                )
                outputs.append(result)

        else:
            raise NotImplementedError(f"Model {model_name} is not supported")

        return outputs

    def prepare_inputs(self, requests_to_batch: List[EngineRequest]):
        """
        Prepares the request data depending on the model type this runner is executing.
        Throws a NotImplementedError if the model type is unknown
        :param requests_to_batch:
        :return:
        """
        inputs = []
        model_name = getattr(self.model_config, 'model_name', None)
        if model_name == "HarmonySpeechEncoder":
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
        elif model_name == "HarmonySpeechSynthesizer":
            for r in requests_to_batch:
                if isinstance(r.request_data, TextToSpeechRequestInput):
                    inputs.append(r.request_data)
                else:
                    raise ValueError(f"request ID {r.request_id} is not of type TextToSpeechRequestInput")
            return self._prepare_harmonyspeech_synthesizer_inputs(inputs)
        elif model_name == "HarmonySpeechVocoder":
            for r in requests_to_batch:
                if (
                    isinstance(r.request_data, TextToSpeechRequestInput) or
                    isinstance(r.request_data, VocodeAudioRequestInput)
                ):
                    inputs.append(r.request_data)
                else:
                    raise ValueError(
                        f"request ID {r.request_id} is not of type TextToSpeechRequestInput or "
                        f"VocodeAudioRequestInput")
            return self._prepare_harmonyspeech_vocoder_inputs(inputs)
        else:
            raise NotImplementedError(f"Cannot provide Inputs for model {model_name}")

    def _prepare_harmonyspeech_encoder_inputs(self, requests_to_batch: List[Union[
        TextToSpeechRequestInput,
        SpeechEmbeddingRequestInput
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

    def _prepare_harmonyspeech_synthesizer_inputs(self, requests_to_batch: List[Union[TextToSpeechRequestInput]]):
        # We're expecting audio in waveform format in the requests
        def prepare(request):
            preprocessed_audio = preprocess_wav(request.input_audio)
            input_frames = get_input_frames(preprocessed_audio)
            # input_frames_tensors = torch.from_numpy(input_frames).to(self.device)
            return input_frames

        with ThreadPoolExecutor() as executor:
            inputs = list(executor.map(prepare, requests_to_batch))

        return inputs

    def _prepare_harmonyspeech_vocoder_inputs(self, requests_to_batch: List[Union[
        TextToSpeechRequestInput,
        VocodeAudioRequestInput
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

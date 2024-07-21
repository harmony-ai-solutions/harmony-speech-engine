import base64
import io
import json
import time
from typing import List

import numpy as np
import soundfile as sf
import torch

from harmonyspeech.common.config import DeviceConfig, ModelConfig
from harmonyspeech.common.outputs import SpeechEmbeddingRequestOutput, SpeechSynthesisRequestOutput, VocodeRequestOutput
from harmonyspeech.common.request import EngineRequest, ExecutorResult
from harmonyspeech.modeling.loader import get_model, get_model_flavour, get_model_config
from harmonyspeech.modeling.models.openvoice.mel_processing import spectrogram_torch
from harmonyspeech.task_handler.inputs import prepare_inputs


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
        inputs = prepare_inputs(self.model_config, requests_to_batch)
        outputs = []

        model_type = getattr(self.model_config, 'model_type', None)
        if model_type == "HarmonySpeechEncoder":
            outputs = self._execute_harmonyspeech_encoder(inputs, requests_to_batch)
        elif model_type == "HarmonySpeechSynthesizer":
            outputs = self._execute_harmonyspeech_synthesizer(inputs, requests_to_batch)
        elif model_type == "HarmonySpeechVocoder":
            outputs = self._execute_harmonyspeech_vocoder(inputs, requests_to_batch)
        elif model_type in ["OpenVoiceV1ToneConverter", "OpenVoiceV2ToneConverter"]:
            outputs = self._execute_openvoice_tone_converter(inputs, requests_to_batch)
        else:
            raise NotImplementedError(f"Model {model_type} is not supported")

        return outputs

    def _build_result(self, initial_request, inference_result):
        request_id = initial_request.request_id
        metrics = initial_request.metrics
        metrics.finished_time = time.time()
        result = ExecutorResult(
            request_id=request_id,
            input_data=initial_request.request_data,
            result_data=SpeechEmbeddingRequestOutput(
                request_id=request_id,
                output=inference_result,
                finish_reason="stop",
                metrics=metrics
            )
        )
        return result

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
            inference_result = embed_utterance(x)
            result = self._build_result(initial_request, inference_result)
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
            inference_result = synthesize_text(x)
            result = self._build_result(initial_request, inference_result)
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
            inference_result = vocode_mel(x)
            result = self._build_result(initial_request, inference_result)
            outputs.append(result)
        return outputs

    def _execute_openvoice_tone_converter(self, inputs, requests_to_batch):
        # Get model flavour if applicable
        flavour = get_model_flavour(self.model_config)
        # Load config
        hf_config = get_model_config(
            self.model_config.model,
            self.model_config.model_type,
            self.model_config.revision,
            flavour
        )


        # FIXME: This is not properly batched
        def run_converter(input_params):
            audio_ref, input_embedding = input_params
            if input_embedding is None:
                # Embed
                y = torch.FloatTensor(audio_ref)
                y = y.to(self.device)
                y = y.unsqueeze(0)
                y = spectrogram_torch(y, hf_config.data.filter_length,
                                      hf_config.data.sampling_rate, hf_config.data.hop_length, hf_config.data.win_length,
                                      center=False).to(self.device)

            else:
                # Convert


            # Encode as WAV and return base64
            with io.BytesIO() as handle:
                sf.write(handle, wav, samplerate=16000, format='wav')
                wav_string = handle.getvalue()
            encoded_wav = base64.b64encode(wav_string).decode('UTF-8')
            return encoded_wav

        outputs = []
        for i, x in enumerate(inputs):
            initial_request = requests_to_batch[i]
            inference_result = run_converter(x)
            result = self._build_result(initial_request, inference_result)
            outputs.append(result)
        return outputs

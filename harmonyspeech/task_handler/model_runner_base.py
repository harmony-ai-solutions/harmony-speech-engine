import base64
import io
import json
import time
from dataclasses import asdict
from typing import List

import numpy as np
import soundfile as sf
import torch

from faster_whisper import BatchedInferencePipeline

from harmonyspeech.common.config import DeviceConfig, ModelConfig
from harmonyspeech.common.inputs import *
from harmonyspeech.common.outputs import *
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
        elif model_type in ["OpenVoiceV1ToneConverterEncoder", "OpenVoiceV2ToneConverterEncoder"]:
            outputs = self._execute_openvoice_tone_converter_encoder(inputs, requests_to_batch)
        elif model_type in ["OpenVoiceV1ToneConverter", "OpenVoiceV2ToneConverter"]:
            outputs = self._execute_openvoice_tone_converter(inputs, requests_to_batch)
        elif model_type == "OpenVoiceV1Synthesizer":
            outputs = self._execute_openvoice_synthesizer(inputs, requests_to_batch)
        elif model_type == "MeloTTSSynthesizer":
            outputs = self._execute_melotts_synthesizer(inputs, requests_to_batch)
        elif model_type == "FasterWhisper":
            outputs = self._execute_faster_whisper(inputs, requests_to_batch)
        else:
            raise NotImplementedError(f"Model {model_type} is not supported")

        return outputs

    def _build_result(self, initial_request, inference_result, result_cls):
        request_id = initial_request.request_id
        metrics = initial_request.metrics
        metrics.finished_time = time.time()
        result = ExecutorResult(
            request_id=request_id,
            input_data=initial_request.request_data,
            result_data=result_cls(
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
            result = self._build_result(initial_request, inference_result, SpeechEmbeddingRequestOutput)
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
            result = self._build_result(initial_request, inference_result, SpeechSynthesisRequestOutput)
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
            result = self._build_result(initial_request, inference_result, VocodeRequestOutput)
            outputs.append(result)
        return outputs

    def _execute_openvoice_tone_converter_encoder(self, inputs, requests_to_batch):
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
        def run_converter_encoder(input_params):
            vad_audio_segments = input_params

            gs = []
            for audio_data in vad_audio_segments:
                y = torch.FloatTensor(audio_data)
                y = y.to(self.device)
                y = y.unsqueeze(0)
                y = spectrogram_torch(y, hf_config.data.filter_length,
                                      hf_config.data.sampling_rate, hf_config.data.hop_length,
                                      hf_config.data.win_length,
                                      center=False).to(self.device)
                with torch.no_grad():
                    g = self.model.ref_enc(y.transpose(1, 2)).unsqueeze(-1)
                    gs.append(g.detach())
            gs = torch.stack(gs).mean(0)

            # Save Embedding
            with io.BytesIO() as handle:
                torch.save(gs.cpu(), handle)
                embedding_string = handle.getvalue()
            encoded_embedding = base64.b64encode(embedding_string).decode('UTF-8')
            return encoded_embedding

        outputs = []
        for i, x in enumerate(inputs):
            initial_request = requests_to_batch[i]
            inference_result = run_converter_encoder(x)
            result = self._build_result(initial_request, inference_result, SpeechEmbeddingRequestOutput)
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
        def run_converter_encoder(input_params):
            audio_ref, input_embedding_ref, source_embedding_ref = input_params

            # Prepare conversion
            target_embedding = torch.load(input_embedding_ref).to(self.device)
            src_embedding = torch.load(source_embedding_ref).to(self.device)
            audio = torch.tensor(audio_ref).float()
            y = torch.FloatTensor(audio).to(self.device)
            y = y.unsqueeze(0)
            spec = spectrogram_torch(y, hf_config.data.filter_length,
                                  hf_config.data.sampling_rate, hf_config.data.hop_length,
                                  hf_config.data.win_length,
                                  center=False).to(self.device)
            spec_lengths = torch.LongTensor([spec.size(-1)]).to(self.device)

            with torch.no_grad():
                output_audio = self.model.voice_conversion(
                    spec, spec_lengths, sid_src=src_embedding, sid_tgt=target_embedding, tau=0.3
                )
                output_audio = output_audio[0][0, 0].data.cpu().float().numpy()

                # Encode as WAV and return base64
                with io.BytesIO() as handle:
                    sf.write(handle, output_audio, samplerate=hf_config.data.sampling_rate, format='wav')
                    wav_string = handle.getvalue()
                encoded_wav = base64.b64encode(wav_string).decode('UTF-8')
                return encoded_wav

        outputs = []
        for i, x in enumerate(inputs):
            initial_request = requests_to_batch[i]
            inference_result = run_converter_encoder(x)
            result = self._build_result(initial_request, inference_result, VoiceConversionRequestOutput)
            outputs.append(result)
        return outputs

    def _execute_faster_whisper(self, inputs, requests_to_batch):
        def run_batched_transcribe(input_params):
            audio_ref = input_params
            batched_model = BatchedInferencePipeline(model=self.model)
            segments, info = batched_model.transcribe(audio_ref, batch_size=16)
            segment_data = []
            text = ""
            for segment in segments:
                text += segment.text
                segment_data.append(asdict(segment))

            response = {
                "text": text,
                "segments": segment_data,
                "info": asdict(info)
            }
            return json.dumps(response)

        outputs = []
        for i, x in enumerate(inputs):
            initial_request = requests_to_batch[i]
            inference_result = run_batched_transcribe(x)
            result = self._build_result(initial_request, inference_result, SpeechTranscriptionRequestOutput)
            outputs.append(result)
        return outputs

    def _execute_openvoice_synthesizer(self, inputs, requests_to_batch):
        # FIXME: This is not properly batched
        # Get model flavour if applicable
        flavour = get_model_flavour(self.model_config)
        # Load config
        hf_config = get_model_config(
            self.model_config.model,
            self.model_config.model_type,
            self.model_config.revision,
            flavour
        )

        def synthesize_text(input_params):
            # Iterate over inputs and add data to list
            audio_segment_list = []
            text_normalized, speaker_id, speed_modifier = input_params
            for text_element in text_normalized:
                # Inference
                with torch.no_grad():
                    x_tst = text_element.unsqueeze(0).to(self.device)
                    x_tst_lengths = torch.LongTensor([text_element.size(0)]).to(self.device)
                    sid = torch.LongTensor([speaker_id]).to(self.device)
                    audio = self.model.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=0.667, noise_scale_w=0.6,
                                             length_scale=1.0 / speed_modifier)[0][0, 0].data.cpu().float().numpy()
                audio_segment_list.append(audio)

            # Concat Output Audio
            audio_segments = []
            for segment_data in audio_segment_list:
                audio_segments += segment_data.reshape(-1).tolist()
                audio_segments += [0] * int((hf_config.data.sampling_rate * 0.05) / speed_modifier)
            audio = np.array(audio_segments).astype(np.float32)

            # Encode as WAV and return base64
            with io.BytesIO() as handle:
                sf.write(handle, audio, samplerate=hf_config.data.sampling_rate, format='wav')
                wav_string = handle.getvalue()
            encoded_wav = base64.b64encode(wav_string).decode('UTF-8')
            return encoded_wav

        outputs = []
        for i, x in enumerate(inputs):
            initial_request = requests_to_batch[i]
            inference_result = synthesize_text(x)
            result = self._build_result(initial_request, inference_result, SpeechSynthesisRequestOutput)
            outputs.append(result)
        return outputs

    def _execute_melotts_synthesizer(self, inputs, requests_to_batch):
        # FIXME: This is not properly batched
        # Get model flavour if applicable
        flavour = get_model_flavour(self.model_config)
        # Load config
        hf_config = get_model_config(
            self.model_config.model,
            self.model_config.model_type,
            self.model_config.revision,
            flavour
        )

        def synthesize_text(input_params):
            # Iterate over inputs and add data to list
            audio_segment_list = []
            inference_items, speaker_id, speed_modifier = input_params
            for items in inference_items:
                # rebuild items from tuple
                bert, ja_bert, phones, tones, lang_ids = items

                # Inference
                with torch.no_grad():
                    x_tst = phones.to(self.device).unsqueeze(0)
                    tones = tones.to(self.device).unsqueeze(0)
                    lang_ids = lang_ids.to(self.device).unsqueeze(0)
                    bert = bert.to(self.device).unsqueeze(0)
                    ja_bert = ja_bert.to(self.device).unsqueeze(0)
                    x_tst_lengths = torch.LongTensor([phones.size(0)]).to(self.device)
                    del phones
                    speakers = torch.LongTensor([speaker_id]).to(self.device)
                    audio = self.model.infer(
                        x_tst,
                        x_tst_lengths,
                        speakers,
                        tones,
                        lang_ids,
                        bert,
                        ja_bert,
                        sdp_ratio=0.2,
                        noise_scale=0.6,
                        noise_scale_w=0.8,
                        length_scale=1.0 / speed_modifier
                    )[0][0, 0].data.cpu().float().numpy()
                    del x_tst, tones, lang_ids, bert, ja_bert, x_tst_lengths, speakers
                audio_segment_list.append(audio)

            # Concat Output Audio
            audio_segments = []
            for segment_data in audio_segment_list:
                audio_segments += segment_data.reshape(-1).tolist()
                audio_segments += [0] * int((hf_config.data.sampling_rate * 0.05) / speed_modifier)
            audio = np.array(audio_segments).astype(np.float32)

            # Encode as WAV and return base64
            with io.BytesIO() as handle:
                sf.write(handle, audio, samplerate=hf_config.data.sampling_rate, format='wav')
                wav_string = handle.getvalue()
            encoded_wav = base64.b64encode(wav_string).decode('UTF-8')
            return encoded_wav

        outputs = []
        for i, x in enumerate(inputs):
            initial_request = requests_to_batch[i]
            inference_result = synthesize_text(x)
            result = self._build_result(initial_request, inference_result, SpeechSynthesisRequestOutput)
            outputs.append(result)
        return outputs

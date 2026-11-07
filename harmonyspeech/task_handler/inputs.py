"""Input preparation: turn request inputs into typed pipeline artifacts.

Phase 2 (artifact bus): every audio-carrying ``prepare_*`` function returns
typed artifacts (:mod:`harmonyspeech.common.artifacts`) instead of numpy arrays,
WAV bytes or wire-encoded strings. The precedence getters in
:mod:`harmonyspeech.common.inputs` guarantee the wire-encoded request field is decoded
**once** and cached on the request; subsequent hops reuse the artifact.

The routing table below is unchanged from Phase 0/1 (``prepare_inputs`` keeps its
``(model_config, requests_to_batch)`` signature and per-request order).

Chatterbox note: the upstream ``prepare_conditionals``/``set_target_voice``
APIs require a file-like object. The handler re-wraps the already-decoded
artifact as WAV bytes in memory (encode-only — no wire encoding, no client-format
decode). This replaces Phase 1's ``ensure_librosa_readable_bytes`` path, which
therefore no longer appears in the pipeline (the helper itself stays in
``common/audio_codec.py`` as a boundary utility).
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from harmonyspeech.common.artifacts import AudioArtifact, MelArtifact, SpeakerArtifact
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
    artifact_store,
    as_request_input,
    decode_b64_json_field,
    decode_embedding_bytes,
    decode_json_field,
    decode_mel_field,
    get_audio_artifact,
    get_speaker_artifact,
    get_vad_artifact,
)
from harmonyspeech.common.inputs import (
    decode_audio_field as _decode_audio_field,
)
from harmonyspeech.common.inputs import (
    decode_conditionals_field as _load_conditionals_payload,
)
from harmonyspeech.common.request import EngineRequest
from harmonyspeech.modeling.loader import get_model_config, get_model_flavour, get_model_speaker
from harmonyspeech.modeling.models.harmonyspeech.synthesizer.inputs import prepare_synthesis_inputs
from harmonyspeech.modeling.models.melo.inputs import normalize_text_inputs as melo_normalize_inputs
from harmonyspeech.modeling.models.openvoice.inputs import normalize_text_inputs as ov1_normalize_inputs

#: HarmonySpeech vocoder frame hop (the vocoder re-inserts silence every 200 frames).
_HARMONYSPEECH_VOCODER_HOP = 200
#: HarmonySpeech vocoder output sample rate.
_HARMONYSPEECH_VOCODER_SR = 16000
#: VoiceFixer operates at 44.1 kHz with a 441-sample hop.
_VOICEFIXER_SR = 44100
_VOICEFIXER_HOP = 441

__all__ = ["prepare_inputs"]

# Re-exported for tests / direct callers that used the Phase 1 private names.
__all__ += ["_decode_audio_field", "_load_conditionals_payload"]


def prepare_inputs(model_config: ModelConfig, requests_to_batch: list[EngineRequest]):
    """
    Prepares the request data depending on the model type this runner is executing.
    Throws a NotImplementedError if the model type is unknown
    :param model_config:
    :param requests_to_batch: EngineRequest objects (carry the artifact store)
    :return:
    """
    inputs = []
    if model_config.model_type == "HarmonySpeechEncoder":
        for r in requests_to_batch:
            if isinstance(r.request_data, TextToSpeechRequestInput) or isinstance(
                r.request_data, SpeechEmbeddingRequestInput
            ):
                inputs.append(r)
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
                inputs.append(r)
            else:
                raise ValueError(
                    f"request ID {r.request_id} is not of type TextToSpeechRequestInput or SynthesisRequestInput"
                )
        return prepare_harmonyspeech_synthesizer_inputs(inputs)
    elif model_config.model_type == "HarmonySpeechVocoder":
        for r in requests_to_batch:
            if isinstance(r.request_data, TextToSpeechRequestInput) or isinstance(r.request_data, VocodeRequestInput):
                inputs.append(r)
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
                inputs.append(r)
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
                inputs.append(r)
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
                inputs.append(r)
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
                inputs.append(r)
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
                inputs.append(r)
            else:
                raise ValueError(
                    f"request ID {r.request_id} is not of type TextToSpeechRequestInput or SynthesisRequestInput"
                )
        return prepare_melotts_synthesizer_inputs(model_config, inputs)
    elif model_config.model_type == "VoiceFixerRestorer":
        for r in requests_to_batch:
            if isinstance(r.request_data, AudioConversionRequestInput):
                inputs.append(r)
            else:
                raise ValueError(f"request ID {r.request_id} is not of type AudioConversionRequestInput")
        return prepare_voicefixer_restorer_inputs(inputs)
    elif model_config.model_type == "VoiceFixerVocoder":
        for r in requests_to_batch:
            if isinstance(r.request_data, AudioConversionRequestInput):
                inputs.append(r)
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
                inputs.append(r)
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
                inputs.append(r)
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
                embed_inputs.append(r)
            elif isinstance(r.request_data, (TextToSpeechRequestInput, SynthesisRequestInput)):
                tts_inputs.append(r)
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
                embed_inputs.append(r)
            elif isinstance(r.request_data, (TextToSpeechRequestInput, SynthesisRequestInput)):
                tts_inputs.append(r)
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
                embed_inputs.append(r)
            elif isinstance(r.request_data, (TextToSpeechRequestInput, SynthesisRequestInput)):
                tts_inputs.append(r)
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
                inputs.append(r)
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
                inputs.append(r)
            else:
                raise ValueError(
                    f"request ID {r.request_id} is not of type SpeechEmbeddingRequestInput or TextToSpeechRequestInput"
                )
        return prepare_chatterbox_embedding_inputs(inputs)
    else:
        raise NotImplementedError(f"Cannot provide Inputs for model {model_config.model_type}")


def prepare_harmonyspeech_encoder_inputs(
    requests_to_batch: list[EngineRequest],
):
    """Return one :class:`AudioArtifact` per request (native rate).

    The preprocessing (``preprocess_wav`` + ``get_input_frames``) moved into the
    handler so the *decoded audio* is the shared artifact: the Whisper VAD step
    produced it once and the encoder no longer re-decodes the same field.
    """

    def prepare(entry):
        request = as_request_input(entry)
        return get_audio_artifact(request, entry, "input_audio")

    with ThreadPoolExecutor() as executor:
        return list(executor.map(prepare, requests_to_batch))


def prepare_voicefixer_restorer_inputs(requests_to_batch: list[AudioConversionRequestInput]):
    """Return one 44.1 kHz :class:`AudioArtifact` per request."""

    def prepare(entry):
        request = as_request_input(entry)
        return get_audio_artifact(request, entry, "source_audio", target_sr=_VOICEFIXER_SR)

    with ThreadPoolExecutor() as executor:
        return list(executor.map(prepare, requests_to_batch))


def prepare_voicefixer_vocoder_inputs(requests_to_batch: list[AudioConversionRequestInput]):
    """Return one :class:`MelArtifact` per request (restorer product or legacy JSON)."""

    def prepare(entry):
        request = as_request_input(entry)
        store = artifact_store(entry)
        existing = store.get("input_mel_spectrogram") if store is not None else None
        if isinstance(existing, MelArtifact):
            return existing

        payload = getattr(request, "input_mel_spectrogram", None)
        if not payload:
            raise ValueError("input_mel_spectrogram: missing mel payload")
        array = decode_mel_field(payload, "input_mel_spectrogram")
        if array.ndim == 4:
            # Legacy restorer output: [batch, channels, frames, bins]; the vocoder
            # only consumes channel 0, so store the canonical [1, bins, frames].
            array = array[0, 0].T[None, ...]
        elif array.ndim == 2:
            array = array[None, ...]
        elif array.ndim != 3:
            raise ValueError(f"input_mel_spectrogram: unexpected mel shape {array.shape}")
        artifact = MelArtifact.from_numpy(array, sample_rate=_VOICEFIXER_SR, hop_length=_VOICEFIXER_HOP)
        if store is not None:
            store["input_mel_spectrogram"] = artifact
        return artifact

    with ThreadPoolExecutor() as executor:
        return list(executor.map(prepare, requests_to_batch))


def prepare_harmonyspeech_synthesizer_inputs(requests_to_batch: list[EngineRequest]):
    # We're receiving a text, a voice embedding and voice modifiers
    def prepare(entry):
        request = as_request_input(entry)
        speaker = get_speaker_artifact(request, entry, "input_embedding", backend="harmonyspeech")
        embedding_array = speaker.embedding.detach().cpu().numpy()
        input_text, input_embedding = prepare_synthesis_inputs(request.input_text, embedding_array)

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
        return list(executor.map(prepare, requests_to_batch))


def prepare_harmonyspeech_vocoder_inputs(requests_to_batch: list[EngineRequest]):
    """Return ``(MelArtifact, breaks)`` per request.

    Artifact path: the synthesizer forwarded its :class:`MelArtifact` (carrying
    ``breaks``) under ``input_audio``. Legacy path: ``input_audio`` is a JSON
    envelope ``{"mel": b64(JSON(list)), "breaks": b64(JSON(list))}``.
    """

    def prepare(entry):
        request = as_request_input(entry)
        store = artifact_store(entry)
        existing = store.get("input_audio") if store is not None else None
        if isinstance(existing, MelArtifact):
            return existing, existing.breaks

        payload = getattr(request, "input_audio", None)
        if not payload:
            raise ValueError("input_audio: missing mel payload")
        envelope = decode_json_field(payload, "input_audio")
        if not isinstance(envelope, dict):
            raise ValueError("input_audio: expected a JSON mel envelope with 'mel' and optional 'breaks'")

        syn_breaks = envelope.get("breaks")
        if syn_breaks is not None:
            syn_breaks = (
                decode_b64_json_field(syn_breaks, "input_audio.breaks")
                if isinstance(syn_breaks, str)
                else syn_breaks
            )
            if not isinstance(syn_breaks, list):
                raise ValueError("input_audio.breaks: expected a list of frame counts")

        syn_mel = envelope.get("mel")
        if not syn_mel:
            raise ValueError("input_audio.mel: missing mel payload")
        mel_array = decode_mel_field(syn_mel, "input_audio.mel")
        if mel_array.ndim == 2:
            mel_array = mel_array[None, ...]
        elif mel_array.ndim != 3:
            raise ValueError(f"input_audio.mel: unexpected mel shape {mel_array.shape}")
        artifact = MelArtifact.from_numpy(
            mel_array,
            sample_rate=_HARMONYSPEECH_VOCODER_SR,
            hop_length=_HARMONYSPEECH_VOCODER_HOP,
            breaks=syn_breaks,
        )
        if store is not None:
            store["input_audio"] = artifact
        return artifact, artifact.breaks

    with ThreadPoolExecutor() as executor:
        return list(executor.map(prepare, requests_to_batch))


def prepare_openvoice_tone_converter_encoder_inputs(
    model_config: ModelConfig, requests_to_batch: list[EngineRequest]
):
    # Get model flavour if applicable
    flavour = get_model_flavour(model_config)
    # Load config
    hf_config = get_model_config(model_config.model, model_config.model_type, model_config.revision, flavour)
    target_sr = int(hf_config.data.sampling_rate)

    # Based on VAD Data we're processing the audio file provided
    def prepare(entry):
        request = as_request_input(entry)
        audio_artifact = get_audio_artifact(request, entry, "input_audio", target_sr=target_sr)
        vad = get_vad_artifact(request, entry, "input_vad_data")
        segments = vad.segments
        # Silero-origin segments carry no transcript (``text`` is empty by
        # design), so the whisper text-length filter would drop every one of
        # them. Keep the duration filter for both backends and only apply the
        # text heuristic to backends that actually produce text (Phase 3-4).
        text_filtered = vad.source != "silero"

        waveform = audio_artifact.numpy
        max_len = waveform.size
        total_ms = int(max_len * 1000 / target_sr)

        # OpenVoice's reference implementation quantises the source to int16
        # before segmenting (se_extractor.split_audio_whisper). Keep that
        # quantisation for embedding parity, but slice the in-memory artifact
        # instead of exporting to a tempfile and decoding it again.
        quantized = np.array(waveform * (1 << 15), dtype=np.int16)

        segment_artifacts: list[AudioArtifact] = []
        start_time = None
        for k, segment in enumerate(segments):
            if k == 0:
                start_time = max(0.0, float(segment["start"]))
            end_time = float(segment["end"])
            text = str(segment.get("text", "")).replace("...", "")

            start_ms = int(start_time * 1000)
            end_ms = min(total_ms, int(end_time * 1000) + 80)
            start_frame = int(start_ms * target_sr / 1000)
            end_frame = int(end_ms * target_sr / 1000)
            segment_samples = quantized[start_frame:end_frame].astype(np.float32) / float(1 << 15)
            duration_s = segment_samples.size / float(target_sr)

            # filter out the segment if shorter than 1.5s and longer than 20s
            text_ok = (2 <= len(text) < 200) if text_filtered else True
            if 1.5 < duration_s < 20.0 and text_ok:
                segment_artifacts.append(AudioArtifact.from_numpy(segment_samples, target_sr))

            if k < len(segments) - 1:
                start_time = max(0.0, float(segments[k + 1]["start"]) - 0.08)

        return segment_artifacts

    with ThreadPoolExecutor() as executor:
        return list(executor.map(prepare, requests_to_batch))


def prepare_openvoice_tone_converter_inputs(
    model_config: ModelConfig,
    requests_to_batch: list[EngineRequest],
):
    # Get model flavour if applicable
    flavour = get_model_flavour(model_config)
    # Load config
    hf_config = get_model_config(model_config.model, model_config.model_type, model_config.revision, flavour)
    target_sr = int(hf_config.data.sampling_rate)

    # We're expecting audio in waveform format in the requests
    def prepare(entry):
        request = as_request_input(entry)
        # Handle both TextToSpeechRequestInput (for voice cloning) and VoiceConversionRequestInput
        # For VoiceConversionRequestInput: source_audio, target_embedding
        # For TextToSpeechRequestInput: input_audio, input_embedding
        if hasattr(request, "source_audio"):
            audio_field = "source_audio"
            embedding_field = "target_embedding"
        else:
            audio_field = "input_audio"
            embedding_field = "input_embedding"

        audio_artifact = get_audio_artifact(request, entry, audio_field, target_sr=target_sr)
        target_speaker = get_speaker_artifact(request, entry, embedding_field, backend="openvoice")

        # For voice conversion, use target embedding directly
        # For voice cloning via TTS, get source embedding from repo
        if hasattr(request, "language_id") and hasattr(request, "voice_id"):
            source_speaker_embedding_file = get_model_speaker(
                model_config.model,
                model_config.model_type,
                model_config.revision,
                request.language_id,
                request.voice_id,
            )
            source_speaker = SpeakerArtifact(
                embedding=decode_embedding_bytes(source_speaker_embedding_file, "source_embedding"),
                backend="openvoice",
            )
        else:
            source_speaker = target_speaker

        return audio_artifact, target_speaker, source_speaker

    with ThreadPoolExecutor() as executor:
        return list(executor.map(prepare, requests_to_batch))


def prepare_faster_whisper_inputs(
    requests_to_batch: list[EngineRequest],
):
    """Return one 16 kHz :class:`AudioArtifact` per request.

    The transcription options live on :class:`SpeechTranscribeRequestInput`
    (``task``, ``language`` hint, ``prompt``) and are resolved once here so both
    execution paths — the sequential ``pipeline.transcribe`` lane and the Phase
    10-2 window batcher — read the same normalized values. ``task`` selects the
    Whisper prompt token (``<|transcribe|>`` / ``<|translate|>``) in both paths;
    an unsupported value becomes this batch's typed prepare error.
    """

    def prepare(entry):
        request = as_request_input(entry)
        task = str(getattr(request, "task", None) or "transcribe").strip().lower()
        if task not in ("transcribe", "translate"):
            raise ValueError(
                f"task: unsupported value '{task}'. Supported tasks: transcribe, translate."
            )
        request.task = task
        language = getattr(request, "language", None)
        request.language = language or None
        prompt = getattr(request, "prompt", None)
        request.prompt = prompt or None
        return get_audio_artifact(request, entry, "input_audio", target_sr=16000)

    with ThreadPoolExecutor() as executor:
        return list(executor.map(prepare, requests_to_batch))


def prepare_openvoice_synthesizer_inputs(model_config: ModelConfig, requests_to_batch: list[EngineRequest]):
    # Get model flavour if applicable
    flavour = get_model_flavour(model_config)
    # Load config
    hf_config = get_model_config(model_config.model, model_config.model_type, model_config.revision, flavour)

    # We're receiving a text, a voice embedding and voice modifiers
    def prepare(entry):
        request = as_request_input(entry)
        input_text = request.input_text
        speaker_id = hf_config.speakers[request.voice_id]

        # Base Input modifiers for OpenVoice V1
        speed_modifier = 1.0
        if request.generation_options:
            if request.generation_options.speed:
                speed_modifier = float(request.generation_options.speed)

        text_normalized = ov1_normalize_inputs(input_text, hf_config, model_config.language)
        return text_normalized, speaker_id, speed_modifier, _openvoice_sampling_params(request)

    with ThreadPoolExecutor() as executor:
        return list(executor.map(prepare, requests_to_batch))


#: OpenVoice V1 defaults, matching the values the handler has always used.
_OPENVOICE_SAMPLING_DEFAULTS = {"noise_scale": 0.667, "noise_scale_w": 0.6, "sdp_ratio": 0.2}
#: MeloTTS defaults, matching the values the handler has always used.
_MELO_SAMPLING_DEFAULTS = {"noise_scale": 0.6, "noise_scale_w": 0.8, "sdp_ratio": 0.2}


def _sampling_params(request, defaults: dict) -> dict:
    """Per-request sampling params: request override, else the family default."""
    options = getattr(request, "generation_options", None)
    resolved = dict(defaults)
    if options is not None:
        for key in defaults:
            value = getattr(options, key, None)
            if value is not None:
                resolved[key] = float(value)
    return resolved


def _openvoice_sampling_params(request) -> dict:
    return _sampling_params(request, _OPENVOICE_SAMPLING_DEFAULTS)


def _melo_sampling_params(request) -> dict:
    return _sampling_params(request, _MELO_SAMPLING_DEFAULTS)



def prepare_melotts_synthesizer_inputs(model_config: ModelConfig, requests_to_batch: list[EngineRequest]):
    # Get model flavour if applicable
    flavour = get_model_flavour(model_config)
    # Load config
    hf_config = get_model_config(model_config.model, model_config.model_type, model_config.revision, flavour)

    # We're receiving a text, a voice embedding and voice modifiers
    def prepare(entry):
        request = as_request_input(entry)
        input_text = request.input_text
        speaker_id = hf_config.data.spk2id[request.voice_id]

        # Base Input modifiers for MeloTTS
        speed_modifier = 1.0
        if request.generation_options:
            if request.generation_options.speed:
                speed_modifier = float(request.generation_options.speed)

        inference_items = melo_normalize_inputs(input_text, hf_config, model_config.language)
        return inference_items, speaker_id, speed_modifier, _melo_sampling_params(request)

    with ThreadPoolExecutor() as executor:
        return list(executor.map(prepare, requests_to_batch))


def prepare_silero_vad_inputs(requests_to_batch: list[EngineRequest]):
    """
    Prepare inputs for Silero VAD model.
    Returns ``(AudioArtifact at 16 kHz, vad_params)`` per request.
    """

    def prepare(entry):
        request = as_request_input(entry)
        audio_artifact = get_audio_artifact(request, entry, "input_audio", target_sr=16000)

        # Extract VAD parameters from request
        vad_params = {
            "threshold": getattr(request, "threshold", 0.5),
            "min_speech_duration_ms": getattr(request, "min_speech_duration_ms", 250),
            "min_silence_duration_ms": getattr(request, "min_silence_duration_ms", 100),
            "speech_pad_ms": getattr(request, "speech_pad_ms", 30),
            "return_seconds": getattr(request, "return_seconds", False),
            "get_timestamps": getattr(request, "get_timestamps", False),
        }

        return audio_artifact, vad_params

    with ThreadPoolExecutor() as executor:
        return list(executor.map(prepare, requests_to_batch))


def prepare_kittentts_synthesizer_inputs(requests_to_batch: list[EngineRequest]):
    """
    Prepare inputs for KittenTTSSynthesizer model.
    Extracts text, voice name, and speed from the request.
    No preprocessing or BERT tokenization needed — KittenTTS handles this internally.
    """

    def prepare(entry):
        request = as_request_input(entry)
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
        return list(executor.map(prepare, requests_to_batch))


def _chatterbox_speaker(entry, field: str = "input_embedding"):
    """Return the speaker conditioning artifact for a Chatterbox request, or ``None``.

    Precedence: pre-computed conditionals/embedding artifact -> reference audio
    artifact -> legacy wire-encoded embedding field. Both artifact forms are produced
    by this module for incoming requests and by the embedding step's forwarding
    (``SpeakerArtifact.conditionals``) for the multi-step path.
    """
    request = as_request_input(entry)
    store = artifact_store(entry)
    existing = store.get(field) if store is not None else None
    if isinstance(existing, SpeakerArtifact):
        return existing

    embedding = getattr(request, field, None)
    if embedding is not None:
        return get_speaker_artifact(request, entry, field, backend="chatterbox")

    audio = getattr(request, "input_audio", None)
    audio_artifact = store.get("input_audio") if store is not None else None
    if audio is not None or isinstance(audio_artifact, AudioArtifact):
        return SpeakerArtifact(
            reference_audio=get_audio_artifact(request, entry, "input_audio"),
            backend="chatterbox",
        )
    return None


def prepare_chatterbox_tts_inputs(requests_to_batch: list[EngineRequest]):
    """
    Prepare inputs for ChatterboxTTS model.

    Returns list of tuples:
        (input_text, speaker_artifact_or_None, exaggeration, cfg_weight, temperature,
         repetition_penalty, top_p, min_p)

    ``speaker_artifact`` carries either ``conditionals`` (pre-computed embedding)
    or ``reference_audio`` (clone from audio) — the handler turns the latter into
    conditionals with ``model.prepare_conditionals``.

    Raises ValueError for:
    - Non-None top_k or norm_loudness (Turbo-only params)
    - Both input_audio AND input_embedding provided (conflict)
    """

    def prepare(entry):
        request = as_request_input(entry)
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

        speaker = _chatterbox_speaker(entry)

        # Apply model-specific defaults for ChatterboxTTS
        exaggeration = opts.exaggeration if opts is not None and opts.exaggeration is not None else 0.5
        cfg_weight = opts.cfg_weight if opts is not None and opts.cfg_weight is not None else 0.5
        temperature = opts.temperature if opts is not None and opts.temperature is not None else 0.8
        repetition_penalty = (
            opts.repetition_penalty if opts is not None and opts.repetition_penalty is not None else 1.2
        )
        top_p = opts.top_p if opts is not None and opts.top_p is not None else 1.0
        min_p = opts.min_p if opts is not None and opts.min_p is not None else 0.05

        return request.input_text, speaker, exaggeration, cfg_weight, temperature, repetition_penalty, top_p, min_p

    with ThreadPoolExecutor() as executor:
        inputs = list(executor.map(prepare, requests_to_batch))
    return inputs


def prepare_chatterbox_turbo_tts_inputs(requests_to_batch: list[EngineRequest]):
    """
    Prepare inputs for ChatterboxTurboTTS model.

    Returns list of tuples:
        (input_text, speaker_artifact_or_None, temperature, repetition_penalty, top_p, top_k, norm_loudness)

    Raises ValueError for:
    - Non-None exaggeration, cfg_weight, or min_p (base-TTS-only params)
    - Both input_audio AND input_embedding provided (conflict)
    """

    def prepare(entry):
        request = as_request_input(entry)
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

        speaker = _chatterbox_speaker(entry)

        # Apply Turbo-specific defaults
        temperature = opts.temperature if opts is not None and opts.temperature is not None else 0.8
        repetition_penalty = (
            opts.repetition_penalty if opts is not None and opts.repetition_penalty is not None else 1.2
        )
        top_p = opts.top_p if opts is not None and opts.top_p is not None else 0.95
        top_k = opts.top_k if opts is not None and opts.top_k is not None else 1000
        norm_loudness = opts.norm_loudness if opts is not None and opts.norm_loudness is not None else True

        return request.input_text, speaker, temperature, repetition_penalty, top_p, top_k, norm_loudness

    with ThreadPoolExecutor() as executor:
        inputs = list(executor.map(prepare, requests_to_batch))
    return inputs


def prepare_chatterbox_multilingual_tts_inputs(
    requests_to_batch: list[EngineRequest],
):
    """
    Prepare inputs for ChatterboxMultilingualTTS model.

    Language validation is handled upstream by the serving engine.
    This function only defaults language_id to 'en' when absent/None.

    Returns list of tuples:
        (input_text, language_id, speaker_artifact_or_None, exaggeration, cfg_weight, temperature,
         repetition_penalty, top_p, min_p)

    Raises ValueError for:
    - Non-None top_k or norm_loudness (Turbo-only params)
    - Both input_audio AND input_embedding provided (conflict)
    """

    def prepare(entry):
        request = as_request_input(entry)
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

        speaker = _chatterbox_speaker(entry)

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
            speaker,
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


def prepare_chatterbox_embedding_inputs(requests_to_batch: list[EngineRequest]):
    """
    Prepare inputs for ChatterboxEmbedding model.

    Returns list of :class:`AudioArtifact` (native rate). The handler re-wraps the
    waveform as in-memory WAV bytes for the upstream ``prepare_conditionals``
    call — the client payload itself is decoded exactly once, at admission.
    """

    def prepare(entry):
        request = as_request_input(entry)
        return get_audio_artifact(request, entry, "input_audio")

    with ThreadPoolExecutor() as executor:
        return list(executor.map(prepare, requests_to_batch))


def prepare_chatterbox_vc_inputs(requests_to_batch: list[EngineRequest]):
    """
    Prepare inputs for ChatterboxVC model.

    Returns list of tuples:
        (source_audio_artifact, target_speaker_artifact_or_None, target_audio_artifact_or_None)

    Raises ValueError for:
    - Both target_audio AND target_embedding provided
    - Neither target_audio NOR target_embedding provided
    """

    def prepare(entry):
        request = as_request_input(entry)
        # Validate: must provide exactly one of target_audio or target_embedding
        target_audio = getattr(request, "target_audio", None)
        target_embedding = getattr(request, "target_embedding", None)

        if target_audio is not None and target_embedding is not None:
            raise ValueError("Provide either target_audio or target_embedding, not both.")
        if target_audio is None and target_embedding is None:
            raise ValueError("ChatterboxVC requires either target_audio or target_embedding.")

        source_artifact = get_audio_artifact(request, entry, "source_audio")

        # Process target
        target_speaker = None
        target_audio_artifact = None

        if target_embedding is not None:
            # Pre-computed Conditionals (safe: weights_only=True)
            target_speaker = get_speaker_artifact(request, entry, "target_embedding", backend="chatterbox")
        else:
            target_audio_artifact = get_audio_artifact(request, entry, "target_audio")

        return source_artifact, target_speaker, target_audio_artifact

    with ThreadPoolExecutor() as executor:
        inputs = list(executor.map(prepare, requests_to_batch))
    return inputs

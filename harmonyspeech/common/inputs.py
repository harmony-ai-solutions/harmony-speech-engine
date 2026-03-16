from dataclasses import dataclass

from harmonyspeech.common.metrics import RequestMetrics
from harmonyspeech.endpoints.openai.protocol import (
    AudioConversionRequest,
    DetectVoiceActivityRequest,
    EmbedSpeakerRequest,
    SpeechTranscribeRequest,
    SynthesizeAudioRequest,
    TextToSpeechRequest,
    VocodeAudioRequest,
    VoiceConversionRequest,
)


@dataclass
class TextToSpeechGenerationOptions:
    seed: int | None
    style: int | None
    speed: float | None
    pitch: float | None
    energy: float | None
    # Chatterbox-specific fields (None = use model default in prepare function)
    exaggeration: float | None = None
    cfg_weight: float | None = None
    temperature: float | None = None
    repetition_penalty: float | None = None
    top_p: float | None = None
    min_p: float | None = None
    top_k: int | None = None
    norm_loudness: bool | None = None


class TextToSpeechAudioOutputOptions:
    format: str | None = "wav"
    sample_rate: int | None = None
    stream: bool | None = False


class RequestInput:
    """
    The base class for model input data
    """

    def __init__(self, request_id: str, requested_model: str, model: str, metrics: RequestMetrics | None = None):
        self.request_id = request_id
        self.requested_model = requested_model
        self.model = model
        self.metrics = metrics


class AudioConversionRequestInput(RequestInput):
    """
    The input data for an Audio Conversion Request
    """

    def __init__(
        self,
        request_id: str,
        requested_model: str,
        model: str,
        source_audio: str,
        input_mel_spectrogram: str | None = None,
        output_options: TextToSpeechAudioOutputOptions | None = None,
        metrics: RequestMetrics | None = None,
    ):
        super().__init__(request_id=request_id, requested_model=requested_model, model=model, metrics=metrics)
        self.source_audio = source_audio
        self.input_mel_spectrogram = input_mel_spectrogram
        self.output_options = output_options

    @classmethod
    def from_openai(cls, request_id: str, request: "AudioConversionRequest"):
        return cls(
            request_id=request_id,
            requested_model=getattr(request, "model", ""),
            model=getattr(request, "model", ""),
            source_audio=getattr(request, "source_audio", ""),
            input_mel_spectrogram=getattr(request, "input_mel_spectrogram", None),
            output_options=getattr(request, "output_options", None),
        )


class VoiceConversionRequestInput(RequestInput):
    """
    The input data for a Voice Conversion Request
    """

    def __init__(
        self,
        request_id: str,
        requested_model: str,
        model: str,
        source_audio: str,
        target_audio: str | None,
        target_embedding: str | None,
        generation_options: TextToSpeechGenerationOptions | None,
        output_options: TextToSpeechAudioOutputOptions | None,
        metrics: RequestMetrics | None = None,
    ):
        super().__init__(request_id=request_id, requested_model=requested_model, model=model, metrics=metrics)
        self.source_audio = source_audio
        self.target_audio = target_audio
        self.target_embedding = target_embedding
        self.generation_options = generation_options
        self.output_options = output_options

    @classmethod
    def from_openai(cls, request_id: str, request: "VoiceConversionRequest"):
        return cls(
            request_id=request_id,
            requested_model=getattr(request, "model", ""),
            model=getattr(request, "model", ""),
            source_audio=getattr(request, "source_audio", ""),
            target_audio=getattr(request, "target_audio", None),
            target_embedding=getattr(request, "target_embedding", None),
            generation_options=getattr(request, "generation_options", None),
            output_options=getattr(request, "output_options", None),
        )


class TextToSpeechRequestInput(RequestInput):
    """
    The input data for a Text-to-Speech Request
    """

    def __init__(
        self,
        request_id: str,
        requested_model: str,
        model: str,
        input_text: str,
        mode: str,
        language_id: str | None = None,
        voice_id: str | None = None,
        input_audio: str | None = None,
        input_vad_mode: str | None = None,
        input_vad_data: str | None = None,
        input_embedding: str | None = None,
        generation_options: TextToSpeechGenerationOptions | None = None,
        output_options: TextToSpeechAudioOutputOptions | None = None,
        post_generation_filters: list[AudioConversionRequestInput] | None = None,
        metrics: RequestMetrics | None = None,
    ):
        super().__init__(request_id=request_id, requested_model=requested_model, model=model, metrics=metrics)
        self.input_text = input_text
        self.mode = mode
        self.language_id = language_id
        self.voice_id = voice_id
        self.input_audio = input_audio
        self.input_vad_mode = input_vad_mode
        self.input_vad_data = input_vad_data
        self.input_embedding = input_embedding
        self.generation_options = generation_options
        self.output_options = output_options
        self.post_generation_filters = post_generation_filters

    @classmethod
    def from_openai(cls, request_id: str, request: "TextToSpeechRequest"):
        return cls(
            request_id=request_id,
            requested_model=getattr(request, "model", ""),
            model=getattr(request, "model", ""),
            input_text=getattr(request, "input", ""),
            mode=getattr(request, "mode", ""),
            language_id=getattr(request, "language", None),
            voice_id=getattr(request, "voice", None),
            input_audio=getattr(request, "input_audio", None),
            input_vad_mode=getattr(request, "input_vad_mode", None),
            input_vad_data=getattr(request, "input_vad_data", None),
            input_embedding=getattr(request, "input_embedding", None),
            generation_options=getattr(request, "generation_options", None),
            output_options=getattr(request, "output_options", None),
            post_generation_filters=getattr(request, "post_generation_filters", None),
        )


class SpeechEmbeddingRequestInput(RequestInput):
    """
    The input data for a Speech Embedding Request
    """

    def __init__(
        self,
        request_id: str,
        requested_model: str,
        model: str,
        input_audio: str | None = None,
        input_vad_mode: str | None = None,
        input_vad_data: str | None = None,
        metrics: RequestMetrics | None = None,
    ):
        super().__init__(request_id=request_id, requested_model=requested_model, model=model, metrics=metrics)
        self.input_audio = input_audio
        self.input_vad_mode = input_vad_mode
        self.input_vad_data = input_vad_data

    @classmethod
    def from_openai(cls, request_id: str, request: "EmbedSpeakerRequest"):
        return cls(
            request_id=request_id,
            requested_model=getattr(request, "model", ""),
            model=getattr(request, "model", ""),
            input_audio=getattr(request, "input_audio", None),
            input_vad_mode=getattr(request, "input_vad_mode", None),
            input_vad_data=getattr(request, "input_vad_data", None),
        )


class SynthesisRequestInput(RequestInput):
    """
    The input data for a Speech Synthesis Request
    """

    def __init__(
        self,
        request_id: str,
        requested_model: str,
        model: str,
        input_text: str = "",
        language_id: str | None = None,
        voice_id: str | None = None,
        input_embedding: str = "",
        generation_options: TextToSpeechGenerationOptions | None = None,
        metrics: RequestMetrics | None = None,
    ):
        super().__init__(request_id=request_id, requested_model=requested_model, model=model, metrics=metrics)
        self.input_text = input_text
        self.language_id = language_id
        self.voice_id = voice_id
        self.input_embedding = input_embedding
        self.generation_options = generation_options

    @classmethod
    def from_openai(cls, request_id: str, request: "SynthesizeAudioRequest"):
        return cls(
            request_id=request_id,
            requested_model=getattr(request, "model", ""),
            model=getattr(request, "model", ""),
            input_text=getattr(request, "input", ""),
            language_id=getattr(request, "language", None),
            voice_id=getattr(request, "voice", None),
            input_embedding=getattr(request, "input_embedding", ""),
            generation_options=getattr(request, "generation_options", None),
        )


class VocodeRequestInput(RequestInput):
    """
    The input data for a Vocoding Request
    """

    def __init__(
        self,
        request_id: str,
        requested_model: str,
        model: str,
        input_audio: str | None = None,
        metrics: RequestMetrics | None = None,
    ):
        super().__init__(request_id=request_id, requested_model=requested_model, model=model, metrics=metrics)
        self.input_audio = input_audio

    @classmethod
    def from_openai(cls, request_id: str, request: "VocodeAudioRequest"):
        return cls(
            request_id=request_id,
            requested_model=getattr(request, "model", ""),
            model=getattr(request, "model", ""),
            input_audio=getattr(request, "input_audio", None),
        )


class SpeechTranscribeRequestInput(RequestInput):
    """
    The input data for a Speech Transcribe Request
    """

    def __init__(
        self,
        request_id: str,
        requested_model: str,
        model: str,
        input_audio: str | None = None,
        get_language: bool | None = False,
        get_timestamps: bool | None = False,
        metrics: RequestMetrics | None = None,
    ):
        super().__init__(request_id=request_id, requested_model=requested_model, model=model, metrics=metrics)
        self.input_audio = input_audio
        self.get_language = get_language
        self.get_timestamps = get_timestamps

    @classmethod
    def from_openai(cls, request_id: str, request: "SpeechTranscribeRequest"):
        return cls(
            request_id=request_id,
            requested_model=getattr(request, "model", ""),
            model=getattr(request, "model", ""),
            input_audio=getattr(request, "input_audio", None),
            get_language=getattr(request, "get_language", False),
            get_timestamps=getattr(request, "get_timestamps", False),
        )


class DetectVoiceActivityRequestInput(RequestInput):
    """
    The input data for a VAD Request
    """

    def __init__(
        self,
        request_id: str,
        requested_model: str,
        model: str,
        input_audio: str | None = None,
        get_timestamps: bool | None = False,
        threshold: float | None = 0.5,
        min_speech_duration_ms: int | None = 250,
        min_silence_duration_ms: int | None = 100,
        speech_pad_ms: int | None = 30,
        return_seconds: bool | None = False,
        metrics: RequestMetrics | None = None,
    ):
        super().__init__(request_id=request_id, requested_model=requested_model, model=model, metrics=metrics)
        self.input_audio = input_audio
        self.get_timestamps = get_timestamps
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        self.return_seconds = return_seconds

    @classmethod
    def from_openai(cls, request_id: str, request: "DetectVoiceActivityRequest"):
        return cls(
            request_id=request_id,
            requested_model=getattr(request, "model", ""),
            model=getattr(request, "model", ""),
            input_audio=getattr(request, "input_audio", None),
            get_timestamps=getattr(request, "get_timestamps", False),
            threshold=getattr(request, "threshold", 0.5),
            min_speech_duration_ms=getattr(request, "min_speech_duration_ms", 250),
            min_silence_duration_ms=getattr(request, "min_silence_duration_ms", 100),
            speech_pad_ms=getattr(request, "speech_pad_ms", 30),
            return_seconds=getattr(request, "return_seconds", False),
        )

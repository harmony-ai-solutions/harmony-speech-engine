from dataclasses import dataclass

from harmonyspeech.common.metrics import RequestMetrics
from harmonyspeech.endpoints.openai.protocol import *


@dataclass
class TextToSpeechGenerationOptions:
    seed: Optional[int]
    style: Optional[int]
    speed: Optional[float]
    pitch: Optional[float]
    energy: Optional[float]


class TextToSpeechAudioOutputOptions:
    format: Optional[str] = "wav"
    sample_rate: Optional[int] = None
    stream: Optional[bool] = False


class RequestInput:
    """
    The base class for model input data
    """
    def __init__(
        self,
        request_id: str,
        requested_model: str,
        model: str,
        metrics: Optional[RequestMetrics] = None,
    ):
        self.request_id = request_id
        self.requested_model = requested_model
        self.model = model
        self.metrics: metrics


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
        target_audio: Optional[str],
        target_embedding: Optional[str],
        generation_options: Optional[TextToSpeechGenerationOptions],
        output_options: Optional[TextToSpeechAudioOutputOptions],
        metrics: Optional[RequestMetrics] = None,
    ):
        super().__init__(
            request_id=request_id,
            requested_model=requested_model,
            model=model,
            metrics=metrics,
        )
        self.source_audio = source_audio
        self.target_audio = target_audio
        self.target_embedding = target_embedding
        self.generation_options = generation_options
        self.output_options = output_options

    @classmethod
    def from_openai(cls, request_id: str, request: "VoiceConversionRequest"):
        return cls(
            request_id=request_id,
            requested_model=getattr(request, 'model', ''),
            model=getattr(request, 'model', ''),
            source_audio=getattr(request, 'source_audio', None),
            target_audio=getattr(request, 'target_audio', None),
            target_embedding=getattr(request, 'target_embedding', None),
            generation_options=getattr(request, 'generation_options', None),
            output_options=getattr(request, 'output_options', None),
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
        language_id: Optional[str] = None,
        voice_id: Optional[str] = None,
        input_audio: Optional[str] = None,
        input_vad_mode: Optional[str] = None,
        input_vad_data: Optional[str] = None,
        input_embedding: Optional[str] = None,
        generation_options: Optional[TextToSpeechGenerationOptions] = None,
        output_options: Optional[TextToSpeechAudioOutputOptions] = None,
        post_generation_filters: Optional[List[Union[VoiceConversionRequestInput]]] = None,
        metrics: Optional[RequestMetrics] = None,
    ):
        super().__init__(
            request_id=request_id,
            requested_model=requested_model,
            model=model,
            metrics=metrics,
        )
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
            requested_model=getattr(request, 'model', ''),
            model=getattr(request, 'model', ''),
            input_text=getattr(request, 'input', ''),
            mode=getattr(request, 'mode', ''),
            language_id=getattr(request, 'language', None),
            voice_id=getattr(request, 'voice', None),
            input_audio=getattr(request, 'input_audio', None),
            input_vad_mode=getattr(request, 'input_vad_mode', None),
            input_vad_data=getattr(request, 'input_vad_data', None),
            input_embedding=getattr(request, 'input_embedding', None),
            generation_options=getattr(request, 'generation_options', None),
            output_options=getattr(request, 'output_options', None),
            post_generation_filters=getattr(request, 'post_generation_filters', None)
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
        input_audio: Optional[str] = None,
        input_vad_mode: Optional[str] = None,
        input_vad_data: Optional[str] = None,
        metrics: Optional[RequestMetrics] = None,
    ):
        super().__init__(
            request_id=request_id,
            requested_model=requested_model,
            model=model,
            metrics=metrics,
        )
        self.input_audio = input_audio
        self.input_vad_mode = input_vad_mode
        self.input_vad_data = input_vad_data

    @classmethod
    def from_openai(cls, request_id: str, request: "EmbedSpeakerRequest"):
        return cls(
            request_id=request_id,
            requested_model=getattr(request, 'model', ''),
            model=getattr(request, 'model', ''),
            input_audio=getattr(request, 'input_audio', None),
            input_vad_mode=getattr(request, 'input_vad_mode', None),
            input_vad_data=getattr(request, 'input_vad_data', None),
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
        language_id: Optional[str] = None,
        voice_id: Optional[str] = None,
        input_embedding: str = None,
        generation_options: TextToSpeechGenerationOptions = None,
        metrics: Optional[RequestMetrics] = None,
    ):
        super().__init__(
            request_id=request_id,
            requested_model=requested_model,
            model=model,
            metrics=metrics,
        )
        self.input_text = input_text
        self.language_id = language_id
        self.voice_id = voice_id
        self.input_embedding = input_embedding
        self.generation_options = generation_options

    @classmethod
    def from_openai(cls, request_id: str, request: "SynthesizeAudioRequest"):
        return cls(
            request_id=request_id,
            requested_model=getattr(request, 'model', ''),
            model=getattr(request, 'model', ''),
            input_text=getattr(request, 'input', ''),
            language_id=getattr(request, 'language', None),
            voice_id=getattr(request, 'voice', None),
            input_embedding=getattr(request, 'input_embedding', None),
            generation_options=getattr(request, 'generation_options', None),
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
        input_audio: Optional[str] = None,
        metrics: Optional[RequestMetrics] = None,
    ):
        super().__init__(
            request_id=request_id,
            requested_model=requested_model,
            model=model,
            metrics=metrics,
        )
        self.input_audio = input_audio

    @classmethod
    def from_openai(cls, request_id: str, request: "VocodeAudioRequest"):
        return cls(
            request_id=request_id,
            requested_model=getattr(request, 'model', ''),
            model=getattr(request, 'model', ''),
            input_audio=getattr(request, 'input_audio', None),
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
        input_audio: Optional[str] = None,
        get_language: Optional[bool] = False,
        get_timestamps: Optional[bool] = False,
        metrics: Optional[RequestMetrics] = None,
    ):
        super().__init__(
            request_id=request_id,
            requested_model=requested_model,
            model=model,
            metrics=metrics,
        )
        self.input_audio = input_audio
        self.get_language = get_language
        self.get_timestamps = get_timestamps

    @classmethod
    def from_openai(cls, request_id: str, request: "SpeechTranscribeRequest"):
        return cls(
            request_id=request_id,
            requested_model=getattr(request, 'model', ''),
            model=getattr(request, 'model', ''),
            input_audio=getattr(request, 'input_audio', None),
            get_language=getattr(request, 'get_language', False),
            get_timestamps=getattr(request, 'get_timestamps', False),
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
        input_audio: Optional[str] = None,
        get_timestamps: Optional[bool] = False,
        metrics: Optional[RequestMetrics] = None,
    ):
        super().__init__(
            request_id=request_id,
            requested_model=requested_model,
            model=model,
            metrics=metrics,
        )
        self.input_audio = input_audio
        self.get_timestamps = get_timestamps

    @classmethod
    def from_openai(cls, request_id: str, request: "DetectVoiceActivityRequest"):
        return cls(
            request_id=request_id,
            requested_model=getattr(request, 'model', ''),
            model=getattr(request, 'model', ''),
            input_audio=getattr(request, 'input_audio', None),
            get_timestamps=getattr(request, 'get_timestamps', False),
        )

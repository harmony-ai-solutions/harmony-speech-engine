from dataclasses import dataclass
from typing import Optional, List

from harmonyspeech.common.metrics import RequestMetrics
from harmonyspeech.endpoints.openai.protocol import VoiceConversionRequest, TextToSpeechRequest, EmbedSpeakerRequest, \
    VocodeAudioRequest


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
        model: str,
        metrics: Optional[RequestMetrics] = None,
    ):
        self.request_id = request_id
        self.model = model
        self.metrics: metrics


class VoiceConversionRequestInput(RequestInput):
    """
    The input data for a Voice Conversion Request
    """

    def __init__(
        self,
        request_id: str,
        model: str,
        source_audio: bytes,
        target_audio: Optional[bytes],
        target_embedding: Optional[bytes],
        generation_options: Optional[TextToSpeechGenerationOptions],
        output_options: Optional[TextToSpeechAudioOutputOptions],
        metrics: Optional[RequestMetrics] = None,
    ):
        super().__init__(
            request_id=request_id,
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
        model: str,
        input_text: str,
        voice_id: Optional[str] = None,
        input_audio: Optional[bytes] = None,
        input_embedding: Optional[bytes] = None,
        generation_options: Optional[TextToSpeechGenerationOptions] = None,
        output_options: Optional[TextToSpeechAudioOutputOptions] = None,
        post_generation_filters: Optional[List[VoiceConversionRequestInput]] = None,
        metrics: Optional[RequestMetrics] = None,
    ):
        super().__init__(
            request_id=request_id,
            model=model,
            metrics=metrics,
        )
        self.input_text = input_text
        self.voice_id = voice_id
        self.input_audio = input_audio
        self.input_embedding = input_embedding
        self.generation_options = generation_options
        self.output_options = output_options
        self.post_generation_filters = post_generation_filters

    @classmethod
    def from_openai(cls, request_id: str, request: "TextToSpeechRequest"):
        return cls(
            request_id=request_id,
            model=getattr(request, 'model', ''),
            input_text=getattr(request, 'input_text', ''),
            voice_id=getattr(request, 'voice_id', None),
            input_audio=getattr(request, 'input_audio', None),
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
        model: str,
        input_audio: Optional[bytes] = None,
        metrics: Optional[RequestMetrics] = None,
    ):
        super().__init__(
            request_id=request_id,
            model=model,
            metrics=metrics,
        )
        self.input_audio = input_audio

    @classmethod
    def from_openai(cls, request_id: str, request: "EmbedSpeakerRequest"):
        return cls(
            request_id=request_id,
            model=getattr(request, 'model', ''),
            input_audio=getattr(request, 'input_audio', None),
        )


class VocodeAudioRequestInput(RequestInput):
    """
    The input data for a Speech Embedding Request
    """

    def __init__(
        self,
        request_id: str,
        model: str,
        input_audio: Optional[bytes] = None,
        metrics: Optional[RequestMetrics] = None,
    ):
        super().__init__(
            request_id=request_id,
            model=model,
            metrics=metrics,
        )
        self.input_audio = input_audio

    @classmethod
    def from_openai(cls, request_id: str, request: "VocodeAudioRequest"):
        return cls(
            request_id=request_id,
            model=getattr(request, 'model', ''),
            input_audio=getattr(request, 'input_audio', None),
        )

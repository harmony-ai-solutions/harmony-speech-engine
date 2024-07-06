from dataclasses import dataclass
from typing import Optional, List

from harmonyspeech.common.request import RequestMetrics
from harmonyspeech.endpoints.openai.protocol import VoiceConversionRequest, TextToSpeechRequest


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
        metrics: Optional[RequestMetrics] = None,
    ):
        self.request_id = request_id
        self.metrics: metrics


class VoiceConversionRequestInput(RequestInput):
    """
    The input data for a Voice Conversion Request
    """

    def __init__(
        self,
        request_id: str,
        source_audio: bytes,
        target_audio: Optional[bytes],
        target_embedding: Optional[bytes],
        generation_options: Optional[TextToSpeechGenerationOptions],
        output_options: Optional[TextToSpeechAudioOutputOptions],
        metrics: Optional[RequestMetrics] = None,
    ):
        super().__init__(
            request_id=request_id,
            metrics=metrics,
        )
        self.source_audio = source_audio
        self.target_audio = target_audio
        self.target_embedding = target_embedding
        self.generation_options = generation_options
        self.output_options = output_options

    @staticmethod
    def from_openai(request_id: str, request: "VoiceConversionRequest"):
        return VoiceConversionRequestInput(
            request_id=request_id,
            source_audio=request.get('source_audio', None),
            target_audio=request.get('target_audio', None),
            target_embedding=request.get('target_embedding', None),
            generation_options=request.get('generation_options', None),
            output_options=request.get('output_options', None),
        )


class TextToSpeechRequestInput(RequestInput):
    """
    The input data for a Text-to-Speech Request
    """

    def __init__(
        self,
        request_id: str,
        input_text: str,
        voice_id: Optional[str] = None,
        target_audio: Optional[bytes] = None,
        target_embedding: Optional[bytes] = None,
        generation_options: Optional[TextToSpeechGenerationOptions] = None,
        output_options: Optional[TextToSpeechAudioOutputOptions] = None,
        post_generation_filters: Optional[List[VoiceConversionRequestInput]] = None,
        metrics: Optional[RequestMetrics] = None,
    ):
        super().__init__(
            request_id=request_id,
            metrics=metrics,
        )
        self.input_text = input_text
        self.voice_id = voice_id
        self.target_audio = target_audio
        self.target_embedding = target_embedding
        self.generation_options = generation_options
        self.output_options = output_options
        self.post_generation_filters = post_generation_filters

    @staticmethod
    def from_openai(request_id: str, request: "TextToSpeechRequest"):
        return TextToSpeechRequestInput(
            request_id=request_id,
            input_text=request.get('input_text', ''),
            voice_id=request.get('voice_id', None),
            target_audio=request.get('target_audio', None),
            target_embedding=request.get('target_embedding', None),
            generation_options=request.get('generation_options', None),
            output_options=request.get('output_options', None),
            post_generation_filters=request.get('post_generation_filters', None)
        )

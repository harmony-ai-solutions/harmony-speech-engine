# Protocol Definition for OpenAI-Like Endpoint
# Inspiration taken from PygmalionAI / Aphrodite Engine

import time
from typing import Dict, Literal, Optional, Union, List

from pydantic import (AliasChoices, BaseModel, Field, conint, model_validator,
                      root_validator)

from harmonyspeech.common.utils import random_uuid


class ErrorResponse(BaseModel):
    object: str = "error"
    message: str
    type: str
    param: Optional[str] = None
    code: int


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "harmony-ai-solutions"
    root: Optional[str] = None
    parent: Optional[str] = None
    # permission: List[ModelPermission] = Field(default_factory=list)


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = Field(default_factory=list)


class ResponseFormat(BaseModel):
    # type must be "json_object" or "text"
    type: str = Literal["text", "json_object"]


class GenerationOptions(BaseModel):
    seed: Optional[int] = None
    style: Optional[int] = None
    speed: Optional[float] = None
    pitch: Optional[float] = None
    energy: Optional[float] = None


class AudioOutputOptions(BaseModel):
    format: Optional[str] = "wav"
    sample_rate: Optional[int] = None
    stream: Optional[bool] = False


class BaseRequest(BaseModel):
    model: str = Field(default="", description="the name of the model")


class VoiceConversionRequest(BaseRequest):
    """
    VoiceConversionRequest
    Used to convert the tone or nature of a voice in a specific way.
    Depending on model selection, the caller may need to provide additional params.
    """
    source_audio: str = Field(default=None, description="Binary audio data to be processed, encoded in base64")
    target_audio: Optional[str] = Field(
        default=None,
        description="Binary audio data of the reference speaker for converting the source, encoded in base64"
    )
    target_embedding: Optional[str] = Field(
        default=None,
        description="Binary embedding of the reference speaker for converting the source, encoded in base64. "
                    "Faster than providing a target audio file"
    )
    generation_options: Optional[GenerationOptions] = Field(
        default=None,
        description="Options for generating a speech request, see documentation for possible values"
    )
    output_options: Optional[AudioOutputOptions] = Field(
        default=None,
        description="Options for returning the generated audio, see documentation for possible values"
    )


class TextToSpeechRequest(BaseRequest):
    """
    TextToSpeechRequest
    Used to trigger a speech generation for the provided text input using the specified model.
    Depending on model selection, the caller may need to provide additional params.
    Based on OpenAI TTS API; extended for Harmony Speech Engine features.
    """
    input: str = Field(default="", description="the text to synthesize")
    language: str = Field(
        default=None,
        description="language to synthesize. Only if the model supports language IDs. "
                    "Please refer to the model's documentation for which values are availiable.")
    voice: Optional[str] = Field(
        default=None,
        description="ID of the voice to synthesize. Only if the model supports voice IDs. "
                    "Please refer to the model's documentation for which values are availiable."
    )
    input_audio: Optional[str] = Field(
        default=None,
        description="Binary audio data of the reference speaker for synthesizing the text, encoded in base64"
    )
    input_embedding: Optional[str] = Field(
        default=None,
        description="Binary embedding of the reference speaker for synthesizing the text, encoded in base64. "
                    "Faster than providing a target audio file"
    )
    generation_options: Optional[GenerationOptions] = Field(
        default=None,
        description="Options for generating a speech request, see documentation for possible values"
    )
    output_options: Optional[AudioOutputOptions] = Field(
        default=None,
        description="Options for returning the generated audio, see documentation for possible values"
    )
    post_generation_filters: Optional[List[VoiceConversionRequest]] = Field(
        default_factory=list,
        description="List of Post-Generation filters to apply to the generated audio."
    )


class BaseResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"{random_uuid()}")
    model: str = Field(default="", description="the name of the model")
    created: int = Field(default_factory=lambda: int(time.time()))


class AudioDataResponse(BaseResponse):
    """
    AudioDataResponse
    Result Audio Data
    """
    data: str = Field(default=None, description="Audio Bytes in requested format of the initial request")


class TextToSpeechResponse(AudioDataResponse):
    """
    TextToSpeechResponse
    Extends AudioDataResponse with a request specific ID
    """
    id: str = Field(default_factory=lambda: f"tts-{random_uuid()}")


class VoiceConversionResponse(AudioDataResponse):
    """
    VoiceConversionResponse
    Extends AudioDataResponse with a request specific ID
    """
    id: str = Field(default_factory=lambda: f"vc-{random_uuid()}")


class SpeechTranscribeRequest(BaseRequest):
    """
    SpeechTranscribeRequest
    Used to apply ASR on a provided audio file
    Depending on model selection, the caller may need to provide additional params.
    Based on OpenAI STT API; extended for Harmony Speech Engine features.
    """
    file: str = Field(default=None, description="Binary audio data to be processed, encoded in base64")
    get_language: Optional[bool] = Field(
        default=False,
        description="whether to return the source language tag. Check model description if supported."
    )
    get_timestamps: Optional[bool] = Field(
        default=False,
        description="whether to return the word timestamps. Check model description if supported."
    )


class SpeechToTextResponse(BaseResponse):
    """
    SpeechToTextResponse
    Result text determined from audio and optional language tag
    """
    id: str = Field(default_factory=lambda: f"stt-{random_uuid()}")
    text: str = Field(default="", description="text retrieved from the input audio")
    language: Optional[str] = Field(default=None, description="tag of the detected language")
    timestamps: Optional[List[str]] = Field(default_factory=list, description="list of the detected timestamps")


class EmbedSpeakerRequest(BaseRequest):
    """
    EmbedSpeakerRequest
    Used to create a Speaker Embedding form a provided audio, which can later be re-used for Text-to-Speech or
    Voice Conversion functionality.
    Please refer to the documentation whether an embedding is compatible between different models.
    """
    input_audio: str = Field(default=None, description="Binary audio data to be processed, encoded in base64")


class EmbedSpeakerResponse(BaseResponse):
    """
    EmbedSpeakerResult
    Result Speaker Embedding
    """
    id: str = Field(default_factory=lambda: f"embed-{random_uuid()}")
    data: str = Field(
        default=None,
        description="Speaker embedding data for the audio provided in the initial request, encoded in base64"
    )


class SynthesizeAudioRequest(BaseRequest):
    """
    SynthesizeAudioRequest
    Used to run a synthesizer model for a provided input audio
    """
    input: str = Field(default="", description="the text to synthesize")
    language: str = Field(
        default=None,
        description="language to synthesize. Only if the model supports language IDs. "
                    "Please refer to the model's documentation for which values are availiable.")
    voice: Optional[str] = Field(
        default=None,
        description="ID of the voice to synthesize. Only if the model supports voice IDs. "
                    "Please refer to the model's documentation for which values are availiable."
    )
    input_embedding: Optional[str] = Field(
        default=None,
        description="Binary embedding of the reference speaker for synthesizing the text, encoded in base64. "
                    "Faster than providing a target audio file"
    )
    generation_options: Optional[GenerationOptions] = Field(
        default=None,
        description="Options for generating a speech request, see documentation for possible values"
    )


class SynthesizeAudioResponse(AudioDataResponse):
    """
    EmbedSpeakerResult
    Result Audio file after vocoding step
    """
    id: str = Field(default_factory=lambda: f"vocode-{random_uuid()}")


class VocodeAudioRequest(BaseRequest):
    """
    VocodeAudioRequest
    Used to run a vocoder model over a provided input audio
    """
    input_audio: str = Field(default=None, description="Binary audio data to be processed, encoded in base64")


class VocodeAudioResponse(AudioDataResponse):
    """
    EmbedSpeakerResult
    Result Audio file after vocoding step
    """
    id: str = Field(default_factory=lambda: f"vocode-{random_uuid()}")

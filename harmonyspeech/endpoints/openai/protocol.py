# Protocol Definition for OpenAI-Like Endpoint
# Inspiration taken from PygmalionAI / Aphrodite Engine

import time
from typing import Dict, Literal, Optional, Union

from pydantic import (AliasChoices, BaseModel, Field, conint, model_validator,
                      root_validator)


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


class ResponseFormat(BaseModel):
    # type must be "json_object" or "text"
    type: str = Literal["text", "json_object"]


class TextToSpeechGenerationOptions(BaseModel):
    seed: Optional[int] = None
    speed: Optional[float] = None
    pitch: Optional[float] = None
    energy: Optional[float] = None


class TextToSpeechOutputOptions(BaseModel):
    format: Optional[str] = "wav"
    sample_rate: Optional[int] = None
    stream: Optional[bool] = False


class TextToSpeechRequest(BaseModel):
    """
    TextToSpeechRequest
    Used to trigger a speech generation for the provided text input using the specified model.
    Depending on model selection, the caller may need to provide additional params.

    Args:
        model: the name of the model
    """
    model: str
    input: str
    voice: Optional[str] = None
    speaker_embedding: Optional[str] = None
    generation_options: TextToSpeechGenerationOptions
    output_options: TextToSpeechOutputOptions



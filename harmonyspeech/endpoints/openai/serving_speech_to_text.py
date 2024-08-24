import base64
import copy
import json
import time
from typing import List, Union, AsyncGenerator, AsyncIterator

from fastapi import Request
from loguru import logger

from harmonyspeech.common.config import ModelConfig
from harmonyspeech.common.inputs import TextToSpeechRequestInput, SpeechEmbeddingRequestInput, \
    SpeechTranscribeRequestInput
from harmonyspeech.common.outputs import TextToSpeechRequestOutput, RequestOutput, SpeechEmbeddingRequestOutput, \
    SpeechTranscriptionRequestOutput
from harmonyspeech.common.utils import random_uuid
from harmonyspeech.endpoints.openai.protocol import TextToSpeechResponse, ErrorResponse, TextToSpeechRequest, \
    EmbedSpeakerRequest, EmbedSpeakerResponse, SpeechTranscribeRequest, SpeechToTextResponse
from harmonyspeech.endpoints.openai.serving_engine import OpenAIServing
from harmonyspeech.engine.async_harmonyspeech import AsyncHarmonySpeech


# Add new model classes which allow handling STT Requests here
# If multiple models need to be initialized to process request, add multiple to the list
_STT_MODEL_TYPES = [
    "FasterWhisper"
]
_STT_MODEL_GROUPS = {
}


class OpenAIServingSpeechToText(OpenAIServing):

    def __init__(
        self,
        engine: AsyncHarmonySpeech,
        available_models: List[str],
    ):
        super().__init__(engine=engine, available_models=available_models)

    @staticmethod
    def models_from_config(configured_models: List[ModelConfig]):
        # create a copy of the model groups and remove each model_type of a group from it's list if an instance exits
        # only groups where all required model types to provide the API function exist will be enabled.
        check_dict = copy.deepcopy(_STT_MODEL_GROUPS)
        for m in configured_models:
            for group_name, remaining_models in check_dict.items():
                if len(remaining_models) == 0:
                    continue
                if m.model_type in remaining_models:
                    check_dict[group_name].remove(m.model_type)

        full_model_groups = [group_name for group_name, remaining_models in check_dict.items() if not remaining_models]
        individual_models = [m.name for m in configured_models if m.model_type in _STT_MODEL_TYPES]
        return individual_models + full_model_groups

    async def create_transcription(
        self, request: SpeechTranscribeRequest, raw_request: Request
    ) -> Union[ErrorResponse, AsyncGenerator[str, None], SpeechToTextResponse]:
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        # TODO: Basic checks for Embedding Generate request

        request_id = f"stt-{random_uuid()}"

        result_generator = self.engine.generate(
            request_id=request_id,
            request_data=SpeechTranscribeRequestInput.from_openai(
                request_id=request_id,
                request=request
            ),
        )

        try:
            return await self.speech_transcription_full_generator(
                request, raw_request, result_generator, request_id)
        except ValueError as e:
            # TODO: Use an aphrodite-specific Validation Error
            return self.create_error_response(str(e))

    async def speech_transcription_full_generator(
        self, request: SpeechTranscribeRequest, raw_request: Request,
        result_generator: AsyncIterator[RequestOutput],
        request_id: str) -> Union[ErrorResponse, SpeechToTextResponse]:

        model_name = request.model
        created_time = int(time.time())
        final_res: RequestOutput = None

        async for res in result_generator:
            if await raw_request.is_disconnected():
                # Abort the request if the client disconnects.
                await self.engine.abort(request_id)
                return self.create_error_response("Client disconnected")
            final_res = res

        # Ensure we're receiving a proper STT Output here
        assert final_res is not None
        assert isinstance(final_res, SpeechTranscriptionRequestOutput)

        # load result data and determine what will be returned
        result_data = json.loads(final_res.output)

        response = SpeechToTextResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            text=result_data["text"] if "text" in result_data else "",
            language=result_data["info"]["language"] if "info" in result_data and request.get_language else None,
            timestamps=result_data["segments"] if "segments" in result_data and request.get_timestamps else None,
        )

        return response



import copy
import time
from typing import List, Union, AsyncGenerator, AsyncIterator

from fastapi import Request
from loguru import logger

from harmonyspeech.common.config import ModelConfig
from harmonyspeech.common.inputs import TextToSpeechRequestInput
from harmonyspeech.common.outputs import TextToSpeechRequestOutput, RequestOutput
from harmonyspeech.common.utils import random_uuid
from harmonyspeech.endpoints.openai.protocol import TextToSpeechResponse, ErrorResponse, TextToSpeechRequest
from harmonyspeech.endpoints.openai.serving_engine import OpenAIServing
from harmonyspeech.engine.async_harmonyspeech import AsyncHarmonySpeech


# Add new model classes which allow handling TTS Requests here
# If multiple models need to be initialized to process request, add multiple to the list
_TTS_MODEL_Groups = {
    "harmonyspeech": ["HarmonySpeechSynthesizer", "HarmonySpeechVocoder"]
}


class OpenAIServingTextToSpeech(OpenAIServing):

    def __init__(
        self,
        engine: AsyncHarmonySpeech,
        available_models: List[str],
    ):
        super().__init__(engine=engine, available_models=available_models)

    @staticmethod
    def models_from_config(configured_models: List[ModelConfig]):
        check_dict = copy.deepcopy(_TTS_MODEL_Groups)
        for m in configured_models:
            for group_name, remaining_models in check_dict.items():
                if len(remaining_models) == 0:
                    continue
                if m.model_type in remaining_models:
                    check_dict[group_name].remove(m.model_type)

        return [group_name for group_name, remaining_models in check_dict.items() if not remaining_models]

    async def create_text_to_speech(
        self, request: TextToSpeechRequest, raw_request: Request
    ) -> Union[ErrorResponse, AsyncGenerator[str, None], TextToSpeechResponse]:
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        # TODO: Basic checks for TTS Generate request

        request_id = f"tts-{random_uuid()}"

        result_generator = self.engine.generate(
            request_id=request_id,
            request_data=TextToSpeechRequestInput.from_openai(
                request_id=request_id,
                request=request
            ),
        )

        if request.output_options.stream:
            # FIXME: Add Stream Output
            error = "Stream output is not yet supported"
            logger.error(error)
            return self.create_error_response(error)
        else:
            try:
                return await self.text_to_speech_full_generator(
                    request, raw_request, result_generator, request_id)
            except ValueError as e:
                # TODO: Use an aphrodite-specific Validation Error
                return self.create_error_response(str(e))

    async def text_to_speech_full_generator(
        self, request: TextToSpeechRequest, raw_request: Request,
        result_generator: AsyncIterator[RequestOutput],
        request_id: str) -> Union[ErrorResponse, TextToSpeechResponse]:

        model_name = request.model
        created_time = int(time.time())
        final_res: RequestOutput = None

        async for res in result_generator:
            if await raw_request.is_disconnected():
                # Abort the request if the client disconnects.
                await self.engine.abort(request_id)
                return self.create_error_response("Client disconnected")
            final_res = res

        # Ensure we're receiving a proper TTS Output here
        assert final_res is not None
        assert isinstance(final_res, TextToSpeechRequestOutput)

        response = TextToSpeechResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            data=final_res.output.data
        )

        return response



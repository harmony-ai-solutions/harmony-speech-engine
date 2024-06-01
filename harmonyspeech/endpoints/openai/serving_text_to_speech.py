import time
from typing import List, Union, AsyncGenerator, AsyncIterator

from fastapi import Request
from loguru import logger

from harmonyspeech.common.outputs import TextToSpeechRequestOutput
from harmonyspeech.common.utils import random_uuid
from harmonyspeech.endpoints.openai.protocol import TextToSpeechResponse, ErrorResponse, TextToSpeechRequest
from harmonyspeech.endpoints.openai.serving_engine import OpenAIServing
from harmonyspeech.engine.async_harmonyspeech import AsyncHarmonySpeech


class OpenAIServingTextToSpeech(OpenAIServing):

    def __init__(
        self,
        engine: AsyncHarmonySpeech,
        available_models: List[str],
    ):
        super().__init__(engine=engine, available_models=available_models)

    async def create_text_to_speech(
        self, request: TextToSpeechRequest, raw_request: Request
    ) -> Union[ErrorResponse, AsyncGenerator[str, None], TextToSpeechResponse]:
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        # TODO: Basic checks for TTS Generate request

        request_id = f"tts-{random_uuid()}"

        result_generator = self.engine.generate_text_to_speech(
            request_id=request_id,
            request_data=request,
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
        result_generator: AsyncIterator[TextToSpeechRequestOutput],
        request_id: str) -> Union[ErrorResponse, TextToSpeechResponse]:

        model_name = request.model
        created_time = int(time.time())
        final_res: TextToSpeechRequestOutput = None

        async for res in result_generator:
            if await raw_request.is_disconnected():
                # Abort the request if the client disconnects.
                await self.engine.abort(request_id)
                return self.create_error_response("Client disconnected")
            final_res = res
        assert final_res is not None

        response = TextToSpeechResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            data=final_res.output
        )

        return response



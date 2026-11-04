import time
from collections.abc import AsyncGenerator, AsyncIterator

from fastapi import Request

from harmonyspeech.common.config import ModelConfig
from harmonyspeech.common.inputs import VoiceConversionRequestInput
from harmonyspeech.common.outputs import RequestOutput, VoiceConversionRequestOutput
from harmonyspeech.common.utils import random_uuid
from harmonyspeech.endpoints.openai.output_encoding import validate_output_options
from harmonyspeech.endpoints.openai.protocol import (
    ErrorResponse,
    ModelCard,
    VoiceConversionRequest,
    VoiceConversionResponse,
)
from harmonyspeech.endpoints.openai.serving_engine import OpenAIServing
from harmonyspeech.endpoints.openai.wire import from_wire_audio, to_wire_audio
from harmonyspeech.engine.async_harmonyspeech import AsyncHarmonySpeech

# Add new model classes which allow handling Voice Conversion Requests here
# If multiple models need to be initialized to process request, add multiple to the list


class OpenAIServingVoiceConversion(OpenAIServing):
    def __init__(self, engine: AsyncHarmonySpeech, available_models: list[ModelCard]):
        super().__init__(engine=engine, available_models=available_models)

    @staticmethod
    def models_from_config(configured_models: list[ModelConfig]):
        # Advertised cards come from the pipeline registry (Phase 3-3).
        return OpenAIServing.model_cards_from_specs(configured_models, "vc")

    async def convert_voice(
        self, request: VoiceConversionRequest, raw_request: Request
    ) -> ErrorResponse | AsyncGenerator[str, None] | VoiceConversionResponse:
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        error_check_options = validate_output_options(request.output_options)
        if error_check_options is not None:
            return self.create_error_response(error_check_options)

        # TODO: Basic checks for Embedding Generate request

        request_id = f"vc-{random_uuid()}"

        request_data = VoiceConversionRequestInput.from_openai(request_id=request_id, request=request)
        artifacts = from_wire_audio(request_data)
        result_generator = self.engine.generate(
            request_id=request_id,
            request_data=request_data,
            artifacts=artifacts,
        )

        try:
            return await self.voice_conversion_full_generator(request, raw_request, result_generator, request_id)
        except ValueError as e:
            # TODO: Use an aphrodite-specific Validation Error
            return self.create_error_response(str(e))

    async def voice_conversion_full_generator(
        self,
        request: VoiceConversionRequest,
        raw_request: Request,
        result_generator: AsyncIterator[RequestOutput],
        request_id: str,
    ) -> ErrorResponse | VoiceConversionResponse:

        model_name = request.model
        created_time = int(time.time())
        final_res: RequestOutput | None = None

        async for res in result_generator:
            if await raw_request.is_disconnected():
                # Abort the request if the client disconnects.
                await self.engine.abort(request_id)
                return self.create_error_response("Client disconnected")
            final_res = res

        # Ensure we're receiving a proper TTS Output here
        assert final_res is not None
        if final_res.finish_reason == "error":
            return self.create_error_response(final_res.error or "Internal inference error")
        assert isinstance(final_res, VoiceConversionRequestOutput)

        encoded = to_wire_audio(final_res, request.output_options)
        response = VoiceConversionResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            data=encoded.data,
            sample_rate=encoded.sample_rate,
        )

        return response

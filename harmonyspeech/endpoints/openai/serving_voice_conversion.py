import copy
from typing import AsyncGenerator, AsyncIterator

from fastapi import Request

from harmonyspeech.common.config import ModelConfig
from harmonyspeech.common.inputs import SpeechEmbeddingRequestInput, VoiceConversionRequestInput
from harmonyspeech.common.outputs import RequestOutput, SpeechEmbeddingRequestOutput, VoiceConversionRequestOutput
from harmonyspeech.endpoints.openai.protocol import *
from harmonyspeech.endpoints.openai.serving_engine import OpenAIServing
from harmonyspeech.engine.async_harmonyspeech import AsyncHarmonySpeech

# Add new model classes which allow handling Embedding Requests here
# If multiple models need to be initialized to process request, add multiple to the list
_VOICE_CONVERSION_MODEL_TYPES = [
    "OpenVoiceV1ToneConverter",
    "OpenVoiceV2ToneConverter",
]
_VOICE_CONVERSION_MODEL_GROUPS = {
    "openvoice_v1": ["OpenVoiceV1ToneConverter"],
    "openvoice_v2": ["OpenVoiceV2ToneConverter"]
}


class OpenAIServingVoiceConversion(OpenAIServing):

    def __init__(
        self,
        engine: AsyncHarmonySpeech,
        available_models: List[str],
    ):
        super().__init__(engine=engine, available_models=available_models)

    @staticmethod
    def models_from_config(configured_models: List[ModelConfig]):
        check_dict = copy.deepcopy(_VOICE_CONVERSION_MODEL_GROUPS)
        for m in configured_models:
            for group_name, remaining_models in check_dict.items():
                if len(remaining_models) == 0:
                    continue
                if m.model_type in remaining_models:
                    check_dict[group_name].remove(m.model_type)

        full_model_groups = [group_name for group_name, remaining_models in check_dict.items() if not remaining_models]
        individual_models = [m.name for m in configured_models if m.model_type in _VOICE_CONVERSION_MODEL_TYPES]
        return individual_models + full_model_groups

    async def convert_voice(
        self, request: VoiceConversionRequest, raw_request: Request
    ) -> Union[ErrorResponse, AsyncGenerator[str, None], TextToSpeechResponse]:
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        # TODO: Basic checks for Embedding Generate request

        request_id = f"vc-{random_uuid()}"

        result_generator = self.engine.generate(
            request_id=request_id,
            request_data=VoiceConversionRequestInput.from_openai(
                request_id=request_id,
                request=request
            ),
        )

        try:
            return await self.voice_conversion_full_generator(
                request, raw_request, result_generator, request_id)
        except ValueError as e:
            # TODO: Use an aphrodite-specific Validation Error
            return self.create_error_response(str(e))

    async def voice_conversion_full_generator(
        self, request: VoiceConversionRequest, raw_request: Request,
        result_generator: AsyncIterator[RequestOutput],
        request_id: str) -> Union[ErrorResponse, VoiceConversionResponse]:

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
        assert isinstance(final_res, VoiceConversionRequestOutput)

        response = VoiceConversionResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            data=final_res.output
        )

        return response



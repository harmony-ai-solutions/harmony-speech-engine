import copy
import time
from typing import List, Union, AsyncGenerator, AsyncIterator

from fastapi import Request

from harmonyspeech.common.config import ModelConfig
from harmonyspeech.common.inputs import SpeechEmbeddingRequestInput
from harmonyspeech.common.outputs import RequestOutput, SpeechEmbeddingRequestOutput
from harmonyspeech.common.utils import random_uuid
from harmonyspeech.endpoints.openai.protocol import *
from harmonyspeech.endpoints.openai.serving_engine import OpenAIServing
from harmonyspeech.engine.async_harmonyspeech import AsyncHarmonySpeech

# Add new model classes which allow handling Embedding Requests here
# If multiple models need to be initialized to process request, add multiple to the list
_EMBEDDING_MODEL_TYPES = [
    "HarmonySpeechEncoder"
]
_EMBEDDING_MODEL_GROUPS = {
    "harmonyspeech": ["HarmonySpeechEncoder"],
    "openvoice_v1": ["FasterWhisper", "OpenVoiceV1ToneConverterEncoder"],
    "openvoice_v2": ["FasterWhisper", "OpenVoiceV2ToneConverterEncoder"]
}


class OpenAIServingVoiceEmbedding(OpenAIServing):

    def __init__(
        self,
        engine: AsyncHarmonySpeech,
        available_models: List[ModelCard],
    ):
        super().__init__(engine=engine, available_models=available_models)

    @staticmethod
    def models_from_config(configured_models: List[ModelConfig]):
        return OpenAIServing.model_cards_from_config_groups(
            configured_models,
            _EMBEDDING_MODEL_TYPES,
            _EMBEDDING_MODEL_GROUPS
        )

    async def create_voice_embedding(
        self, request: EmbedSpeakerRequest, raw_request: Request
    ) -> Union[ErrorResponse, AsyncGenerator[str, None], TextToSpeechResponse]:
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        # TODO: Basic checks for Embedding Generate request

        request_id = f"emb-{random_uuid()}"

        result_generator = self.engine.generate(
            request_id=request_id,
            request_data=SpeechEmbeddingRequestInput.from_openai(
                request_id=request_id,
                request=request
            ),
        )

        try:
            return await self.voice_embedding_full_generator(
                request, raw_request, result_generator, request_id)
        except ValueError as e:
            # TODO: Use an aphrodite-specific Validation Error
            return self.create_error_response(str(e))

    async def voice_embedding_full_generator(
        self, request: EmbedSpeakerRequest, raw_request: Request,
        result_generator: AsyncIterator[RequestOutput],
        request_id: str) -> Union[ErrorResponse, EmbedSpeakerResponse]:

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
        assert isinstance(final_res, SpeechEmbeddingRequestOutput)

        response = EmbedSpeakerResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            data=final_res.output
        )

        return response

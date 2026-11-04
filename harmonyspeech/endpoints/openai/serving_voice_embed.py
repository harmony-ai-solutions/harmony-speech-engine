import time
from collections.abc import AsyncGenerator, AsyncIterator

from fastapi import Request

from harmonyspeech.common.config import ModelConfig
from harmonyspeech.common.inputs import SpeechEmbeddingRequestInput
from harmonyspeech.common.outputs import RequestOutput, SpeechEmbeddingRequestOutput
from harmonyspeech.common.utils import random_uuid
from harmonyspeech.endpoints.openai.protocol import EmbedSpeakerRequest, EmbedSpeakerResponse, ErrorResponse, ModelCard
from harmonyspeech.endpoints.openai.serving_engine import OpenAIServing
from harmonyspeech.endpoints.openai.wire import from_wire_audio, to_wire_embedding
from harmonyspeech.engine.async_harmonyspeech import AsyncHarmonySpeech


class OpenAIServingVoiceEmbedding(OpenAIServing):
    def __init__(self, engine: AsyncHarmonySpeech, available_models: list[ModelCard]):
        super().__init__(engine=engine, available_models=available_models)

    @staticmethod
    def models_from_config(configured_models: list[ModelConfig]):
        # Advertised cards come from the pipeline registry (Phase 3-3).
        return OpenAIServing.model_cards_from_specs(configured_models, "embed")

    async def create_voice_embedding(
        self, request: EmbedSpeakerRequest, raw_request: Request
    ) -> ErrorResponse | AsyncGenerator[str, None] | EmbedSpeakerResponse:
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        # TODO: Basic checks for Embedding Generate request

        request_id = f"emb-{random_uuid()}"

        request_data = SpeechEmbeddingRequestInput.from_openai(request_id=request_id, request=request)
        artifacts = from_wire_audio(request_data)
        result_generator = self.engine.generate(
            request_id=request_id,
            request_data=request_data,
            artifacts=artifacts,
        )

        try:
            return await self.voice_embedding_full_generator(request, raw_request, result_generator, request_id)
        except ValueError as e:
            # TODO: Use an aphrodite-specific Validation Error
            return self.create_error_response(str(e))

    async def voice_embedding_full_generator(
        self,
        request: EmbedSpeakerRequest,
        raw_request: Request,
        result_generator: AsyncIterator[RequestOutput],
        request_id: str,
    ) -> ErrorResponse | EmbedSpeakerResponse:

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
        assert isinstance(final_res, SpeechEmbeddingRequestOutput)

        # Serialize the speaker artifact at the boundary in the pre-Phase-2
        # format (raw float32 / torch.save tensor / 0-1-safe conditionals).
        response = EmbedSpeakerResponse(
            id=request_id, created=created_time, model=model_name, data=to_wire_embedding(final_res)
        )

        return response

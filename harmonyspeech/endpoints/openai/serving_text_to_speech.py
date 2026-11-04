import time
from collections.abc import AsyncGenerator, AsyncIterator

from fastapi import Request
from loguru import logger

from harmonyspeech.common.config import ModelConfig
from harmonyspeech.common.inputs import TextToSpeechRequestInput
from harmonyspeech.common.outputs import RequestOutput, SpeechSynthesisRequestOutput, TextToSpeechRequestOutput
from harmonyspeech.common.utils import random_uuid
from harmonyspeech.endpoints.openai.output_encoding import validate_output_options
from harmonyspeech.endpoints.openai.protocol import ErrorResponse, ModelCard, TextToSpeechRequest, TextToSpeechResponse
from harmonyspeech.endpoints.openai.serving_engine import OpenAIServing
from harmonyspeech.endpoints.openai.wire import from_wire_audio, to_wire_audio
from harmonyspeech.engine.async_harmonyspeech import AsyncHarmonySpeech


class OpenAIServingTextToSpeech(OpenAIServing):
    def __init__(self, engine: AsyncHarmonySpeech, available_models: list[ModelCard]):
        super().__init__(engine=engine, available_models=available_models)

    @staticmethod
    def models_from_config(configured_models: list[ModelConfig]):
        # Advertised cards come from the pipeline registry (Phase 3-3); this
        # endpoint serves the "tts" capability.
        return OpenAIServing.model_cards_from_specs(configured_models, "tts")

    async def create_text_to_speech(
        self, request: TextToSpeechRequest, raw_request: Request
    ) -> ErrorResponse | AsyncGenerator[str, None] | TextToSpeechResponse:

        error_check_model = await self._check_model(request)
        if error_check_model is not None:
            return error_check_model

        # Validate output_options at admission: never start synthesis for a
        # format/sample-rate combination this runtime cannot produce.
        error_check_options = validate_output_options(request.output_options)
        if error_check_options is not None:
            return self.create_error_response(error_check_options)

        if request.output_options and request.output_options.stream:
            # TODO: Add Stream Output
            error = "Stream output is not yet supported"
            logger.error(error)
            return self.create_error_response(error)

        # TODO: Basic checks for TTS Generate request

        request_id = f"tts-{random_uuid()}"

        request_data = TextToSpeechRequestInput.from_openai(request_id=request_id, request=request)
        # Boundary decode-once: the reference audio becomes an artifact now, so
        # no pipeline step re-reads the base64 field.
        artifacts = from_wire_audio(request_data)
        result_generator = self.engine.generate(
            request_id=request_id,
            request_data=request_data,
            artifacts=artifacts,
        )

        try:
            return await self.text_to_speech_full_generator(request, raw_request, result_generator, request_id)
        except ValueError as e:
            # TODO: Use an aphrodite-specific Validation Error
            return self.create_error_response(str(e))

    async def text_to_speech_full_generator(
        self,
        request: TextToSpeechRequest,
        raw_request: Request,
        result_generator: AsyncIterator[RequestOutput],
        request_id: str,
    ) -> ErrorResponse | TextToSpeechResponse:

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

        # Handle both TextToSpeechRequestOutput (KittenTTS) and SpeechSynthesisRequestOutput (MeloTTS)
        if isinstance(final_res, TextToSpeechRequestOutput) or isinstance(final_res, SpeechSynthesisRequestOutput):
            pass
        else:
            raise ValueError(
                f"Expected TextToSpeechRequestOutput or SpeechSynthesisRequestOutput, got {type(final_res).__name__}"
            )

        # Encode from the typed artifact when present (no base64 WAV decode);
        # fall back to the legacy payload for pre-artifact producers.
        encoded = to_wire_audio(final_res, request.output_options)
        response = TextToSpeechResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            data=encoded.data,
            sample_rate=encoded.sample_rate,
        )

        return response

import json
import time
from collections.abc import AsyncIterator

from fastapi import Request
from fastapi.responses import PlainTextResponse

from harmonyspeech.common.config import ModelConfig
from harmonyspeech.common.inputs import SpeechTranscribeRequestInput
from harmonyspeech.common.outputs import RequestOutput, SpeechTranscriptionRequestOutput
from harmonyspeech.common.utils import random_uuid
from harmonyspeech.endpoints.openai.protocol import (
    ErrorResponse,
    ModelCard,
    SpeechToTextResponse,
    SpeechTranscribeRequest,
    SpeechTranslationRequest,
)
from harmonyspeech.endpoints.openai.serving_engine import OpenAIServing
from harmonyspeech.endpoints.openai.stt_formats import (
    normalize_response_format,
    segments_to_srt,
    segments_to_vtt,
    validate_response_format,
)
from harmonyspeech.endpoints.openai.wire import from_wire_audio
from harmonyspeech.engine.async_harmonyspeech import AsyncHarmonySpeech

# Add new model classes which allow handling STT Requests here
# If multiple models need to be initialized to process request, add multiple to the list


class OpenAIServingSpeechToText(OpenAIServing):
    def __init__(self, engine: AsyncHarmonySpeech, available_models: list[ModelCard]):
        super().__init__(engine=engine, available_models=available_models)

    @staticmethod
    def models_from_config(configured_models: list[ModelConfig]) -> list[ModelCard]:
        # Advertised cards come from the pipeline registry (Phase 3-3). The same
        # Whisper executors serve transcription and translation, so both
        # capabilities drive the card set (Phase 10-4).
        return OpenAIServing.model_cards_from_specs(configured_models, ("stt", "translation"))

    async def create_transcription(
        self, request: SpeechTranscribeRequest, raw_request: Request
    ) -> ErrorResponse | PlainTextResponse | SpeechToTextResponse:
        return await self._create_stt(request, raw_request, task="transcribe")

    async def create_translation(
        self, request: SpeechTranslationRequest, raw_request: Request
    ) -> ErrorResponse | PlainTextResponse | SpeechToTextResponse:
        return await self._create_stt(request, raw_request, task="translate")

    async def _create_stt(
        self, request: SpeechTranscribeRequest, raw_request: Request, *, task: str
    ) -> ErrorResponse | PlainTextResponse | SpeechToTextResponse:
        """Shared transcription/translation entry point.

        ``task`` is ``"transcribe"`` or ``"translate"`` and is threaded onto the
        request input so the executor (both the sequential pipeline and the
        Phase 10-2 window batcher) selects the matching Whisper prompt token.
        """
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        # Validate response_format at admission (Phase 1-2 pattern): an unknown
        # format is a 400 before any inference is queued.
        error_check_format = validate_response_format(getattr(request, "response_format", "json"))
        if error_check_format is not None:
            return self.create_error_response(error_check_format)

        # TODO: Basic checks for Embedding Generate request

        request_id = f"stt-{random_uuid()}"

        request_data = SpeechTranscribeRequestInput.from_openai(
            request_id=request_id, request=request, task=task
        )
        artifacts = from_wire_audio(request_data)
        result_generator = self.engine.generate(
            request_id=request_id,
            request_data=request_data,
            artifacts=artifacts,
        )

        try:
            return await self.speech_transcription_full_generator(request, raw_request, result_generator, request_id)
        except ValueError as e:
            # TODO: Use an aphrodite-specific Validation Error
            return self.create_error_response(str(e))

    async def speech_transcription_full_generator(
        self,
        request: SpeechTranscribeRequest,
        raw_request: Request,
        result_generator: AsyncIterator[RequestOutput],
        request_id: str,
    ) -> ErrorResponse | PlainTextResponse | SpeechToTextResponse:

        model_name = request.model
        created_time = int(time.time())
        final_res: RequestOutput | None = None

        async for res in result_generator:
            if await raw_request.is_disconnected():
                # Abort the request if the client disconnects.
                await self.engine.abort(request_id)
                return self.create_error_response("Client disconnected")
            final_res = res

        # Ensure we're receiving a proper STT Output here
        assert final_res is not None
        if final_res.finish_reason == "error":
            return self.create_error_response(final_res.error or "Internal inference error")
        assert isinstance(final_res, SpeechTranscriptionRequestOutput)

        # load result data and determine what will be returned
        result_data = json.loads(final_res.output)

        response_format = normalize_response_format(getattr(request, "response_format", "json"))

        if response_format == "json":
            # Byte-identical to the pre-10-4 shape: do not add, rename or
            # reorder fields here.
            response = SpeechToTextResponse(
                id=request_id,
                created=created_time,
                model=model_name,
                text=result_data.get("text", ""),
                language=result_data["info"]["language"] if "info" in result_data and request.get_language else None,
                timestamps=result_data["segments"] if "segments" in result_data and request.get_timestamps else None,
            )
            return response

        if response_format == "text":
            return PlainTextResponse(result_data.get("text", ""))

        segments = result_data.get("segments") or []
        if response_format == "srt":
            return PlainTextResponse(segments_to_srt(segments))
        return PlainTextResponse(segments_to_vtt(segments))

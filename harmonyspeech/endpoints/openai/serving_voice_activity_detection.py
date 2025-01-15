import json
from typing import AsyncGenerator, AsyncIterator

from fastapi import Request

from harmonyspeech.common.config import ModelConfig
from harmonyspeech.common.inputs import DetectVoiceActivityRequestInput
from harmonyspeech.common.outputs import RequestOutput, DetectVoiceActivityRequestOutput, SpeechTranscriptionRequestOutput
from harmonyspeech.endpoints.openai.protocol import *
from harmonyspeech.endpoints.openai.serving_engine import OpenAIServing
from harmonyspeech.engine.async_harmonyspeech import AsyncHarmonySpeech

# Add new model classes which allow handling VAD Requests here
# If multiple models need to be initialized to process request, add multiple to the list
_VAD_MODEL_TYPES = [
    "FasterWhisper"
]
_VAD_MODEL_GROUPS = {
}


class OpenAIServingVoiceActivityDetection(OpenAIServing):

    def __init__(
        self,
        engine: AsyncHarmonySpeech,
        available_models: List[ModelCard],
    ):
        super().__init__(engine=engine, available_models=available_models)

    @staticmethod
    def models_from_config(configured_models: List[ModelConfig]) -> List[ModelCard]:
        return OpenAIServing.model_cards_from_config_groups(
            configured_models,
            _VAD_MODEL_TYPES,
            _VAD_MODEL_GROUPS
        )

    async def check_voice_activity(
        self, request: DetectVoiceActivityRequest, raw_request: Request
    ) -> Union[ErrorResponse, AsyncGenerator[str, None], DetectVoiceActivityResponse]:
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        # TODO: Basic checks for Embedding Generate request

        request_id = f"vad-{random_uuid()}"

        result_generator = self.engine.generate(
            request_id=request_id,
            request_data=DetectVoiceActivityRequestInput.from_openai(
                request_id=request_id,
                request=request
            ),
        )

        try:
            return await self.voice_activity_check_full_generator(
                request, raw_request, result_generator, request_id)
        except ValueError as e:
            # TODO: Use an aphrodite-specific Validation Error
            return self.create_error_response(str(e))

    async def voice_activity_check_full_generator(
        self, request: DetectVoiceActivityRequest, raw_request: Request,
        result_generator: AsyncIterator[RequestOutput],
        request_id: str) -> Union[ErrorResponse, DetectVoiceActivityResponse]:

        model_name = request.model
        created_time = int(time.time())
        final_res: RequestOutput = None

        async for res in result_generator:
            if await raw_request.is_disconnected():
                # Abort the request if the client disconnects.
                await self.engine.abort(request_id)
                return self.create_error_response("Client disconnected")
            final_res = res

        # Ensure we're receiving a proper VAD Output here
        # We may also receive transcription output instead, which is technically also a VAD result
        assert final_res is not None
        assert isinstance(final_res, DetectVoiceActivityRequestOutput) or isinstance(final_res, SpeechTranscriptionRequestOutput)

        # load result data and determine what will be returned
        result_data = json.loads(final_res.output)

        # Check for Text in case of Transcription result
        if isinstance(final_res, SpeechTranscriptionRequestOutput):
            result_data["speech_activity"] = len(result_data["text"]) > 0 if "text" in result_data else False
            del result_data["text"]

        response = DetectVoiceActivityResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            speech_activity=result_data["speech_activity"] if "speech_activity" in result_data else "",
            timestamps=result_data["segments"] if "segments" in result_data and request.get_timestamps else None,
        )

        return response



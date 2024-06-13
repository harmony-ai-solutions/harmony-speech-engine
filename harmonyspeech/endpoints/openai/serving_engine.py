import asyncio
import json
from dataclasses import dataclass
from http import HTTPStatus
from typing import Dict, List, Optional, Tuple, Union

from loguru import logger

from harmonyspeech.endpoints.openai.protocol import ModelList, ModelCard, ErrorResponse
from harmonyspeech.engine.async_harmonyspeech import AsyncHarmonySpeech


class OpenAIServing:

    def __init__(
        self,
        engine: AsyncHarmonySpeech,
        available_models: List[str],
    ):
        self.engine = engine
        self.available_models = available_models

        try:
            event_loop = asyncio.get_running_loop()
        except RuntimeError:
            event_loop = None

        if event_loop is not None and event_loop.is_running():
            # If the current is instanced by Ray Serve,
            # there is already a running event loop
            event_loop.create_task(self._post_init())
        else:
            # When using single HarmonySpeech without engine_use_ray
            asyncio.run(self._post_init())

    async def _post_init(self):
        engine_model_config = await self.engine.get_model_configs()

    async def show_available_models(self) -> ModelList:
        model_cards = [ModelCard(id=x, root=x) for x in self.available_models]
        return ModelList(data=model_cards)

    def create_error_response(
            self,
            message: str,
            err_type: str = "BadRequestError",
            status_code: HTTPStatus = HTTPStatus.BAD_REQUEST) -> ErrorResponse:
        return ErrorResponse(message=message,
                             type=err_type,
                             code=status_code.value)

    def create_streaming_error_response(
            self,
            message: str,
            err_type: str = "BadRequestError",
            status_code: HTTPStatus = HTTPStatus.BAD_REQUEST) -> str:
        json_str = json.dumps({
            "error":
            self.create_error_response(message=message,
                                       err_type=err_type,
                                       status_code=status_code).model_dump()
        })
        return json_str

    async def _check_model(self, request) -> Optional[ErrorResponse]:
        if request.model in [self.available_models]:
            return
        return self.create_error_response(
            message=f"The model `{request.model}` does not exist.",
            err_type="NotFoundError",
            status_code=HTTPStatus.NOT_FOUND)

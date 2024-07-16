import asyncio
import importlib
import inspect
import os
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import Optional

import fastapi
import uvicorn
from fastapi import APIRouter, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

import harmonyspeech
from harmonyspeech.common.config import EngineConfig
from harmonyspeech.common.logger import UVICORN_LOG_CONFIG
from harmonyspeech.endpoints.openai.args import make_arg_parser
from harmonyspeech.endpoints.openai.protocol import TextToSpeechRequest, ErrorResponse, EmbedSpeakerRequest
from harmonyspeech.endpoints.openai.serving_text_to_speech import OpenAIServingTextToSpeech
from harmonyspeech.endpoints.openai.serving_voice_embed import OpenAIServingVoiceEmbedding
from harmonyspeech.engine.async_harmonyspeech import AsyncHarmonySpeech
from harmonyspeech.engine.args_tools import AsyncEngineArgs
from fastapi.responses import (HTMLResponse, JSONResponse, Response, StreamingResponse)
from prometheus_client import make_asgi_app

TIMEOUT_KEEP_ALIVE = 5  # seconds

engine: Optional[AsyncHarmonySpeech] = None
engine_args: Optional[AsyncEngineArgs] = None
engine_config: Optional[EngineConfig] = None
openai_serving_tts: OpenAIServingTextToSpeech = None
# TODO: STT
# TODO: VC
openai_serving_embedding: OpenAIServingVoiceEmbedding = None

router = APIRouter()


@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):

    async def _force_log():
        while True:
            await asyncio.sleep(10)
            await engine.do_log_stats()

    if not engine_args.disable_log_stats:
        asyncio.create_task(_force_log())

    yield

# Add prometheus asgi middleware to route /metrics requests
metrics_app = make_asgi_app()
router.mount("/metrics", metrics_app)


@router.get("/health")
async def health() -> Response:
    """Health check."""
    await openai_serving_tts.engine.check_health()
    # TODO: STT
    # TODO: VC
    # TODO: Embed
    return Response(status_code=200)


@router.get("/version", description="Fetch the Harmony Speech Engine version.")
async def show_version(x_api_key: Optional[str] = Header(None)):
    ver = {"version": harmonyspeech.__version__}
    return JSONResponse(content=ver)


# Based on: https://platform.openai.com/docs/api-reference/audio/createSpeech
@router.post("/v1/audio/speech")
async def create_speech(request: TextToSpeechRequest,
                        raw_request: Request,
                        x_api_key: Optional[str] = Header(None)):
    generator = await openai_serving_tts.create_text_to_speech(request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(), status_code=generator.code)
    if request.output_options and request.output_options.stream:
        return StreamingResponse(content=generator, media_type="text/event-stream")
    else:
        return JSONResponse(content=generator.model_dump())


@router.get("/v1/audio/models")
async def show_available_models(x_api_key: Optional[str] = Header(None)):
    models = await openai_serving_tts.show_available_models()
    return JSONResponse(content=models.model_dump())


@router.post("/v1/embed/speaker")
async def create_embedding(request: EmbedSpeakerRequest,
                        raw_request: Request,
                        x_api_key: Optional[str] = Header(None)):
    generator = await openai_serving_embedding.create_voice_embedding(request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(), status_code=generator.code)
    # if request.stream:
    #     return StreamingResponse(content=generator, media_type="text/event-stream")
    else:
        return JSONResponse(content=generator.model_dump())


@router.get("/v1/embed/models")
async def show_available_models(x_api_key: Optional[str] = Header(None)):
    models = await openai_serving_embedding.show_available_models()
    return JSONResponse(content=models.model_dump())


def build_app(args):
    app = fastapi.FastAPI(lifespan=lifespan)
    app.include_router(router)
    app.root_path = args.root_path

    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(_, exc):
        err = openai_serving_tts.create_error_response(message=str(exc))
        return JSONResponse(err.model_dump(),
                            status_code=HTTPStatus.BAD_REQUEST)

    if token := os.environ.get("HARMONYSPEECH_API_KEY") or args.api_keys:
        admin_key = os.environ.get("HARMONYSPEECH_ADMIN_KEY") or args.admin_key

        if admin_key is None:
            logger.warning("Admin key not provided. Admin operations will "
                           "be disabled.")

        @app.middleware("http")
        async def authentication(request: Request, call_next):
            excluded_paths = ["/api"]
            if any(
                    request.url.path.startswith(path)
                    for path in excluded_paths):
                return await call_next(request)
            if not request.url.path.startswith("/v1"):
                return await call_next(request)

            # Browsers may send OPTIONS requests to check CORS headers
            # before sending the actual request. We should allow these
            # requests to pass through without authentication.
            # See https://github.com/PygmalionAI/aphrodite-engine/issues/434
            if request.method == "OPTIONS":
                return await call_next(request)

            auth_header = request.headers.get("Authorization")
            api_key_header = request.headers.get("x-api-key")

            if auth_header != "Bearer " + token and api_key_header != token:
                return JSONResponse(content={"error": "Unauthorized"},
                                    status_code=401)
            return await call_next(request)

    for middleware in args.middleware:
        module_path, object_name = middleware.rsplit(".", 1)
        imported = getattr(importlib.import_module(module_path), object_name)
        if inspect.isclass(imported):
            app.add_middleware(imported)
        elif inspect.iscoroutinefunction(imported):
            app.middleware("http")(imported)
        else:
            raise ValueError(f"Invalid middleware {middleware}. "
                             f"Must be a function or a class.")

    return app


def run_server(args):
    app = build_app(args)

    logger.debug(f"args: {args}")

    global engine, engine_args, engine_config, openai_serving_tts, openai_serving_embedding
    # TODO: Add other Endpoint serving classes here

    if args.config_file_path is not None:
        config_file_path = args.config_file_path
    else:
        config_file_path = "config.yml"

    # Load Args from Config file if there is one
    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine_config = EngineConfig.load_config_from_yaml(config_file_path)
    engine = AsyncHarmonySpeech.from_engine_args_and_config(engine_args, engine_config)

    openai_serving_tts = OpenAIServingTextToSpeech(
        engine,
        OpenAIServingTextToSpeech.models_from_config(engine_config.model_configs)
    )
    openai_serving_embedding = OpenAIServingVoiceEmbedding(
        engine,
        OpenAIServingVoiceEmbedding.models_from_config(engine_config.model_configs)
    )
    # TODO: Init other Endpoint serving classes here

    try:
        uvicorn.run(app,
                    host=args.host,
                    port=args.port,
                    log_level="info",
                    timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
                    ssl_keyfile=args.ssl_keyfile,
                    ssl_certfile=args.ssl_certfile,
                    log_config=UVICORN_LOG_CONFIG)
    except KeyboardInterrupt:
        logger.info("API server stopped by user. Exiting.")
    except asyncio.exceptions.CancelledError:
        logger.info("API server stopped due to a cancelled request. Exiting.")


if __name__ == "__main__":
    # NOTE:
    # This section should be in sync with aphrodite/endpoints/cli.py
    # for CLI entrypoints.
    parser = make_arg_parser()
    args = parser.parse_args()
    run_server(args)

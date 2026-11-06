import asyncio
import http
import importlib
import inspect
import os
from contextlib import asynccontextmanager
from http import HTTPStatus
from threading import Thread

import fastapi
import uvicorn
from fastapi import APIRouter, Header, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from loguru import logger
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

import harmonyspeech

# Harmony Auth
from auth.apikeys import ApiKeyCacheManager
from harmonyspeech.common import audio_codec
from harmonyspeech.common.config import EngineConfig
from harmonyspeech.common.logger import UVICORN_LOG_CONFIG
from harmonyspeech.endpoints.openai.args import make_arg_parser
from harmonyspeech.endpoints.openai.protocol import (
    AudioConversionRequest,
    AudioConversionResponse,
    DetectVoiceActivityRequest,
    DetectVoiceActivityResponse,
    EmbedSpeakerRequest,
    EmbedSpeakerResponse,
    ErrorResponse,
    ModelList,
    SpeechToTextResponse,
    SpeechTranscribeRequest,
    TextToSpeechRequest,
    TextToSpeechResponse,
    VoiceConversionRequest,
    VoiceConversionResponse,
)
from harmonyspeech.endpoints.openai.request_envelope import (
    UploadTooLargeError,
    dual_body_openapi,
    parse_dual_request,
    resolve_max_upload_bytes,
)
from harmonyspeech.endpoints.openai.serving_audio_conversion import OpenAIServingAudioConversion
from harmonyspeech.endpoints.openai.serving_speech_to_text import OpenAIServingSpeechToText
from harmonyspeech.endpoints.openai.serving_text_to_speech import OpenAIServingTextToSpeech
from harmonyspeech.endpoints.openai.serving_voice_activity_detection import OpenAIServingVoiceActivityDetection
from harmonyspeech.endpoints.openai.serving_voice_conversion import OpenAIServingVoiceConversion
from harmonyspeech.endpoints.openai.serving_voice_embed import OpenAIServingVoiceEmbedding
from harmonyspeech.engine.args_tools import AsyncEngineArgs
from harmonyspeech.engine.async_harmonyspeech import AsyncHarmonySpeech

SERVICE_NAME = os.environ.get("SERVICE_NAME", "harmony-speech-engine")

TIMEOUT_KEEP_ALIVE = 5  # seconds

engine: AsyncHarmonySpeech | None = None
engine_args: AsyncEngineArgs | None = None
engine_config: EngineConfig | None = None
openai_serving_tts: OpenAIServingTextToSpeech = None
openai_serving_stt: OpenAIServingSpeechToText = None
openai_serving_vc: OpenAIServingVoiceConversion = None
openai_serving_embedding: OpenAIServingVoiceEmbedding = None
openai_serving_vad: OpenAIServingVoiceActivityDetection = None
openai_serving_ac: OpenAIServingAudioConversion = None

router = APIRouter()

# Upload part names accepted per request-model field, in priority order. The
# first entry is the canonical/documented name; the aliases keep OpenAI-style
# `curl -F file=@audio.mp3` clients working without a second route.
_TTS_UPLOAD_FIELDS = {"input_audio": ("input_audio", "file", "audio")}
_AUDIO_UPLOAD_FIELDS = {"input_audio": ("file", "audio", "input_audio")}
_VC_UPLOAD_FIELDS = {
    "source_audio": ("source_audio", "file", "audio"),
    "target_audio": ("target_audio",),
}
_AC_UPLOAD_FIELDS = {"source_audio": ("source_audio", "file", "audio")}

_REFERENCE_AUDIO_DOC = "Reference audio file (wav/flac/ogg/mp3/aac…), sent as a file part"
_SOURCE_AUDIO_DOC = "Source audio file (wav/flac/ogg/mp3/aac…), sent as a file part"

_TTS_UPLOAD_DOCS = {"input_audio": _REFERENCE_AUDIO_DOC, "file": "Alias for `input_audio`"}
_AUDIO_UPLOAD_DOCS = {"input_audio": _SOURCE_AUDIO_DOC, "file": "Alias for `input_audio`", "audio": "Alias for `input_audio`"}
_VC_UPLOAD_DOCS = {
    "source_audio": _SOURCE_AUDIO_DOC,
    "file": "Alias for `source_audio`",
    "audio": "Alias for `source_audio`",
    "target_audio": _REFERENCE_AUDIO_DOC,
}
_AC_UPLOAD_DOCS = {"source_audio": _SOURCE_AUDIO_DOC, "file": "Alias for `source_audio`", "audio": "Alias for `source_audio`"}


@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):

    async def _force_log():
        while True:
            await asyncio.sleep(10)
            await engine.do_log_stats()

    if not engine_args.disable_log_stats:
        asyncio.create_task(_force_log())

    yield


# Prometheus metrics: the default registry's series plus the engine's VRAM and
# embedding-cache gauges/counters (Phase 4-5, 7-3). Rendered by hand so a scrape
# never needs a running GPU — or even a live cache — to return a well-formed
# body with every series present.
@router.get("/metrics", description="Prometheus metrics (default registry + VRAM + embedding cache).")
async def metrics() -> Response:
    from harmonyspeech.engine.embedding_cache_metrics import render_embedding_cache_metrics
    from harmonyspeech.engine.vram_metrics import render_vram_metrics

    body = generate_latest().decode("utf-8", errors="replace")
    engine_ref = None
    for serving in (openai_serving_tts, openai_serving_stt, openai_serving_vc, openai_serving_embedding):
        engine_ref = getattr(serving, "engine", None)
        if engine_ref is not None:
            break
    if engine_ref is not None:
        try:
            body += render_vram_metrics(engine_ref.get_vram_stats())
        except Exception as exc:  # noqa: BLE001 - metrics must never break a scrape
            logger.warning(f"Failed to render VRAM metrics: {exc}")
        try:
            cache_stats = engine_ref.get_embedding_cache_stats()
        except Exception as exc:  # noqa: BLE001 - metrics must never break a scrape
            logger.warning(f"Failed to read embedding cache stats: {exc}")
            cache_stats = None
        # Disabled/unavailable cache still renders every series (as zeros) so
        # dashboards never see an absent metric family.
        body += render_embedding_cache_metrics(cache_stats)
    else:
        body += render_embedding_cache_metrics(None)
    return PlainTextResponse(body, media_type=CONTENT_TYPE_LATEST)


@router.get("/health", response_model=dict)
async def health() -> JSONResponse:
    """Health check endpoint to verify the status of all engine serving objects."""
    health_status = {
        "text_to_speech": "unknown",
        "speech_to_text": "unknown",
        "voice_conversion": "unknown",
        "voice_embedding": "unknown",
    }

    status_code = http.HTTPStatus.OK
    try:
        result = await openai_serving_tts.engine.check_health()
        if isinstance(result, dict) and result.get("vram"):
            # Phase 4-5: per-model VRAM accounting rides along with /health.
            health_status["vram"] = result["vram"]
        if isinstance(result, dict) and result.get("embedding_cache") is not None:
            # Phase 7-3: cross-request embedding cache accounting.
            health_status["embedding_cache"] = result["embedding_cache"]
        if isinstance(result, dict) and not result.get("healthy", True):
            health_status["text_to_speech"] = f"unhealthy: {result}"
            status_code = http.HTTPStatus.INTERNAL_SERVER_ERROR
        else:
            health_status["text_to_speech"] = "healthy"
    except Exception as e:
        health_status["text_to_speech"] = f"unhealthy: {str(e)}"
        status_code = http.HTTPStatus.INTERNAL_SERVER_ERROR

    try:
        result = await openai_serving_stt.engine.check_health()
        if isinstance(result, dict) and not result.get("healthy", True):
            health_status["speech_to_text"] = f"unhealthy: {result}"
            status_code = http.HTTPStatus.INTERNAL_SERVER_ERROR
        else:
            health_status["speech_to_text"] = "healthy"
    except Exception as e:
        health_status["speech_to_text"] = f"unhealthy: {str(e)}"
        status_code = http.HTTPStatus.INTERNAL_SERVER_ERROR

    try:
        result = await openai_serving_vc.engine.check_health()
        if isinstance(result, dict) and not result.get("healthy", True):
            health_status["voice_conversion"] = f"unhealthy: {result}"
            status_code = http.HTTPStatus.INTERNAL_SERVER_ERROR
        else:
            health_status["voice_conversion"] = "healthy"
    except Exception as e:
        health_status["voice_conversion"] = f"unhealthy: {str(e)}"
        status_code = http.HTTPStatus.INTERNAL_SERVER_ERROR

    try:
        result = await openai_serving_embedding.engine.check_health()
        if isinstance(result, dict) and not result.get("healthy", True):
            health_status["voice_embedding"] = f"unhealthy: {result}"
            status_code = http.HTTPStatus.INTERNAL_SERVER_ERROR
        else:
            health_status["voice_embedding"] = "healthy"
    except Exception as e:
        health_status["voice_embedding"] = f"unhealthy: {str(e)}"
        status_code = http.HTTPStatus.INTERNAL_SERVER_ERROR

    try:
        result = await openai_serving_vad.engine.check_health()
        if isinstance(result, dict) and not result.get("healthy", True):
            health_status["voice_activity_detection"] = f"unhealthy: {result}"
            status_code = http.HTTPStatus.INTERNAL_SERVER_ERROR
        else:
            health_status["voice_activity_detection"] = "healthy"
    except Exception as e:
        health_status["voice_activity_detection"] = f"unhealthy: {str(e)}"
        status_code = http.HTTPStatus.INTERNAL_SERVER_ERROR

    return JSONResponse(content=health_status, status_code=status_code)


@router.get("/version", description="Fetch the Harmony Speech Engine version.", response_model=dict)
async def show_version(x_api_key: str | None = Header(None)):
    """Fetch the Harmony Speech Engine version."""
    ver = {"version": harmonyspeech.__version__}
    return JSONResponse(content=ver)


# Based on: https://platform.openai.com/docs/api-reference/audio/createSpeech
@router.post(
    "/v1/audio/speech",
    response_model=TextToSpeechResponse,
    openapi_extra=dual_body_openapi(TextToSpeechRequest, _TTS_UPLOAD_DOCS),
)
async def create_speech(raw_request: Request, x_api_key: str | None = Header(None)):
    """Generate speech Audio from text.

    Accepts a JSON body (primary, OpenAI-style) or a multipart/form-data body
    for voice cloning, where the reference audio is uploaded as a file part
    instead of base64 inside JSON.
    """
    request = await parse_dual_request(raw_request, TextToSpeechRequest, upload_fields=_TTS_UPLOAD_FIELDS)
    generator = await openai_serving_tts.create_text_to_speech(request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(), status_code=generator.code)
    # if request.output_options and request.output_options.stream:
    #     return StreamingResponse(content=generator, media_type="text/event-stream")
    else:
        return JSONResponse(content=generator.model_dump())


@router.get("/v1/audio/speech/models", response_model=ModelList)
async def show_available_speech_models(x_api_key: str | None = Header(None)):
    """Show available speech models."""
    models = await openai_serving_tts.show_available_models()
    return JSONResponse(content=models.model_dump())


@router.post(
    "/v1/embed/speaker",
    response_model=EmbedSpeakerResponse,
    openapi_extra=dual_body_openapi(EmbedSpeakerRequest, _AUDIO_UPLOAD_DOCS),
)
async def create_embedding(raw_request: Request, x_api_key: str | None = Header(None)):
    """Create a speaker embedding.

    Accepts a JSON body or a multipart/form-data upload (`file`/`audio`/`input_audio`).
    """
    request = await parse_dual_request(raw_request, EmbedSpeakerRequest, upload_fields=_AUDIO_UPLOAD_FIELDS)
    generator = await openai_serving_embedding.create_voice_embedding(request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(), status_code=generator.code)
    # if request.stream:
    #     return StreamingResponse(content=generator, media_type="text/event-stream")
    else:
        return JSONResponse(content=generator.model_dump())


@router.get("/v1/embed/models", response_model=ModelList)
async def show_available_embedding_models(x_api_key: str | None = Header(None)):
    """Show available embedding models."""
    models = await openai_serving_embedding.show_available_models()
    return JSONResponse(content=models.model_dump())


# Based on: https://platform.openai.com/docs/api-reference/audio/createTranscription
@router.post(
    "/v1/audio/transcriptions",
    response_model=SpeechToTextResponse,
    openapi_extra=dual_body_openapi(SpeechTranscribeRequest, _AUDIO_UPLOAD_DOCS),
)
async def create_transcription(raw_request: Request, x_api_key: str | None = Header(None)):
    """Create a transcription from audio.

    Accepts the OpenAI-style multipart upload (`file=@audio.mp3`) or a JSON body
    with base64 `input_audio` (legacy alternative).
    """
    request = await parse_dual_request(raw_request, SpeechTranscribeRequest, upload_fields=_AUDIO_UPLOAD_FIELDS)
    generator = await openai_serving_stt.create_transcription(request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(), status_code=generator.code)
    # if request.output_options and request.output_options.stream:
    #     return StreamingResponse(content=generator, media_type="text/event-stream")
    else:
        return JSONResponse(content=generator.model_dump())


@router.get("/v1/audio/transcriptions/models", response_model=ModelList)
async def show_available_transcription_models(x_api_key: str | None = Header(None)):
    """Show available transcription models."""
    models = await openai_serving_stt.show_available_models()
    return JSONResponse(content=models.model_dump())


@router.post(
    "/v1/voice/convert",
    response_model=VoiceConversionResponse,
    openapi_extra=dual_body_openapi(VoiceConversionRequest, _VC_UPLOAD_DOCS),
)
async def convert_voice(raw_request: Request, x_api_key: str | None = Header(None)):
    """Convert the voice in an audio file or stream to a desired target voice.

    Accepts multipart uploads (`source_audio`/`file` plus optional
    `target_audio`) or a JSON body with base64 audio fields.
    """
    request = await parse_dual_request(raw_request, VoiceConversionRequest, upload_fields=_VC_UPLOAD_FIELDS)
    generator = await openai_serving_vc.convert_voice(request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(), status_code=generator.code)
    # if request.output_options and request.output_options.stream:
    #     return StreamingResponse(content=generator, media_type="text/event-stream")
    else:
        return JSONResponse(content=generator.model_dump())


@router.get("/v1/voice/convert/models")
async def show_available_voice_conversion_models(x_api_key: str | None = Header(None)):
    """Show available voice conversion models."""
    models = await openai_serving_vc.show_available_models()
    return JSONResponse(content=models.model_dump())


@router.post(
    "/v1/audio/vad",
    response_model=DetectVoiceActivityResponse,
    openapi_extra=dual_body_openapi(DetectVoiceActivityRequest, _AUDIO_UPLOAD_DOCS),
)
async def create_vad(raw_request: Request, x_api_key: str | None = Header(None)):
    """Detect voice activity in audio.

    Accepts a multipart upload (`file`/`audio`/`input_audio`) or a JSON body.
    """
    request = await parse_dual_request(raw_request, DetectVoiceActivityRequest, upload_fields=_AUDIO_UPLOAD_FIELDS)
    generator = await openai_serving_vad.check_voice_activity(request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(), status_code=generator.code)
    # if request.output_options and request.output_options.stream:
    #     return StreamingResponse(content=generator, media_type="text/event-stream")
    else:
        return JSONResponse(content=generator.model_dump())


@router.get("/v1/audio/vad/models", response_model=ModelList)
async def show_available_vad_models(x_api_key: str | None = Header(None)):
    """Show available vad models."""
    models = await openai_serving_vad.show_available_models()
    return JSONResponse(content=models.model_dump())


@router.post(
    "/v1/audio/convert",
    response_model=AudioConversionResponse,
    openapi_extra=dual_body_openapi(AudioConversionRequest, _AC_UPLOAD_DOCS),
)
async def convert_audio(raw_request: Request, x_api_key: str | None = Header(None)):
    """Convert/restore audio.

    Accepts a multipart upload (`source_audio`/`file`) or a JSON body.
    """
    request = await parse_dual_request(raw_request, AudioConversionRequest, upload_fields=_AC_UPLOAD_FIELDS)
    generator = await openai_serving_ac.convert_audio(request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(), status_code=generator.code)
    # if request.output_options and request.output_options.stream:
    #     return StreamingResponse(content=generator, media_type="text/event-stream")
    else:
        return JSONResponse(content=generator.model_dump())


@router.get("/v1/audio/convert/models", response_model=ModelList)
async def show_available_audio_conversion_models(x_api_key: str | None = Header(None)):
    """Show available vad models."""
    models = await openai_serving_ac.show_available_models()
    return JSONResponse(content=models.model_dump())


# Register event with Rate limiting backend
# Use separate thread to not interfere with request processing
def register_event(api_key, event_name, client_ip):
    register_thread = Thread(
        target=ApiKeyCacheManager.register_rate_limiting_event, args=(api_key, SERVICE_NAME, event_name, client_ip)
    )
    register_thread.start()


def build_app(args):
    app = fastapi.FastAPI(
        title="Harmony Speech Engine API", description="API for the Harmony Speech Engine.", lifespan=lifespan
    )
    app.include_router(router)
    app.root_path = args.root_path

    # Upload/request size guard for multipart audio uploads (and any oversized
    # JSON body). Configurable per deployment.
    max_upload_mb = getattr(args, "max_upload_size_mb", None)
    if max_upload_mb is None:
        try:
            max_upload_mb = int(os.environ.get("HARMONYSPEECH_MAX_UPLOAD_MB", 50))
        except ValueError:
            max_upload_mb = 50
    app.state.max_upload_bytes = int(max_upload_mb) * 1024 * 1024

    # Custom OpenAPI schema generation for Official clients
    default_schema = app.openapi

    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        openapi_schema = default_schema()
        openapi_schema["info"]["x-go-client-name"] = "HarmonySpeechAPI"
        openapi_schema["info"]["x-go-package"] = "harmonyspeech"
        app.openapi_schema = openapi_schema
        return app.openapi_schema

    app.openapi = custom_openapi

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
        return JSONResponse(err.model_dump(), status_code=HTTPStatus.BAD_REQUEST)

    @app.exception_handler(UploadTooLargeError)
    async def upload_too_large_handler(_, exc):
        err = ErrorResponse(
            message=str(exc),
            type="PayloadTooLargeError",
            code=HTTPStatus.REQUEST_ENTITY_TOO_LARGE.value,
        )
        return JSONResponse(err.model_dump(), status_code=HTTPStatus.REQUEST_ENTITY_TOO_LARGE)

    @app.middleware("http")
    async def enforce_upload_size(request: Request, call_next):
        if request.method in ("POST", "PUT", "PATCH"):
            limit = resolve_max_upload_bytes(request.app)
            content_length = request.headers.get("content-length")
            if content_length:
                try:
                    declared = int(content_length)
                except ValueError:
                    declared = 0
                if declared > limit:
                    err = ErrorResponse(
                        message=(
                            f"Request body of {declared} bytes exceeds the maximum upload size "
                            f"of {limit // (1024 * 1024)} MiB."
                        ),
                        type="PayloadTooLargeError",
                        code=HTTPStatus.REQUEST_ENTITY_TOO_LARGE.value,
                    )
                    return JSONResponse(err.model_dump(), status_code=HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
        return await call_next(request)

    if token := os.environ.get("HARMONYSPEECH_API_KEY") or args.api_keys:
        admin_key = os.environ.get("HARMONYSPEECH_ADMIN_KEY") or args.admin_key

        if admin_key is None:
            logger.warning("Admin key not provided. Admin operations will be disabled.")

        @app.middleware("http")
        async def authentication(request: Request, call_next):
            excluded_paths = ["/api"]
            if any(request.url.path.startswith(path) for path in excluded_paths):
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
            api_key_header = request.headers.get("Api-Key")

            # If Harmony Auth Key is set, this takes precedence
            if api_key_header and api_key_header != token:
                harmony_auth_valid, error = ApiKeyCacheManager.check_request_allowed_by_rate_limit(
                    api_key=api_key_header, service=SERVICE_NAME
                )
                if not harmony_auth_valid:
                    return JSONResponse(content={"error": error}, status_code=401)

                # Log event to rate limiting
                client_ip = request.client.host
                client_source_ip = request.headers.get("X-Real-Ip")
                if client_source_ip:
                    client_ip = client_source_ip
                register_event(api_key=api_key_header, event_name=request.url.path, client_ip=client_ip)
            else:
                # Default API Key for local instance - make sure to also set this if using harmony auth;
                # to ensure Access is blocked if client does not send an API key
                if api_key_header != token or auth_header != f"Bearer {token}":
                    return JSONResponse(content={"error": "Unauthorized"}, status_code=401)

            return await call_next(request)

    for middleware in args.middleware:
        module_path, object_name = middleware.rsplit(".", 1)
        imported = getattr(importlib.import_module(module_path), object_name)
        if inspect.isclass(imported):
            app.add_middleware(imported)
        elif inspect.iscoroutinefunction(imported):
            app.middleware("http")(imported)
        else:
            raise ValueError(f"Invalid middleware {middleware}. Must be a function or a class.")

    return app


def run_server(args):
    app = build_app(args)

    logger.debug(f"args: {args}")

    global engine, engine_args, engine_config
    global openai_serving_tts
    global openai_serving_embedding
    global openai_serving_stt
    global openai_serving_vc
    global openai_serving_vad
    global openai_serving_ac
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
        engine, OpenAIServingTextToSpeech.models_from_config(engine_config.model_configs)
    )
    openai_serving_embedding = OpenAIServingVoiceEmbedding(
        engine, OpenAIServingVoiceEmbedding.models_from_config(engine_config.model_configs)
    )
    openai_serving_stt = OpenAIServingSpeechToText(
        engine, OpenAIServingSpeechToText.models_from_config(engine_config.model_configs)
    )
    openai_serving_vc = OpenAIServingVoiceConversion(
        engine, OpenAIServingVoiceConversion.models_from_config(engine_config.model_configs)
    )
    openai_serving_vad = OpenAIServingVoiceActivityDetection(
        engine, OpenAIServingVoiceActivityDetection.models_from_config(engine_config.model_configs)
    )
    openai_serving_ac = OpenAIServingAudioConversion(
        engine, OpenAIServingAudioConversion.models_from_config(engine_config.model_configs)
    )
    # TODO: Init other Endpoint serving classes here

    audio_codec.refresh_output_formats()
    logger.info(
        "Audio I/O contract — input formats: {} (ffmpeg fallback: {}); output formats: {}",
        ", ".join(sorted(audio_codec.SUPPORTED_INPUT_FORMATS)),
        "available" if audio_codec.FFMPEG_AVAILABLE else "unavailable",
        ", ".join(sorted(audio_codec.SUPPORTED_OUTPUT_FORMATS)),
    )

    try:
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level="info",
            timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
            ssl_keyfile=args.ssl_keyfile,
            ssl_certfile=args.ssl_certfile,
            log_config=UVICORN_LOG_CONFIG,
        )
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

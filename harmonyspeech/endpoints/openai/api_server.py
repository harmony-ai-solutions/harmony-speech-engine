import asyncio
from contextlib import asynccontextmanager
from typing import Optional

import fastapi
import uvicorn
from fastapi import APIRouter, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from harmonyspeech.engine.async_harmonyspeech import AsyncHarmonySpeech
from fastapi.responses import (HTMLResponse, JSONResponse, Response, StreamingResponse)
from prometheus_client import make_asgi_app

TIMEOUT_KEEP_ALIVE = 5  # seconds

engine: Optional[AsyncHarmonySpeech] = None
engine_args: Optional[AsyncEngineArgs] = None
openai_serving_tts =

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
    await openai_serving_chat.engine.check_health()
    await openai_serving_completion.engine.check_health()
    return Response(status_code=200)


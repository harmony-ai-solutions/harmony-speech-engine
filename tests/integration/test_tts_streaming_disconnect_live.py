"""Phase 14-5: a real HTTP client disconnect against a live uvicorn server.

The in-process disconnect gates (``test_tts_streaming.py`` /
``test_tts_streaming_sse.py``) cancel the response generator directly. This
test closes the **TCP connection** instead, through uvicorn's ASGI server and
Starlette's disconnect listener, and asserts the production seam:

* ``engine.abort`` fires for the chunk that was in flight (the abort-call
  seam — the stub executor has no AR row / cache slot, so the 9-4/9-2 stats
  surfaces are not reachable hermetically; see the task log);
* the frame source never submits another chunk (the harness model's request
  log is the discipline check);
* the stream is classified ``aborted`` in the 14-5 metrics.

Hermetic and CPU-only: the 12-2 ``EngineHarness`` boots the real
``AsyncHarmonySpeech`` loop with a stub Chatterbox model and a deterministic
per-chunk delay. No GPU, no weights.
"""

import asyncio
import contextlib
import socket
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest

from harmonyspeech.endpoints.openai import api_server
from harmonyspeech.endpoints.openai.args import make_arg_parser
from harmonyspeech.endpoints.openai.streaming_metrics import (
    get_stream_stats,
    reset_stream_metrics,
)
from tests.integration.test_tts_chunking import (
    ChatterboxStubModel,
    EngineHarness,
    _model_config,
)
from tests.integration.test_tts_streaming import (
    CHUNK_SIZE,
    SENTENCE_A,
    SENTENCE_C,
    TEXT_3,
)

#: Per-chunk synthesis time. Long enough that the client can abort while chunk 1
#: is in flight, short enough that the test stays fast.
EXECUTOR_DELAY_S = 0.25


@contextlib.asynccontextmanager
async def _live_app(serving, engine):
    """Serve the wire app on a loopback socket in the current event loop."""
    import uvicorn

    args = make_arg_parser().parse_args([])
    with (
        patch.object(api_server, "openai_serving_tts", serving),
        patch.object(api_server, "engine", engine),
        patch.object(
            api_server,
            "engine_args",
            SimpleNamespace(disable_log_stats=True),
        ),
    ):
        app = api_server.build_app(args)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        sock.listen(2048)
        port = sock.getsockname()[1]
        config = uvicorn.Config(
            app,
            host="127.0.0.1",
            port=port,
            log_level="warning",
            lifespan="off",
        )
        server = uvicorn.Server(config)
        # The loop runs in the pytest main thread; do not install process
        # signal handlers for a test server.
        server.capture_signals = lambda: contextlib.nullcontext()
        task = asyncio.create_task(server.serve(sockets=[sock]))
        try:
            deadline = asyncio.get_running_loop().time() + 15
            while not server.started and asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(0.01)
            assert server.started, "uvicorn did not start"
            yield f"http://127.0.0.1:{port}"
        finally:
            server.should_exit = True
            with contextlib.suppress(Exception):
                await asyncio.wait_for(task, timeout=15)
            sock.close()


def _payload() -> dict:
    return {
        "model": "chatterbox",
        "mode": "single_speaker_tts",
        "input": TEXT_3,
        "generation_options": {"split_text": True, "chunk_size": CHUNK_SIZE},
        "output_options": {"stream_format": "audio", "format": "wav"},
    }


@pytest.mark.integration
@pytest.mark.asyncio
async def test_tcp_disconnect_aborts_the_in_flight_chunk_and_stops_synthesis():
    reset_stream_metrics()
    config = _model_config("chatterbox", "ChatterboxTTS")
    model = ChatterboxStubModel()
    harness = EngineHarness(
        [config], {"chatterbox": model}, executor_delay_s=EXECUTOR_DELAY_S
    )

    async with harness:
        aborts: list[str] = []
        original_abort = harness.engine.abort

        async def spy_abort(request_id):
            aborts.append(request_id)
            return await original_abort(request_id)

        harness.engine.abort = spy_abort

        async with _live_app(harness.serving, harness.engine) as base_url:
            async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
                async with client.stream(
                    "POST", "/v1/audio/speech", json=_payload()
                ) as response:
                    assert response.status_code == 200
                    first = None
                    async for chunk in response.aiter_raw():
                        first = chunk
                        break
                    assert first is not None and first[:4] == b"RIFF"
                    # Chunk 0 is in hand; the generator has resumed and chunk 1
                    # is now in flight. Closing the context closes the socket.
                    await asyncio.sleep(0.05)

        # Give uvicorn/Starlette a moment to deliver http.disconnect and the
        # serving layer to abort the in-flight chunk.
        for _ in range(200):
            if aborts:
                break
            await asyncio.sleep(0.01)

    assert aborts, "engine.abort was never called for the disconnected stream"
    assert aborts[0].endswith("-chunk-1"), aborts
    # Chunk 0 was delivered. Chunk 1 was in flight when the client left: it may
    # finish synthesis before the abort is processed, but its audio is never
    # delivered, and chunk 2 is never submitted (sequential submission is the
    # discipline gate).
    assert SENTENCE_A in model.generate_calls
    assert SENTENCE_C not in model.generate_calls
    assert len(model.generate_calls) <= 2
    assert get_stream_stats()["chunks"] == {"completed": 0, "aborted": 1, "error": 0}

"""OpenAI-compatible text-to-speech serving (Phase 12-2 long-form chunking).

Single-shot requests keep the pre-12 behaviour byte-for-byte: when the input
splits into zero or one chunk the chunker is skipped entirely and the original
request goes through the engine exactly as before.

Long-form requests (``split_text: true`` or an explicit ``chunk_size``) are
split into sentence-aware chunks, **submitted concurrently** so the engine can
admit them into one batch, and stitched back in chunk-index order (never
completion order). The serving layer owns the cross-family post-processing:

* the reference is conditioned **once** for a voice-cloning request and reused
  by every chunk (no per-chunk embed step) — through exactly the mechanism the
  single-shot path uses for that pipeline: a standalone embedding step where
  one exists, and no conditioning at all where the single-shot path never reads
  the reference (see :meth:`_reference_conditioning_mode`);
* a watermark is applied **exactly once**, on the stitched buffer (chunk
  synthesis runs with the watermark suppressed);
* ``speed`` is applied with a post-stitch phase-vocoder stretch only for
  families that do not consume ``speed`` natively;
* any chunk failure fails the whole request (no partial audio with a 200);
* the opt-in Phase B filters (Phase 12-3) run here, not in the runner: DC
  removal per chunk **before** stitching, then lead/trail trim and
  internal-silence fixing on the **final stitched buffer** (before the speed
  stretch and the single watermark). All three default off.

Streaming requests (Phase 14) bypass the buffered dispatch: one shared frame
source (:meth:`OpenAIServingTextToSpeech._stream_audio_segments`) synthesises
the chunks sequentially and emits float32 segments, which are wrapped by one of
two envelopes — ``stream_format: "audio"`` (binary WAV/PCM frames, 14-3) or
``stream_format: "sse"`` (``speech.audio.delta``/``done``/``error`` events,
14-4). Only the framing differs; the audio is identical by construction.
"""

import asyncio
import base64
import contextlib
import copy
import dataclasses
import threading
import time
from collections.abc import AsyncGenerator, AsyncIterator
from http import HTTPStatus

import numpy as np
from fastapi import Request
from loguru import logger
from starlette.concurrency import run_in_threadpool

from harmonyspeech.common import audio_codec
from harmonyspeech.common.artifacts import AudioArtifact
from harmonyspeech.common.audio_utils import (
    DEFAULT_CHUNK_CROSSFADE_MS,
    DEFAULT_CHUNK_SENTENCE_PAUSE_MS,
    CrossfadeStitcher,
    crossfade_stitch,
    fix_internal_silence,
    remove_dc_offset,
    split_text_into_chunks,
    time_stretch,
    trim_silence,
)
from harmonyspeech.common.config import ModelConfig
from harmonyspeech.common.inputs import SpeechEmbeddingRequestInput, TextToSpeechRequestInput
from harmonyspeech.common.outputs import (
    RequestOutput,
    SpeechSynthesisRequestOutput,
    TextToSpeechRequestOutput,
)
from harmonyspeech.common.utils import random_uuid
from harmonyspeech.endpoints.openai.output_encoding import normalize_format, validate_output_options
from harmonyspeech.endpoints.openai.protocol import ErrorResponse, ModelCard, TextToSpeechRequest, TextToSpeechResponse
from harmonyspeech.endpoints.openai.serving_engine import OpenAIServing
from harmonyspeech.endpoints.openai.standard_voice import standard_voice_wire_payload
from harmonyspeech.endpoints.openai.streaming import (
    SSE_DONE_FRAME,
    SSE_EVENT_DELTA,
    SSE_EVENT_DONE,
    SSE_EVENT_ERROR,
    SSE_MEDIA_TYPE,
    STREAMING_AUDIO_FORMATS,
    StreamResampler,
    delta_payload,
    done_payload,
    error_payload,
    keepalive,
    pcm16_frame,
    resolve_stream_format,
    sse_event,
    streaming_response_headers,
    wav_stream_header,
)
from harmonyspeech.endpoints.openai.streaming_metrics import (
    observe_first_frame,
    record_stream_bytes,
    record_stream_outcome,
)
from harmonyspeech.endpoints.openai.wire import from_wire_audio, to_wire_audio
from harmonyspeech.engine.async_harmonyspeech import AsyncHarmonySpeech
from harmonyspeech.modeling.voice_catalog import (
    lookup_standard_voice,
    request_model_types,
)
from harmonyspeech.processing.pipeline_registry import get_spec

#: Default character budget per chunk when a client enables ``split_text``
#: without naming a ``chunk_size`` (Phase 12-2 owner decision, 2026-11-09).
#: 300 characters is one to three spoken sentences: small enough to keep the
#: autoregressive families' per-request token budget and peak memory bounded,
#: large enough that prosody and the crossfade are not dominated by joins.
#:
#: Phase 12-7/12-8: the old per-model overrides (Turbo 150, Multilingual 75)
#: were workarounds for bugs that are now fixed at the source — Turbo's
#: 768-token KV-slot truncation and Multilingual's premature analyzer EOS. Every
#: family uses this unified 300-char default again.
DEFAULT_CHUNK_SIZE = 300

#: Model types that consume ``generation_options.speed`` natively today (each
#: maps ``speed`` onto a duration predictor / ``length_scale`` / ``alpha``).
#: Everything else (all Chatterbox families) ignores ``speed``; on the chunked
#: path the serving layer applies a whole-buffer ``time_stretch`` for those.
#: Verified 2026-11-09 from ``common/inputs.py`` and the handlers:
#: * ``HarmonySpeechSynthesizer`` -> ``alpha`` (``generate``);
#: * ``OpenVoiceV1Synthesizer`` -> ``length_scale=1/speed`` (``infer``);
#: * ``MeloTTSSynthesizer`` -> ``length_scale=1/speed`` (``infer``);
#: * ``KittenTTSSynthesizer`` -> ``speed`` argument of the ONNX model.
NATIVE_SPEED_MODEL_TYPES: frozenset[str] = frozenset(
    {
        "HarmonySpeechSynthesizer",
        "OpenVoiceV1Synthesizer",
        "MeloTTSSynthesizer",
        "KittenTTSSynthesizer",
    }
)

#: Serializes serving-side watermark application. The engine's model threads
#: are the only other caller; the watermark forward is read-only, but a single
#: lock keeps concurrent serving coroutines from racing each other.
_WATERMARK_LOCK = threading.Lock()

#: How long the SSE envelope waits for one frame before emitting a
#: ``: keepalive`` comment (Phase 14-4, owner decision 2026-11-10).
#:
#: 15 s is safely under the common 60 s proxy/browser idle timeouts while
#: staying far above a normal chunk's synthesis time, so a healthy stream never
#: carries a comment. The race covers **every** frame wait, the first one
#: included: a cold start whose first sentence takes longer than this keeps the
#: connection warm instead of being dropped by an intermediary.
#:
#: Module-level on purpose: tests inject a tiny value (``monkeypatch.setattr``)
#: rather than sleeping 15 s. ``_encode_sse_frames`` resolves it at call time,
#: so patching the module attribute is enough.
SSE_KEEPALIVE_TIMEOUT_S = 15.0

#: Model types that can produce a reference's conditioning in a standalone
#: embedding step even when the request names the executor **directly** (no
#: pipeline id): ``task_handler.inputs.prepare_inputs`` accepts a
#: ``SpeechEmbeddingRequestInput`` for these and the handler returns a
#: ``SpeakerArtifact``. Only the Chatterbox families qualify — their single-shot
#: clone path computes the *same* conditionals inline in ``generate``
#: (``prepare_conditionals``), so hoisting that one call out of the chunk loop
#: is faithful (the embed step is the same computation, and it is also the step
#: a Chatterbox toolchain request runs).
#:
#: Every other concrete synthesizer (``MeloTTSSynthesizer``, the OpenVoice V1
#: synthesizer, ``KittenTTSSynthesizer``, ``HarmonySpeechSynthesizer``) never
#: reads ``input_audio`` on the single-shot path: its ``prepare_inputs`` accepts
#: only ``TextToSpeechRequestInput``/``SynthesisRequestInput``, so a standalone
#: embedding sub-request would (a) invent conditioning the single-shot path
#: never applies and (b) be rejected with the 2026-11-09 benchmark error
#: (``ov2-synthesizer-en`` + ``voice_cloning`` → ``-embed`` rejected by MeloTTS).
STANDALONE_EMBED_MODEL_TYPES: frozenset[str] = frozenset(
    {
        "ChatterboxEmbedding",
        "ChatterboxTTS",
        "ChatterboxTurboTTS",
        "ChatterboxMultilingualTTS",
    }
)


def _option(options, name: str, default=None):
    return getattr(options, name, default) if options is not None else default


def _output_filter_flags(options) -> tuple[bool, bool, bool]:
    """Return ``(remove_dc_offset, trim_silence, fix_internal_silence)``.

    Phase 12-3 opt-in output filters. All default off; when all three are false
    the serving layer never touches the samples (byte-identity gate).
    """
    return (
        bool(_option(options, "remove_dc_offset", False)),
        bool(_option(options, "trim_silence", False)),
        bool(_option(options, "fix_internal_silence", False)),
    )


def streaming_admission_error(request, *, uses_native_speed: bool) -> str | None:
    """Return the admission error for a streaming request, or ``None``.

    Phase 14-1 (owner decisions 2026-11-08 / 2026-11-09). Every rule fails
    **before synthesis starts**; the serving layer turns the message into the
    usual JSON 400. The restrictions exist because the streaming path emits
    each chunk as it completes, while these options are defined on the whole
    stitched buffer:

    * streaming emits ``wav``/``pcm`` frames only — whole-buffer encoders such
      as mp3/flac cannot be framed incrementally;
    * ``trim_silence`` / ``fix_internal_silence`` gate on the **global** RMS
      peak of the final stitched buffer;
    * non-native ``speed`` is a phase-vocoder stretch of the final buffer
      (families in :data:`NATIVE_SPEED_MODEL_TYPES` consume ``speed``
      themselves, so they are exempt).

    ``sample_rate`` and ``remove_dc_offset`` compose with streaming: the
    stateful resampler lands in 14-2, and DC removal is per-chunk/stateless.
    """
    output_options = getattr(request, "output_options", None)
    if resolve_stream_format(output_options) is None:
        return None

    output_format = normalize_format(getattr(output_options, "format", None))
    if output_format not in STREAMING_AUDIO_FORMATS:
        return (
            "Streaming output is only supported for the 'wav' and 'pcm' formats, "
            f"not '{output_format}'. Set output_options.format to 'wav' or 'pcm', "
            "or remove stream_format/stream to use buffered output."
        )

    options = getattr(request, "generation_options", None)
    _remove_dc, trim, fix_internal = _output_filter_flags(options)
    if trim:
        return (
            "trim_silence is only supported for whole-buffer (non-streaming) output: "
            "it trims the final stitched buffer relative to its global peak, so it cannot "
            "be applied to incrementally streamed audio. Disable trim_silence or streaming."
        )
    if fix_internal:
        return (
            "fix_internal_silence is only supported for whole-buffer (non-streaming) output: "
            "it collapses pauses on the final stitched buffer, so it cannot be applied to "
            "incrementally streamed audio. Disable fix_internal_silence or streaming."
        )

    speed = _option(options, "speed", None)
    if speed is not None and float(speed) != 1.0 and not uses_native_speed:
        return (
            f"speed={speed!r} is only supported for whole-buffer (non-streaming) output on "
            "model families that do not consume speed natively: the speed change is a "
            "phase-vocoder stretch of the final stitched buffer. Use speed 1 (or omit it), "
            "choose a native-speed model, or disable streaming."
        )

    return None


def _resolve_join_timing(options) -> tuple[int, int]:
    """Return ``(crossfade_ms, sentence_pause_ms)`` for a chunked request.

    Phase 12-5: only reached when chunking is on (the serving layer already
    produced 2+ chunks), so an unset field gets the tuned default — 40 ms
    crossfade and 120 ms sentence pause. An explicit value (including 0) always
    wins. The stitcher edge-conditions every chunk either way.
    """
    crossfade = _option(options, "crossfade_ms", None)
    pause = _option(options, "sentence_pause_ms", None)
    return (
        DEFAULT_CHUNK_CROSSFADE_MS if crossfade is None else int(crossfade),
        DEFAULT_CHUNK_SENTENCE_PAUSE_MS if pause is None else int(pause),
    )


def _clone_generation_options(options, **updates):
    """Return a per-chunk copy of a pydantic or dataclass options object."""
    if options is None:
        return None
    model_copy = getattr(options, "model_copy", None)
    if callable(model_copy):
        return model_copy(update=updates)
    try:
        return dataclasses.replace(options, **updates)
    except TypeError:  # pragma: no cover - non-dataclass stand-ins
        clone = copy.copy(options)
        for key, value in updates.items():
            setattr(clone, key, value)
        return clone


class StreamSynthesisError(Exception):
    """A streaming synthesis failure raised before the body is committed.

    The binary envelope sends headers lazily, with the first frame: everything
    that can fail **before** that first frame is raised as this exception and
    turned into an ordinary JSON error response (normal HTTP status), never a
    truncated 200 body. A failure after the first byte cannot be signalled on a
    binary body — that is the known weakness of ``stream_format: "audio"`` and
    the reason the SSE envelope (14-4) exists.
    """


@dataclasses.dataclass
class StreamState:
    """Mutable accounting for one streaming request, shared by both envelopes.

    The 14-3 binary generator and the 14-4 SSE wrapper consume the same frame
    source (:meth:`OpenAIServingTextToSpeech._stream_audio_segments`) and read
    their ``done`` payload from this object.

    * ``source_rate`` — the model's native rate, learned from the first chunk;
    * ``target_rate`` — the effective wire rate (requested ``sample_rate`` or
      the source rate). This is what ``X-Audio-Sample-Rate`` and the streaming
      WAV header carry;
    * ``emitted_samples`` — the generator's **own** counter of stitched
      source-rate samples handed to the wire (``feed()`` + ``flush()`` lengths).
      ``CrossfadeStitcher`` exposes no cumulative counter and the wire length
      differs once resampling is in play, so this is the only honest source for
      ``duration_s`` (14-4's ``speech.audio.done``).
    """

    chunks: int = 0
    source_rate: int | None = None
    target_rate: int | None = None
    emitted_samples: int = 0
    segments: int = 0
    #: Monotonic start of the request's frame source; the first-frame histogram
    #: observes ``time.monotonic() - started_at`` (Phase 14-5).
    started_at: float = dataclasses.field(default_factory=time.monotonic)
    #: Set when the client is known to be gone (a cancelled response task, a
    #: closed generator, or the per-chunk ``is_disconnected`` poll). Additive:
    #: it changes no envelope bytes; it lets the metrics classify a polled
    #: disconnect as ``aborted`` instead of a normal ``completed`` (14-4's note).
    aborted: bool = False
    #: True once the first *audio* frame was emitted (keepalives excluded).
    first_frame_observed: bool = False

    @property
    def duration_s(self) -> float:
        """Streamed duration in seconds (source-rate samples ÷ source rate)."""
        if not self.source_rate:
            return 0.0
        return self.emitted_samples / float(self.source_rate)


@dataclasses.dataclass
class StreamingAudioResponse:
    """The binary streaming envelope, ready for ``StreamingResponse``.

    The serving layer owns the wire contract (14-1/14-2: header names, WAV
    header, PCM quantization), so it returns the body iterator plus the exact
    header set and media type; ``api_server.create_speech`` only wraps it.
    """

    body: AsyncIterator[bytes | str]
    media_type: str
    headers: dict[str, str]
    #: The stream's own accounting (emitted samples, duration, chunk count).
    #: The binary envelope has no trailer, so this is where 14-4's SSE ``done``
    #: payload (and any future stream metrics) read ``duration_s`` from.
    state: StreamState | None = None


#: Media types for the two binary containers (``X-Audio-Format`` carries the
#: machine-readable value; this is the conventional Content-Type).
_STREAM_MEDIA_TYPES: dict[str, str] = {"wav": "audio/wav", "pcm": "audio/pcm"}

async def _prepend_frames[FrameT](first: FrameT, rest: AsyncIterator[FrameT]) -> AsyncGenerator[FrameT, None]:
    """Re-yield an already-awaited first frame, then the rest of the body.

    Generic over the frame type: ``bytes`` for the binary envelope, ``str`` for
    the SSE event stream (Starlette UTF-8 encodes it).
    """
    yield first
    async for frame in rest:
        yield frame


async def _encode_stream_frames(
    segments: AsyncIterator[np.ndarray],
    *,
    audio_format: str,
    state: StreamState,
) -> AsyncGenerator[bytes, None]:
    """Turn float32 segments into the binary envelope's byte frames.

    One ``pcm16_frame`` call per segment (in a worker thread — never on the
    event loop). The first yielded frame carries the streaming WAV header for
    ``wav`` (``0xFFFFFFFF`` sizes at the **target** rate, which the frame source
    has already put on ``state``); ``pcm`` omits it. Header and first PCM frame
    are one yield on purpose: the caller primes this generator, so headers only
    reach the client once the first chunk's audio is in hand.
    """
    normalized = str(audio_format).strip().lower()
    header_sent = False
    outcome: str | None = None
    try:
        async for segment in segments:
            rate = int(state.target_rate or state.source_rate or 0)
            payload = await run_in_threadpool(pcm16_frame, segment, rate)
            if not payload:
                # No audio in this segment: never let an empty frame consume the
                # header (it rides with the first real audio, never alone).
                continue
            if not header_sent:
                if normalized == "wav":
                    payload = await run_in_threadpool(wav_stream_header, rate) + payload
                header_sent = True
                # Phase 14-5: the first audio frame is the TTFA observation.
                # Pre-header-priming by design (see streaming_metrics).
                observe_first_frame(time.monotonic() - state.started_at)
                state.first_frame_observed = True
            record_stream_bytes(len(payload))
            yield payload
        outcome = "aborted" if state.aborted else "completed"
    except asyncio.CancelledError:
        # Client disconnect: the frame source aborts its in-flight chunk.
        state.aborted = True
        outcome = "aborted"
        raise
    except GeneratorExit:
        # Closed at a yield (the response task was torn down): same meaning.
        state.aborted = True
        outcome = "aborted"
        raise
    except Exception:
        outcome = "error"
        raise
    finally:
        if outcome is None:  # pragma: no cover - defensive: closed before start
            state.aborted = True
            outcome = "aborted"
        record_stream_outcome(outcome)


def _pcm16_base64(audio, sample_rate: int | None = None) -> str:
    """One segment → base64 s16le PCM (the SSE delta's ``audio`` field).

    :func:`pcm16_frame` is the one canonical quantizer (14-2); base64 is pure
    CPU, so both run together in a worker thread. Module-level so the thread
    gate can monkeypatch ``pcm16_frame`` exactly like the binary encoder's.
    """
    return base64.b64encode(pcm16_frame(audio, sample_rate)).decode("ascii")


async def _encode_sse_frames(
    segments: AsyncIterator[np.ndarray],
    *,
    state: StreamState,
    request_id: str,
    model: str,
    created: int,
    usage=None,
    keepalive_timeout: float | None = None,
) -> AsyncGenerator[str, None]:
    """Turn float32 segments into the SSE envelope's text frames (Phase 14-4).

    Consumes the **same** frame source as the binary envelope
    (:meth:`OpenAIServingTextToSpeech._stream_audio_segments`), so ordering,
    crossfading, per-chunk DC removal, per-segment watermarking and resampling
    are identical by construction — only the framing differs:

    * one ``speech.audio.delta`` per non-empty segment, ``audio`` = base64
      s16le PCM (already at the target rate when ``sample_rate`` was requested)
      and additive ``chunk_index`` (0-based, so a dropped frame is detectable);
    * one terminal ``speech.audio.done`` (``id``/``model``/``created``/
      ``sample_rate``/``format``/``chunks``/``duration_s``/``usage``), where
      ``duration_s`` comes from ``state``'s emitted-sample counter (source-rate
      samples ÷ source rate — never the wire length, which differs once
      resampling is in play) and ``sample_rate`` is the **target** rate;
    * then :data:`~harmonyspeech.endpoints.openai.streaming.SSE_DONE_FRAME`
      (``data: [DONE]``) as the very last line.

    SSE always carries headerless PCM: the delta ``audio`` field is raw s16le
    even when ``format: "wav"`` was requested, so ``done.format`` is ``"pcm"``
    (the binary envelope is where a requested container header lives).

    Terminal semantics — **exactly one** of ``done`` / ``error``:

    * a failure before anything was yielded is re-raised, so the caller can
      still answer with an ordinary JSON error (headers are not committed yet);
    * a failure after at least one frame (a keepalive counts) is reported as
      one ``speech.audio.error`` event and the stream closes — no ``done`` and
      no ``[DONE]`` follow it;
    * a cancellation (client disconnect) is **not** an error: it propagates
      untouched after the frame source has aborted its in-flight chunk.

    Keepalives are a timeout race around the *frame wait* only
    (``asyncio.wait`` on the pending ``__anext__``): the synthesis task is
    never cancelled or re-issued, so a slow chunk costs no CPU and loses no
    frame. ``keepalive_timeout=None`` resolves
    :data:`SSE_KEEPALIVE_TIMEOUT_S` at call time (the test seam).
    """
    timeout = SSE_KEEPALIVE_TIMEOUT_S if keepalive_timeout is None else float(keepalive_timeout)
    timeout = timeout if timeout > 0 else None

    frames = segments.__aiter__()
    pending: asyncio.Task | None = None
    sent = False
    first_audio = False
    chunk_index = 0
    outcome: str | None = None
    try:
        while True:
            if pending is None:
                pending = asyncio.ensure_future(frames.__anext__())
            done, _still_pending = await asyncio.wait({pending}, timeout=timeout)
            if not done:
                # The frame is still being synthesised: keep the connection
                # warm and keep waiting on the *same* task (no busy poll).
                sent = True
                yield keepalive()
                continue
            try:
                segment = pending.result()
            except StopAsyncIteration:
                break
            pending = None
            rate = int(state.target_rate or state.source_rate or 0)
            payload = await run_in_threadpool(_pcm16_base64, segment, rate)
            if not payload:
                # No audio in this segment (the stitcher's first feed can be
                # empty while the carry fills): skip it, never emit an empty
                # delta.
                continue
            sent = True
            if not first_audio:
                # Phase 14-5: TTFA is the first *audio* frame, not a keepalive.
                # Pre-header-priming by design (see streaming_metrics).
                first_audio = True
                observe_first_frame(time.monotonic() - state.started_at)
                state.first_frame_observed = True
            # Count the raw s16le bytes, not the larger base64 event text.
            record_stream_bytes(len(base64.b64decode(payload)))
            yield sse_event(SSE_EVENT_DELTA, delta_payload(payload, chunk_index))
            chunk_index += 1
        outcome = "aborted" if state.aborted else "completed"
    except asyncio.CancelledError:
        # Client disconnect: the frame source aborts its in-flight chunk on the
        # cancellation; cancel is cancel, never a mid-stream error event.
        state.aborted = True
        outcome = "aborted"
        raise
    except GeneratorExit:
        state.aborted = True
        outcome = "aborted"
        raise
    except Exception as exc:
        outcome = "error"
        if not sent:
            # Nothing on the wire yet: the caller can answer with a normal JSON
            # error because no header has been committed.
            raise
        yield sse_event(SSE_EVENT_ERROR, error_payload(exc))
        return
    finally:
        # A generator closed at a yield (early client close) must not leave the
        # frame source's task running: cancel it, let its own cleanup abort the
        # chunk in flight, then close the source.
        if pending is not None and not pending.done():
            pending.cancel()
            with contextlib.suppress(BaseException):
                await pending
        aclose = getattr(frames, "aclose", None)
        if aclose is not None:
            with contextlib.suppress(BaseException):
                await aclose()
        if outcome is None:  # pragma: no cover - defensive: closed before start
            state.aborted = True
            outcome = "aborted"
        record_stream_outcome(outcome)

    yield sse_event(
        SSE_EVENT_DONE,
        done_payload(
            id=request_id,
            model=model,
            created=created,
            sample_rate=int(state.target_rate or state.source_rate or 0),
            # SSE audio is headerless PCM regardless of the requested container.
            format="pcm",
            # Number of delta frames the client actually received (its
            # chunk_index range is 0..chunks-1), not the planned synthesis
            # chunk count: the stitcher's flush tail is one more frame.
            chunks=chunk_index,
            duration_s=state.duration_s,
            usage=usage,
        ),
    )
    yield SSE_DONE_FRAME


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
    ) -> ErrorResponse | StreamingAudioResponse | TextToSpeechResponse:

        # Phase 13-1: resolve a standard voice pack name ONCE per HTTP request,
        # before the model check and before ``from_wire_audio``/
        # ``_resolve_chunks``. The resolved pack WAV becomes the request's
        # ``input_audio`` (+ cloning mode), so the single-shot and the 12-2
        # chunked clone paths see exactly a client-supplied reference, and the
        # 7-2 embedding cache keys on the WAV content hash unchanged.
        voice_lookup = self._lookup_standard_voice(request)
        if voice_lookup.error is not None:
            return self.create_error_response(voice_lookup.error)

        # ``_check_model`` is shared by every serving module (CRITICAL blast
        # radius), so it is not changed: the pack voice is validated through an
        # effective copy of the request that carries the reference and cloning
        # mode the serving layer will actually dispatch.
        validation_request = request
        if voice_lookup.resolution is not None:
            payload = standard_voice_wire_payload(voice_lookup.resolution.name)
            if payload is None:  # pragma: no cover - resolution validated the path
                return self.create_error_response(
                    f"Standard voice '{voice_lookup.resolution.name}' is missing from the "
                    "voice pack installation."
                )
            validation_request = request.model_copy(
                update={
                    "input_audio": payload,
                    "mode": "voice_cloning",
                    "voice": voice_lookup.native_default,
                }
            )

        error_check_model = await self._check_model(validation_request)
        if error_check_model is not None:
            return error_check_model

        # Validate output_options at admission: never start synthesis for a
        # format/sample-rate combination this runtime cannot produce.
        error_check_options = validate_output_options(request.output_options)
        if error_check_options is not None:
            return self.create_error_response(error_check_options)

        # Phase 14-1: streaming admission. Every restriction fails here, before
        # any synthesis, with the usual JSON 400 shape. ``stream_format`` selects
        # streaming on its own; the deprecated ``stream: true`` alias maps to
        # "audio" (see ``streaming.resolve_stream_format``).
        error_streaming = self._validate_streaming_admission(request)
        if error_streaming is not None:
            return self.create_error_response(error_streaming)

        # Phase 14-3/14-4: envelope selection. ``stream_format`` selects
        # streaming on its own; the deprecated ``stream: true`` alias maps to
        # "audio" (see ``streaming.resolve_stream_format``). Both envelopes
        # consume the same frame source and are opened below, once the request
        # data (and therefore the chunk plan) exists.
        stream_envelope = resolve_stream_format(request.output_options)

        # TODO: Basic checks for TTS Generate request

        request_id = f"tts-{random_uuid()}"

        request_data = TextToSpeechRequestInput.from_openai(request_id=request_id, request=request)
        if voice_lookup.resolution is not None:
            apply_error = self._apply_standard_voice(
                request_data, voice_lookup.resolution, voice_lookup.native_default
            )
            if apply_error is not None:
                return self.create_error_response(apply_error)
        # Boundary decode-once: the reference audio becomes an artifact now, so
        # no pipeline step re-reads the base64 field.
        artifacts = from_wire_audio(request_data)

        # Long-form dispatch (Phase 12-2). A short input (0/1 chunks) returns
        # None and keeps the single-shot path below byte-identical.
        try:
            chunks = self._resolve_chunks(request_data)
        except ValueError as e:
            return self.create_error_response(str(e))

        # Phase 14-3/14-4: the streaming envelopes. They are chosen before the
        # buffered dispatch because even a one-chunk request must stream (its
        # own framing); inside, a 0/1-chunk list mirrors the buffered
        # single-shot path (no stitcher, byte-identical samples). Both consume
        # the one frame source ``_stream_audio_segments``.
        if stream_envelope == "audio":
            return await self._open_audio_stream(
                request=request,
                raw_request=raw_request,
                request_id=request_id,
                request_data=request_data,
                artifacts=artifacts,
                chunks=chunks,
            )

        if stream_envelope == "sse":
            return await self._open_sse_stream(
                request=request,
                raw_request=raw_request,
                request_id=request_id,
                request_data=request_data,
                artifacts=artifacts,
                chunks=chunks,
            )

        if chunks is not None:
            return await self._chunked(
                request=request,
                raw_request=raw_request,
                request_id=request_id,
                request_data=request_data,
                artifacts=artifacts,
                chunks=chunks,
            )

        return await self._single(
            request=request,
            raw_request=raw_request,
            request_id=request_id,
            request_data=request_data,
            artifacts=artifacts,
        )

    async def _single(
        self,
        *,
        request: TextToSpeechRequest,
        raw_request: Request,
        request_id: str,
        request_data: TextToSpeechRequestInput,
        artifacts: dict,
    ) -> ErrorResponse | TextToSpeechResponse:
        """The pre-12 single-shot path, unchanged (byte-identity gate)."""
        result_generator = self.engine.generate(
            request_id=request_id,
            request_data=request_data,
            artifacts=artifacts,
        )

        try:
            return await self.text_to_speech_full_generator(
                request, raw_request, result_generator, request_id, request_data=request_data
            )
        except ValueError as e:
            # TODO: Use an aphrodite-specific Validation Error
            return self.create_error_response(str(e))

    # ------------------------------------------------------------------
    # Long-form chunking (Phase 12-2)
    # ------------------------------------------------------------------
    def _request_model_types(self, request_data) -> set[str]:
        """Return the configured model types backing this request.

        Mirrors :meth:`_request_uses_native_speed`: a pipeline sentinel id is
        resolved through its spec's steps (intersected with what is actually
        configured), while a concrete configured model name is looked up by
        ``config.name``. Returns an empty set when nothing can be resolved
        (e.g. a bare serving object in a unit test), so callers fall back to
        the global defaults.
        """
        requested = getattr(request_data, "requested_model", None)
        model_name = getattr(request_data, "model", None) or requested
        return request_model_types(
            self._engine_model_configs(),
            requested,
            model_name,
            only_configured=True,
        )

    def _engine_model_configs(self) -> list[ModelConfig]:
        """Return the engine's configured models (empty for test doubles)."""
        engine = getattr(self, "engine", None)
        engine = getattr(engine, "engine", None) or engine
        return list(getattr(engine, "model_configs", None) or [])

    def _standard_voice_pack_enabled(self) -> bool:
        """Global ``standard_voice_pack`` switch (default on)."""
        engine = getattr(self, "engine", None)
        engine = getattr(engine, "engine", None) or engine
        return bool(getattr(engine, "standard_voice_pack", True))

    def _lookup_standard_voice(self, request):
        """Resolve the request's ``voice`` as a standard pack name (Phase 13-1).

        Returns a :class:`~harmonyspeech.modeling.voice_catalog.StandardVoiceLookup`.
        A non-pack name yields an empty lookup, so native voices and the
        pre-13 paths are untouched.
        """
        model = getattr(request, "model", "") or ""
        return lookup_standard_voice(
            self._engine_model_configs(),
            model,
            getattr(request, "voice", None),
            language_id=getattr(request, "language", None),
            enabled=self._standard_voice_pack_enabled(),
            model_exists=self.get_model_by_id(model) is not None,
            has_explicit_reference=bool(
                getattr(request, "input_audio", None)
                or getattr(request, "input_embedding", None)
                or getattr(request, "input_vad_data", None)
            ),
        )

    @staticmethod
    def _apply_standard_voice(request_data, resolution, native_default) -> str | None:
        """Turn a resolved pack voice into the request's cloning reference.

        The pack WAV is injected as ``input_audio`` (the same wire form a client
        supplies), the mode is derived as ``voice_cloning``, and any native
        voice id is replaced with the model's default source voice. Returns an
        error string only if the pack file vanished between listing and reading.
        """
        payload = standard_voice_wire_payload(resolution.name)
        if payload is None:  # pragma: no cover - resolution validated the path
            return (
                f"Standard voice '{resolution.name}' is missing from the voice pack "
                "installation."
            )
        request_data.input_audio = payload
        request_data.input_embedding = None
        request_data.mode = "voice_cloning"
        request_data.voice_id = native_default
        return None

    def _resolve_default_chunk_size(self, request_data) -> int:
        """Return the default character budget for this request's family.

        Phase 12-7/12-8: every family uses the unified
        :data:`DEFAULT_CHUNK_SIZE` (300). Reached only when the client left
        ``chunk_size`` unset — an explicit value always wins.
        """
        return DEFAULT_CHUNK_SIZE

    def _resolve_chunks(self, request_data: TextToSpeechRequestInput) -> list[str] | None:
        """Return the chunk list for a long-form request, or ``None`` to skip.

        ``None`` covers both "chunking not requested" and the degenerate
        0/1-chunk case, which is the byte-identity gate: the chunker pipeline is
        not entered at all, so a short request is indistinguishable from the
        pre-12 path.
        """
        options = getattr(request_data, "generation_options", None)
        if options is None:
            return None

        split_text = bool(_option(options, "split_text", False))
        chunk_size = _option(options, "chunk_size", None)
        if not split_text and chunk_size is None:
            return None

        if chunk_size is not None:
            if int(chunk_size) <= 0:
                raise ValueError(f"chunk_size must be positive, got {chunk_size!r}")
            budget = int(chunk_size)
        else:
            budget = self._resolve_default_chunk_size(request_data)

        crossfade_ms = _option(options, "crossfade_ms", None)
        pause_ms = _option(options, "sentence_pause_ms", None)
        if crossfade_ms is not None and int(crossfade_ms) < 0:
            raise ValueError(f"crossfade_ms must be >= 0, got {crossfade_ms!r}")
        if pause_ms is not None and int(pause_ms) < 0:
            raise ValueError(f"sentence_pause_ms must be >= 0, got {pause_ms!r}")

        chunks = split_text_into_chunks(
            getattr(request_data, "input_text", "") or "",
            chunk_size=budget,
            language=getattr(request_data, "language_id", None),
        )
        return chunks if len(chunks) > 1 else None

    async def _chunked(
        self,
        *,
        request: TextToSpeechRequest,
        raw_request: Request,
        request_id: str,
        request_data: TextToSpeechRequestInput,
        artifacts: dict,
        chunks: list[str],
    ) -> ErrorResponse | TextToSpeechResponse:
        """The Phase 12-2 buffered long-form path, unchanged (byte-identity gate)."""
        options = getattr(request_data, "generation_options", None)
        # Phase 12-5 tuned defaults (40 ms fade / 120 ms pause) apply only here:
        # this method is reached only when the request produced 2+ chunks, and
        # an explicit client value (including 0) always wins.
        crossfade_ms, pause_ms = _resolve_join_timing(options)

        # Per-chunk deadline override (Phase 12-2 follow-up, 2026-11-09). Chunk
        # sub-requests inherit the engine's global per-request timeout, but a
        # chunk can legitimately wait for its siblings ahead of it in the queue
        # (e.g. a max_batch_size=1 "loop" executor under concurrent load).
        # Queueing delay is not synthesis failure, so every chunk gets a
        # deadline that covers the worst-case wait of this request's own chunks
        # plus its own synthesis (``chunk_count + 1``). Single-shot requests
        # never reach this method, so their behaviour is untouched.
        chunk_timeout_s = self._chunk_request_timeout_s(len(chunks))

        if await raw_request.is_disconnected():
            return self.create_error_response("Client disconnected")

        chunk_artifacts = dict(artifacts or {})
        clear_clone_fields = False

        has_audio = chunk_artifacts.get("input_audio") is not None or getattr(
            request_data, "input_audio", None
        )
        has_embedding = chunk_artifacts.get("input_embedding") is not None or getattr(
            request_data, "input_embedding", None
        )

        # Clone mode: condition on the reference exactly the way the
        # single-shot path does for this request's pipeline (see
        # ``_reference_conditioning_mode``), once per HTTP request, and reuse
        # that conditioning for every chunk. Re-embedding per chunk is the
        # silent failure mode this branch prevents; pre-embedding for an
        # executor that never reads the reference is the 2026-11-09 benchmark
        # failure this branch also prevents.
        conditioning_mode = "none"
        if has_audio and not has_embedding:
            conditioning_mode = self._reference_conditioning_mode(request_data)
        if conditioning_mode == "embed":
            embedding, error = await self._resolve_reference_embedding(
                request_id,
                request_data,
                chunk_artifacts,
                request_timeout_s=chunk_timeout_s,
            )
            if error is not None:
                return self.create_error_response(error)
            chunk_artifacts["input_embedding"] = embedding
            has_embedding = True
        # ``"none"``: the single-shot path never conditions on ``input_audio``
        # for this executor, so the chunks carry the request fields unchanged
        # (N chunks behave exactly like N single-shot requests would).

        if has_audio and has_embedding:
            # The embedding supersedes the reference clip; forwarding the clip
            # would make each chunk re-run the embed step.
            chunk_artifacts.pop("input_audio", None)
            chunk_artifacts.pop("input_vad_data", None)
            clear_clone_fields = True

        chunk_requests: list[tuple[int, str, TextToSpeechRequestInput]] = []
        for index, text in enumerate(chunks):
            chunk_id = f"{request_id}-chunk-{index}"
            chunk_options = _clone_generation_options(
                options,
                # Chunk sub-requests are ordinary single-shot requests; the
                # serving layer already did the splitting.
                split_text=False,
                chunk_size=None,
                crossfade_ms=None,
                sentence_pause_ms=None,
                **_chunk_seed_update(options, index),
            )
            chunk_data = self._build_chunk_request(
                request_data,
                text,
                chunk_options,
                clear_clone_fields=clear_clone_fields,
                chunk_id=chunk_id,
                request_timeout_s=chunk_timeout_s,
            )
            chunk_requests.append((index, chunk_id, chunk_data))

        # Submit every chunk sub-request before consuming any stream. Each
        # ``add_request`` completes without yielding to the event loop, so all
        # chunk requests are queued before the engine's first step: they are
        # ordinary admitted requests and can co-batch by themselves (Phase 12-2
        # concurrency rule), and the stitch below is by chunk index.
        streams = [
            (index, await self._submit_chunk(chunk_id, chunk_data, chunk_artifacts))
            for index, chunk_id, chunk_data in chunk_requests
        ]
        results = await asyncio.gather(
            *[self._consume_stream(index, stream) for index, stream in streams],
            return_exceptions=True,
        )

        if await raw_request.is_disconnected():
            return self.create_error_response("Client disconnected")

        ordered: dict[int, TextToSpeechRequestOutput] = {}
        errors: list[str] = []
        for item in results:
            if isinstance(item, Exception):
                errors.append(str(item))
                continue
            index, final = item
            if final is None:
                errors.append(f"chunk {index} produced no output")
            elif getattr(final, "finish_reason", None) == "error":
                errors.append(getattr(final, "error", None) or f"chunk {index} failed")
            elif getattr(final, "audio_artifact", None) is None:
                errors.append(f"chunk {index} produced no audio")
            else:
                ordered[index] = final

        if errors:
            # All-or-nothing: never return partial audio with a 200.
            return self.create_error_response("Long-form synthesis failed: " + "; ".join(errors))
        if len(ordered) != len(chunks):
            return self.create_error_response(
                f"Long-form synthesis produced {len(ordered)} of {len(chunks)} chunks"
            )

        outputs = [ordered[index] for index in range(len(chunks))]
        sample_rate = int(outputs[0].audio_artifact.sample_rate)
        waveforms = [output.audio_artifact.numpy for output in outputs]

        remove_dc, trim, fix_internal = _output_filter_flags(options)
        if remove_dc:
            # Per chunk, before stitching: each synthesis run can carry its own
            # DC offset, and a whole-buffer mean would smear the chunk joins.
            waveforms = [remove_dc_offset(waveform) for waveform in waveforms]

        stitched = crossfade_stitch(
            waveforms,
            crossfade_ms=crossfade_ms,
            sentence_pause_ms=pause_ms,
            sample_rate=sample_rate,
        )

        # Trim / internal-silence run on the *final* stitched buffer (Phase
        # 12-3): a chunk boundary must never be mistaken for lead/trail silence.
        # Ordering pinned by tests: stitch -> trim -> fix internal -> speed ->
        # watermark (the watermark stays the last, once-per-request step).
        if trim:
            stitched = trim_silence(stitched, sample_rate)
        if fix_internal:
            stitched = fix_internal_silence(stitched, sample_rate)

        speed = _option(options, "speed", None)
        if (
            speed is not None
            and float(speed) > 0
            and float(speed) != 1.0
            and not self._request_uses_native_speed(request_data)
        ):
            # Whole-buffer, not per chunk: rate > 1 is faster/shorter, matching
            # the native families' ``length_scale = 1 / speed``.
            stitched = time_stretch(stitched, float(speed), sample_rate)

        # Exactly one watermark, on the final buffer. Chunk synthesis ran with
        # the watermark suppressed (see ``_build_chunk_request``).
        final_model_name = getattr(chunk_requests[0][2], "model", None)
        stitched = self._apply_final_watermark(final_model_name, stitched, sample_rate)

        final_output = TextToSpeechRequestOutput(
            request_id=request_id,
            text=getattr(request_data, "input_text", "") or "",
            output="",
            finish_reason="stop",
            audio_artifact=AudioArtifact.from_numpy(stitched, sample_rate),
        )
        encoded = to_wire_audio(final_output, request.output_options)
        return TextToSpeechResponse(
            id=request_id,
            created=int(time.time()),
            model=request.model,
            data=encoded.data,
            sample_rate=encoded.sample_rate,
        )

    def _build_chunk_request(
        self,
        source: TextToSpeechRequestInput,
        text: str,
        options,
        *,
        clear_clone_fields: bool,
        chunk_id: str,
        request_timeout_s: float | None = None,
    ) -> TextToSpeechRequestInput:
        return TextToSpeechRequestInput(
            request_id=chunk_id,
            requested_model=getattr(source, "requested_model", ""),
            # Keep the original model name: for a concrete executor this is the
            # routing target; for a pipeline family id the engine overwrites it
            # with the resolved step's executor name.
            model=getattr(source, "model", ""),
            input_text=text,
            mode=getattr(source, "mode", ""),
            language_id=getattr(source, "language_id", None),
            voice_id=getattr(source, "voice_id", None),
            input_audio=None if clear_clone_fields else getattr(source, "input_audio", None),
            input_vad_mode=getattr(source, "input_vad_mode", None),
            input_vad_data=None if clear_clone_fields else getattr(source, "input_vad_data", None),
            input_embedding=getattr(source, "input_embedding", None),
            generation_options=options,
            output_options=getattr(source, "output_options", None),
            post_generation_filters=getattr(source, "post_generation_filters", None),
            # The serving layer watermarks the stitched buffer once.
            suppress_watermark=True,
            # Extended deadline covering this chunk's worst-case queue wait
            # (Phase 12-2 follow-up); None keeps the engine-wide timeout.
            request_timeout_s=request_timeout_s,
        )

    def _engine_request_timeout_s(self) -> float | None:
        """Return the engine's configured per-request deadline (seconds)."""
        engine = getattr(self.engine, "engine", None) or self.engine
        return getattr(engine, "request_timeout_s", None)

    def _chunk_request_timeout_s(self, chunk_count: int) -> float | None:
        """Return the deadline for a long-form chunk sub-request.

        A chunk may wait for every sibling of its own request ahead of it in
        the queue in the worst case (``max_batch_size=1`` "loop" executors under
        concurrent load), and queueing delay is not synthesis failure; ``+1``
        covers the chunk's own synthesis. A disabled engine-wide timeout
        (``None``/0) stays disabled: returning ``None`` means "inherit", which
        the engine reads as the same no-deadline setting.
        """
        base = self._engine_request_timeout_s()
        if not base:
            return None
        return base * (chunk_count + 1)

    def _configured_model(self, name: str | None):
        """Return the configured model with ``name`` (or ``None``)."""
        if not name:
            return None
        engine = getattr(self.engine, "engine", None) or self.engine
        for config in getattr(engine, "model_configs", None) or []:
            if getattr(config, "name", None) == name:
                return config
        return None

    @staticmethod
    def _spec_produces_embedding(spec) -> bool:
        """True when the pipeline has a step that produces a speaker embedding."""
        return any(
            artifact == "embedding_artifact"
            for step in getattr(spec, "steps", ()) or ()
            for artifact, _field in step.produces
        )

    def _reference_conditioning_mode(self, request_data: TextToSpeechRequestInput) -> str:
        """How the *single-shot* path conditions on ``input_audio``: embed or none.

        ``"embed"`` — one standalone embedding sub-request reproduces the
        single-shot conditioning exactly:

        * the request names a **pipeline id** whose steps include an embedding
          producer (OpenVoice v1/v2 ``vad → embed`` tone-converter encoder,
          HarmonySpeech ``encoder``, Chatterbox ``embed``). The sub-request is a
          ``SpeechEmbeddingRequestInput``, which every such pipeline's entry
          rules route to that step (Phase 3-4 registry);
        * the request names a **Chatterbox executor directly** (no pipeline id):
          its ``prepare_inputs`` accepts a standalone
          ``SpeechEmbeddingRequestInput`` and the handler computes the same
          ``prepare_conditionals`` the single-shot path computes inline
          (``STANDALONE_EMBED_MODEL_TYPES``).

        ``"none"`` — the single-shot path never reads ``input_audio`` for this
        request: it names a concrete synthesizer (MeloTTS / OpenVoice V1
        synthesizer / KittenTTS / HarmonySpeech synthesizer) whose
        ``prepare_inputs`` rejects a ``SpeechEmbeddingRequestInput`` outright.
        Building one would fail the whole request with a request-type error
        (the 2026-11-09 Phase 12-4 benchmark failure) and, even if it did not,
        would apply conditioning the single-shot path never applies. The
        chunked path therefore forwards the request fields to every chunk
        unchanged — N chunks behave exactly like N single-shot requests.
        """
        spec = get_spec(getattr(request_data, "requested_model", None))
        if spec is not None and self._spec_produces_embedding(spec):
            return "embed"
        config = self._configured_model(getattr(request_data, "model", None))
        if getattr(config, "model_type", None) in STANDALONE_EMBED_MODEL_TYPES:
            return "embed"
        return "none"

    async def _resolve_reference_embedding(
        self,
        request_id: str,
        source: TextToSpeechRequestInput,
        artifacts: dict,
        *,
        request_timeout_s: float | None = None,
    ) -> tuple[object | None, str | None]:
        """Run the pipeline's embed step once for a clone request."""
        embed_id = f"{request_id}-embed"
        embed_request = SpeechEmbeddingRequestInput(
            request_id=embed_id,
            requested_model=getattr(source, "requested_model", ""),
            model=getattr(source, "model", ""),
            input_audio=None,
            input_vad_mode=getattr(source, "input_vad_mode", None),
            input_vad_data=None,
            generation_options=getattr(source, "generation_options", None),
            # The one-time embed sub-request queues with the chunk burst too;
            # it inherits the same extended deadline.
            request_timeout_s=request_timeout_s,
        )
        _index, final_output = await self._consume_stream(
            embed_id, await self._submit_chunk(embed_id, embed_request, artifacts)
        )
        if final_output is None:
            return None, "Reference embedding step produced no output"
        if getattr(final_output, "finish_reason", None) == "error":
            return None, getattr(final_output, "error", None) or "Reference embedding step failed"
        embedding = getattr(final_output, "embedding_artifact", None)
        if embedding is None:
            return None, "Reference embedding step produced no embedding"
        return embedding, None

    async def _submit_chunk(self, request_id: str, request_data, artifacts: dict):
        """Add one chunk request to the engine and return its output stream."""
        return await self.engine.add_request(
            request_id=request_id,
            request_data=request_data,
            artifacts=dict(artifacts),
        )

    async def _consume_stream(self, index: int, stream) -> tuple[int, RequestOutput | None]:
        final: RequestOutput | None = None
        async for output in stream:
            final = output
        return index, final

    # ------------------------------------------------------------------
    # Streaming envelopes (Phase 14-3 binary / Phase 14-4 SSE)
    # ------------------------------------------------------------------
    def _stream_chunk_request_timeout_s(self) -> float | None:
        """Deadline for one **sequentially submitted** streaming chunk.

        The buffered chunker stamps ``base * (chunk_count + 1)`` on every chunk
        (:meth:`_chunk_request_timeout_s`) because its whole burst queues at
        once. Sequential streaming never has its own siblings ahead of it in the
        queue — a chunk waits only for itself plus whatever other requests are
        already ahead — so the engine-wide ``base`` is the MVP rule. Recorded
        next to the 14-5 TTFA numbers for review.
        """
        base = self._engine_request_timeout_s()
        if not base:
            return None
        return base

    async def _open_audio_stream(
        self,
        *,
        request: TextToSpeechRequest,
        raw_request: Request,
        request_id: str,
        request_data: TextToSpeechRequestInput,
        artifacts: dict,
        chunks: list[str] | None,
    ) -> ErrorResponse | StreamingAudioResponse:
        """Start the binary envelope, priming the first frame before headers.

        Headers must not be committed before the first chunk's audio is in hand,
        otherwise a first-chunk failure would be indistinguishable from a
        successful stream that got truncated. Starlette's ``StreamingResponse``
        sends ``http.response.start`` before iterating, so this method pulls the
        first frame itself: a pre-first-byte :class:`StreamSynthesisError`
        becomes an ordinary JSON error response (normal HTTP status), and only
        then is the response object built. Failures *after* the first byte
        truncate the body — the documented weakness of binary streaming, fixed
        by the SSE envelope in 14-4.
        """
        output_options = getattr(request, "output_options", None)
        audio_format = normalize_format(getattr(output_options, "format", None))
        requested_rate = getattr(output_options, "sample_rate", None)

        state = StreamState()
        segments = self._stream_audio_segments(
            raw_request=raw_request,
            request_id=request_id,
            request_data=request_data,
            artifacts=artifacts,
            chunks=chunks,
            state=state,
            requested_rate=requested_rate,
        )
        body = _encode_stream_frames(segments, audio_format=audio_format, state=state)

        try:
            first = await anext(body)
        except StopAsyncIteration:
            return self.create_error_response(
                "Streaming synthesis produced no audio.", status_code=HTTPStatus.INTERNAL_SERVER_ERROR
            )
        except StreamSynthesisError as exc:
            return self.create_error_response(str(exc))

        headers = streaming_response_headers(
            audio_format=audio_format,
            sample_rate=int(state.target_rate or state.source_rate or 0),
            request_id=request_id,
        )
        return StreamingAudioResponse(
            body=_prepend_frames(first, body),
            media_type=_STREAM_MEDIA_TYPES.get(audio_format, "application/octet-stream"),
            headers=headers,
            state=state,
        )

    async def _open_sse_stream(
        self,
        *,
        request: TextToSpeechRequest,
        raw_request: Request,
        request_id: str,
        request_data: TextToSpeechRequestInput,
        artifacts: dict,
        chunks: list[str] | None,
    ) -> ErrorResponse | StreamingAudioResponse:
        """Start the SSE envelope over the shared frame source (Phase 14-4).

        Priming follows 14-3: the first frame of the event stream is pulled
        here, so a failure before **anything** reaches the wire becomes an
        ordinary JSON error response (normal HTTP status) and no header is ever
        committed for a stream that produced nothing. The first frame is
        normally the first ``speech.audio.delta``; when the first chunk takes
        longer than :data:`SSE_KEEPALIVE_TIMEOUT_S` it is a ``: keepalive``
        comment instead — the connection is warm and the response is committed,
        so from that point on failures are reported as a
        ``speech.audio.error`` event (still exactly one terminal event).

        Headers: the full 14-1 streaming set, with ``X-Audio-Format: pcm``
        always — SSE delta audio is headerless s16le even when
        ``format: "wav"`` was requested, so none of the seven headers is
        WAV-specific. ``Content-Type`` is
        :data:`~harmonyspeech.endpoints.openai.streaming.SSE_MEDIA_TYPE`; no
        compression is added here (``Cache-Control: no-cache, no-transform`` and
        ``X-Accel-Buffering: no`` already tell intermediaries to pass the body
        through untouched — gzip would break incremental delivery).
        """
        output_options = getattr(request, "output_options", None)
        requested_rate = getattr(output_options, "sample_rate", None)

        state = StreamState()
        segments = self._stream_audio_segments(
            raw_request=raw_request,
            request_id=request_id,
            request_data=request_data,
            artifacts=artifacts,
            chunks=chunks,
            state=state,
            requested_rate=requested_rate,
        )
        usage = {"input_characters": len(getattr(request_data, "input_text", "") or "")}
        body = _encode_sse_frames(
            segments,
            state=state,
            request_id=request_id,
            model=request.model,
            created=int(time.time()),
            usage=usage,
        )

        try:
            first = await anext(body)
        except StopAsyncIteration:
            return self.create_error_response(
                "Streaming synthesis produced no audio.", status_code=HTTPStatus.INTERNAL_SERVER_ERROR
            )
        except StreamSynthesisError as exc:
            return self.create_error_response(str(exc))

        headers = streaming_response_headers(
            # SSE carries headerless PCM in every delta; the requested container
            # (wav/pcm) only selects the binary envelope's framing.
            audio_format="pcm",
            sample_rate=int(state.target_rate or state.source_rate or 0),
            request_id=request_id,
        )
        return StreamingAudioResponse(
            body=_prepend_frames(first, body),
            media_type=SSE_MEDIA_TYPE,
            headers=headers,
            state=state,
        )

    async def _stream_audio_segments(
        self,
        *,
        raw_request: Request,
        request_id: str,
        request_data: TextToSpeechRequestInput,
        artifacts: dict,
        chunks: list[str] | None,
        state: StreamState,
        requested_rate: int | None,
    ) -> AsyncGenerator[np.ndarray, None]:
        """The shared frame source: engine chunks → float32 segments at the wire rate.

        Sequential chunk submission (submit chunk *k*, await it, emit, then
        *k+1*): ordering holds by construction and memory stays bounded to one
        chunk. Pipelining is a measured later optimisation, not MVP.

        Per chunk, in the buffered chunked path's order (12-2, delta item 5):

        1. ``remove_dc_offset`` per chunk **before** the stitcher (the one
           post-option that composes; a whole-buffer mean would smear joins);
        2. ``CrossfadeStitcher.feed`` — it conditions edges and applies the
           two-sided pause fade **internally** (never pre- or double-condition);
           the 40 ms carry (960 samples @ 24 kHz) is held back, so the first
           frame trails by exactly that hold-back;
        3. one perth ``apply_watermark`` call per **emitted segment** (owner
           decision 2026-11-09; the buffered path marks the whole clip once, so
           the two diverge by design and the 14-5 equality axis is pre-watermark);
        4. the stateful :class:`StreamResampler` when ``sample_rate`` was asked
           for (continuous across chunk boundaries, never per-chunk one-shot).

        Then the final ``stitcher.flush()`` tail (the last chunk's conditioned
        tail lives only there) and the resampler's filter tail are emitted
        through the same chain.

        A 0/1-chunk request bypasses the stitcher entirely: that mirrors
        ``crossfade_stitch``'s single-chunk byte-identity rule and keeps the
        streamed samples identical to the buffered single-shot path.

        Disconnect is the correctness item: Starlette cancels the response task
        (and therefore this generator) on client disconnect. The in-flight chunk
        is aborted on ``CancelledError`` **and** from the ``finally``, so a
        closed tab stops synthesising the remaining chunks — they are never even
        submitted (sequential submission).
        """
        options = getattr(request_data, "generation_options", None)
        chunk_texts = list(chunks) if chunks else [getattr(request_data, "input_text", "") or ""]
        state.chunks = len(chunk_texts)

        crossfade_ms, pause_ms = _resolve_join_timing(options)
        remove_dc, _trim, _fix_internal = _output_filter_flags(options)
        # One shared deadline rule for the embed sub-request and every chunk.
        chunk_timeout_s = self._stream_chunk_request_timeout_s()

        chunk_artifacts = dict(artifacts or {})
        stitcher: CrossfadeStitcher | None = None
        resampler: StreamResampler | None = None
        watermark_model: str | None = None
        in_flight: str | None = None
        client_gone = False

        try:
            # Clone conditioning runs once per HTTP request and is itself an
            # engine sub-request (``-embed``); a disconnect while it runs must
            # abort it too, hence the pre-set in-flight id (aborting an unknown
            # id is a documented no-op).
            in_flight = f"{request_id}-embed"
            clear_clone_fields, clone_error = await self._stream_clone_artifacts(
                request_id=request_id,
                request_data=request_data,
                chunk_artifacts=chunk_artifacts,
                request_timeout_s=chunk_timeout_s,
            )
            in_flight = None
            if clone_error is not None:
                raise StreamSynthesisError(clone_error)

            for index, text in enumerate(chunk_texts):
                if await raw_request.is_disconnected():
                    # Pre-first-byte: an honest HTTP error. Post-first-byte the
                    # client is gone, so stop submitting further chunks and do
                    # not bother flushing the stitcher/resampler tails.
                    if state.segments == 0:
                        raise StreamSynthesisError("Client disconnected")
                    client_gone = True
                    # Phase 14-5: the stream ended early. Additive flag only —
                    # no envelope bytes change — so the metrics can classify a
                    # polled disconnect as `aborted` (14-4's note).
                    state.aborted = True
                    break

                chunk_id = f"{request_id}-chunk-{index}"
                chunk_options = _clone_generation_options(
                    options,
                    split_text=False,
                    chunk_size=None,
                    crossfade_ms=None,
                    sentence_pause_ms=None,
                    **_chunk_seed_update(options, index),
                )
                chunk_data = self._build_chunk_request(
                    request_data,
                    text,
                    chunk_options,
                    clear_clone_fields=clear_clone_fields,
                    chunk_id=chunk_id,
                    # Sequential submission: base deadline, not base x (n + 1).
                    request_timeout_s=chunk_timeout_s,
                )
                if watermark_model is None:
                    watermark_model = getattr(chunk_data, "model", None)

                # Mark before submitting: a cancellation during add_request must
                # still abort the chunk it was about to hand to the engine.
                in_flight = chunk_id
                stream = await self._submit_chunk(chunk_id, chunk_data, chunk_artifacts)
                final: RequestOutput | None = None
                async for output in stream:
                    final = output
                in_flight = None

                if final is None:
                    raise StreamSynthesisError(f"chunk {index} produced no output")
                if getattr(final, "finish_reason", None) == "error":
                    raise StreamSynthesisError(
                        getattr(final, "error", None) or f"chunk {index} failed"
                    )

                waveform, source_rate = await run_in_threadpool(self._result_waveform, final)
                if waveform.size == 0:
                    raise StreamSynthesisError(f"chunk {index} produced no audio")

                if state.source_rate is None:
                    state.source_rate = int(source_rate)
                    state.target_rate = (
                        int(requested_rate) if requested_rate else state.source_rate
                    )
                    if len(chunk_texts) > 1:
                        stitcher = CrossfadeStitcher(
                            crossfade_ms=crossfade_ms,
                            sentence_pause_ms=pause_ms,
                            sample_rate=state.source_rate,
                        )
                    if state.target_rate != state.source_rate:
                        resampler = StreamResampler(state.source_rate, state.target_rate)
                elif int(source_rate) != state.source_rate:
                    raise StreamSynthesisError(
                        f"chunk {index} reported sample rate {source_rate} Hz; the stream "
                        f"started at {state.source_rate} Hz"
                    )

                segment, source_samples = await run_in_threadpool(
                    self._emit_stream_segment,
                    waveform,
                    stitcher=stitcher,
                    resampler=resampler,
                    remove_dc=remove_dc,
                    model_name=watermark_model,
                    sample_rate=state.source_rate,
                )
                state.emitted_samples += source_samples
                if segment.size:
                    state.segments += 1
                    yield segment

            if stitcher is not None and not client_gone:
                # The last chunk's conditioned tail is only in flush(): the
                # stitcher never materialises a trailing pause.
                tail, source_samples = await run_in_threadpool(
                    self._finish_stream_segment,
                    stitcher.flush(),
                    resampler=resampler,
                    model_name=watermark_model,
                    sample_rate=state.source_rate,
                )
                state.emitted_samples += source_samples
                if tail.size:
                    state.segments += 1
                    yield tail

            if resampler is not None and not client_gone:
                # The resampler's filter tail: a continuation of already-marked
                # audio, so it is not watermarked again.
                resampler_tail = await run_in_threadpool(resampler.flush)
                if resampler_tail.size:
                    state.segments += 1
                    yield np.ascontiguousarray(resampler_tail, dtype=np.float32)
        except asyncio.CancelledError:
            # Starlette cancels the response task on client disconnect (anyio
            # task group under ASGI spec < 2.4; newer servers end the response
            # with an exception and rely on the ``finally`` below). Abort the
            # chunk that is actually in flight, then let the cancellation
            # propagate.
            state.aborted = True
            if in_flight is not None:
                await self._abort_stream_request(in_flight)
                in_flight = None
            raise
        finally:
            # Belt and braces: a generator closed at a yield (early client
            # close on a newer server) still aborts whatever chunk it owns.
            if in_flight is not None:
                state.aborted = True
                await self._abort_stream_request(in_flight)

    async def _abort_stream_request(self, request_id: str) -> None:
        """Abort one in-flight chunk without masking the reason we got here."""
        try:
            await self.engine.abort(request_id)
        except Exception as exc:  # pragma: no cover - engine already torn down
            logger.warning(f"Failed to abort streaming chunk {request_id}: {exc}")

    async def _stream_clone_artifacts(
        self,
        *,
        request_id: str,
        request_data: TextToSpeechRequestInput,
        chunk_artifacts: dict,
        request_timeout_s: float | None,
    ) -> tuple[bool, str | None]:
        """Clone conditioning for the streaming path, once per HTTP request.

        Same rule as the buffered chunked path (see :meth:`_chunked`): condition
        on the reference exactly the way the single-shot path does for this
        request's pipeline — a standalone embedding step where one exists, none
        where the single-shot path never reads the reference. Returns
        ``(clear_clone_fields, error)``.
        """
        has_audio = chunk_artifacts.get("input_audio") is not None or getattr(
            request_data, "input_audio", None
        )
        has_embedding = chunk_artifacts.get("input_embedding") is not None or getattr(
            request_data, "input_embedding", None
        )

        if (
            has_audio
            and not has_embedding
            and self._reference_conditioning_mode(request_data) == "embed"
        ):
            embedding, error = await self._resolve_reference_embedding(
                request_id,
                request_data,
                chunk_artifacts,
                request_timeout_s=request_timeout_s,
            )
            if error is not None:
                return False, error
            chunk_artifacts["input_embedding"] = embedding
            has_embedding = True

        clear_clone_fields = False
        if has_audio and has_embedding:
            # The embedding supersedes the reference clip; forwarding the clip
            # would make each chunk re-run the embed step.
            chunk_artifacts.pop("input_audio", None)
            chunk_artifacts.pop("input_vad_data", None)
            clear_clone_fields = True
        return clear_clone_fields, None

    @staticmethod
    def _result_waveform(final: RequestOutput) -> tuple[np.ndarray, int]:
        """Return ``(mono float32 waveform, sample rate)`` for one chunk result.

        Production results carry a typed :class:`AudioArtifact` (the verified
        representation, ``audio_artifact.numpy``); the legacy base64 WAV payload
        is decoded through the codec layer (one decoder, never a second one).
        Runs in a worker thread: ``.numpy`` copies device→CPU and the decode is
        CPU-heavy.
        """
        artifact = getattr(final, "audio_artifact", None)
        if artifact is not None:
            waveform = getattr(artifact, "numpy", None)
            if waveform is not None:
                return np.asarray(waveform, dtype=np.float32).reshape(-1), int(
                    artifact.sample_rate
                )
        legacy = getattr(final, "output", "") or ""
        if not legacy:
            return np.empty(0, dtype=np.float32), 0
        waveform, sample_rate = audio_codec.decode_wav_b64(legacy)
        return np.asarray(waveform, dtype=np.float32).reshape(-1), int(sample_rate)

    def _emit_stream_segment(
        self,
        waveform: np.ndarray,
        *,
        stitcher: CrossfadeStitcher | None,
        resampler: StreamResampler | None,
        remove_dc: bool,
        model_name: str | None,
        sample_rate: int,
    ) -> tuple[np.ndarray, int]:
        """One chunk through the post-chain; returns ``(segment, source_samples)``.

        Runs in a worker thread (DC removal, watermark forward, resampler) so no
        CPU-heavy work ever blocks the event loop. ``source_samples`` counts the
        stitched source-rate samples handed on (the duration counter's unit).
        """
        if remove_dc:
            waveform = remove_dc_offset(waveform)
        # 0/1-chunk request (stitcher None): crossfade_stitch's byte-identity rule.
        piece = (
            stitcher.feed(waveform)
            if stitcher is not None
            else np.ascontiguousarray(waveform, dtype=np.float32)
        )
        return self._finish_stream_segment(
            piece,
            resampler=resampler,
            model_name=model_name,
            sample_rate=sample_rate,
        )

    def _finish_stream_segment(
        self,
        piece: np.ndarray,
        *,
        resampler: StreamResampler | None,
        model_name: str | None,
        sample_rate: int,
    ) -> tuple[np.ndarray, int]:
        """Watermark one emittable segment, then resample it (worker thread)."""
        piece = np.ascontiguousarray(piece, dtype=np.float32)
        source_samples = int(piece.shape[0])
        if source_samples:
            # One perth call per emitted segment (owner decision 2026-11-09).
            piece = self._apply_final_watermark(model_name, piece, sample_rate)
        if resampler is not None:
            piece = resampler.feed(piece)
        return np.ascontiguousarray(piece, dtype=np.float32), source_samples


    def _validate_streaming_admission(self, request: TextToSpeechRequest) -> str | None:
        """Phase 14-1 streaming admission; returns a 400 message or ``None``.

        Called before any synthesis (and before ``from_wire_audio``), so an
        inadmissible streaming request never reaches the engine. 14-3's
        dispatcher reuses this method: it is the single admission gate for both
        streaming envelopes.
        """
        if resolve_stream_format(getattr(request, "output_options", None)) is None:
            return None
        model_name = getattr(request, "model", "") or ""
        return streaming_admission_error(
            request,
            uses_native_speed=self._uses_native_speed(model_name, model_name),
        )

    def _uses_native_speed(self, requested_model: str | None, model_name: str | None) -> bool:
        """True when this synthesis family consumes ``speed`` itself.

        ``requested_model`` is the wire id (a pipeline alias or a concrete
        executor name); ``model_name`` is the resolved executor name. Shared by
        the chunked path and the pre-synthesis streaming admission gate.
        """
        spec = get_spec(requested_model)
        engine = getattr(self.engine, "engine", None) or self.engine
        configs = getattr(engine, "model_configs", None) or []
        available = {getattr(config, "model_type", None) for config in configs}
        if spec is not None:
            return any(
                model_type in NATIVE_SPEED_MODEL_TYPES and model_type in available
                for step in spec.steps
                for model_type in step.requires
            )
        for config in configs:
            if config.name == model_name:
                return config.model_type in NATIVE_SPEED_MODEL_TYPES
        return False

    def _request_uses_native_speed(self, request_data: TextToSpeechRequestInput) -> bool:
        """True when this request's synthesis family consumes ``speed`` itself."""
        requested = getattr(request_data, "requested_model", None)
        model_name = getattr(request_data, "model", None) or requested
        return self._uses_native_speed(requested, model_name)

    def _resolve_serving_model(self, model_name: str | None):
        """Return the loaded model object that produced the final step."""
        if not model_name:
            return None
        engine = getattr(self.engine, "engine", None) or self.engine
        executors = getattr(engine, "model_executors", None) or {}
        executor = executors.get(model_name)
        if executor is None:
            return None
        runner = getattr(executor, "runner", None)
        if runner is None:
            runner = getattr(getattr(executor, "driver_worker", None), "model_runner", None)
        return getattr(runner, "model", None)

    def _apply_final_watermark(self, model_name: str | None, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply the producing model's watermarker once to the stitched buffer."""
        model = self._resolve_serving_model(model_name)
        watermarker = getattr(model, "watermarker", None) if model is not None else None
        if watermarker is None:
            return waveform
        sr = int(getattr(model, "sr", sample_rate) or sample_rate)
        with _WATERMARK_LOCK:
            marked = watermarker.apply_watermark(waveform, sample_rate=sr)
        return np.asarray(marked, dtype=np.float32)

    async def text_to_speech_full_generator(
        self,
        request: TextToSpeechRequest,
        raw_request: Request,
        result_generator: AsyncIterator[RequestOutput],
        request_id: str,
        request_data: TextToSpeechRequestInput | None = None,
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
        if isinstance(final_res, (TextToSpeechRequestOutput, SpeechSynthesisRequestOutput)):
            pass
        else:
            raise ValueError(
                f"Expected TextToSpeechRequestOutput or SpeechSynthesisRequestOutput, got {type(final_res).__name__}"
            )

        # Phase 12-3 opt-in output filters, single-shot path: all three run on
        # the model's final buffer (DC -> trim -> internal). Families that
        # watermark inside ``generate`` have already marked it here; the chunked
        # path filters before its serving-side watermark, and the difference is
        # recorded in the task log.
        options = getattr(request_data, "generation_options", None)
        remove_dc, trim, fix_internal = _output_filter_flags(options)
        artifact = getattr(final_res, "audio_artifact", None)
        if artifact is not None and (remove_dc or trim or fix_internal):
            sample_rate = int(artifact.sample_rate)
            waveform = artifact.numpy
            if remove_dc:
                waveform = remove_dc_offset(waveform)
            if trim:
                waveform = trim_silence(waveform, sample_rate)
            if fix_internal:
                waveform = fix_internal_silence(waveform, sample_rate)
            final_res.audio_artifact = AudioArtifact.from_numpy(waveform, sample_rate)

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


def _chunk_seed_update(options, index: int) -> dict:
    """Derive a chunk's seed deterministically from the request seed."""
    seed = _option(options, "seed", None)
    if seed is None:
        return {}
    return {"seed": int(seed) + index}

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
"""

import asyncio
import copy
import dataclasses
import threading
import time
from collections.abc import AsyncGenerator, AsyncIterator

import numpy as np
from fastapi import Request
from loguru import logger

from harmonyspeech.common.artifacts import AudioArtifact
from harmonyspeech.common.audio_utils import (
    DEFAULT_CHUNK_CROSSFADE_MS,
    DEFAULT_CHUNK_SENTENCE_PAUSE_MS,
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
from harmonyspeech.endpoints.openai.output_encoding import validate_output_options
from harmonyspeech.endpoints.openai.protocol import ErrorResponse, ModelCard, TextToSpeechRequest, TextToSpeechResponse
from harmonyspeech.endpoints.openai.serving_engine import OpenAIServing
from harmonyspeech.endpoints.openai.wire import from_wire_audio, to_wire_audio
from harmonyspeech.engine.async_harmonyspeech import AsyncHarmonySpeech
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

        # Long-form dispatch (Phase 12-2). A short input (0/1 chunks) returns
        # None and keeps the single-shot path below byte-identical.
        try:
            chunks = self._resolve_chunks(request_data)
        except ValueError as e:
            return self.create_error_response(str(e))
        if chunks is not None:
            return await self._create_chunked_response(
                request=request,
                raw_request=raw_request,
                request_id=request_id,
                request_data=request_data,
                artifacts=artifacts,
                chunks=chunks,
            )

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
        spec = get_spec(requested)
        engine = getattr(self, "engine", None)
        engine = getattr(engine, "engine", None) or engine
        configs = getattr(engine, "model_configs", None) or []
        available = {getattr(config, "model_type", None) for config in configs}
        available.discard(None)
        model_types: set[str] = set()
        if spec is not None:
            model_types.update(
                model_type
                for step in spec.steps
                for model_type in step.requires
                if model_type in available
            )
        # Also honour a concrete config named exactly like the request: this is
        # the "request names the executor directly" path, and it stays correct
        # even when a same-named pipeline sentinel points at a different family.
        model_name = getattr(request_data, "model", None) or requested
        model_types.update(
            config.model_type
            for config in configs
            if config.name == model_name and getattr(config, "model_type", None)
        )
        return model_types

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

    async def _create_chunked_response(
        self,
        *,
        request: TextToSpeechRequest,
        raw_request: Request,
        request_id: str,
        request_data: TextToSpeechRequestInput,
        artifacts: dict,
        chunks: list[str],
    ) -> ErrorResponse | TextToSpeechResponse:
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

    def _request_uses_native_speed(self, request_data: TextToSpeechRequestInput) -> bool:
        """True when this request's synthesis family consumes ``speed`` itself."""
        requested = getattr(request_data, "requested_model", None)
        spec = get_spec(requested)
        engine = getattr(self.engine, "engine", None) or self.engine
        configs = getattr(engine, "model_configs", None) or []
        available = {getattr(config, "model_type", None) for config in configs}
        if spec is not None:
            return any(
                model_type in NATIVE_SPEED_MODEL_TYPES and model_type in available
                for step in spec.steps
                for model_type in step.requires
            )
        model_name = getattr(request_data, "model", None) or requested
        for config in configs:
            if config.name == model_name:
                return config.model_type in NATIVE_SPEED_MODEL_TYPES
        return False

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

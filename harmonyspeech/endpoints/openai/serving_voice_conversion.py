import time
from collections.abc import AsyncGenerator, AsyncIterator

from fastapi import Request

from harmonyspeech.common.config import ModelConfig
from harmonyspeech.common.inputs import SpeechEmbeddingRequestInput, VoiceConversionRequestInput
from harmonyspeech.common.outputs import RequestOutput, VoiceConversionRequestOutput
from harmonyspeech.common.utils import random_uuid
from harmonyspeech.endpoints.openai.output_encoding import validate_output_options
from harmonyspeech.endpoints.openai.protocol import (
    ErrorResponse,
    ModelCard,
    VoiceConversionRequest,
    VoiceConversionResponse,
)
from harmonyspeech.endpoints.openai.serving_engine import OpenAIServing
from harmonyspeech.endpoints.openai.standard_voice import standard_voice_wire_payload
from harmonyspeech.endpoints.openai.wire import from_wire_audio, to_wire_audio
from harmonyspeech.engine.async_harmonyspeech import AsyncHarmonySpeech
from harmonyspeech.modeling.voice_catalog import (
    VC_TARGET_EMBEDDING,
    lookup_standard_voice,
    resolve_target_embedding_route,
)

# Add new model classes which allow handling Voice Conversion Requests here
# If multiple models need to be initialized to process request, add multiple to the list


class OpenAIServingVoiceConversion(OpenAIServing):
    def __init__(self, engine: AsyncHarmonySpeech, available_models: list[ModelCard]):
        super().__init__(engine=engine, available_models=available_models)

    @staticmethod
    def models_from_config(configured_models: list[ModelConfig]):
        # Advertised cards come from the pipeline registry (Phase 3-3).
        return OpenAIServing.model_cards_from_specs(configured_models, "vc")

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
        """Resolve the request's ``target_voice`` as a standard pack name."""
        model = getattr(request, "model", "") or ""
        return lookup_standard_voice(
            self._engine_model_configs(),
            model,
            getattr(request, "target_voice", None),
            target=True,
            enabled=self._standard_voice_pack_enabled(),
            model_exists=self.get_model_by_id(model) is not None,
            has_explicit_reference=bool(
                getattr(request, "target_audio", None)
                or getattr(request, "target_embedding", None)
            ),
        )

    async def convert_voice(
        self, request: VoiceConversionRequest, raw_request: Request
    ) -> ErrorResponse | AsyncGenerator[str, None] | VoiceConversionResponse:
        # Phase 13-1/13-1 addendum: a standard pack `target_voice` resolves to
        # the shipped reference clip before the model check and before the
        # boundary decode. Two capability kinds exist:
        #   * target-audio VC (ChatterboxVC): the clip becomes `target_audio`,
        #     exactly like a client-supplied reference;
        #   * target-embedding VC (OpenVoice tone converters): the clip is run
        #     once through the pipeline's `vad → embed` steps and the returned
        #     artifact becomes `target_embedding`. `target_audio` is never set,
        #     because the tone converter ignores it.
        voice_lookup = self._lookup_standard_voice(request)
        if voice_lookup.error is not None:
            return self.create_error_response(voice_lookup.error)

        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        error_check_options = validate_output_options(request.output_options)
        if error_check_options is not None:
            return self.create_error_response(error_check_options)

        # TODO: Basic checks for Embedding Generate request

        request_id = f"vc-{random_uuid()}"

        request_data = VoiceConversionRequestInput.from_openai(request_id=request_id, request=request)
        embed_target = voice_lookup.vc_target_kind == VC_TARGET_EMBEDDING
        if voice_lookup.resolution is not None and not embed_target:
            payload = standard_voice_wire_payload(voice_lookup.resolution.name)
            if payload is None:  # pragma: no cover - resolution validated the path
                return self.create_error_response(
                    f"Standard voice '{voice_lookup.resolution.name}' is missing from the "
                    "voice pack installation."
                )
            request_data.target_audio = payload
            request_data.target_embedding = None
        artifacts = from_wire_audio(request_data)
        if voice_lookup.resolution is not None and embed_target:
            embedding, embed_error = await self._resolve_target_embedding(
                request_id, request, voice_lookup.resolution.name
            )
            if embed_error is not None:
                return self.create_error_response(embed_error)
            # Typed artifact: the tone converter's prepare reads the artifact
            # store first (``get_speaker_artifact``), so no wire round-trip.
            artifacts["target_embedding"] = embedding
            request_data.target_audio = None
            request_data.target_embedding = None
        result_generator = self.engine.generate(
            request_id=request_id,
            request_data=request_data,
            artifacts=artifacts,
        )

        try:
            return await self.voice_conversion_full_generator(request, raw_request, result_generator, request_id)
        except ValueError as e:
            # TODO: Use an aphrodite-specific Validation Error
            return self.create_error_response(str(e))

    async def _resolve_target_embedding(
        self, request_id: str, request: VoiceConversionRequest, voice_name: str
    ) -> tuple[object | None, str | None]:
        """Embed a pack ``target_voice`` through the OpenVoice pipeline once.

        Routes one ``SpeechEmbeddingRequestInput`` at the pipeline id so the
        engine runs its ``vad → embed`` steps (a concrete encoder name would
        skip the VAD step the encoder requires). The encoder artifact rides the
        7-2/13-2 cache exactly like the TTS clone path, so a warm pack voice
        performs zero encoder forwards.
        """
        model = getattr(request, "model", "") or ""
        route = resolve_target_embedding_route(
            self._engine_model_configs(), model, model
        )
        if route is None:
            return None, (
                f"Target voice '{voice_name}' cannot be embedded for model '{model}': "
                "its voice-conversion pipeline has no configured encoder/VAD. Provide "
                "`target_embedding`, or configure the OpenVoice encoder and VAD."
            )
        payload = standard_voice_wire_payload(voice_name)
        if payload is None:  # pragma: no cover - resolution validated the path
            return None, (
                f"Standard voice '{voice_name}' is missing from the voice pack installation."
            )
        embed_id = f"{request_id}-target-embed"
        embed_request = SpeechEmbeddingRequestInput(
            request_id=embed_id,
            requested_model=route.pipeline_id,
            model=route.pipeline_id,
            input_audio=payload,
            input_vad_mode=None,
            input_vad_data=None,
            generation_options=getattr(request, "generation_options", None),
        )
        final_output = None
        stream = await self.engine.add_request(
            request_id=embed_id, request_data=embed_request
        )
        async for output in stream:
            final_output = output
        if final_output is None:
            return None, "Target voice embedding step produced no output"
        if getattr(final_output, "finish_reason", None) == "error":
            return None, (
                getattr(final_output, "error", None) or "Target voice embedding step failed"
            )
        embedding = getattr(final_output, "embedding_artifact", None)
        if embedding is None:
            return None, "Target voice embedding step produced no embedding"
        return embedding, None

    async def voice_conversion_full_generator(
        self,
        request: VoiceConversionRequest,
        raw_request: Request,
        result_generator: AsyncIterator[RequestOutput],
        request_id: str,
    ) -> ErrorResponse | VoiceConversionResponse:

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
        assert isinstance(final_res, VoiceConversionRequestOutput)

        encoded = to_wire_audio(final_res, request.output_options)
        response = VoiceConversionResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            data=encoded.data,
            sample_rate=encoded.sample_rate,
        )

        return response

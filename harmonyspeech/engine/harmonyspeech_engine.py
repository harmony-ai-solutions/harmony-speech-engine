import os
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor

from loguru import logger

import harmonyspeech
from harmonyspeech.common.config import ModelConfig
from harmonyspeech.common.inputs import (
    AudioConversionRequestInput,
    RequestInput,
    SpeechEmbeddingRequestInput,
    SpeechTranscribeRequestInput,
    SynthesisRequestInput,
    TextToSpeechRequestInput,
    VocodeRequestInput,
    VoiceConversionRequestInput,
)
from harmonyspeech.common.logger import setup_logger
from harmonyspeech.common.outputs import (
    AudioConversionRequestOutput,
    RequestOutput,
    SpeechEmbeddingRequestOutput,
    SpeechSynthesisRequestOutput,
    SpeechTranscriptionRequestOutput,
    TextToSpeechRequestOutput,
)
from harmonyspeech.common.request import EngineRequest, ExecutorResult, RequestStatus
from harmonyspeech.engine.args_tools import EngineArgs
from harmonyspeech.engine.metrics import StatLogger, Stats
from harmonyspeech.processing.pipeline_registry import (
    FAMILY_REQUIREMENTS,
    missing_family_requirements,
    validate_families,
)
from harmonyspeech.processing.scheduler import Scheduler, SchedulerOutputs

_LOCAL_LOGGING_INTERVAL_SEC = int(os.environ.get("HARMONYSPEECH_LOCAL_LOGGING_INTERVAL_SEC", "5"))


class HarmonySpeechEngine:
    """
    An Inference Engine for AI Speech that receives requests and generates outputs.

    """

    def __init__(
        self,
        model_configs: list[ModelConfig] | None,
        log_stats: bool,
        request_timeout_s: float | None = 120.0,
    ):
        self.model_configs = model_configs
        self.log_stats = log_stats
        # Per-request deadline (seconds); 0/None disables deadline enforcement.
        self.request_timeout_s = request_timeout_s
        # Requests rejected before ever entering the scheduler (unresolvable
        # routing, expired deadlines). Drained into the next step's outputs so
        # their HTTP call resolves with an error instead of hanging.
        self._pending_error_outputs: list[RequestOutput] = []
        self._verify_args()

        logger.info(
            f"Initializing Harmony Speech Engine (v{harmonyspeech.__version__}) with the following config:\nPreloaded Models = \n"
            # TODO
            # f"Available Models = "
            # f"Device = {device_config.device}\n"
        )

        """
        For each provided model config we will create a separate executor.
        Models may allocate across multiple devices based on DeviceConfig.
        """
        self.model_executors = {}
        self.init_custom_executors()

        # Startup validation: which declared toolchains are complete?
        self.available_families: list[str] = self._validate_family_configuration()

        # Initialize Scheduler for requests across models
        self.scheduler = Scheduler(model_configs=self.model_configs)

        # Metric Logging.
        if self.log_stats:
            self.stat_logger = StatLogger(local_interval=_LOCAL_LOGGING_INTERVAL_SEC, labels=dict())

    def _validate_family_configuration(self) -> list[str]:
        """Log incomplete toolchains and return the complete family ids.

        A family is complete when every required ``model_type`` is configured.
        Boot does not fail on incomplete families (operators may run subsets),
        but the serving layer only advertises complete ones.
        """
        complete_families = validate_families(self.model_configs)
        for family, missing_types in missing_family_requirements(self.model_configs).items():
            logger.warning(
                f"Model toolchain '{family}' is not fully configured — missing model_type(s): "
                f"{', '.join(missing_types)}. Requests routed to '{family}' will fail with an error and "
                f"the toolchain is not advertised on /v1/models."
            )
        logger.info(
            "Complete model toolchains: "
            f"{', '.join(complete_families) if complete_families else 'none (no family fully configured)'}"
        )
        return complete_families

    def init_custom_executors(self) -> None:
        """
        Initialize custom executors for each provided ModelConfig.
        """
        for model_cfg in self.model_configs:
            # Determine Executor class based on Device config for the model
            if model_cfg.device_config.device_type == "cpu":
                from harmonyspeech.executor.cpu_executor import CPUExecutor

                executor_class = CPUExecutor
            else:
                from harmonyspeech.executor.gpu_executor import GPUExecutorAsync

                executor_class = GPUExecutorAsync

            # Create a new executor for each ModelConfig
            executor = executor_class(model_config=model_cfg)
            # Append the created executor to the dict of model executors
            self.model_executors[model_cfg.name] = executor

    @classmethod
    def from_engine_args(cls, engine_args: EngineArgs) -> "HarmonySpeechEngine":
        """Creates an LLM engine from the engine arguments."""
        # Create the engine configs.
        engine_config = engine_args.create_engine_config()

        # Create the LLM engine.
        engine = cls(**engine_config.to_dict(), log_stats=not engine_args.disable_log_stats)
        return engine

    def __reduce__(self):
        # This is to ensure that the HarmonySpeechEngine is not referenced in
        # the closure used to initialize Ray worker actors
        raise RuntimeError("HarmonySpeechEngine should not be pickled!")

    def _verify_args(self) -> None:
        # for model_cfg in self.model_configs:
        #     model_cfg.verify_with_parallel_config(self.parallel_config)
        return None

    def reroute_request_harmonyspeech(self, request: RequestInput):
        if isinstance(request, SpeechEmbeddingRequestInput) or (
            isinstance(request, TextToSpeechRequestInput)
            and request.input_audio is not None  #
            and request.input_embedding is None  # Output of Embedding becomes input for synthesis
        ):
            for cfg in self.model_configs:
                if cfg.model_type == "HarmonySpeechEncoder":
                    request.model = cfg.name
                    break
        elif isinstance(request, SynthesisRequestInput) or (
            isinstance(request, TextToSpeechRequestInput)
            and request.input_audio is None  # Removing Input audio after Embedding Step
            and request.input_embedding is not None  # Output of Embedding becomes input for synthesis
        ):
            for cfg in self.model_configs:
                if cfg.model_type == "HarmonySpeechSynthesizer":
                    request.model = cfg.name
                    break
        elif isinstance(request, VocodeRequestInput) or (
            isinstance(request, TextToSpeechRequestInput)
            and request.input_audio is not None  # Output of synthesis becomes Input for vocoder
            and request.input_embedding is not None
        ):
            for cfg in self.model_configs:
                if cfg.model_type == "HarmonySpeechVocoder":
                    request.model = cfg.name
                    break

    def reroute_request_openvoice_v1(self, request: RequestInput):
        """
        OpenVoice V1 uses a multi-step-embedding process for improving accuracy.
        This process involves retrieving VAD information for the provided audio if that's not provided.
        The default variant is to use whisper for this Process; however there's also a SileroTTS variant
        provided in the OpenVoice repo.

        When processing OpenVoice V2 Embedding or TTS requests, the routing rule is the following:
        1. VAD using Whisper or Silero* (*not implemented yet) to get Audio segments
        2. Embedding using ToneConverter in Encoder mode using Audio and Timestamp segments returned from VAD step.
        3. TTS using base Synthesizer model
        4. Tone Conversion for Output of MeloTTS based on Speaker ID embedding
        """
        if (
            isinstance(request, SpeechTranscribeRequestInput)
            or (
                isinstance(request, TextToSpeechRequestInput)
                and request.mode == "voice_cloning"  # Only required when voice cloning
                and request.input_audio is not None  #
                and request.input_vad_data is None  # VAD Data not set, key condition for Transcribe Action
                and request.input_embedding is None  # Output of Embedding becomes input for synthesis
            )
            or (
                isinstance(request, SpeechEmbeddingRequestInput)
                and request.input_audio is not None  #
                and request.input_vad_data is None  # VAD Data not set, key condition for Transcribe Action
            )
        ):
            for cfg in self.model_configs:
                # TODO: add switch for silero here using request.input_vad_mode
                if cfg.model_type == "FasterWhisper":
                    request.model = cfg.name
                    break
        elif isinstance(request, SpeechEmbeddingRequestInput) or (
            isinstance(request, TextToSpeechRequestInput)
            and request.mode == "voice_cloning"  # Only required when voice cloning
            and request.input_audio is not None  #
            and request.input_vad_data is not None  # VAD Data set, key condition for Embedding Action
            and request.input_embedding is None  # Output of Embedding becomes input for synthesis
        ):
            for cfg in self.model_configs:
                if cfg.model_type == "OpenVoiceV1ToneConverterEncoder":
                    request.model = cfg.name
                    break
        elif isinstance(request, SynthesisRequestInput) or (
            isinstance(request, TextToSpeechRequestInput)
            and request.input_audio is None  # Removing Input audio after Embedding Step
            and (
                request.input_embedding is not None  # Output of Embedding becomes input for synthesis
                or request.mode == "single_speaker_tts"  # FIXME: This code needs refactor
            )
        ):
            for cfg in self.model_configs:
                if cfg.model_type == "OpenVoiceV1Synthesizer" and cfg.language == request.language_id:
                    request.model = cfg.name
                    break
        elif isinstance(request, VoiceConversionRequestInput) or (
            isinstance(request, TextToSpeechRequestInput)
            and request.mode == "voice_cloning"  # Only required when voice cloning
            and request.input_audio is not None  # Output of synthesis becomes Input for voice converter
            and request.input_embedding is not None
        ):
            for cfg in self.model_configs:
                if cfg.model_type == "OpenVoiceV1ToneConverter":
                    request.model = cfg.name
                    break

    def reroute_request_openvoice_v2(self, request: RequestInput):
        """
        OpenVoice V2 uses a multi-step-embedding process for improving accuracy.
        This process involves retrieving VAD information for the provided audio if that's not provided.
        The default variant is to use whisper for this Process; however there's also a SileroTTS variant
        provided in the OpenVoice repo.

        When processing OpenVoice V2 Embedding or TTS requests, the routing rule is the following:
        1. VAD using Whisper or Silero* (*not implemented yet) to get Audio segments
        2. Embedding using ToneConverter in Encoder mode using Audio and Timestamp segments returned from VAD step.
        3. TTS using Melo
        4. Tone Conversion for Output of MeloTTS based on Speaker ID embedding
        """
        if (
            isinstance(request, SpeechTranscribeRequestInput)
            or (
                isinstance(request, TextToSpeechRequestInput)
                and request.mode == "voice_cloning"  # Only required when voice cloning
                and request.input_audio is not None  #
                and request.input_vad_data is None  # VAD Data not set, key condition for Transcribe Action
                and request.input_embedding is None  # Output of Embedding becomes input for synthesis
            )
            or (
                isinstance(request, SpeechEmbeddingRequestInput)
                and request.input_audio is not None  #
                and request.input_vad_data is None  # VAD Data not set, key condition for Transcribe Action
            )
        ):
            for cfg in self.model_configs:
                # TODO: add switch for silero here using request.input_vad_mode
                if cfg.model_type == "FasterWhisper":
                    request.model = cfg.name
                    break
        elif isinstance(request, SpeechEmbeddingRequestInput) or (
            isinstance(request, TextToSpeechRequestInput)
            and request.mode == "voice_cloning"  # Only required when voice cloning
            and request.input_audio is not None  #
            and request.input_vad_data is not None  # VAD Data set, key condition for Embedding Action
            and request.input_embedding is None  # Output of Embedding becomes input for synthesis
        ):
            for cfg in self.model_configs:
                if cfg.model_type == "OpenVoiceV2ToneConverterEncoder":
                    request.model = cfg.name
                    break
        elif isinstance(request, SynthesisRequestInput) or (
            isinstance(request, TextToSpeechRequestInput)
            and request.input_audio is None  # Removing Input audio after Embedding Step
            and (
                request.input_embedding is not None  # Output of Embedding becomes input for synthesis
                or request.mode == "single_speaker_tts"  # FIXME: This code needs refactor
            )
        ):
            for cfg in self.model_configs:
                if cfg.model_type == "MeloTTSSynthesizer" and cfg.language == request.language_id:
                    request.model = cfg.name
                    break
        elif isinstance(request, VoiceConversionRequestInput) or (
            isinstance(request, TextToSpeechRequestInput)
            and request.mode == "voice_cloning"  # Only required when voice cloning
            and request.input_audio is not None  # Output of synthesis becomes Input for voice converter
            and request.input_embedding is not None
        ):
            for cfg in self.model_configs:
                if cfg.model_type == "OpenVoiceV2ToneConverter":
                    request.model = cfg.name
                    break

    def reroute_request_voicefixer(self, request: RequestInput):
        """
        VoiceFixer uses a two-step process for audio restoration:
        1. VoiceFixerRestorer: Audio → Enhanced Mel Spectrogram
        2. VoiceFixerVocoder: Enhanced Mel Spectrogram → Restored Audio
        """
        if isinstance(request, AudioConversionRequestInput):
            if request.input_mel_spectrogram is None:
                # Step 1: Route to Restorer (Audio → Mel)
                for cfg in self.model_configs:
                    if cfg.model_type == "VoiceFixerRestorer":
                        request.model = cfg.name
                        break
            else:
                # Step 2: Route to Vocoder (Mel → Audio)
                for cfg in self.model_configs:
                    if cfg.model_type == "VoiceFixerVocoder":
                        request.model = cfg.name
                        break

    def reroute_request_chatterbox(self, request: RequestInput):
        # Map sentinel name → TTS model_type
        _CHATTERBOX_TTS_TYPE_MAP = {
            "chatterbox": "ChatterboxTTS",
            "chatterbox_turbo": "ChatterboxTurboTTS",
            "chatterbox_multilingual": "ChatterboxMultilingualTTS",
        }

        if isinstance(request, SpeechEmbeddingRequestInput) or (
            isinstance(request, TextToSpeechRequestInput)
            and request.input_audio is not None
            and request.input_embedding is None
        ):
            # Prefer dedicated ChatterboxEmbedding executor if configured
            for cfg in self.model_configs:
                if cfg.model_type == "ChatterboxEmbedding":
                    request.model = cfg.name
                    break
            else:
                # Fallback: route embedding request to the TTS model executor
                # (ChatterboxTTS also has prepare_conditionals — no separate model needed)
                tts_type = _CHATTERBOX_TTS_TYPE_MAP.get(request.requested_model, "ChatterboxTTS")
                for cfg in self.model_configs:
                    if cfg.model_type == tts_type:
                        request.model = cfg.name
                        break
        elif (
            isinstance(request, TextToSpeechRequestInput)
            and (request.input_audio is None)  # direct path or Step 2 after embedding
        ):
            # Route to the appropriate TTS variant
            tts_type = _CHATTERBOX_TTS_TYPE_MAP.get(request.requested_model)
            if tts_type:
                for cfg in self.model_configs:
                    if cfg.model_type == tts_type:
                        request.model = cfg.name
                        break

    def reroute_request_chatterbox_vc(self, request: RequestInput):
        if isinstance(request, VoiceConversionRequestInput):
            for cfg in self.model_configs:
                if cfg.model_type == "ChatterboxVC":
                    request.model = cfg.name
                    break

    def check_reroute_request_to_model(self, request: RequestInput):
        # Harmonyspeech TTS Processing
        if request.requested_model == "harmonyspeech":
            self.reroute_request_harmonyspeech(request)
        # OpenVoice V1 TTS & VC Processing
        if request.requested_model == "openvoice_v1":
            self.reroute_request_openvoice_v1(request)
        # OpenVoice V2 TTS & VC Processing
        if request.requested_model == "openvoice_v2":
            self.reroute_request_openvoice_v2(request)
        # VoiceFixer Audio Conversion Processing
        if request.requested_model == "voicefixer":
            self.reroute_request_voicefixer(request)
        # Chatterbox TTS (standard)
        if request.requested_model == "chatterbox":
            self.reroute_request_chatterbox(request)
        # Chatterbox Turbo TTS
        if request.requested_model == "chatterbox_turbo":
            self.reroute_request_chatterbox(request)
        # Chatterbox Multilingual TTS
        if request.requested_model == "chatterbox_multilingual":
            self.reroute_request_chatterbox(request)
        # Chatterbox Voice Conversion
        if request.requested_model == "chatterbox_vc":
            self.reroute_request_chatterbox_vc(request)

    def check_forward_processing(self, result: ExecutorResult):
        new_status = RequestStatus.FINISHED_STOPPED
        input_data = result.input_data
        returning_model_cfg = None
        requested_model = input_data.requested_model
        for cfg in self.model_configs:
            if input_data.model == cfg.name:
                returning_model_cfg = cfg
                break

        # If inference produced an error result (e.g. bad input, model failure),
        # do NOT attempt to forward it through the multi-step pipeline. Mark the
        # request as finished so the scheduler can clean it up, and let the error
        # output flow back to the caller. Without this, a failed request would
        # either be incorrectly re-forwarded or leak in the scheduler's running
        # queue forever.
        if isinstance(result.result_data, RequestOutput) and result.result_data.finish_reason == "error":
            self.scheduler.update_request_status(result.request_id, RequestStatus.FINISHED_STOPPED)
            return RequestStatus.FINISHED_STOPPED, None

        # Explicit model selected, no forwarding needed
        if returning_model_cfg == requested_model:
            return new_status, None

        forwarding_request = None

        # Multi-Step Embedding Processing (OpenVoice)
        if isinstance(input_data, SpeechEmbeddingRequestInput) and requested_model in ["openvoice_v1", "openvoice_v2"]:
            forwarding_request = input_data

            if isinstance(result.result_data, SpeechTranscriptionRequestOutput):
                # Transcription results used By:
                # - OpenVoice V1
                # - OpenVoice V2
                # Mark as forwarded in scheduler
                new_status = RequestStatus.FINISHED_FORWARDED
                self.scheduler.update_request_status(result.request_id, new_status)
                # Transcription Step Finished, provide VAD Result Data for Embedding Step
                forwarding_request.input_vad_data = result.result_data.output
                self.add_request(result.request_id, forwarding_request)

        # Multi-Step TTS Processing (Harmonyspeech, OpenVoice)
        if isinstance(input_data, TextToSpeechRequestInput) and requested_model in [
            "harmonyspeech",
            "openvoice_v1",
            "openvoice_v2",
        ]:
            forwarding_request = input_data

            if isinstance(result.result_data, SpeechTranscriptionRequestOutput):
                # Transcription results used By:
                # - OpenVoice V1
                # - OpenVoice V2

                # Mark as forwarded in scheduler
                new_status = RequestStatus.FINISHED_FORWARDED
                self.scheduler.update_request_status(result.request_id, new_status)
                # Transcription Step Finished, provide VAD Result Data for Embedding Step
                forwarding_request.input_vad_data = result.result_data.output
                self.add_request(result.request_id, forwarding_request)
            elif isinstance(result.result_data, SpeechEmbeddingRequestOutput):
                # Embedding results used By:
                # - OpenVoice V1
                # - OpenVoice V2
                # - Harmony Speech

                # Mark as forwarded in scheduler
                new_status = RequestStatus.FINISHED_FORWARDED
                self.scheduler.update_request_status(result.request_id, new_status)
                # Embedding step finished, update input data and continue with synthesis step
                forwarding_request.input_audio = None
                forwarding_request.input_embedding = result.result_data.output
                self.add_request(result.request_id, forwarding_request)
            elif isinstance(result.result_data, SpeechSynthesisRequestOutput) and input_data.mode == "voice_cloning":
                # Synthesis results used By:
                # - OpenVoice V1
                # - OpenVoice V2
                # - Harmony Speech

                # Mark as forwarded in scheduler
                new_status = RequestStatus.FINISHED_FORWARDED
                self.scheduler.update_request_status(result.request_id, new_status)
                # Embedding step finished, update input data and continue with synthesis step
                forwarding_request.input_audio = result.result_data.output
                self.add_request(result.request_id, forwarding_request)
            else:  # There should be only final processing results here...
                # Vocoding / Voice Conversion / Single Speaker TTS results used By:
                # - OpenVoice V1
                # - OpenVoice V2
                # - Harmony Speech

                # Mark as finished in scheduler
                new_status = RequestStatus.FINISHED_STOPPED
                self.scheduler.update_request_status(result.request_id, new_status)
                # Convert result to TTS output - Assuming VocodeRequestOutput
                tts_result = TextToSpeechRequestOutput(
                    request_id=result.request_id,
                    text=input_data.input_text,
                    output=result.result_data.output,
                    finish_reason=result.result_data.finish_reason,
                    metrics=result.result_data.metrics if result.result_data.metrics else None,
                )
                result.result_data = tts_result

        # Multi-step TTS Processing (Chatterbox — embed then synthesize)
        if isinstance(input_data, TextToSpeechRequestInput) and requested_model in [
            "chatterbox",
            "chatterbox_turbo",
            "chatterbox_multilingual",
        ]:
            forwarding_request = input_data

            if isinstance(result.result_data, SpeechEmbeddingRequestOutput):
                # Embedding step finished — forward to TTS synthesize step
                # Mark as forwarded in scheduler
                new_status = RequestStatus.FINISHED_FORWARDED
                self.scheduler.update_request_status(result.request_id, new_status)
                # Embedding result becomes input_embedding; clear input_audio
                forwarding_request.input_audio = None
                forwarding_request.input_embedding = result.result_data.output
                self.add_request(result.request_id, forwarding_request)

        # Multi-Step Audio Conversion Processing (VoiceFixer)
        if isinstance(input_data, AudioConversionRequestInput) and requested_model == "voicefixer":
            forwarding_request = input_data

            if (
                isinstance(result.result_data, AudioConversionRequestOutput)
                and input_data.input_mel_spectrogram is None
            ):
                # Restorer step finished, forward to Vocoder
                new_status = RequestStatus.FINISHED_FORWARDED
                self.scheduler.update_request_status(result.request_id, new_status)
                # Update request with mel spectrogram output from restorer
                forwarding_request.input_mel_spectrogram = result.result_data.output
                self.add_request(result.request_id, forwarding_request)
            else:
                # Vocoder step finished, mark as completed
                new_status = RequestStatus.FINISHED_STOPPED
                self.scheduler.update_request_status(result.request_id, new_status)

        # TODO: Post-Generation Processing

        return new_status, forwarding_request

    def _pending_errors(self) -> list[RequestOutput]:
        """Return the pre-step error queue, creating it for shell instances."""
        pending = getattr(self, "_pending_error_outputs", None)
        if pending is None:
            pending = []
            self._pending_error_outputs = pending
        return pending

    def _queue_pending_errors(self, outputs: list[RequestOutput]) -> None:
        self._pending_errors().extend(outputs)

    def _drain_pending_errors(self) -> list[RequestOutput]:
        pending = self._pending_errors()
        drained = list(pending)
        pending.clear()
        return drained

    def _unresolvable_model_error(self, request_data: RequestInput) -> str:
        """Describe why a request cannot be routed to a configured executor."""
        requested = getattr(request_data, "requested_model", None) or getattr(request_data, "model", "unknown")
        required_types = FAMILY_REQUIREMENTS.get(requested)
        if required_types is None:
            return (
                f"Request routed to unknown model '{getattr(request_data, 'model', 'unknown')}', "
                f"which is not configured on this engine."
            )
        configured_types = {cfg.model_type for cfg in (self.model_configs or [])}
        missing = [model_type for model_type in required_types if model_type not in configured_types]
        detail = f"missing one of: {', '.join(missing)}" if missing else "no executor registered"
        return (
            f"Model family '{requested}' is not fully configured on this engine "
            f"({detail})"
        )

    def _expire_overdue_requests(self) -> list[RequestOutput]:
        """Resolve requests whose per-request deadline has passed as errors.

        Requests are not cancelled mid-forward (that would corrupt CUDA state);
        once control returns to the engine they resolve as typed timeout errors
        instead of being rescheduled forever.
        """
        now = time.monotonic()
        timeout_s = getattr(self, "request_timeout_s", None)
        expired: list[RequestOutput] = []
        for queue in (self.scheduler.waiting, self.scheduler.running):
            for request in list(queue):
                deadline = getattr(request, "deadline", None)
                if deadline is None or deadline > now:
                    continue
                message = (
                    f"request exceeded timeout of {timeout_s:g}s"
                    if timeout_s
                    else "request exceeded its deadline"
                )
                logger.error(f"Request {request.request_id} timed out: {message}")
                request.metrics.finished_time = time.time()
                self.scheduler.update_request_status(request.request_id, RequestStatus.FINISHED_ABORTED)
                expired.append(
                    RequestOutput(
                        request_id=request.request_id,
                        finish_reason="error",
                        error=message,
                        metrics=request.metrics,
                    )
                )
        return expired

    def add_request(self, request_id: str, request_data: RequestInput, arrival_time: float | None = None) -> None:
        """Add a Text-to-Speech request to the engine's request pool.

        The request is added to the request pool and will be processed by the
        scheduler as `engine.step()` is called. The exact scheduling policy is
        determined by the scheduler.

        Args:
            request_id: The unique ID of the request.
            request_data: The incoming Request data.
            arrival_time: The arrival time of the request. If None, we use
                the current monotonic time.

        Details:
            - Set arrival_time to the current time if it is None.
            - Stamp a per-request deadline (`request_timeout_s` from now).
            - Reject requests that cannot be routed to a configured executor
              immediately, as a typed error output instead of leaving them to
              wait forever in the scheduler (Phase 0-4).
        """

        if arrival_time is None:
            arrival_time = time.monotonic()

        # Route to correct model if necessary
        self.check_reroute_request_to_model(request=request_data)

        model_executors = getattr(self, "model_executors", {})
        if request_data.model not in model_executors:
            error = self._unresolvable_model_error(request_data)
            logger.error(f"Request {request_id} rejected: {error}")
            self._queue_pending_errors(
                [RequestOutput(request_id=request_id, finish_reason="error", error=error)]
            )
            return

        timeout_s = getattr(self, "request_timeout_s", None)
        deadline = time.monotonic() + timeout_s if timeout_s else None

        # Add the request to the scheduler.
        self.scheduler.add_request(
            EngineRequest(
                request_id=request_id,
                request_data=request_data,
                arrival_time=arrival_time,
                deadline=deadline,
            )
        )

    def abort_request(self, request_id: str | Iterable[str]) -> None:
        """Aborts a request(s) with the given ID.

        Args:
            request_id: The ID(s) of the request to abort.

        Details:
            - Refer to the
              :meth:`~aphrodite.processing.scheduler.Scheduler.abort_seq_group`
              from class :class:`~aphrodite.processing.scheduler.Scheduler`.

        Example:
            >>> # initialize engine and add a request with request_id
            >>> request_id = str(0)
            >>> # abort the request
            >>> engine.abort_request(request_id)
        """
        self.scheduler.update_request_status(request_id, RequestStatus.FINISHED_ABORTED)

    def get_model_configs(self) -> list[ModelConfig]:
        """Gets the model configuration."""
        return self.model_configs

    def get_num_unfinished_requests(self) -> int:
        """Gets the number of unfinished requests."""
        return self.scheduler.get_num_unfinished_requests()

    def has_unfinished_requests(self) -> bool:
        """Returns True if there are unfinished requests."""
        return self.scheduler.has_unfinished_requests()

    def _process_model_outputs(
        self, outputs: list[ExecutorResult], ignored_requests: list[EngineRequest]
    ) -> tuple[list[RequestOutput], list[RequestInput]]:
        """Apply the model output to the sequences in the scheduled seq groups.

        Returns RequestOutputs that can be returned to the client.
        """

        # Forward Processing or create the outputs and update the requests.
        # Pre-step errors (unresolvable routing, expired deadlines) were never
        # scheduled; resolve them first so the caller sees them immediately.
        request_outputs: list[RequestOutput] = self._drain_pending_errors()
        forwarded_requests: list[RequestInput] = []
        for result in outputs:
            # Check for Re-Schedule conditions
            new_status, forwarding_request = self.check_forward_processing(result)
            if new_status == RequestStatus.FINISHED_STOPPED:
                # Finished normally, return output
                request_outputs.append(result.result_data)
            if forwarding_request is not None:
                forwarded_requests.append(forwarding_request)

        for request in ignored_requests:
            # Mark as Finished / Ignored
            self.scheduler.update_request_status(request.request_id, RequestStatus.FINISHED_IGNORED)
            model_name = getattr(request.request_data, "model", None)
            if model_name not in getattr(self, "model_executors", {}):
                # Defence in depth: a request that reached the scheduler with an
                # unknown model (e.g. a future routing path) must error out
                # instead of being dropped silently.
                request_output = RequestOutput(
                    request_id=request.request_id,
                    finish_reason="error",
                    error=f"Request routed to unknown model '{model_name}', which is not configured on this engine.",
                )
            else:
                request_output = RequestOutput(
                    request_id=request.request_id, finish_reason=RequestStatus.get_finished_reason(request.status)
                )
            request_outputs.append(request_output)

        # Free the finished requests.
        self.scheduler.free_finished_requests()

        return request_outputs, forwarded_requests

    def step(self) -> tuple[list[RequestOutput], list[RequestInput]]:
        """Performs one decoding iteration and returns newly generated results.

        .. figure:: https://i.imgur.com/sv2HssD.png
            :alt: Overview of the step function
            :align: center

            Overview of the step function.

        Details:
            - Step 1: Schedules the sequences to be executed in the next
              iteration and the token blocks to be swapped in/out/copy.

                - Depending on the scheduling policy,
                  sequences may be `preempted/reordered`.
                - A Sequence Group (SG) refer to a group of sequences
                  that are generated from the same prompt.

            - Step 2: Calls the distributed executor to execute the model.
            - Step 3: Processes the model output. This mainly includes:

                - Decodes the relevant outputs.
                - Updates the scheduled sequence groups with model outputs
                  based on its `sampling parameters` (`use_beam_search` or not).
                - Frees the finished sequence groups.

            - Finally, it creates and returns the newly generated results.

        Example:
            >>> # Please see the example/ folder for more detailed examples.
            >>>
            >>> # initialize engine and request arguments
            >>> engine = AphroditeEngine.from_engine_args_and_config(engine_args)
            >>> example_inputs = [(0, "What is LLM?",
            >>>    SamplingParams(temperature=0.0))]
            >>>
            >>> # Start the engine with an event loop
            >>> while True:
            >>>     if example_inputs:
            >>>         req_id, prompt, sampling_params = example_inputs.pop(0)
            >>>         engine.add_request(str(req_id), prompt, sampling_params)
            >>>
            >>>     # continue the request processing
            >>>     request_outputs = engine.step()
            >>>     for request_output in request_outputs:
            >>>         if request_output.finished:
            >>>             # return or show the request output
            >>>
            >>>     if not (engine.has_unfinished_requests() or example_inputs):
            >>>         break
        """
        # Enforce per-request deadlines before scheduling another batch: an
        # expired request must resolve as an error, never be rescheduled forever.
        self._queue_pending_errors(self._expire_overdue_requests())

        scheduler_outputs = self.scheduler.schedule()

        output = []
        if not scheduler_outputs.is_empty():
            # Process per executor, in parallel (TODO: Verify if that works or GIL makes problems here)
            with ThreadPoolExecutor(len(scheduler_outputs.scheduled_requests_per_model.keys())) as ex:
                futures = []
                for model_name, model_requests in scheduler_outputs.scheduled_requests_per_model.items():
                    futures.append(ex.submit(self.model_executors[model_name].execute_model, model_requests))

                for future in futures:
                    model_results = future.result()
                    output.extend(model_results)

        request_outputs, forwarded_requests = self._process_model_outputs(
            outputs=output, ignored_requests=scheduler_outputs.ignored_requests
        )

        # Log stats.
        # if self.log_stats:
        #     self.stat_logger.log(
        #         self._get_stats(scheduler_outputs, model_output=output))

        return request_outputs, forwarded_requests

    def do_log_stats(self) -> None:
        """Forced log when no requests active."""
        if self.log_stats:
            self.stat_logger.log(self._get_stats(scheduler_outputs=None))

    def _get_stats(self, scheduler_outputs: SchedulerOutputs | None) -> Stats:
        """Get Stats to be Logged to Prometheus.
        Args:
            scheduler_outputs: Optional, used to populate metrics related to
                the scheduled batch,
            model_output: Optional, used to emit speculative decoding metrics
                which are created by the workers.
        """
        now = time.monotonic()

        # Scheduler State
        num_running = len(self.scheduler.running)
        num_waiting = len(self.scheduler.waiting)

        return Stats(now=now, num_running=num_running, num_waiting=num_waiting)

    def check_health(self) -> dict:
        """Probe every model executor and return a structured health payload.

        Returns ``{"healthy": bool, "models": {<executor name>: {"healthy": bool,
        "model_type": str, "error": str | None}}}``. Per-model detail is the
        extension point for Phase 4-5 VRAM reporting.
        """
        models: dict[str, dict] = {}
        healthy = True
        for name, model_executor in getattr(self, "model_executors", {}).items():
            model_config = getattr(model_executor, "model_config", None)
            detail: dict = {"model_type": getattr(model_config, "model_type", None), "healthy": True}
            try:
                model_executor.check_health()
            except Exception as e:  # noqa: BLE001 - health probes must not raise
                healthy = False
                detail["healthy"] = False
                detail["error"] = str(e)
            models[name] = detail
        return {"healthy": healthy, "models": models}


setup_logger()

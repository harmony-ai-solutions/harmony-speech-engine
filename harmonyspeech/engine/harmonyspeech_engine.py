import os
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor

from loguru import logger

import harmonyspeech
from harmonyspeech.common.config import ModelConfig
from harmonyspeech.common.device import resolve_device
from harmonyspeech.common.inputs import (
    RequestInput,
    TextToSpeechRequestInput,
    prefill_audio_artifacts,
)
from harmonyspeech.common.logger import setup_logger
from harmonyspeech.common.outputs import RequestOutput, TextToSpeechRequestOutput
from harmonyspeech.common.request import EngineRequest, ExecutorResult, RequestStatus
from harmonyspeech.engine.args_tools import EngineArgs
from harmonyspeech.engine.embedding_cache_metrics import embedding_cache_hit_ratio
from harmonyspeech.engine.metrics import StatLogger, Stats
from harmonyspeech.engine.vram_budget import (
    allowed_windows,
    evaluate_whisper_budget,
    projected_whisper_peak,
)
from harmonyspeech.processing.pipeline_registry import (
    PipelineExecution,
    PipelineRoutingError,
    PipelineSpec,
    PipelineState,
    advance,
    completed_step_state,
    get_spec,
    next_step,
    resolve_entry,
    validate_families,
    validate_registry,
)
from harmonyspeech.processing.scheduler import (
    AR_INFERENCE_MODEL_TYPES,
    Scheduler,
    SchedulerOutputs,
)

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
        strict_pipelines: bool = False,
        embedding_cache_enabled: bool = True,
        embedding_cache_ttl_s: float = 300.0,
        embedding_cache_max_entries: int = 1000,
        embedding_cache_max_bytes: int = 1073741824,
        whisper_vram_budget_bytes: int | None = None,
        whisper_max_audio_seconds: float = 300.0,
        ar_cache_enabled: bool = True,
        ar_cache_budget_bytes: int = 1073741824,
        ar_cache_default_cap_tokens: int = 768,
    ):
        self.model_configs = model_configs
        self.log_stats = log_stats
        # Per-request deadline (seconds); 0/None disables deadline enforcement.
        self.request_timeout_s = request_timeout_s
        # Turn "pipeline incomplete" startup warnings into a boot error.
        self.strict_pipelines = strict_pipelines
        # Whisper VRAM governance (Phase 10-3): optional hard device-memory
        # budget for the projected Whisper footprint, and the longest audio a
        # single request may carry (the probe's max shape).
        self.whisper_vram_budget_bytes = whisper_vram_budget_bytes
        self.whisper_max_audio_seconds = whisper_max_audio_seconds
        # Boot-time governance report, populated after the executors load.
        self.whisper_vram_report: dict | None = None
        # Requests rejected before ever entering the scheduler (unresolvable
        # routing, expired deadlines). Drained into the next step's outputs so
        # their HTTP call resolves with an error instead of hanging.
        self._pending_error_outputs: list[RequestOutput] = []
        self._verify_args()

        # One shared-weight registry per engine (Phase 4-3): executor configs
        # that opt into the same `backbone_group` load shared backbone modules
        # once instead of duplicating them.
        from harmonyspeech.modeling.weight_registry import WeightRegistry

        self.weight_registry = WeightRegistry()

        # One cross-request embedding/conditionals cache per engine (Phase 7-1),
        # threaded to the runners with the same plumbing as the weight registry.
        # `embedding_cache_enabled=False` (or `ttl_s=0`) leaves it None: handlers
        # then take the no-cache fast path and behavior matches pre-Phase-7.
        from harmonyspeech.processing.embedding_cache import EmbeddingCache

        self.embedding_cache = (
            EmbeddingCache(
                ttl_s=embedding_cache_ttl_s,
                max_entries=embedding_cache_max_entries,
                max_bytes=embedding_cache_max_bytes,
            )
            if embedding_cache_enabled and embedding_cache_ttl_s and embedding_cache_ttl_s > 0
            else None
        )

        # One device-wide AR KV-cache pool per engine (Phase 9-2), threaded to
        # the runners exactly like the weight registry and the embedding cache.
        # `--no-ar-cache` leaves it None: consumers then take their sequential
        # fallback and /metrics keeps every series present as zeros.
        from harmonyspeech.engine.ar_cache import ARCachePool

        self.ar_cache_enabled = bool(ar_cache_enabled)
        self.ar_cache_budget_bytes = ar_cache_budget_bytes
        self.ar_cache_default_cap_tokens = ar_cache_default_cap_tokens
        self.ar_cache_pool = (
            ARCachePool(
                budget_bytes=ar_cache_budget_bytes,
                default_cap_tokens=ar_cache_default_cap_tokens,
            )
            if self.ar_cache_enabled
            else None
        )

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
        # The probe needs the engine-level audio cap; attach it to the Whisper
        # configs before the executors (and their runners) are constructed.
        self._apply_whisper_probe_defaults()
        self.model_executors = {}
        self.init_custom_executors()

        # Phase 9-2: log the AR cache arena table once the executors (and their
        # runners) had the chance to register arenas during model load, the same
        # way the weight registry reports its shared components at boot.
        self._log_ar_cache_arena_table()

        # Startup validation: which declared toolchains are complete?
        self.available_families: list[str] = self._validate_family_configuration()

        # Per-AR-model decode progress accounting (Phase 9-4): cumulative
        # tokens/steps plus the last token count seen per in-flight request, so
        # `get_ar_decode_stats()` can report a real tokens/step figure without
        # the scheduler having to know about metrics.
        self._ar_progress: dict[str, dict] = {}

        # Initialize Scheduler for requests across models. Phase 10-1: the
        # Whisper lanes consult the Phase 10-3 memory governor at admission, so
        # lane credit × capacity × measured VRAM compose into one width.
        # Phase 9-4: AR admission consults the 9-2 pool's per-family capacity
        # (free slots) the same injected-callable way.
        self.scheduler = Scheduler(
            model_configs=self.model_configs,
            whisper_admission=self.get_whisper_admission,
            ar_admission=self.get_ar_cache_capacity,
        )

        # Whisper VRAM governance (Phase 10-3): compare the measured/projected
        # Whisper footprint with the free device headroom and the optional
        # budget. May raise when `whisper_vram_budget_bytes` is exceeded.
        self.whisper_vram_report = self._evaluate_whisper_memory_governance()

        # Metric Logging.
        if self.log_stats:
            self.stat_logger = StatLogger(local_interval=_LOCAL_LOGGING_INTERVAL_SEC, labels=dict())

    def _apply_whisper_probe_defaults(self) -> None:
        """Give Whisper configs the engine-level audio cap the probe needs.

        10-1/10-2 add per-model worker/window/beam knobs; until then the probe
        reads them through ``getattr`` defaults. The audio cap is engine-level
        (``whisper_max_audio_seconds``), so it is stamped onto each Whisper
        config before the runner is built.
        """
        for config in self.model_configs or []:
            if getattr(config, "model_type", None) != "FasterWhisper":
                continue
            if not hasattr(config, "whisper_max_audio_seconds"):
                config.whisper_max_audio_seconds = self.whisper_max_audio_seconds


    def _validate_family_configuration(self) -> list[str]:
        """Log incomplete toolchains and return the complete family ids.

        A pipeline is complete when every mandatory step can be resolved to at
        least one configured model. Boot does not fail on an incomplete pipeline
        (operators may run subsets), but the serving layer only advertises
        complete ones — unless ``strict_pipelines`` is enabled, in which case the
        same message becomes a boot error.
        """
        incomplete = validate_registry(self.model_configs)
        for spec_id, missing_steps in incomplete.items():
            logger.warning(
                f"Pipeline '{spec_id}' incomplete — missing step(s): {', '.join(missing_steps)}. "
                f"Requests routed to '{spec_id}' will fail with an error and the pipeline is not "
                f"advertised on /v1/models."
            )
        complete_families = validate_families(self.model_configs)
        logger.info(
            "Complete model toolchains: "
            f"{', '.join(complete_families) if complete_families else 'none (no family fully configured)'}"
        )
        if getattr(self, "strict_pipelines", False) and incomplete:
            raise RuntimeError(
                "Pipeline configuration is incomplete and --strict-pipelines is enabled: "
                + "; ".join(
                    f"pipeline '{spec_id}' missing step(s): {', '.join(steps)}"
                    for spec_id, steps in incomplete.items()
                )
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
            executor = executor_class(
                model_config=model_cfg,
                weight_registry=self.weight_registry,
                embedding_cache=self.embedding_cache,
                ar_cache_pool=getattr(self, "ar_cache_pool", None),
            )
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

    # ------------------------------------------------------------------
    # Declarative pipeline routing (Phase 3)
    # ------------------------------------------------------------------
    #
    # Both former if/elif ladders are gone: routing is `next_step()` on the
    # request's PipelineState, forwarding is `advance()` over the step's
    # declared artifacts. Everything a pipeline can do is declared once in
    # `harmonyspeech.processing.pipeline_registry.PIPELINE_SPECS`.

    def _model_config_by_name(self, name: str | None):
        """Return the configured model with ``name``, or ``None``."""
        for config in self.model_configs or []:
            if config.name == name:
                return config
        return None

    def _pipeline_incomplete_message(self, spec: PipelineSpec, state: PipelineState | None) -> str:
        """Explain which step/model is missing for ``spec`` (typed routing error)."""
        incomplete = validate_registry(self.model_configs)
        step = spec.step_at(state.step_index) if state is not None else None
        parts: list[str] = []
        if step is not None:
            parts.append(
                f"step '{step.name}' has no matching model "
                f"(missing one of: {', '.join(step.requires)})"
            )
        missing_steps = incomplete.get(spec.id, [])
        if missing_steps:
            parts.append(f"incomplete pipeline steps: {', '.join(missing_steps)}")
        detail = "; ".join(parts) if parts else "no executor registered"
        return f"Pipeline '{spec.id}' is not fully configured on this engine — {detail}"

    def _unresolvable_model_error(self, request_data: RequestInput, state: PipelineState | None = None) -> str:
        """Describe why a request cannot be routed to a configured executor."""
        spec = get_spec(getattr(request_data, "requested_model", None))
        if spec is None:
            return (
                f"Request routed to unknown model '{getattr(request_data, 'model', 'unknown')}', "
                f"which is not configured on this engine."
            )
        return self._pipeline_incomplete_message(spec, state)

    def _route_request(
        self,
        request_data: RequestInput,
        artifacts: dict | None = None,
        pipeline_state: PipelineState | None = None,
    ) -> tuple[PipelineState | None, str | None]:
        """Resolve ``request_data.model`` for the request's current step.

        Returns ``(state, error_message)``. A forwarded request arrives with its
        state and never re-resolves entry rules; a fresh request gets its state
        from the pipeline's first matching :class:`EntryRule`.

        ``(None, None)`` means "not a pipeline request" — a concrete model name
        was requested directly (single-step semantics) or the name is unknown;
        both are handled by the existing admission checks.
        """
        spec = get_spec(getattr(request_data, "requested_model", None))
        if spec is None:
            return None, None

        state = pipeline_state
        if state is None:
            state = resolve_entry(spec, request_data, artifacts, self.model_configs)
        if state is None:
            return None, (
                f"Request shape is not supported by pipeline '{spec.id}' "
                f"(model '{getattr(request_data, 'requested_model', '')}')."
            )

        try:
            config = next_step(spec, state, request_data, self.model_configs, artifacts)
        except PipelineRoutingError as exc:
            return state, exc.message

        if config is None:
            return state, self._unresolvable_model_error(request_data, state)
        request_data.model = config.name
        return state, None

    def check_reroute_request_to_model(
        self,
        request: RequestInput,
        artifacts: dict | None = None,
        pipeline_state: PipelineState | None = None,
    ) -> PipelineState | None:
        """Set ``request.model`` for the request's current pipeline step.

        Thin wrapper around :meth:`_route_request` kept as the documented
        routing entry point; the admission path calls ``_route_request``
        directly so it can surface the typed error message.
        """
        state, _error = self._route_request(request, artifacts, pipeline_state)
        return state

    def _find_engine_request(self, request_id: str):
        """Return the in-flight ``EngineRequest`` for ``request_id`` (or ``None``)."""
        scheduler = getattr(self, "scheduler", None)
        for attr in ("waiting", "running"):
            try:
                queue = list(getattr(scheduler, attr, None) or [])
            except TypeError:  # mocked scheduler in unit tests
                continue
            for request in queue:
                if getattr(request, "request_id", None) == request_id:
                    return request
        return None

    # ------------------------------------------------------------------
    # Multi-step decode plumbing (Phase 9-4)
    # ------------------------------------------------------------------
    @staticmethod
    def _is_unfinished_result(result: ExecutorResult) -> bool:
        """True when an executor result is a partial decode result.

        Keyed on the explicit ``unfinished`` marker, not a null finish reason:
        an error result also carries ``finish_reason`` and must go through the
        normal terminal path.
        """
        data = getattr(result, "result_data", None)
        return bool(getattr(data, "unfinished", False)) and getattr(data, "finish_reason", None) is None

    def _ar_progress_state(self) -> dict:
        """Return the per-model decode-progress accumulator (created lazily)."""
        state = getattr(self, "_ar_progress", None)
        if state is None:
            state = {}
            self._ar_progress = state
        return state

    def _record_unfinished_progress(self, result: ExecutorResult) -> None:
        """Record a partial decode result and keep the request RUNNING.

        The client never sees this result: no ``RequestOutput`` is returned and
        the scheduler status stays RUNNING, so the request is scheduled again
        next step. Progress is request-scoped and cleared on the terminal
        transition (Phase 9-4).
        """
        progress = dict(getattr(result.result_data, "progress", None) or {})
        request = self._find_engine_request(result.request_id)
        if request is not None:
            request.set_progress(progress)
        model_name = getattr(getattr(result, "input_data", None), "model", None)
        self._note_ar_progress(model_name, result.request_id, progress)

    def _note_ar_progress(self, model_name, request_id: str, progress: dict) -> None:
        """Accumulate the token delta from one partial/final result."""
        if not model_name:
            return
        state = self._ar_progress_state().setdefault(
            model_name, {"tokens_total": 0, "steps_total": 0, "last_tokens": {}}
        )
        try:
            tokens = int(progress.get("tokens") or 0)
        except (TypeError, ValueError):  # pragma: no cover - defensive on odd progress
            tokens = 0
        last = int(state["last_tokens"].get(request_id, 0) or 0)
        if tokens > last:
            state["tokens_total"] += tokens - last
            state["last_tokens"][request_id] = tokens

    def _count_ar_step(self, model_name) -> None:
        """Count one engine step in which an AR model advanced at least one row."""
        if not model_name:
            return
        state = self._ar_progress_state().setdefault(
            model_name, {"tokens_total": 0, "steps_total": 0, "last_tokens": {}}
        )
        state["steps_total"] += 1

    def _forget_ar_progress(self, request_id: str) -> None:
        """Drop the per-request token watermark once a request is terminal."""
        for entry in self._ar_progress_state().values():
            entry.get("last_tokens", {}).pop(request_id, None)

    def _release_request_resources(self, request_ids) -> None:
        """Tell executors a request will never be scheduled again.

        Duck-typed hook (``release_request(request_id)``): the AR runner (9-3)
        uses it to drop a decode row and release its 9-2 ``ARCacheSlot`` when a
        client disconnects or a deadline expires mid-decode. Executors without
        the hook are unaffected, and a hook failure is logged rather than
        allowed to break the engine loop — aborting one request must never
        affect the others.
        """
        ids = [request_ids] if isinstance(request_ids, str) else list(request_ids)
        if not ids:
            return
        for executor in getattr(self, "model_executors", {}).values():
            hook = getattr(executor, "release_request", None)
            if not callable(hook):
                continue
            for request_id in ids:
                try:
                    hook(request_id)
                except Exception as error:  # noqa: BLE001 - cleanup must not break the loop
                    logger.warning(f"Failed to release resources for request {request_id}: {error}")
        for request_id in ids:
            self._forget_ar_progress(request_id)

    def _finalize_pipeline_output(self, result: ExecutorResult, input_data: RequestInput) -> None:
        """Map the final step's output onto the endpoint's response type.

        Speech endpoints speak ``TextToSpeechRequestOutput``; every other
        endpoint keeps the executor's own output class. This is the one
        spec-independent conversion left in the forwarding path.
        """
        if not isinstance(input_data, TextToSpeechRequestInput):
            return
        if isinstance(result.result_data, TextToSpeechRequestOutput):
            return
        result.result_data = TextToSpeechRequestOutput(
            request_id=result.request_id,
            text=getattr(input_data, "input_text", "") or "",
            output=result.result_data.output,
            finish_reason=result.result_data.finish_reason,
            metrics=result.result_data.metrics if result.result_data.metrics else None,
            audio_artifact=getattr(result.result_data, "audio_artifact", None),
        )

    def check_forward_processing(self, result: ExecutorResult):
        input_data = result.input_data
        requested_model = getattr(input_data, "requested_model", None)
        returning_model_cfg = self._model_config_by_name(getattr(input_data, "model", None))

        # If inference produced an error result (e.g. bad input, model failure),
        # do NOT attempt to forward it through the multi-step pipeline. Mark the
        # request as finished so the scheduler can clean it up, and let the error
        # output flow back to the caller. Without this, a failed request would
        # either be incorrectly re-forwarded or leak in the scheduler's running
        # queue forever.
        if isinstance(result.result_data, RequestOutput) and result.result_data.finish_reason == "error":
            self.scheduler.update_request_status(result.request_id, RequestStatus.FINISHED_STOPPED)
            return RequestStatus.FINISHED_STOPPED, None

        # Explicit model selected (a concrete configured executor name equal to
        # the requested model), no forwarding needed.
        if returning_model_cfg is not None and returning_model_cfg.name == requested_model:
            self.scheduler.update_request_status(result.request_id, RequestStatus.FINISHED_STOPPED)
            return RequestStatus.FINISHED_STOPPED, None

        spec = get_spec(requested_model)
        if spec is None:
            # Unknown pipeline / direct single-step executor: nothing to forward.
            self.scheduler.update_request_status(result.request_id, RequestStatus.FINISHED_STOPPED)
            return RequestStatus.FINISHED_STOPPED, None

        engine_request = self._find_engine_request(result.request_id)
        context = engine_request if engine_request is not None else PipelineExecution(request_data=input_data)
        if getattr(context, "pipeline_state", None) is None and returning_model_cfg is not None:
            # Defensive fallback (the EngineRequest is normally found): the step
            # that just ran is unambiguous from the returned model's type.
            entry_state = resolve_entry(spec, input_data, getattr(context, "artifacts", None), self.model_configs)
            context.pipeline_state = completed_step_state(
                spec,
                returning_model_cfg.model_type,
                stop_after=entry_state.stop_after if entry_state is not None else None,
            )

        try:
            done = advance(spec, context, result.result_data)
        except PipelineRoutingError as exc:
            # A step failed to produce its declared artifact: fail the request
            # naming the step instead of forwarding a half-built state.
            logger.error(f"Request {result.request_id} failed in pipeline '{spec.id}': {exc.message}")
            result.result_data.finish_reason = "error"
            result.result_data.error = exc.message
            self.scheduler.update_request_status(result.request_id, RequestStatus.FINISHED_STOPPED)
            return RequestStatus.FINISHED_STOPPED, None

        if done:
            self._finalize_pipeline_output(result, input_data)
            self.scheduler.update_request_status(result.request_id, RequestStatus.FINISHED_STOPPED)
            return RequestStatus.FINISHED_STOPPED, None

        # Snapshot the artifact references before the finished request releases
        # its store (``update_request_status`` -> ``clear_artifacts``).
        artifacts = dict(getattr(context, "artifacts", None) or {})
        self.scheduler.update_request_status(result.request_id, RequestStatus.FINISHED_FORWARDED)
        self.add_request(
            result.request_id,
            input_data,
            artifacts=artifacts or None,
            pipeline_state=context.pipeline_state,
        )
        return RequestStatus.FINISHED_FORWARDED, input_data

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
                # Phase 9-4: an unfinished AR row may still hold a pool slot and
                # decoder row. The executor will not see this request again, so
                # release its resources here (the loop itself stays alive).
                self._release_request_resources([request.request_id])
                expired.append(
                    RequestOutput(
                        request_id=request.request_id,
                        finish_reason="error",
                        error=message,
                        metrics=request.metrics,
                    )
                )
        return expired

    def add_request(
        self,
        request_id: str,
        request_data: RequestInput,
        arrival_time: float | None = None,
        artifacts: dict | None = None,
        pipeline_state: PipelineState | None = None,
    ) -> None:
        """Add a Text-to-Speech request to the engine's request pool.

        The request is added to the request pool and will be processed by the
        scheduler as `engine.step()` is called. The exact scheduling policy is
        determined by the scheduler.

        Args:
            request_id: The unique ID of the request.
            request_data: The incoming Request data.
            arrival_time: The arrival time of the request. If None, we use
                the current monotonic time.
            artifacts: Typed in-process artifacts carried over from a forwarding
                step (Phase 2-3). When None, inbound audio/VAD fields are decoded
                once here (Phase 2-2 admission decode).
            pipeline_state: Routing position carried over from a forwarding step.
                When None, the pipeline's entry rules decide where this request
                starts (Phase 3-4). Forwarded requests always pass it explicitly,
                so they never re-resolve entry.

        Details:
            - Set arrival_time to the current time if it is None.
            - Stamp a per-request deadline (`request_timeout_s` from now).
            - Reject requests that cannot be routed to a configured executor
              immediately, as a typed error output instead of leaving them to
              wait forever in the scheduler (Phase 0-4).
        """

        if arrival_time is None:
            arrival_time = time.monotonic()

        artifact_store = dict(artifacts) if artifacts else {}

        # Route to correct model if necessary. The pipeline state decides the
        # step; routing treats a cached artifact as a present field, so forwarded
        # steps need no legacy string sentinel.
        pipeline_state, routing_error = self._route_request(request_data, artifact_store, pipeline_state)

        model_executors = getattr(self, "model_executors", {})
        if routing_error is not None or request_data.model not in model_executors:
            error = routing_error or self._unresolvable_model_error(request_data, pipeline_state)
            logger.error(f"Request {request_id} rejected: {error}")
            self._queue_pending_errors(
                [RequestOutput(request_id=request_id, finish_reason="error", error=error)]
            )
            return

        # Boundary decode-once: the wire-encoded audio fields are turned into typed
        # artifacts exactly once, at admission. This is idempotent for forwarded
        # requests (fields already cached are skipped).
        artifact_store = prefill_audio_artifacts(request_data, artifact_store)

        timeout_s = getattr(self, "request_timeout_s", None)
        deadline = time.monotonic() + timeout_s if timeout_s else None

        # Add the request to the scheduler.
        self.scheduler.add_request(
            EngineRequest(
                request_id=request_id,
                request_data=request_data,
                arrival_time=arrival_time,
                deadline=deadline,
                artifacts=artifact_store,
                pipeline_state=pipeline_state,
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
        request_ids = [request_id] if isinstance(request_id, str) else list(request_id)
        self.scheduler.update_request_status(request_ids, RequestStatus.FINISHED_ABORTED)
        # Phase 9-4: a client disconnect/abort mid-decode must free the AR row
        # and its 9-2 cache slot immediately; only that request is affected.
        self._release_request_resources(request_ids)

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
        advanced_models: set[str] = set()
        for result in outputs:
            # Phase 9-4: a partial decode result keeps the request RUNNING for
            # the next step and is never handed to the client.
            if self._is_unfinished_result(result):
                self._record_unfinished_progress(result)
                model_name = getattr(getattr(result, "input_data", None), "model", None)
                if model_name:
                    advanced_models.add(model_name)
                continue
            # A terminal result may optionally carry the final progress payload
            # (the runner knows the last token count); count it before the
            # request state is cleared.
            final_progress = getattr(result.result_data, "progress", None)
            if final_progress:
                model_name = getattr(getattr(result, "input_data", None), "model", None)
                self._note_ar_progress(model_name, result.request_id, dict(final_progress))
                if model_name:
                    advanced_models.add(model_name)
            # Check for Re-Schedule conditions
            new_status, forwarding_request = self.check_forward_processing(result)
            if new_status == RequestStatus.FINISHED_STOPPED:
                # Finished normally, return output
                request_outputs.append(result.result_data)
            if forwarding_request is not None:
                forwarded_requests.append(forwarding_request)
            self._forget_ar_progress(result.request_id)

        # One engine step per model that advanced at least one AR row, so
        # tokens/steps approximates the decode batch width (Phase 9-4 metrics).
        for model_name in advanced_models:
            self._count_ar_step(model_name)

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
        "model_type": str, "vram_bytes": int, "error": str | None}},
        "vram": {...}, "whisper_vram": {...} | None, "embedding_cache": {...},
        "ar_cache": {...}}``.
        ``vram`` is the Phase 4-5/10-3 summary (see :meth:`get_vram_stats`);
        ``whisper_vram`` is the Phase 10-3 boot governance report;
        ``embedding_cache`` is the Phase 7-3 cache accounting (see
        :meth:`get_embedding_cache_stats`); ``ar_cache`` is the Phase 9-2 pool
        accounting (see :meth:`get_ar_cache_stats`).
        """
        vram = self.get_vram_stats()
        models: dict[str, dict] = {}
        healthy = True
        for name, model_executor in getattr(self, "model_executors", {}).items():
            model_config = getattr(model_executor, "model_config", None)
            vram_detail = vram["models"].get(name, {})
            detail: dict = {
                "model_type": getattr(model_config, "model_type", None),
                "healthy": True,
                "vram_bytes": vram_detail.get("vram_bytes", 0),
            }
            # Phase 10-3: carry the measured (or explicitly skipped) Whisper
            # probe so /health never implies an attribution it does not have.
            if "peak_activation_bytes" in vram_detail:
                detail["peak_activation_bytes"] = vram_detail.get("peak_activation_bytes", 0)
                detail["probe_measured"] = vram_detail.get("probe_measured", False)
                detail["probe_reason"] = vram_detail.get("probe_reason")
                detail["per_window_bytes"] = vram_detail.get("per_window_bytes", 0)
            try:
                model_executor.check_health()
            except Exception as e:  # noqa: BLE001 - health probes must not raise
                healthy = False
                detail["healthy"] = False
                detail["error"] = str(e)
            models[name] = detail
        return {
            "healthy": healthy,
            "models": models,
            "vram": vram,
            "whisper_vram": self.get_whisper_vram_report(),
            # Phase 11-0: the governor's live admission decision, so the
            # runtime budget trip is observable on /health (it used to be
            # inferable only from the effect).
            "whisper_admission": self.get_whisper_admission(),
            "embedding_cache": self.get_embedding_cache_stats(),
            "ar_cache": self.get_ar_cache_stats(),
            # Phase 9-4: per-AR-model decode progress (active rows, slot
            # occupancy, tokens/step). Present with zeros when no AR model is
            # configured, so the shape never changes.
            "ar_decode": self.get_ar_decode_stats(),
        }

    # ------------------------------------------------------------------
    # Cross-request embedding cache accounting (Phase 7-3)
    # ------------------------------------------------------------------
    def get_embedding_cache_stats(self) -> dict:
        """Return the embedding-cache accounting used by ``/health`` + ``/metrics``.

        ``enabled`` is ``False`` and every value ``0`` when the cache is disabled
        at boot, so the exporter can keep every Prometheus series present.
        Robust on a shell engine (``__new__`` without ``__init__``) and on
        deployments that never constructed a cache.
        """
        cache = getattr(self, "embedding_cache", None)
        if cache is None:
            return {
                "enabled": False,
                "entries": 0,
                "bytes": 0,
                "hits": 0,
                "misses": 0,
                "stores": 0,
                "ttl_expirations": 0,
                "fifo_evictions": 0,
                "byte_evictions": 0,
                "hit_ratio": 0.0,
            }
        stats = dict(cache.stats())
        stats["enabled"] = True
        # Policy knobs ride along so an operator can see what the counters mean
        # without re-reading the config file.
        stats["ttl_s"] = getattr(cache, "ttl_s", None)
        stats["max_entries"] = getattr(cache, "max_entries", None)
        stats["max_bytes"] = getattr(cache, "max_bytes", None)
        stats["hit_ratio"] = embedding_cache_hit_ratio(stats)
        return stats

    # ------------------------------------------------------------------
    # Shared AR KV-cache pool accounting (Phase 9-2)
    # ------------------------------------------------------------------
    def _log_ar_cache_arena_table(self) -> None:
        """Log the registered arena table (family, slots, bytes) at startup.

        Runs after the executors load, so any arena a runner registered during
        ``load_model`` is included. With no consumer yet (9-3) the table is
        empty and the log states that explicitly — silence would be ambiguous.
        """
        pool = getattr(self, "ar_cache_pool", None)
        if pool is None:
            logger.info("AR KV-cache pool disabled (--no-ar-cache / ar_inference.enabled: false).")
            return
        rows = pool.arena_table()
        logger.info(
            f"AR KV-cache pool: budget {pool.budget_bytes / 2**20:.1f} MiB per device, "
            f"default device {pool.device}, "
            f"default cap {pool.default_cap_tokens} tokens, {len(rows)} arena(s), "
            f"{pool.pool_bytes / 2**20:.1f} MiB resident."
        )
        if not rows:
            logger.info("AR KV-cache arena table: empty (no AR family registered an arena at load).")
            return
        logger.info(
            f"{'family':<30} {'device':>9} {'slots':>7} {'cap':>7} {'slab MiB':>10} "
            f"{'arena MiB':>10} {'dtype':>8}"
        )
        for row in rows:
            logger.info(
                f"{row['model_key']:<30} {row.get('device', '-'):>9} {row['slots']:>7} "
                f"{row['cap_tokens']:>7} {row['slab_bytes'] / 2**20:>10.1f} "
                f"{row['arena_bytes'] / 2**20:>10.1f} {row['dtype']:>8}"
            )

    def get_ar_cache_stats(self) -> dict:
        """Return the AR KV-cache pool accounting used by ``/health`` + ``/metrics``.

        ``enabled`` is ``False`` and every value ``0`` when the pool is disabled
        at boot, so the exporter can keep every Prometheus series present.
        Robust on a shell engine (``__new__`` without ``__init__``) and on
        deployments that never constructed a pool.
        """
        pool = getattr(self, "ar_cache_pool", None)
        if pool is None:
            return {
                "enabled": False,
                "budget_bytes": getattr(self, "ar_cache_budget_bytes", None),
                "default_cap_tokens": getattr(self, "ar_cache_default_cap_tokens", None),
                "device": None,
                "devices": {},
                "pool_bytes": 0,
                "allocated_bytes": 0,
                "slots": 0,
                "slots_active": 0,
                "slots_free": 0,
                "active_rows": 0,
                "grants_total": 0,
                "refusals_total": 0,
                "releases_total": 0,
                "arenas": {},
            }
        stats = dict(pool.stats())
        stats["enabled"] = True
        return stats

    def get_ar_cache_capacity(self, model_key: str | None = None) -> dict:
        """Capacity-based admission input for Phase 9-4 (slots, not batch width).

        Returns the pool's per-family capacity row (or the aggregate with
        ``arenas`` when ``model_key`` is omitted). A disabled/absent pool reports
        zeros with ``enabled: False`` so a caller can always branch on the flag.
        """
        pool = getattr(self, "ar_cache_pool", None)
        if pool is None:
            return {
                "enabled": False,
                "device": None,
                "devices": {},
                "slots": 0,
                "slots_active": 0,
                "slots_free": 0,
                "active_rows": 0,
                "pool_bytes": 0,
                "allocated_bytes": 0,
                "arenas": {},
            }
        return pool.capacity(model_key)

    def get_ar_decode_stats(self) -> dict:
        """Per-AR-model decode progress (Phase 9-4).

        ``active_rows`` and slot occupancy come from the 9-2 pool's capacity
        (per model/arena; the scheduler's running-row count is the fallback when
        no arena is registered yet); ``tokens_total`` / ``steps_total`` are
        cumulative counters fed by every partial executor result, and
        ``tokens_per_step`` is their ratio. Exposed on ``/health`` (``ar_decode``)
        and rendered on ``/metrics``. With no AR model configured the payload is
        ``{"enabled": False, "models": {}}`` so the shape never changes.
        """
        running_rows: dict[str, int] = {}
        scheduler = getattr(self, "scheduler", None)
        for request in getattr(scheduler, "running", None) or []:
            model_name = getattr(getattr(request, "request_data", None), "model", None)
            if model_name is not None:
                running_rows[model_name] = running_rows.get(model_name, 0) + 1

        progress_state = getattr(self, "_ar_progress", None) or {}
        models: dict[str, dict] = {}
        for config in getattr(self, "model_configs", None) or []:
            if getattr(config, "model_type", None) not in AR_INFERENCE_MODEL_TYPES:
                continue
            name = config.name
            try:
                capacity = self.get_ar_cache_capacity(name) or {}
            except KeyError:
                # No arena registered for this family yet (9-3).
                capacity = {}
            if capacity.get("enabled"):
                active_rows = int(capacity.get("active_rows", 0) or 0)
                slots = int(capacity.get("slots", 0) or 0)
                slots_active = int(capacity.get("slots_active", 0) or 0)
                slots_free = int(capacity.get("slots_free", 0) or 0)
            else:
                # Boolean/sequential fallback: rows are still real, slots are not.
                active_rows = running_rows.get(name, 0)
                slots = slots_active = slots_free = 0
            entry = progress_state.get(name) or {}
            tokens_total = int(entry.get("tokens_total", 0) or 0)
            steps_total = int(entry.get("steps_total", 0) or 0)
            models[name] = {
                "model_type": getattr(config, "model_type", None),
                "pool_enabled": bool(capacity.get("enabled")),
                "active_rows": active_rows,
                "slots": slots,
                "slots_active": slots_active,
                "slots_free": slots_free,
                "tokens_total": tokens_total,
                "steps_total": steps_total,
                "tokens_per_step": (tokens_total / steps_total) if steps_total else 0.0,
            }
        return {"enabled": bool(models), "models": models}

    # ------------------------------------------------------------------
    # VRAM accounting (Phase 4-5) and device-level truth (Phase 10-3)
    # ------------------------------------------------------------------
    def _vram_device_index(self) -> int | None:
        """CUDA index the device-level VRAM series describe (Phase 11-0).

        Multi-device placement means "the process's active device" is no longer
        a safe default: a worker for an explicit ``cuda:1`` model sets it, and
        the last worker to initialise wins. Report the device the catalogue
        actually uses — the first CUDA runner's resolved device, then the first
        CUDA config, then the process device. ``None`` on a CPU-only engine.
        """
        import torch

        for executor in (getattr(self, "model_executors", None) or {}).values():
            runner = getattr(getattr(executor, "driver_worker", None), "model_runner", None)
            raw_device = getattr(runner, "device", None)
            if not isinstance(raw_device, (str, torch.device)):
                # Test doubles / shell engines may carry a mock here.
                continue
            device = torch.device(raw_device)
            if device.type == "cuda" and device.index is not None:
                return int(device.index)
        for config in getattr(self, "model_configs", None) or []:
            raw_device = getattr(getattr(config, "device_config", None), "device", None)
            if not isinstance(raw_device, (str, torch.device)):
                continue
            device = resolve_device(raw_device)
            if device.type == "cuda" and device.index is not None:
                return int(device.index)
        if torch.cuda.is_available():
            return int(torch.cuda.current_device())
        return None

    def get_vram_stats(self) -> dict:
        """Return per-model VRAM accounting plus shared-backbone group totals.

        Per-model ``vram_bytes`` is the CUDA allocation delta measured while
        that model loaded (``GPUModelRunner.model_memory_usage``); on CPU it is
        ``0`` and ``weights_bytes`` (host-resident parameter/buffer size, or the
        checkpoint size for native CTranslate2 models) is still reported.
        Shared backbone components are listed once under ``shared`` and
        aggregated per ``backbone_group`` so a deduplicated deployment is not
        counted twice.

        Device-level truth (Phase 10-3): ``device_used_bytes`` / ``free_bytes``
        come from ``torch.cuda.mem_get_info`` and include everything on the
        device — torch allocations, CTranslate2 (Whisper) weights/activations
        and CUDA contexts. ``headroom_bytes`` is therefore the **true free
        device memory** (a documented breaking change: it used to be
        ``total - torch_reserved`` and overstated free memory whenever a
        non-torch model held memory). The torch-scoped counters
        (``allocated_bytes`` / ``reserved_bytes`` / ``peak_allocated_bytes``)
        are kept because they still answer "what does torch hold".

        Safe on CPU-only builds: CUDA fields are ``0``/``None`` and nothing
        here raises.
        """
        import torch

        cuda = False
        allocated = reserved = peak = total = 0
        free_bytes = device_used_bytes = 0
        device_used_source = None
        device_name = None
        device_index = self._vram_device_index()
        if torch.cuda.is_available():
            try:
                cuda = True
                allocated = int(torch.cuda.memory_allocated(device_index))
                reserved = int(torch.cuda.memory_reserved(device_index))
                peak = int(torch.cuda.max_memory_allocated(device_index))
                total = int(torch.cuda.get_device_properties(device_index).total_memory)
                device_name = torch.cuda.get_device_name(device_index)
                try:
                    # Device-level truth: captures torch + CTranslate2 + CUDA
                    # contexts. `mem_get_info` returns (free, total).
                    free_raw, total_raw = torch.cuda.mem_get_info(device_index)
                    free_bytes = int(free_raw)
                    total = int(total_raw)
                    device_used_bytes = max(0, total - free_bytes)
                    device_used_source = "mem_get_info"
                except Exception:  # pragma: no cover - defensive on odd driver states
                    # Best-effort fallback: the old torch-scoped number is still
                    # an honest lower bound on non-torch usage (it cannot see
                    # CTranslate2), so mark the source explicitly.
                    free_bytes = max(0, total - reserved)
                    device_used_bytes = max(0, total - free_bytes)
                    device_used_source = "torch_reserved_fallback"
            except Exception:  # pragma: no cover - defensive on odd driver states
                cuda = False
                free_bytes = device_used_bytes = 0
                device_used_source = None

        registry = getattr(self, "weight_registry", None)
        shared = registry.stats() if registry is not None else {
            "entries": 0,
            "total_bytes": 0,
            "components": {},
        }

        # Map checkpoint repo -> backbone group name so shared bytes can be
        # attributed to the group operators recognise in the config. Repo ids
        # are normalised (case/hyphen-insensitive): configs spell them
        # lowercase while the HF cache and registry keys carry the on-disk name.
        def _repo_key(repo: str | None) -> str:
            return str(repo or "").lower().replace("-", "").replace("_", "")

        repo_to_group: dict[str, str] = {}
        models: dict[str, dict] = {}
        for name, model_executor in getattr(self, "model_executors", {}).items():
            model_config = getattr(model_executor, "model_config", None)
            runner = getattr(getattr(model_executor, "driver_worker", None), "model_runner", None)
            model = getattr(runner, "model", None)
            weights_bytes = 0
            if model is not None:
                try:
                    from harmonyspeech.modeling.weight_registry import model_nbytes

                    weights_bytes = model_nbytes(model)
                except Exception:  # pragma: no cover - non-module natives
                    weights_bytes = 0
            group = getattr(model_config, "backbone_group", None)
            repo = getattr(model_config, "model", None)
            if group and repo:
                repo_to_group[_repo_key(repo)] = group
            # A native model can be configured for CUDA but fall back to CPU
            # (CTranslate2/cuDNN guard); report where it actually runs so the
            # governance does not count host-resident Whisper as VRAM.
            native_device = getattr(getattr(model, "model", None), "device", None)
            detail = {
                "model_type": getattr(model_config, "model_type", None),
                "device": str(getattr(getattr(model_config, "device_config", None), "device", "")),
                "runtime_device": str(native_device) if native_device else None,
                "dtype": str(getattr(model_config, "dtype", "")) or None,
                "vram_bytes": int(getattr(runner, "model_memory_usage", 0) or 0),
                "weights_bytes": int(weights_bytes),
                "backbone_group": group,
                "max_batch_size": getattr(model_config, "max_batch_size", None),
            }
            # Phase 10-3: CTranslate2 Whisper memory is invisible to torch, so
            # the runner records a measured max-shape probe instead. Absent or
            # skipped probes stay explicit ("not-probed") — never guessed.
            probe = getattr(runner, "whisper_probe", None)
            if probe is not None:
                detail.update(
                    {
                        "peak_activation_bytes": int(probe.get("peak_activation_bytes", 0) or 0),
                        "per_window_bytes": int(probe.get("per_window_bytes", 0) or 0),
                        "probe_measured": bool(probe.get("measured")),
                        "probe_reason": probe.get("reason"),
                        "probe_method": probe.get("method"),
                        "workers": probe.get("workers"),
                        "window_batch": probe.get("window_batch"),
                        "beam_size": probe.get("beam_size"),
                        "max_audio_seconds": probe.get("max_audio_seconds"),
                        "windows_probed": probe.get("windows_probed"),
                    }
                )
            else:
                detail.update(
                    {
                        "peak_activation_bytes": 0,
                        "per_window_bytes": 0,
                        "probe_measured": False,
                        "probe_reason": "not-probed",
                    }
                )
            models[name] = detail

        groups: dict[str, int] = {}
        dedup_saved = 0
        for key_text, detail in shared.get("components", {}).items():
            repo = key_text.split("@", 1)[0]
            group = repo_to_group.get(_repo_key(repo))
            if group:
                groups[group] = groups.get(group, 0) + int(detail.get("bytes", 0))
            # Bytes that would have been re-loaded per additional acquirer.
            refs = int(detail.get("refs", 1))
            if refs > 1:
                dedup_saved += int(detail.get("bytes", 0)) * (refs - 1)

        return {
            "cuda": cuda,
            "device": device_name,
            "total_bytes": total,
            "allocated_bytes": allocated,
            "reserved_bytes": reserved,
            "peak_allocated_bytes": peak,
            # Phase 10-3: device-level truth (torch + CT2 + CUDA contexts).
            "device_used_bytes": device_used_bytes if cuda else 0,
            "free_bytes": free_bytes if cuda else 0,
            "device_used_source": device_used_source,
            # Redefined in place: true device free memory, not total-reserved.
            "headroom_bytes": free_bytes if cuda else 0,
            "resident_models": len(models),
            "models": models,
            "shared": shared,
            "groups": groups,
            "dedup_saved_bytes": dedup_saved,
            "whisper_budget_bytes": getattr(self, "whisper_vram_budget_bytes", None),
            "whisper_governance": getattr(self, "whisper_vram_report", None),
            # Phase 9-2: the AR pool's slabs are ordinary torch allocations, so
            # they are already inside `allocated_bytes` / `device_used_bytes`
            # above. Report them once here (device-wide pool, not attributable
            # to a single model) so the footprint is explicit, never hidden.
            "ar_cache": self.get_ar_cache_stats(),
        }

    # ------------------------------------------------------------------
    # Whisper memory governance (Phase 10-3)
    # ------------------------------------------------------------------
    def _whisper_vram_entries(self, stats: dict | None = None) -> list[dict]:
        """Return the CUDA-placed Whisper model entries from VRAM stats."""
        stats = stats if stats is not None else self.get_vram_stats()
        entries = []
        for name, detail in (stats.get("models") or {}).items():
            if detail.get("model_type") != "FasterWhisper":
                continue
            # Prefer the actual device: a CUDA-configured Whisper that fell back
            # to CPU (cuDNN guard) must not be counted as VRAM.
            device = detail.get("runtime_device") or detail.get("device")
            if not str(device or "").startswith("cuda"):
                continue
            entries.append({"name": name, **detail})
        return entries

    def _evaluate_whisper_memory_governance(self) -> dict:
        """Boot-time Whisper budget check + probe table (Phase 10-3).

        Compares the projected Whisper footprint (checkpoint weights + measured
        max-shape probe peaks) with the true free device headroom. With
        ``whisper_vram_budget_bytes`` configured an overshoot is a hard boot
        error; without it the check only warns. On a CPU-only build there is no
        device-level VRAM story, so the check is skipped explicitly.
        """
        stats = self.get_vram_stats()
        entries = self._whisper_vram_entries(stats)
        if not stats.get("cuda"):
            report = {
                "enabled": False,
                "status": "cpu",
                "message": "Device-level VRAM governance is unavailable without CUDA.",
                "projected_peak_bytes": 0,
                "free_headroom_bytes": 0,
                "budget_bytes": getattr(self, "whisper_vram_budget_bytes", None),
                "budget_configured": getattr(self, "whisper_vram_budget_bytes", None) is not None,
                "exceeded_budget": False,
                "exceeded_headroom": False,
                "models": [],
                "measured_models": 0,
            }
            logger.info(report["message"])
            return report

        if not entries:
            report = {
                "enabled": False,
                "status": "no-whisper",
                "message": "No CUDA-placed Whisper config is resident; VRAM governance is idle.",
                "projected_peak_bytes": 0,
                "free_headroom_bytes": int(stats.get("free_bytes") or 0),
                "budget_bytes": getattr(self, "whisper_vram_budget_bytes", None),
                "budget_configured": getattr(self, "whisper_vram_budget_bytes", None) is not None,
                "exceeded_budget": False,
                "exceeded_headroom": False,
                "models": [],
                "measured_models": 0,
            }
            logger.info(report["message"])
            return report

        projected = projected_whisper_peak(entries)
        free_headroom = int(stats.get("free_bytes") or 0)
        budget = getattr(self, "whisper_vram_budget_bytes", None)
        if stats.get("device_used_source") == "torch_reserved_fallback":
            logger.warning(
                "Device-level VRAM truth (torch.cuda.mem_get_info) is unavailable; "
                "Whisper governance is using the torch-reserved fallback, so the "
                "free-headroom figure may be too high while CTranslate2 holds memory."
            )
        verdict = evaluate_whisper_budget(projected, free_headroom, budget)

        logger.info("Whisper VRAM governance (Phase 10-3):")
        logger.info(
            f"{'config':<34} {'workers':>7} {'windows':>7} {'beam':>4} "
            f"{'max audio s':>11} {'weights MiB':>11} {'probe peak MiB':>14} {'probe':>10}"
        )
        for entry in entries:
            if entry.get("probe_measured"):
                probe_state = "measured"
            else:
                probe_state = f"skipped:{entry.get('probe_reason') or 'unknown'}"
            logger.info(
                f"{str(entry.get('name')):<34} "
                f"{str(entry.get('workers') if entry.get('workers') is not None else '-'):>7} "
                f"{str(entry.get('window_batch') if entry.get('window_batch') is not None else '-'):>7} "
                f"{str(entry.get('beam_size') if entry.get('beam_size') is not None else '-'):>4} "
                f"{float(entry.get('max_audio_seconds') or 0.0):>11.0f} "
                f"{int(entry.get('weights_bytes') or 0) / 2**20:>11.1f} "
                f"{int(entry.get('peak_activation_bytes') or 0) / 2**20:>14.1f} "
                f"{probe_state:>10}"
            )

        report = {
            "enabled": True,
            **verdict,
            "models": entries,
            "measured_models": sum(1 for entry in entries if entry.get("probe_measured")),
        }
        if verdict["status"] == "error":
            raise RuntimeError(verdict["message"])
        if verdict["status"] == "warn":
            logger.warning(verdict["message"])
        else:
            logger.info(verdict["message"])
        return report

    def get_whisper_vram_report(self) -> dict | None:
        """Return the boot-time Whisper governance report (``None`` on a shell engine)."""
        return getattr(self, "whisper_vram_report", None)

    def get_whisper_admission(self, active_windows: int = 0, *, min_windows: int = 1) -> dict:
        """Runtime admission-scaling hook for 10-1/10-2 (Phase 10-3).

        Reads device-level used memory (``torch.cuda.mem_get_info``), the
        optional budget (or the device total when unset) and the measured
        per-window cost, then returns how many Whisper windows may be in flight.
        The decision itself is the pure
        :func:`~harmonyspeech.engine.vram_budget.allowed_windows` function, so
        10-5 can drive it with a mocked ``mem_get_info``.

        ``min_windows`` is the draining floor (default ``1``); pass ``0`` for a
        hard stop when testing exhaustion behaviour.

        Returns a dict with ``enabled``, ``allowed_windows`` and every input
        used, so the caller (and tests) can explain the decision.
        """
        stats = self.get_vram_stats()
        entries = self._whisper_vram_entries(stats)
        if not entries:
            return {
                "enabled": False,
                "allowed_windows": 0,
                "cap": 0,
                "budget_bytes": getattr(self, "whisper_vram_budget_bytes", None),
                "budget_source": "none",
                "device_used_bytes": int(stats.get("device_used_bytes") or 0),
                "free_bytes": int(stats.get("free_bytes") or 0),
                "per_window_bytes": 0,
                "active_windows": int(active_windows or 0),
            }

        cap = max(int(entry.get("max_batch_size") or 0) for entry in entries) or 1
        cost = max(int(entry.get("per_window_bytes") or 0) for entry in entries)
        budget = getattr(self, "whisper_vram_budget_bytes", None)
        if budget is None:
            budget = int(stats.get("total_bytes") or 0)
            budget_source = "device_total"
        else:
            budget_source = "config"
        used = int(stats.get("device_used_bytes") or 0)
        allowed = allowed_windows(
            budget,
            used,
            active_windows,
            cap,
            per_window_bytes=cost,
            min_windows=min_windows,
        )
        return {
            "enabled": True,
            "allowed_windows": allowed,
            "cap": cap,
            "budget_bytes": budget,
            "budget_source": budget_source,
            "device_used_bytes": used,
            "free_bytes": int(stats.get("free_bytes") or 0),
            "per_window_bytes": cost,
            "active_windows": int(active_windows or 0),
        }

    # ------------------------------------------------------------------
    # Whisper execution lanes (Phase 10-1)
    # ------------------------------------------------------------------
    def get_whisper_lane_stats(self) -> dict:
        """Return the Whisper lane accounting rendered by ``/metrics``.

        Shape::

            {"enabled": bool,
             "models": {<executor name>: {
                 "workers": int,              # configured CT2 inter_threads
                 "in_flight": int,            # transcribe calls executing now
                 "lane_short_depth": int,     # waiting <=30 s requests
                 "lane_long_depth": int,      # waiting >30 s requests
                 "queue_wait": {"sum", "count", "max", "buckets"},
                 "ct2": {"available", "queued", "active"},
                 "batch_fallbacks": int,        # window-batcher → sequential fallbacks
             }}}

        Robust on a shell engine (``__new__`` without ``__init__``) and on a
        deployment without a Whisper config: ``enabled`` is then ``False`` and
        the renderer emits explicit zero samples.
        """
        lane_depths: dict[str, dict[str, int]] = {}
        scheduler = getattr(self, "scheduler", None)
        if scheduler is not None:
            try:
                lane_depths = scheduler.whisper_lane_depths() or {}
            except Exception as error:  # noqa: BLE001 - metrics must never break a scrape
                logger.warning(f"Failed to read Whisper lane depths: {error}")

        models: dict[str, dict] = {}
        for name, model_executor in getattr(self, "model_executors", {}).items():
            config = getattr(model_executor, "model_config", None)
            if getattr(config, "model_type", None) != "FasterWhisper":
                continue
            runner = getattr(getattr(model_executor, "driver_worker", None), "model_runner", None)
            if runner is None:
                # Test harnesses and shell engines expose the runner directly.
                runner = getattr(model_executor, "runner", None)
            stats: dict = {}
            reader = getattr(runner, "whisper_lane_stats", None)
            if callable(reader):
                try:
                    stats = dict(reader() or {})
                except Exception as error:  # noqa: BLE001 - metrics must never break a scrape
                    logger.warning(f"Failed to read Whisper lane stats for '{name}': {error}")
            workers = stats.get("workers")
            if not workers:
                workers = getattr(config, "num_workers", 1)
            depths = lane_depths.get(name) or {}
            models[name] = {
                "workers": max(1, int(workers or 1)),
                "in_flight": int(stats.get("in_flight") or 0),
                "lane_short_depth": int(depths.get("short") or 0),
                "lane_long_depth": int(depths.get("long") or 0),
                "queue_wait": stats.get("queue_wait")
                or {"sum": 0.0, "count": 0, "max": 0.0, "buckets": []},
                "ct2": stats.get("ct2") or {"available": False, "queued": 0, "active": 0},
                "batch_fallbacks": int(stats.get("batch_fallbacks") or 0),
            }
        return {"enabled": bool(models), "models": models}


setup_logger()

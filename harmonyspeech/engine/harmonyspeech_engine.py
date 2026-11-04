import os
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor

from loguru import logger

import harmonyspeech
from harmonyspeech.common.config import ModelConfig
from harmonyspeech.common.inputs import (
    RequestInput,
    TextToSpeechRequestInput,
    prefill_audio_artifacts,
)
from harmonyspeech.common.logger import setup_logger
from harmonyspeech.common.outputs import RequestOutput, TextToSpeechRequestOutput
from harmonyspeech.common.request import EngineRequest, ExecutorResult, RequestStatus
from harmonyspeech.engine.args_tools import EngineArgs
from harmonyspeech.engine.metrics import StatLogger, Stats
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
        strict_pipelines: bool = False,
    ):
        self.model_configs = model_configs
        self.log_stats = log_stats
        # Per-request deadline (seconds); 0/None disables deadline enforcement.
        self.request_timeout_s = request_timeout_s
        # Turn "pipeline incomplete" startup warnings into a boot error.
        self.strict_pipelines = strict_pipelines
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

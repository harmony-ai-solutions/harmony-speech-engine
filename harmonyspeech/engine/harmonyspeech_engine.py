import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Type, Union, Iterable, Dict

from loguru import logger

import harmonyspeech
from harmonyspeech.common.config import ModelConfig
from harmonyspeech.common.inputs import RequestInput, SpeechEmbeddingRequestInput
from harmonyspeech.common.logger import setup_logger
from harmonyspeech.common.outputs import RequestOutput
from harmonyspeech.common.request import EngineRequest, ExecutorResult, RequestStatus
from harmonyspeech.engine.args_tools import EngineArgs
from harmonyspeech.engine.metrics import Stats, StatLogger
from harmonyspeech.processing.scheduler import Scheduler, SchedulerOutputs

_LOCAL_LOGGING_INTERVAL_SEC = int(os.environ.get("HARMONYSPEECH_LOCAL_LOGGING_INTERVAL_SEC", "5"))


class HarmonySpeechEngine:
    """
    An Inference Engine for AI Speech that receives requests and generates outputs.

    """

    def __init__(
        self,
        model_configs: Optional[List[ModelConfig]],
        log_stats: bool,
    ):
        self.model_configs = model_configs
        self.log_stats = log_stats
        self._verify_args()

        logger.info(
            f"Initializing Harmony Speech Engine (v{harmonyspeech.__version__}) "
            "with the following config:\n"
            f"Preloaded Models = "
            f"Available Models = "
            # f"Device = {device_config.device}\n"
        )

        """
        For each provided model config we will create a separate executor.
        Models may allocate across multiple devices based on DeviceConfig.        
        """
        self.model_executors = {}
        self.init_custom_executors()

        # Initialize Scheduler for requests across models
        self.scheduler = Scheduler(model_configs=self.model_configs)

        # Metric Logging.
        if self.log_stats:
            self.stat_logger = StatLogger(
                local_interval=_LOCAL_LOGGING_INTERVAL_SEC,
                labels=dict()
            )

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
            )
            # Append the created executor to the dict of model executors
            self.model_executors[model_cfg.name] = executor

    @classmethod
    def from_engine_args(cls, engine_args: EngineArgs) -> "HarmonySpeechEngine":
        """Creates an LLM engine from the engine arguments."""
        # Create the engine configs.
        engine_config = engine_args.create_engine_config()

        # Create the LLM engine.
        engine = cls(**engine_config.to_dict(),
                     log_stats=not engine_args.disable_log_stats)
        return engine

    def __reduce__(self):
        # This is to ensure that the HarmonySpeechEngine is not referenced in
        # the closure used to initialize Ray worker actors
        raise RuntimeError("HarmonySpeechEngine should not be pickled!")

    def _verify_args(self) -> None:
        # for model_cfg in self.model_configs:
        #     model_cfg.verify_with_parallel_config(self.parallel_config)
        return None

    def check_reroute_request_to_model(self, request: RequestInput):
        if request.model == "harmonyspeech":
            if isinstance(request, SpeechEmbeddingRequestInput):
                for cfg in self.model_configs:
                    if cfg.model_type == "HarmonySpeechEncoder":
                        request.model = cfg.name

    def add_request(
        self,
        request_id: str,
        request_data: RequestInput,
        arrival_time: Optional[float] = None,
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

        Details:
            - Set arrival_time to the current time if it is None.
        """

        if arrival_time is None:
            arrival_time = time.monotonic()

        # Route to correct model if necessary
        self.check_reroute_request_to_model(request=request_data)

        # Add the request to the scheduler.
        self.scheduler.add_request(EngineRequest(
            request_id=request_id,
            request_data=request_data,
            arrival_time=arrival_time
        ))

    def abort_request(self, request_id: Union[str, Iterable[str]]) -> None:
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

    def get_model_configs(self) -> List[ModelConfig]:
        """Gets the model configuration."""
        return self.model_configs

    def get_num_unfinished_requests(self) -> int:
        """Gets the number of unfinished requests."""
        return self.scheduler.get_num_unfinished_requests()

    def has_unfinished_requests(self) -> bool:
        """Returns True if there are unfinished requests."""
        return self.scheduler.has_unfinished_requests()

    def _process_model_outputs(
        self, outputs: List[ExecutorResult],
        ignored_requests: List[EngineRequest]
    ) -> List[RequestOutput]:
        """Apply the model output to the sequences in the scheduled seq groups.

        Returns RequestOutputs that can be returned to the client.
        """
        # Check for Re-Schedule conditions
        # TODO


        # Create the outputs and update the requests
        request_outputs: List[RequestOutput] = []
        for result in outputs:
            self.scheduler.update_request_status(result.request_id, RequestStatus.FINISHED_STOPPED)
            request_outputs.append(result.request_data)

        for request in ignored_requests:
            # Mark as Finished / Ignored
            self.scheduler.update_request_status(request.request_id, RequestStatus.FINISHED_IGNORED)
            request_output = RequestOutput(
                request_id=request.request_id,
                finish_reason=RequestStatus.get_finished_reason(request.status)
            )
            request_outputs.append(request_output)

        # Free the finished requests.
        self.scheduler.free_finished_requests()

        return request_outputs

    def step(self) -> List[RequestOutput]:
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

        request_outputs = self._process_model_outputs(
            outputs=output,
            ignored_requests=scheduler_outputs.ignored_requests
        )

        # Log stats.
        # if self.log_stats:
        #     self.stat_logger.log(
        #         self._get_stats(scheduler_outputs, model_output=output))

        return request_outputs

    def do_log_stats(self) -> None:
        """Forced log when no requests active."""
        if self.log_stats:
            self.stat_logger.log(self._get_stats(scheduler_outputs=None))

    def _get_stats(
            self,
            scheduler_outputs: Optional[SchedulerOutputs]) -> Stats:
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

        return Stats(
            now=now,
            num_running=num_running,
            num_waiting=num_waiting,
        )

    def check_health(self) -> None:
        for model_executor in self.model_executors:
            model_executor.check_health()


setup_logger()

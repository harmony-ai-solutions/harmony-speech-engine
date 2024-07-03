import os
import time
from typing import List, Optional, Type, Union, Iterable

from loguru import logger

import harmonyspeech
from harmonyspeech.common.config import EngineConfig, ModelConfig, ParallelConfig, DeviceConfig
from harmonyspeech.common.logger import setup_logger
from harmonyspeech.common.outputs import RequestOutput
from harmonyspeech.common.sequence import EngineRequest
from harmonyspeech.endpoints.openai.protocol import GenerationOptions, AudioOutputOptions, VoiceConversionRequest, \
    TextToSpeechRequest
from harmonyspeech.engine.args_tools import EngineArgs
from harmonyspeech.executor.executor_base import ExecutorBase


_LOCAL_LOGGING_INTERVAL_SEC = int(os.environ.get("HARMONYSPEECH_LOCAL_LOGGING_INTERVAL_SEC", "5"))


class HarmonySpeechEngine:
    """
    An Inference Engine for AI Speech that receives requests and generates outputs.

    """

    def __init__(
        self,
        engine_config: EngineConfig,
        model_configs: Optional[List[ModelConfig]],
        parallel_config: ParallelConfig,
        device_config: DeviceConfig,
        executor_class: Type[ExecutorBase],
        log_stats: bool,
    ):
        logger.info(
            f"Initializing Harmony Speech Engine (v{harmonyspeech.__version__}) "
            "with the following config:\n"
            f"Preloaded Models = "
            f"Available Models = "
            f"Number of GPUs = {parallel_config.tensor_parallel_size}\n"
            f"Disable Custom All-Reduce = "
            f"{parallel_config.disable_custom_all_reduce}\n"
            f"Device = {device_config.device}\n"
        )

        self.engine_config = engine_config
        self.model_configs = model_configs
        self.parallel_config = parallel_config
        self.device_config = device_config
        self.log_stats = log_stats
        self._verify_args()

        """
        For each provided model config we will create a separate executor.
        Models may replicate across devices based on ParallelConfig and DeviceConfig.        
        """
        self.model_executors = {}
        self.init_custom_executors(executor_class)

    def init_custom_executors(self, executor_class: Type[ExecutorBase]) -> None:
        """
        Initialize custom executors for each provided ModelConfig.
        """
        for model_cfg in self.model_configs:
            # Create a new executor for each ModelConfig
            executor = executor_class(
                model_config=model_cfg,
                parallel_config=self.parallel_config,
                device_config=self.device_config,
            )
            # Append the created executor to the dict of model executors
            self.model_executors[model_cfg.model] = executor

    @classmethod
    def from_engine_args(cls, engine_args: EngineArgs) -> "HarmonySpeechEngine":
        """Creates an LLM engine from the engine arguments."""
        # Create the engine configs.
        engine_config = engine_args.create_engine_config()

        # Initialize the cluster and specify the executor class.
        if engine_config.device_config.device_type == "neuron":
            from harmonyspeech.executor.neuron_executor import NeuronExecutor
            executor_class = NeuronExecutor
        elif engine_config.device_config.device_type == "cpu":
            from harmonyspeech.executor.cpu_executor import CPUExecutor
            executor_class = CPUExecutor
        elif engine_config.parallel_config.worker_use_ray:
            initialize_ray_cluster(engine_config.parallel_config)
            from harmonyspeech.executor.ray_gpu_executor import RayGPUExecutor
            executor_class = RayGPUExecutor
        else:
            assert engine_config.parallel_config.world_size == 1, (
                "Ray is required if parallel_config.world_size > 1.")
            from harmonyspeech.executor.gpu_executor import GPUExecutor
            executor_class = GPUExecutor

        # Create the LLM engine.
        engine = cls(**engine_config.to_dict(),
                     executor_class=executor_class,
                     log_stats=not engine_args.disable_log_stats)
        return engine

    def __reduce__(self):
        # This is to ensure that the HarmonySpeechEngine is not referenced in
        # the closure used to initialize Ray worker actors
        raise RuntimeError("HarmonySpeechEngine should not be pickled!")

    def _verify_args(self) -> None:
        for model_cfg in self.model_configs:
            model_cfg.verify_with_parallel_config(self.parallel_config)

    def add_text_to_speech_request(
        self,
        request_id: str,
        request_data: TextToSpeechRequest,
        arrival_time: Optional[float] = None,
    ) -> None:

        """Add a Text-to-Speech request to the engine's request pool.

        The request is added to the request pool and will be processed by the
        scheduler as `engine.step()` is called. The exact scheduling policy is
        determined by the scheduler.

        Args:
            request_id: The unique ID of the request.
            request_data: The incoming TextToSpeechRequest data.
            arrival_time: The arrival time of the request. If None, we use
                the current monotonic time.

        Details:
            - Set arrival_time to the current time if it is None.
        """

        if arrival_time is None:
            arrival_time = time.monotonic()

        # Add the sequence group to the scheduler.
        self.scheduler.add_request(request_data)

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
        self.scheduler.abort_request(request_id)

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
        self, output: List[SamplerOutput],
        scheduled_seq_groups: List[EngineRequest],
        ignored_seq_groups: List[EngineRequest]) -> List[RequestOutput]:
        """Apply the model output to the sequences in the scheduled seq groups.

        Returns RequestOutputs that can be returned to the client.
        """
        now = time.time()
        # Organize outputs by [sequence group][step] instead of
        # [step][sequence group].
        output_by_sequence_group = create_output_by_sequence_group(
            sampler_outputs=output, num_seq_groups=len(scheduled_seq_groups))
        # Update the scheduled sequence groups with the model outputs.
        for scheduled_seq_group, outputs in zip(scheduled_seq_groups,
                                                output_by_sequence_group):
            seq_group = scheduled_seq_group.seq_group
            seq_group.update_num_computed_tokens(
                scheduled_seq_group.token_chunk_size)
            # If all sequences in the sequence group are in DECODE, then we can
            # process the output tokens. Otherwise, they are (chunked) prefill
            # samples and should not be processed.
            stages = [seq.data._stage for seq in seq_group.seqs_dict.values()]
            if all(stage == SequenceStage.DECODE for stage in stages):
                self.output_processor.process_outputs(seq_group, outputs)

        # Free the finished sequence groups.
        self.scheduler.free_finished_requests()

        # Create the outputs.
        request_outputs: List[RequestOutput] = []
        for scheduled_seq_group in scheduled_seq_groups:
            seq_group = scheduled_seq_group.seq_group
            seq_group.maybe_set_first_token_time(now)
            request_output = RequestOutput.from_seq_group(seq_group)
            request_outputs.append(request_output)
        for seq_group in ignored_seq_groups:
            request_output = RequestOutput.from_seq_group(seq_group)
            request_outputs.append(request_output)

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
            >>> engine = AphroditeEngine.from_engine_args(engine_args)
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
        seq_group_metadata_list, scheduler_outputs = self.scheduler.schedule()

        if not scheduler_outputs.is_empty():
            output = self.model_executor.execute_model(
                seq_group_metadata_list=seq_group_metadata_list,
                blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
                blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
                blocks_to_copy=scheduler_outputs.blocks_to_copy,
                num_lookahead_slots=scheduler_outputs.num_lookahead_slots)
        else:
            output = []

        request_outputs = self._process_model_outputs(
            output, scheduler_outputs.scheduled_seq_groups,
            scheduler_outputs.ignored_seq_groups)

        # Log stats.
        if self.log_stats:
            self.stat_logger.log(
                self._get_stats(scheduler_outputs, model_output=output))

        return request_outputs

    def check_health(self) -> None:
        for model_executor in self.model_executors:
            model_executor.check_health()


setup_logger()

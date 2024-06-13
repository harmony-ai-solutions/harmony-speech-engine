from typing import Dict, List, Set

from loguru import logger

from harmonyspeech.executor.executor_base import ExecutorAsyncBase, ExecutorBase
from harmonyspeech.common.sequence import SamplerOutput, SequenceGroupMetadata
from harmonyspeech.common.utils import (
    get_ip,
    get_open_port,
    get_distributed_init_method,
    make_async,
)


class GPUExecutor(ExecutorBase):

    def _init_executor(self) -> None:
        """Initialize the worker and load the model.
        If speculative decoding is enabled, we instead create the speculative
        worker.
        """
        self._init_worker()

    def _init_worker(self):
        # Lazy import the Worker to avoid importing torch.cuda/xformers
        # before CUDA_VISIBLE_DEVICES is set in the Worker
        from harmonyspeech.task_handler.worker import Worker

        assert (self.parallel_config.world_size == 1
                ), "GPUExecutor only supports single GPU."

        distributed_init_method = get_distributed_init_method(
            get_ip(), get_open_port())
        self.driver_worker = Worker(
            model_config=self.model_config,
            parallel_config=self.parallel_config,
            device_config=self.device_config,
            local_rank=0,
            rank=0,
            distributed_init_method=distributed_init_method,
            is_driver_worker=True,
        )
        self.driver_worker.init_device()
        self.driver_worker.load_model()

    def execute_model(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
        num_lookahead_slots: int,
    ) -> List[SamplerOutput]:
        output = self.driver_worker.execute_model(
            seq_group_metadata_list=seq_group_metadata_list,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            num_lookahead_slots=num_lookahead_slots,
        )
        return output

    def check_health(self) -> None:
        # GPUExecutor will always be healthy as long as
        # it's running.
        return


class GPUExecutorAsync(GPUExecutor, ExecutorAsyncBase):

    async def execute_model_async(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
        num_lookahead_slots: int,
    ) -> SamplerOutput:
        output = await make_async(self.driver_worker.execute_model)(
            seq_group_metadata_list=seq_group_metadata_list,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            num_lookahead_slots=num_lookahead_slots,
        )
        return output

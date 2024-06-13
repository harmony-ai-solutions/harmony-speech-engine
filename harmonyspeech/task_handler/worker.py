"""A GPU worker class."""
import gc
import os
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.distributed
from loguru import logger

from harmonyspeech.common.config import (
    DeviceConfig,
    ModelConfig,
    ParallelConfig
)
from harmonyspeech.common.sequence import SamplerOutput, SequenceGroupMetadata
from harmonyspeech.common.utils import in_wsl
from harmonyspeech.distributed import (
    broadcast_tensor_dict,
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from harmonyspeech.distributed.device_communicators import pynccl_utils
from harmonyspeech.distributed.device_communicators.custom_all_reduce import (
    init_custom_ar )
from harmonyspeech.modeling import set_random_seed
from harmonyspeech.task_handler.model_runner import ModelRunner
from harmonyspeech.task_handler.worker_base import WorkerBase


class Worker(WorkerBase):
    """A worker class that executes the model on a GPU.

    Each worker is associated with a single GPU. The worker is responsible for
    executing the model on the GPU.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        device_config: DeviceConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ) -> None:
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.device_config = device_config
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.is_driver_worker = is_driver_worker
        if self.is_driver_worker:
            assert self.rank == 0, "The driver worker must have rank 0."

        self.model_runner = ModelRunner(
            model_config,
            parallel_config,
            device_config,
            is_driver_worker=is_driver_worker
        )
        # Uninitialized cache engine. Will be initialized by
        # initialize_cache
        self.cache_engine = None
        self.gpu_cache = None

    def init_device(self) -> None:
        if self.device_config.device.type == "cuda":
            # torch.distributed.all_reduce does not free the input tensor until
            # the synchronization point. This causes the memory usage to grow
            # as the number of all_reduce calls increases. This env var disables
            # this behavior.
            # Related issue:
            # https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573
            os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

            # This env var set by Ray causes exceptions with graph building.
            os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)

            # Patch for torch.cuda.is_available() unexpected error in WSL;
            # always call torch.cuda.device_count() before initialising device
            if in_wsl():
                torch.cuda.device_count()
            self.device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(self.device)

            _check_if_gpu_supports_dtype(self.model_config.dtype)
            torch.cuda.empty_cache()
            self.init_gpu_memory = torch.cuda.mem_get_info()[0]
        else:
            raise RuntimeError(
                f"Not support device type: {self.device_config.device}")
        # Initialize the distributed environment.
        init_worker_distributed_environment(self.parallel_config, self.rank,
                                            self.distributed_init_method,
                                            self.local_rank)
        # Set random seed
        set_random_seed(self.model_config.seed)

    def load_model(self):
        self.model_runner.load_model()

    def _warm_up_model(self) -> None:
        if not self.model_config.enforce_eager:
            self.model_runner.capture_model(self.gpu_cache)
        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)

    @torch.inference_mode()
    def execute_model(
        self,
        seq_group_metadata_list: Optional[List[SequenceGroupMetadata]] = None,
        blocks_to_swap_in: Optional[Dict[int, int]] = None,
        blocks_to_swap_out: Optional[Dict[int, int]] = None,
        blocks_to_copy: Optional[Dict[int, List[int]]] = None,
        num_lookahead_slots: int = 0,
    ) -> List[SamplerOutput]:
        if self.is_driver_worker:
            assert seq_group_metadata_list is not None
            num_seq_groups = len(seq_group_metadata_list)
            assert blocks_to_swap_in is not None
            assert blocks_to_swap_out is not None
            assert blocks_to_copy is not None
            data = {
                "num_seq_groups": num_seq_groups,
                "blocks_to_swap_in": blocks_to_swap_in,
                "blocks_to_swap_out": blocks_to_swap_out,
                "blocks_to_copy": blocks_to_copy,
            }
            broadcast_tensor_dict(data, src=0)
        else:
            data = broadcast_tensor_dict(src=0)
            num_seq_groups = data["num_seq_groups"]
            blocks_to_swap_in = data["blocks_to_swap_in"]
            blocks_to_swap_out = data["blocks_to_swap_out"]
            blocks_to_copy = data["blocks_to_copy"]

        self.cache_swap(blocks_to_swap_in, blocks_to_swap_out, blocks_to_copy)

        # If there is no input, we don't need to execute the model.
        if num_seq_groups == 0:
            return []

        output = self.model_runner.execute_model(seq_group_metadata_list,
                                                 self.gpu_cache)
        # Worker only supports single-step execution. Wrap the output in a list
        # to conform to interface.
        return [output]


def init_worker_distributed_environment(
    parallel_config: ParallelConfig,
    rank: int,
    distributed_init_method: Optional[str] = None,
    local_rank: int = -1,
) -> None:
    """Initialize the distributed environment."""
    init_distributed_environment(parallel_config.world_size, rank,
                                 distributed_init_method, local_rank)

    if pynccl_utils.is_initialized():
        pynccl_world_size = pynccl_utils.get_world_size()
        if pynccl_world_size != parallel_config.world_size:
            raise RuntimeError(
                "pynccl is already initialized but the pynccl world "
                "size does not match parallel_config.world_size "
                f"({pynccl_world_size} vs. {parallel_config.world_size}).")
    elif parallel_config.world_size > 1:
        # NOTE: We don't initialize pynccl process group when world size
        # is 1.
        pynccl_utils.init_process_group(
            world_size=parallel_config.world_size,
            local_rank=local_rank,
            rank=rank,
            init_method=distributed_init_method,
        )

    ensure_model_parallel_initialized(parallel_config.tensor_parallel_size,
                                      parallel_config.pipeline_parallel_size)

    # Initialize a custom fast all-reduce implementation.
    if not parallel_config.disable_custom_all_reduce:
        init_custom_ar()

    # A small all_reduce for warmup.
    torch.distributed.all_reduce(torch.zeros(1).cuda())
    if pynccl_utils.is_initialized():
        pynccl_utils.all_reduce(torch.zeros(1).cuda())


def _check_if_gpu_supports_dtype(torch_dtype: torch.dtype):
    # Check if the GPU supports the dtype.
    if torch_dtype == torch.bfloat16:
        compute_capability = torch.cuda.get_device_capability()
        if compute_capability[0] < 8:
            gpu_name = torch.cuda.get_device_name()
            raise ValueError(
                "Bfloat16 is only supported on GPUs with compute capability "
                f"of at least 8.0. Your {gpu_name} GPU has compute capability "
                f"{compute_capability[0]}.{compute_capability[1]}. "
                "You can use float16 instead by explicitly setting the"
                "`dtype` flag in CLI, for example: --dtype=half.")


def raise_if_cache_size_invalid(num_gpu_blocks, block_size,
                                max_model_len) -> None:
    if num_gpu_blocks <= 0:
        raise ValueError("No available memory for the cache blocks. "
                         "Try increasing `gpu_memory_utilization` when "
                         "initializing the engine.")
    max_seq_len = block_size * num_gpu_blocks
    logger.info(f"Maximum sequence length allowed in the cache: "
                f"{max_seq_len}")
    if max_model_len > max_seq_len:
        raise ValueError(
            f"The model's max seq len ({max_model_len}) "
            "is larger than the maximum number of tokens that can be "
            f"stored in KV cache ({max_seq_len}). Try increasing "
            "`gpu_memory_utilization` or decreasing `max_model_len` when "
            "initializing the engine.")

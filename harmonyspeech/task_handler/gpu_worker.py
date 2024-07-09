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
)
from harmonyspeech.common.request import EngineRequest, ExecutorResult
from harmonyspeech.common.utils import in_wsl

from harmonyspeech.modeling import set_random_seed
from harmonyspeech.task_handler.gpu_model_runner import GPUModelRunner
from harmonyspeech.task_handler.worker_base import WorkerBase


class GPUWorker(WorkerBase):
    """A worker class that executes the model on a GPU.

    Each worker is associated with a single GPU. The worker is responsible for
    executing the model on the GPU.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        device_config: DeviceConfig,
        local_rank: int,
        rank: int,
        is_driver_worker: bool = False,
    ) -> None:
        self.model_config = model_config
        self.device_config = device_config
        self.local_rank = local_rank
        self.rank = rank
        self.is_driver_worker = is_driver_worker
        if self.is_driver_worker:
            assert self.rank == 0, "The driver worker must have rank 0."

        self.model_runner = GPUModelRunner(
            model_config,
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
        # Set random seed
        set_random_seed(self.model_config.seed)

    def load_model(self):
        self.model_runner.load_model()

    def _warm_up_model(self) -> None:
        # if not self.model_config.enforce_eager:
        #     self.model_runner.capture_model(self.gpu_cache)
        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)

    @torch.inference_mode()
    def execute_model(
        self,
        requests_to_batch: List[EngineRequest]
    ) -> List[ExecutorResult]:
        output = self.model_runner.execute_model(requests_to_batch=requests_to_batch)
        # Worker only supports single-step execution. Wrap the output in a list
        # to conform to interface.
        return output


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

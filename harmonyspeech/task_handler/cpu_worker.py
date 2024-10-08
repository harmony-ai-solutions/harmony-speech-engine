"""A CPU worker class."""
from typing import Dict, List, Optional

import torch
import torch.distributed

from harmonyspeech.common.config import (
    DeviceConfig,
    ModelConfig,
)
from harmonyspeech.common.request import EngineRequest, ExecutorResult
from harmonyspeech.modeling import set_random_seed
from harmonyspeech.task_handler.cpu_model_runner import CPUModelRunner
from harmonyspeech.task_handler.worker_base import WorkerBase


class CPUWorker(WorkerBase):
    """A worker class that executes (a partition of) the model on a CPU socket.
    Each worker is associated with a single CPU socket. The worker is 
    responsible for maintaining the KV cache and executing the model on the 
    CPU. In case of distributed inference, each worker is assigned a partition
    of the model.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        device_config: DeviceConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ) -> None:
        self.model_config = model_config
        self.device_config = device_config
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.is_driver_worker = is_driver_worker
        if self.is_driver_worker:
            assert self.rank == 0, "The driver worker must have rank 0."

        self.model_runner = CPUModelRunner(model_config,
                                           device_config,
                                           is_driver_worker=is_driver_worker)
        # Uninitialized cache engine. Will be initialized by
        # initialize_cache
        self.cache_engine = None
        self.cpu_cache = None

    def init_device(self) -> None:
        # Set random seed.
        set_random_seed(self.model_config.seed)

    def load_model(self):
        self.model_runner.load_model()

    @torch.inference_mode()
    def execute_model(
        self,
        requests_to_batch: List[EngineRequest]
    ) -> List[ExecutorResult]:
        output = self.model_runner.execute_model(requests_to_batch=requests_to_batch)
        # CPU worker only supports single-step execution.
        return output


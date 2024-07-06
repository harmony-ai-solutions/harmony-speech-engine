import os
from typing import Dict, List, Set

import torch
from loguru import logger

from harmonyspeech.common.config import ModelConfig
from harmonyspeech.common.request import EngineRequest, ExecutorResult
from harmonyspeech.common.utils import (get_distributed_init_method, get_ip, get_open_port, make_async)
from harmonyspeech.executor.executor_base import ExecutorBase


class CPUExecutor(ExecutorBase):

    def _init_executor(self) -> None:
        assert self.device_config.device_type == "cpu"
        self.model_config = _verify_and_get_model_config(self.model_config)

        # Instantiate the worker and load the model to CPU.
        self._init_worker()

    def _init_worker(self):
        from harmonyspeech.task_handler.cpu_worker import CPUWorker

        distributed_init_method = get_distributed_init_method(get_ip(), get_open_port())
        self.driver_worker = CPUWorker(
            model_config=self.model_config,
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
        requests_to_batch: List[EngineRequest]
    ) -> List[ExecutorResult]:
        output = self.driver_worker.execute_model(
            requests_to_batch=requests_to_batch
        )
        return output

    async def execute_model_async(
        self,
        requests_to_batch: List[EngineRequest],
    ) -> List[ExecutorResult]:
        output = await make_async(self.driver_worker.execute_model)(
            requests_to_batch=requests_to_batch,
        )
        return output

    def check_health(self) -> None:
        # CPUExecutor will always be healthy as long as
        # it's running.
        return


def _verify_and_get_model_config(config: ModelConfig) -> ModelConfig:
    if config.dtype == torch.float16:
        logger.warning("float16 is not supported on CPU, casting to bfloat16.")
        config.dtype = torch.bfloat16
    if not config.enforce_eager:
        logger.warning(
            "CUDA graph is not supported on CPU, fallback to the eager "
            "mode.")
        config.enforce_eager = True
    return config

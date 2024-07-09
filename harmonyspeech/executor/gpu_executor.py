from typing import Dict, List, Set

from loguru import logger

from harmonyspeech.executor.executor_base import ExecutorAsyncBase, ExecutorBase
from harmonyspeech.common.request import EngineRequest, ExecutorResult
from harmonyspeech.common.utils import make_async


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
        from harmonyspeech.task_handler.gpu_worker import GPUWorker

        self.driver_worker = GPUWorker(
            model_config=self.model_config,
            device_config=self.device_config,
            local_rank=0,
            rank=0,
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

    def check_health(self) -> None:
        # GPUExecutor will always be healthy as long as
        # it's running.
        return


class GPUExecutorAsync(GPUExecutor, ExecutorAsyncBase):

    async def execute_model_async(
        self,
        requests_to_batch: List[EngineRequest],
    ) -> List[ExecutorResult]:
        output = await make_async(self.driver_worker.execute_model)(
            requests_to_batch=requests_to_batch,
        )
        return output

from abc import ABC, abstractmethod

from loguru import logger

from harmonyspeech.common.config import ModelConfig
from harmonyspeech.common.request import EngineRequest, ExecutorResult


class ExecutorBase(ABC):
    """Base class for all executors.
    An executor is responsible for executing the model on a specific device
    type (e.g., CPU, GPU, Neuron, etc.). Or it can be a distributed executor
    that can execute the model on multiple devices.
    """

    def __init__(
        self, model_config: ModelConfig, weight_registry=None, embedding_cache=None, ar_cache_pool=None
    ) -> None:
        self.model_config = model_config
        self.device_config = model_config.device_config
        # Engine-owned shared-weight registry (Phase 4-3). Executors for configs
        # with the same `backbone_group` acquire the same module objects.
        self.weight_registry = weight_registry
        # Engine-owned cross-request embedding cache (Phase 7-1); None means
        # disabled — every consumer must stay None-safe.
        self.embedding_cache = embedding_cache
        # Engine-owned shared AR KV-cache pool (Phase 9-2); None means disabled —
        # consumers fall back to their sequential path.
        self.ar_cache_pool = ar_cache_pool

        self._init_executor()
        logger.info(
            f"Successfully initialized executor for model {self.model_config.name} ({self.model_config.model_type}) on device {self.device_config.device}"
        )

    @abstractmethod
    def _init_executor(self) -> None:
        pass

    @abstractmethod
    def execute_model(self, requests_to_batch: list[EngineRequest]) -> list[ExecutorResult]:
        """Executes one model step on the given sequences."""
        raise NotImplementedError

    @abstractmethod
    def check_health(self) -> None:
        """Checks if the executor is healthy. If not, it should raise an
        exception."""
        raise NotImplementedError

    def release_request(self, request_id: str) -> None:
        """Tell the runner a request will never be scheduled again (Phase 9-4).

        Duck-typed hook the engine calls on abort/disconnect and deadline expiry
        mid-decode. The AR runner (9-3) uses it to drop the decode row and free
        its 9-2 cache slot. A runner without the hook is a no-op, so no executor
        is forced to implement it.
        """
        worker = getattr(self, "driver_worker", None)
        runner = getattr(worker, "model_runner", None)
        hook = getattr(runner, "release_request", None)
        if callable(hook):
            hook(request_id)


class ExecutorAsyncBase(ExecutorBase):
    @abstractmethod
    async def execute_model_async(self, requests_to_batch: list[EngineRequest]) -> list[ExecutorResult]:
        """Executes one model step on the given sequences."""
        raise NotImplementedError

    async def check_health_async(self) -> None:
        """Checks if the executor is healthy. If not, it should raise an
        exception."""
        self.check_health()

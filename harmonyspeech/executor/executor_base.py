from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set, Tuple
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
        self,
        model_config: ModelConfig,
    ) -> None:
        self.model_config = model_config
        self.device_config = model_config.device_config

        self._init_executor()
        logger.info(
            f"Successfully initialized executor for model {self.model_config.name} ({self.model_config.model_type}) "
            f"on device {self.device_config.device}"
        )

    @abstractmethod
    def _init_executor(self) -> None:
        pass

    @abstractmethod
    def execute_model(
        self,
        requests_to_batch: List[EngineRequest],
    ) -> List[ExecutorResult]:
        """Executes one model step on the given sequences."""
        raise NotImplementedError

    @abstractmethod
    def check_health(self) -> None:
        """Checks if the executor is healthy. If not, it should raise an
        exception."""
        raise NotImplementedError


class ExecutorAsyncBase(ExecutorBase):

    @abstractmethod
    async def execute_model_async(
        self,
        requests_to_batch: List[EngineRequest],
    ) -> List[ExecutorResult]:
        """Executes one model step on the given sequences."""
        raise NotImplementedError

    async def check_health_async(self) -> None:
        """Checks if the executor is healthy. If not, it should raise an
        exception."""
        self.check_health()

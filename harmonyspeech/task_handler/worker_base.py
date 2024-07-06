from abc import ABC, abstractmethod
from typing import Dict, List

from harmonyspeech.common.request import EngineRequest, ExecutorResult


class WorkerBase(ABC):
    """Worker interface that allows Aphrodite to cleanly separate
    implementations for different hardware.
    """

    @abstractmethod
    def init_device(self) -> None:
        """Initialize device state, such as loading the model or other on-device
        memory allocations.
        """
        raise NotImplementedError

    @abstractmethod
    def execute_model(
        self,
        requests_to_batch: List[EngineRequest]
    ) -> List[ExecutorResult]:
        """Executes at least one model step on the given sequences, unless no
        sequences are provided."""
        raise NotImplementedError

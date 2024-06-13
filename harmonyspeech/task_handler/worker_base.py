from abc import ABC, abstractmethod
from typing import Dict, List

from harmonyspeech.common.sequence import SamplerOutput, SequenceGroupMetadata


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
            self, seq_group_metadata_list: List[SequenceGroupMetadata],
            blocks_to_swap_in: Dict[int, int], blocks_to_swap_out: Dict[int,
                                                                        int],
            blocks_to_copy: Dict[int, List[int]]) -> List[SamplerOutput]:
        """Executes at least one model step on the given sequences, unless no
        sequences are provided."""
        raise NotImplementedError

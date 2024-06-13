"""A Neuron worker class."""
from typing import List

import torch
import torch.distributed

from harmonyspeech.common.config import (
    DeviceConfig,
    ModelConfig,
    ParallelConfig,
)
from harmonyspeech.common.sequence import SamplerOutput, SequenceGroupMetadata
from harmonyspeech.modeling import set_random_seed
from harmonyspeech.task_handler.neuron_model_runner import NeuronModelRunner
from harmonyspeech.task_handler.worker_base import WorkerBase


class NeuronWorker(WorkerBase):
    """A worker class that executes the model on a group of neuron cores."""

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        device_config: DeviceConfig,
    ) -> None:
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.device_config = device_config

        self.model_runner = NeuronModelRunner(model_config, parallel_config, device_config)

    def init_device(self) -> None:
        # Set random seed.
        set_random_seed(self.model_config.seed)

    def load_model(self):
        self.model_runner.load_model()

    @torch.inference_mode()
    def execute_model(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> List[SamplerOutput]:
        num_seq_groups = len(seq_group_metadata_list)

        # If there is no input, we don't need to execute the model.
        if num_seq_groups == 0:
            return []

        output = self.model_runner.execute_model(seq_group_metadata_list)
        # Neuron worker only supports single-step output. Wrap the output in a
        # list to conform to interface.
        return [output]

    def get_cache_block_size_bytes(self) -> int:
        """Determine the size in bytes of a cache block.
        This is required for speculative decoding; it is not yet implemented.
        """
        raise NotImplementedError

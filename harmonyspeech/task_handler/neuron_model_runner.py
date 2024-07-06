from typing import Dict, List, Optional, Tuple

import torch
from loguru import logger

from harmonyspeech.common.config import DeviceConfig, ModelConfig
from harmonyspeech.modeling.neuron_loader import get_neuron_model
from harmonyspeech.common.request import EngineRequest, ExecutorResult
from harmonyspeech.common.utils import (async_tensor_h2d, is_pin_memory_available,
                                    make_tensor_with_pad, maybe_expand_dim)


class NeuronModelRunner:

    def __init__(
        self,
        model_config: ModelConfig,
        device_config: DeviceConfig,
    ):
        self.model_config = model_config

        if model_config is not None and model_config.get_sliding_window():
            logger.warning("Sliding window is not supported on Neuron. "
                           "The model will run without sliding window.")
        self.device_config = (device_config
                              if device_config is not None else DeviceConfig())
        self.device = self.device_config.device
        self.model = None
        self.pin_memory = is_pin_memory_available()

    def load_model(self) -> None:
        self.model = get_neuron_model(
            self.model_config,
        )

    @torch.inference_mode()
    def execute_model(
        self,
        requests_to_batch: List[EngineRequest]
    ) -> List[ExecutorResult]:

        (input_tokens, input_positions, input_block_ids, sampling_metadata
         ) = self.prepare_input_tensors(seq_group_metadata_list)

        hidden_states = self.model(
            input_ids=input_tokens,
            positions=input_positions,
            input_block_ids=input_block_ids,
        )

        # Compute the logits.
        logits = self.model.compute_logits(hidden_states, sampling_metadata)

        # Sample the next token.
        output = self.model.sample(
            logits=logits,
            sampling_metadata=sampling_metadata,
        )
        return output

from typing import Dict, List, Optional, Tuple

import torch

from harmonyspeech.common.config import (
    DeviceConfig,
    ModelConfig,
)
from harmonyspeech.common.request import EngineRequest, ExecutorResult
from harmonyspeech.modeling.loader import get_model

_PAD_SLOT_ID = -1


class CPUModelRunner:

    def __init__(
        self,
        model_config: ModelConfig,
        device_config: DeviceConfig,
        is_driver_worker: bool = False,
        *args,
        **kwargs,
    ):
        self.model_config = model_config
        self.is_driver_worker = is_driver_worker

        # model_config can be None in tests/samplers/test_sampler.py.
        # FIXME: This is a hack to make the tests work. Refactor this.
        self.sliding_window = (model_config.get_sliding_window()
                               if model_config is not None else None)
        self.device_config = (device_config
                              if device_config is not None else DeviceConfig())
        self.device = self.device_config.device

        self.model = None
        self.block_size = None  # Set after initial profiling.

        self.attn_backend = get_attn_backend(
            self.model_config.dtype if model_config is not None else None)

    def load_model(self) -> None:
        self.model = get_model(self.model_config,
                               self.device_config,
                               parallel_config=self.parallel_config)

    @torch.inference_mode()
    def execute_model(
        self,
        requests_to_batch: List[EngineRequest]
    ) -> List[ExecutorResult]:

        (input_tokens, input_positions, attn_metadata, sampling_metadata
         ) = self.prepare_input_tensors(seq_group_metadata_list)

        model_executable = self.model
        execute_model_kwargs = {
            "input_ids": input_tokens,
            "positions": input_positions,
            "kv_caches": kv_caches,
            "attn_metadata": attn_metadata,
        }

        hidden_states = model_executable(**execute_model_kwargs)

        # Compute the logits.
        logits = self.model.compute_logits(hidden_states, sampling_metadata)

        # Only perform sampling in the driver worker.
        if not sampling_metadata.perform_sampling:
            return None

        # Sample the next token.
        output = self.model.sample(
            logits=logits,
            sampling_metadata=sampling_metadata,
        )
        return output

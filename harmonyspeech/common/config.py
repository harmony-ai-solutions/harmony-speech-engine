from typing import Optional, TYPE_CHECKING, Union
from loguru import logger

import torch

from harmonyspeech.common.utils import is_neuron, is_cpu, is_hip

if TYPE_CHECKING:
    from ray.util.placement_group import PlacementGroup

class EngineConfig:
    """
    EngineConfig defines setup params for the Speech Engine

    Args:


    """

    def __init__(
        self,
        device_ids=None
    ) -> None:
        self.device_ids = device_ids

        if self.device_ids is None:
            self.device_ids = ["cpu"]


class DeviceConfig:

    def __init__(self, device: str = "auto") -> None:
        if device == "auto":
            # Automated device type detection
            if torch.cuda.is_available():
                self.device_type = "cuda"
            elif is_neuron():
                self.device_type = "neuron"
            elif is_cpu():
                self.device_type = "cpu"
            else:
                raise RuntimeError("No supported device detected.")
        else:
            # Device type is assigned explicitly
            self.device_type = device

        # Some device types require processing inputs on CPU
        if self.device_type in ["neuron"]:
            self.device = torch.device("cpu")
        else:
            # Set device with device type
            self.device = torch.device(self.device_type)


class ModelConfig:
    """
    Configuration for a model

    Args:
        model: model name
        max_batch_size: max batch size per processing step
        device_config: device config for this model's executor and worker
        dtype: Data type for model weights and activations. The "auto" option
            will use FP16 precision for FP32 and FP16 models, and BF16 precision
            for BF16 models.
        seed: Random seed for reproducibility.
        enforce_eager: Whether to enforce eager execution. If True, we will
            disable CUDA graph and always execute the model in eager mode.
            If False, we will use CUDA graph and eager execution in hybrid.
    """

    def __init__(
        self,
        model: str,
        max_batch_size: int,
        device_config: DeviceConfig,
        dtype: Union[str, torch.dtype],
        seed: int,
        enforce_eager: bool = True,
    ) -> None:
        self.model = model
        self.max_batch_size = max_batch_size
        self.device_config = device_config
        self.dtype = dtype  # Remark: in Aphrodite this is: _get_and_verify_dtype(self.hf_text_config, dtype)
        self.seed = seed
        self.enforce_eager = enforce_eager

import torch

from harmonyspeech.common.utils import is_neuron, is_cpu


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


class ModelConfig:
    """
    Configuration for a model

    Args:


    """

    def __init__(
        self,
        model: str,
    ) -> None:
        self.model = model


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


from dataclasses import dataclass, fields
from typing import Optional, Union, List
from loguru import logger

import torch
import yaml

from harmonyspeech.common.utils import is_cpu, is_hip


class DeviceConfig:

    def __init__(self, device: str = "auto") -> None:
        if device == "auto":
            # Automated device type detection
            if torch.cuda.is_available():
                self.device_type = "cuda"
            elif is_cpu():
                self.device_type = "cpu"
            else:
                raise RuntimeError("No supported device detected.")
        else:
            # Device type is assigned explicitly
            self.device_type = device

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
        name: str,
        model: str,
        model_type: str,
        max_batch_size: int,
        device_config: DeviceConfig,
        language: Optional[str] = None,
        voices: Optional[List[str]] = None,
        trust_remote_code: Optional[bool] = False,
        download_dir: Optional[str] = None,
        load_format: Optional[str] = "auto",
        dtype: Optional[Union[str, torch.dtype]] = "bfloat16",
        seed: Optional[int] = 0,
        revision: Optional[str] = None,
        code_revision: Optional[str] = None,
        enforce_eager: bool = True,
    ) -> None:
        self.name = name
        self.model = model
        self.model_type = model_type
        self.max_batch_size = max_batch_size
        self.device_config = device_config
        self.language = language
        self.voices = voices
        self.trust_remote_code = trust_remote_code
        self.download_dir = download_dir
        self.load_format = load_format
        self.dtype = dtype  # Remark: in Aphrodite this is: _get_and_verify_dtype(self.hf_text_config, dtype)
        self.seed = seed
        self.revision = revision
        self.code_revision = code_revision
        self.enforce_eager = enforce_eager

        self.dtype = _get_and_verify_dtype(dtype)

        self._verify_load_format()

    def _verify_load_format(self) -> None:
        load_format = self.load_format.lower()
        supported_load_format = [
            "auto", "pt", "pth", "safetensors", "npcache", "dummy"
        ]
        rocm_not_supported_load_format = []
        if load_format not in supported_load_format:
            raise ValueError(
                f"Unknown load format: {self.load_format}. Must be one of "
                "'auto', 'pt', 'pth', 'safetensors', 'npcache', or 'dummy'.")
        if is_hip() and load_format in rocm_not_supported_load_format:
            rocm_supported_load_format = [
                f for f in supported_load_format
                if (f not in rocm_not_supported_load_format)
            ]
            raise ValueError(
                f"load format \'{load_format}\' is not supported in ROCm. "
                f"Supported load format are "
                f"{rocm_supported_load_format}")

        self.load_format = load_format


@dataclass(frozen=True)
class EngineConfig:
    """
    Dataclass which contains all engine-related configuration. This
    simplifies passing around the distinct configurations in the codebase.
    """

    model_configs: List[ModelConfig]

    def __post_init__(self):
        """Verify configs are valid & consistent with each other.
        """
        # self.model_config.verify_with_parallel_config(self.parallel_config)

    def to_dict(self):
        """Return the configs as a dictionary, for use in **kwargs.
        """
        return dict((field.name, getattr(self, field.name)) for field in fields(self))

    @classmethod
    def load_config_from_yaml(cls, yaml_file_path: str) -> "EngineConfig":
        with open(yaml_file_path, 'r') as file:
            config_data = yaml.safe_load(file)

        model_configs = []
        for model_cfg in config_data["model_configs"]:
            device_config = DeviceConfig(**model_cfg["device_config"])
            del model_cfg["device_config"]
            model_configs.append(ModelConfig(device_config=device_config, **model_cfg))

        return EngineConfig(model_configs=model_configs)


_STR_DTYPE_TO_TORCH_DTYPE = {
    "half": torch.float16,
    "float16": torch.float16,
    "float": torch.float32,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}

_ROCM_NOT_SUPPORTED_DTYPE = ["float", "float32"]


def _get_and_verify_dtype(
    dtype: Union[str, torch.dtype],
) -> torch.dtype:

    config_dtype = torch.float32

    if isinstance(dtype, str):
        dtype = dtype.lower()
        if dtype == "auto":
            if config_dtype == torch.float32:
                # Following the common practice, we use float16 for float32
                # models.
                torch_dtype = torch.float16
            else:
                torch_dtype = config_dtype
        else:
            if dtype not in _STR_DTYPE_TO_TORCH_DTYPE:
                raise ValueError(f"Unknown dtype: {dtype}")
            torch_dtype = _STR_DTYPE_TO_TORCH_DTYPE[dtype]
    elif isinstance(dtype, torch.dtype):
        torch_dtype = dtype
    else:
        raise ValueError(f"Unknown dtype: {dtype}")

    # if is_hip() and torch_dtype == torch.float32:
    #     rocm_supported_dtypes = [
    #         k for k, v in _STR_DTYPE_TO_TORCH_DTYPE.items()
    #         if (k not in _ROCM_NOT_SUPPORTED_DTYPE)
    #     ]
    #     raise ValueError(f"dtype \'{dtype}\' is not supported in ROCm. "
    #                      f"Supported dtypes are {rocm_supported_dtypes}")

    # Verify the dtype.
    if torch_dtype != config_dtype:
        if torch_dtype == torch.float32:
            # Upcasting to float32 is allowed.
            pass
        elif config_dtype == torch.float32:
            # Downcasting from float32 to float16 or bfloat16 is allowed.
            pass
        else:
            # Casting between float16 and bfloat16 is allowed with a warning.
            logger.warning(f"Casting {config_dtype} to {torch_dtype}.")

    return torch_dtype

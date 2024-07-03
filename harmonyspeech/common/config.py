from typing import Optional, TYPE_CHECKING
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


class ModelConfig:
    """
    Configuration for a model

    Args:


    """

    def __init__(
        self,
        model: str,
        max_batch_size: int
    ) -> None:
        self.model = model
        self.max_batch_size = max_batch_size


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


class ParallelConfig:
    """Configuration for the distributed execution.

    Args:
        pipeline_parallel_size: Number of pipeline parallel groups.
        tensor_parallel_size: Number of tensor parallel groups.
        worker_use_ray: Whether to use Ray for model workers. Will be set to
            True if either pipeline_parallel_size or tensor_parallel_size is
            greater than 1.
        max_parallel_loading_workers: Maximum number of multiple batches
            when load model sequentially. To avoid RAM OOM when using tensor
            parallel and large models.
        disable_custom_all_reduce: Disable the custom all-reduce kernel and
            fall back to NCCL.
        ray_workers_use_nsight: Whether to profile Ray workers with nsight, see
            https://docs.ray.io/en/latest/ray-observability/user-guides/profiling.html#profiling-nsight-profiler.
    """

    def __init__(
        self,
        pipeline_parallel_size: int,
        tensor_parallel_size: int,
        worker_use_ray: bool,
        max_parallel_loading_workers: Optional[int] = None,
        disable_custom_all_reduce: bool = False,
        ray_workers_use_nsight: bool = False,
        placement_group: Optional["PlacementGroup"] = None,
    ) -> None:
        self.pipeline_parallel_size = pipeline_parallel_size
        self.tensor_parallel_size = tensor_parallel_size
        self.worker_use_ray = worker_use_ray
        self.max_parallel_loading_workers = max_parallel_loading_workers
        self.disable_custom_all_reduce = disable_custom_all_reduce
        self.ray_workers_use_nsight = ray_workers_use_nsight
        self.placement_group = placement_group

        self.world_size = pipeline_parallel_size * self.tensor_parallel_size
        if self.world_size > 1:
            self.worker_use_ray = True
        self._verify_args()

    def _verify_args(self) -> None:
        if self.pipeline_parallel_size > 1:
            raise NotImplementedError(
                "Pipeline parallelism is not supported yet.")
        if not self.disable_custom_all_reduce and self.world_size > 1:
            if is_hip():
                self.disable_custom_all_reduce = True
                logger.info(
                    "Disabled the custom all-reduce kernel because it is not "
                    "supported on AMD GPUs.")
            elif self.pipeline_parallel_size > 1:
                self.disable_custom_all_reduce = True
                logger.info(
                    "Disabled the custom all-reduce kernel because it is not "
                    "supported with pipeline parallelism.")
        if self.ray_workers_use_nsight and not self.worker_use_ray:
            raise ValueError("Unable to use nsight profiling unless workers "
                             "run with Ray.")
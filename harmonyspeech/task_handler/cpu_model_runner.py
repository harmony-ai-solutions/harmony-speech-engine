from typing import Dict, List, Optional, Tuple

import torch

from harmonyspeech.common.config import (
    DeviceConfig,
    ModelConfig,
)
from harmonyspeech.common.request import EngineRequest, ExecutorResult
from harmonyspeech.task_handler.model_runner_base import ModelRunnerBase

_PAD_SLOT_ID = -1


class CPUModelRunner(ModelRunnerBase):

    def __init__(
        self,
        model_config: ModelConfig,
        device_config: DeviceConfig,
        is_driver_worker: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(model_config, device_config, is_driver_worker, *args, **kwargs)

    def load_model(self) -> None:
        self.model = self._load_model()

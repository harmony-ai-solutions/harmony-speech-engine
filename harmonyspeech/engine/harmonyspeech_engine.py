import os
from typing import List, Optional

from loguru import logger

import harmonyspeech
from harmonyspeech.common.config import EngineConfig, ModelConfig


_LOCAL_LOGGING_INTERVAL_SEC = int(os.environ.get("HARMONYSPEECH_LOCAL_LOGGING_INTERVAL_SEC", "5"))


class HarmonySpeechEngine:
    """
    An Inference Engine for AI Speech that receives requests and generates outputs.

    """

    def __init__(
        self,
        engine_config: EngineConfig,
        model_config: Optional[List[ModelConfig]],
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        executor_class: Type[ExecutorBase],
        log_stats: bool,
    ):
        logger.info(
            f"Initializing Harmony Speech Engine (v{harmonyspeech.__version__}) "
            "with the following config:\n"
            f"Preloaded Models = "
            f"Available Models = "
        )

        self.engine_config = engine_config
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config

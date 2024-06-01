from typing import List, Optional

from loguru import logger

import harmonyspeech
from harmonyspeech.common.config import EngineConfig, ModelConfig


class HarmonySpeechEngine:

    def __init__(
        self,
        engine_config: EngineConfig,
        model_config: Optional[List[ModelConfig]]
    ):
        logger.info(
            f"Initializing Harmony Speech Engine (v{harmonyspeech.__version__}) "
            "with the following config:\n"
            f"Preloaded Models = "
            f"Available Models = "
        )

        self.engine_config = engine_config
        self.model_config = model_config

from dataclasses import dataclass
from typing import List


@dataclass
class EngineArgs:
    """Arguments For HarmonySpeechEngine"""

    models: List[str]
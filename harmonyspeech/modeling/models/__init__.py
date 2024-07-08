import importlib
from typing import Optional, Type, Dict, List
from loguru import logger

import torch.nn as nn

from harmonyspeech.common.utils import is_hip

# Architecture -> (module, class).
_MODELS = {
    "OpenVoice": ("openvoice", "SynthesizerTrn"),
    "Melo": ("melo", "SynthesizerTrn"),
    "HarmonySpeechEncoder": ("harmonyspeech", "SpeakerEncoder"),
    "HarmonySpeechSynthesizer": ("harmonyspeech", "ForwardTacotron"),
    "HarmonySpeechVocoder": ("harmonyspeech", "MelGANGenerator"),
}

# Architecture -> type.
# out of tree models
_OOT_MODELS: Dict[str, Type[nn.Module]] = {}

# Models not supported by ROCm.
_ROCM_UNSUPPORTED_MODELS = []

# Models partially supported by ROCm.
# Architecture -> Reason.
_ROCM_PARTIALLY_SUPPORTED_MODELS = {
    # "Qwen2ForCausalLM":
    # "Sliding window attention is not yet supported in ROCm's flash attention",
    # "MistralForCausalLM":
    # "Sliding window attention is not yet supported in ROCm's flash attention",
    # "MixtralForCausalLM":
    # "Sliding window attention is not yet supported in ROCm's flash attention",
}


class ModelRegistry:

    @staticmethod
    def load_model_cls(model_arch: str) -> Optional[Type[nn.Module]]:
        if model_arch in _OOT_MODELS:
            return _OOT_MODELS[model_arch]
        if model_arch not in _MODELS:
            return None
        if is_hip():
            if model_arch in _ROCM_UNSUPPORTED_MODELS:
                raise ValueError(
                    f"Model architecture {model_arch} is not supported by "
                    "ROCm for now.")
            if model_arch in _ROCM_PARTIALLY_SUPPORTED_MODELS:
                logger.warning(
                    f"Model architecture {model_arch} is partially supported "
                    "by ROCm: " + _ROCM_PARTIALLY_SUPPORTED_MODELS[model_arch])

        module_name, model_cls_name = _MODELS[model_arch]
        module = importlib.import_module(
            f"harmonyspeech.modeling.models.{module_name}")
        return getattr(module, model_cls_name, None)

    @staticmethod
    def get_supported_archs() -> List[str]:
        return list(_MODELS.keys())

    @staticmethod
    def register_model(model_arch: str, model_cls: Type[nn.Module]):
        if model_arch in _MODELS:
            logger.warning(
                f"Model architecture {model_arch} is already registered, "
                "and will be overwritten by the new model "
                f"class {model_cls.__name__}.")
        global _OOT_MODELS
        _OOT_MODELS[model_arch] = model_cls


__all__ = [
    "ModelRegistry",
]

from typing import Optional

from transformers import AutoConfig, PretrainedConfig

from harmonyspeech.config_utils.configs.harmonyspeech import *
from harmonyspeech.config_utils.configs.openvoice_v1 import *
from harmonyspeech.config_utils.configs.openvoice_v2 import *

_CONFIG_REGISTRY = {
    "HarmonySpeechEncoder": HarmonySpeechEncoderConfig,
    "HarmonySpeechSynthesizer": HarmonySpeechSynthesizerConfig,
    "HarmonySpeechVocoder": HarmonySpeechVocoderConfig,
    "OpenVoiceV1Synthesizer": OpenVoiceV1SynthesizerConfig,
    "OpenVoiceV1ToneConverter": OpenVoiceV1ToneConverterConfig,
    "OpenVoiceV2Synthesizer": OpenVoiceV2SynthesizerConfig,
    "OpenVoiceV2ToneConverter": OpenVoiceV2ToneConverterConfig,
}


def get_config(model: str,
               trust_remote_code: bool,
               revision: Optional[str] = None,
               code_revision: Optional[str] = None) -> PretrainedConfig:
    try:
        config = AutoConfig.from_pretrained(
            model,
            trust_remote_code=trust_remote_code,
            revision=revision,
            code_revision=code_revision)
    except ValueError as e:
        if (not trust_remote_code and
                "requires you to execute the configuration file" in str(e)):
            err_msg = (
                "Failed to load the model config. If the model is a custom "
                "model not yet available in the HuggingFace transformers "
                "library, consider setting `trust_remote_code=True` in LLM "
                "or using the `--trust-remote-code` flag in the CLI.")
            raise RuntimeError(err_msg) from e
        else:
            raise e
    if config.model_type in _CONFIG_REGISTRY:
        config_class = _CONFIG_REGISTRY[config.model_type]
        config = config_class.from_pretrained(model,
                                              revision=revision,
                                              code_revision=code_revision)
    return config
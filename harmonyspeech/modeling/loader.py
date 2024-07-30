"""Utilities for selecting and loading models."""
import contextlib
from typing import Type, Optional

import torch
import torch.nn as nn
from faster_whisper import WhisperModel

from harmonyspeech.common.config import DeviceConfig, ModelConfig
from harmonyspeech.modeling.models import ModelRegistry
from harmonyspeech.modeling.hf_downloader import initialize_dummy_weights, load_or_download_model, \
    load_or_download_config


@contextlib.contextmanager
def _set_default_torch_dtype(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(old_dtype)


def _get_model_cls(model_config: ModelConfig) -> Type[nn.Module]:
    model_type = getattr(model_config, 'model_type', None)
    model_cls = ModelRegistry.load_model_cls(model_type)
    if model_cls is not None:
        return model_cls
    else:
        raise ValueError(
            f"Model of type {model_type} is not supported for now. "
            f"Supported models: {ModelRegistry.get_supported_archs()}")


_MODEL_CONFIGS = {
    # OpenVoice V1
    "OpenVoiceV1Synthesizer": {
        "EN": "base_speakers/EN/config.json",
        "ZH": "base_speakers/ZH/config.json",
    },
    "OpenVoiceV1ToneConverter": {
        "default": "converter/config.json",
    },
    "OpenVoiceV1ToneConverterEncoder": {
        "default": "converter/config.json",
    },
    # OpenVoice V2 / MeloTTS - Different Repos but same structure per language
    "MeloTTSSynthesizer": {
        "default": "config.json",
    },
    "OpenVoiceV2ToneConverter": {
        "default": "converter/config.json",
    },
    "OpenVoiceV2ToneConverterEncoder": {
        "default": "converter/config.json",
    },
    # Harmony Speech V1
    "HarmonySpeechEncoder": {
        "default": "encoder/config.json"
    },
    "HarmonySpeechSynthesizer": {
        "default": "synthesizer/config.json"
    },
    "HarmonySpeechVocoder": {
        "default": "vocoder/config.json"
    },
    # Faster-Whisper
    "FasterWhisper": {
        "default": "native"
    }
}

_MODEL_WEIGHTS = {
    # OpenVoice V1
    "OpenVoiceV1Synthesizer": {
        "EN": "base_speakers/EN/checkpoint.pth",
        "ZH": "base_speakers/ZH/checkpoint.pth",
    },
    "OpenVoiceV1ToneConverter": {
        "default": "converter/checkpoint.pth",
    },
    "OpenVoiceV1ToneConverterEncoder": {
        "default": "converter/checkpoint.pth",
    },
    # OpenVoice V2 / MeloTTS - Different Repos but same structure per language
    "MeloTTSSynthesizer": {
        "default": "checkpoint.pth",
    },
    "OpenVoiceV2ToneConverter": {
        "default": "converter/checkpoint.pth",
    },
    "OpenVoiceV2ToneConverterEncoder": {
        "default": "converter/checkpoint.pth",
    },
    # Harmony Speech V1
    "HarmonySpeechEncoder": {
        "default": "encoder/encoder.pt"
    },
    "HarmonySpeechSynthesizer": {
        "default": "synthesizer/synthesizer.pt"
    },
    "HarmonySpeechVocoder": {
        "default": "vocoder/vocoder.pt"
    },
    # Faster-Whisper
    "FasterWhisper": {
        "default": "native"
    }
}


def get_model_config(
    model_name_or_path: str,
    model_type: str,
    revision: Optional[str] = None,
    flavour: str = "default"
):
    if model_type not in _MODEL_CONFIGS:
        raise NotImplementedError(f"model type {model_type} is not implemented.")
    if flavour not in _MODEL_CONFIGS[model_type]:
        raise NotImplementedError(f"model subtype {flavour} for model {model_type} does not exist.")

    # Bailout for native Model implementations
    if _MODEL_CONFIGS[model_type][flavour] == "native":
        return "native"

    # Get Config
    config_data = load_or_download_config(model_name_or_path, _MODEL_CONFIGS[model_type][flavour], revision)
    return config_data


def get_model_weights(
    model_name_or_path: str,
    model_type: str,
    revision: Optional[str] = None,
    device: str = "cpu",
    flavour: str = "default"
):
    if model_type not in _MODEL_WEIGHTS:
        raise NotImplementedError(f"model type {model_type} is not implemented.")
    if flavour not in _MODEL_WEIGHTS[model_type]:
        raise NotImplementedError(f"model subtype {flavour} for model {model_type} does not exist.")

    # Get Base checkpoint
    checkpoint = load_or_download_model(model_name_or_path, device, _MODEL_WEIGHTS[model_type][flavour], revision)
    return checkpoint


def get_model_flavour(model_config: ModelConfig):
    if model_config.model_type in ["OpenVoiceV1Synthesizer"]:
        # OpenVoice V1 use Different Singe-Speaker-TTS weights per language within same Repo
        return model_config.language
    return "default"


def get_model(model_config: ModelConfig, device_config: DeviceConfig, **kwargs):
    model_class = _get_model_cls(model_config)

    with _set_default_torch_dtype(model_config.dtype):
        # Get model flavour if applicable
        flavour = get_model_flavour(model_config)
        # Load config
        hf_config = get_model_config(model_config.model, model_config.model_type, model_config.revision, flavour)

        # Create a model instance.
        # The weights will be initialized as empty tensors.
        with torch.device(device_config.device):

            # Bailout for native models
            if model_class == "native" and hf_config == "native":
                if model_config.model_type == "FasterWhisper":
                    model = WhisperModel(model_config.model)
                    return model

            # Initialize the model using config
            if hasattr(hf_config, "model"):
                # Model class initialization for Harmony Speech Models and OpenVoice
                if model_config.model_type in [
                    "OpenVoiceV1Synthesizer",
                    "OpenVoiceV1ToneConverter",
                    "OpenVoiceV1ToneConverterEncoder",
                    "OpenVoiceV2ToneConverter",
                    "OpenVoiceV2ToneConverterEncoder"
                ]:
                    # Dynamic Parameters for OpenVoice
                    hf_config.model.n_vocab = len(getattr(hf_config, 'symbols', []))
                    hf_config.model.spec_channels = hf_config.data.filter_length // 2 + 1
                    hf_config.model.n_speakers = hf_config.data.n_speakers
                if model_config.model_type in [
                    "MeloTTSSynthesizer",
                ]:
                    # Dynamic Parameters for MeloTTS
                    hf_config.model.n_vocab = len(getattr(hf_config, 'symbols', []))
                    hf_config.model.spec_channels = hf_config.data.filter_length // 2 + 1
                    hf_config.model.segment_size = hf_config.train.segment_size // hf_config.data.hop_length
                    hf_config.model.n_speakers = hf_config.data.n_speakers
                    hf_config.model.num_tones = hf_config.num_tones
                    hf_config.model.num_languages = hf_config.num_languages

                model = model_class(**hf_config.model)
            else:
                model = model_class(**hf_config)

        if model_config.load_format == "dummy":
            # NOTE: For accurate performance evaluation, we assign
            # random values to the weights.
            initialize_dummy_weights(model)
        else:
            # Load the weights from the cached or downloaded files.
            checkpoint = get_model_weights(
                model_config.model,
                model_config.model_type,
                model_config.revision,
                device_config.device,
                flavour
            )
            # Run the load weight method of the model
            # Every model needs to implement this seperately, because state dicts can differ
            # and post-init task may apply
            model.load_weights(checkpoint, hf_config)

        # if isinstance(linear_method, BNBLinearMethod):
        #     replace_quant_params(
        #         model,
        #         quant_config=linear_method.quant_config,
        #         modules_to_not_convert="lm_head",
        #     )
        #     torch.cuda.synchronize()
        #     if linear_method.quant_config.from_float:
        #         model = model.cuda()
        #     gc.collect()
        #     torch.cuda.empty_cache()
        #     tp = get_tensor_model_parallel_world_size()
        #     logger.info(
        #         "Memory allocated for converted model: {} GiB x {} = {} "
        #         "GiB".format(
        #             round(
        #                 torch.cuda.memory_allocated(
        #                     torch.cuda.current_device()) /
        #                 (1024 * 1024 * 1024),
        #                 2,
        #             ),
        #             tp,
        #             round(
        #                 torch.cuda.memory_allocated(
        #                     torch.cuda.current_device()) * tp /
        #                 (1024 * 1024 * 1024),
        #                 2,
        #             ),
        #         ))
        #     logger.info(
        #         "Memory reserved for converted model: {} GiB x {} = {} "
        #         "GiB".format(
        #             round(
        #                 torch.cuda.memory_reserved(torch.cuda.current_device())
        #                 / (1024 * 1024 * 1024),
        #                 2,
        #             ),
        #             tp,
        #             round(
        #                 torch.cuda.memory_reserved(torch.cuda.current_device())
        #                 * tp / (1024 * 1024 * 1024),
        #                 2,
        #             ),
        #         ))
    return model.eval()

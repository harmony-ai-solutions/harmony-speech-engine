"""Utilities for selecting and loading models."""
import contextlib
from typing import Type

import torch
import torch.nn as nn

from harmonyspeech.common.config import DeviceConfig, ModelConfig
from harmonyspeech.modeling.models import ModelRegistry
from harmonyspeech.modeling.hf_downloader import initialize_dummy_weights


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


def get_model(model_config: ModelConfig, device_config: DeviceConfig, **kwargs) -> nn.Module:
    model_class = _get_model_cls(model_config)

    # Get the (maybe quantized) linear method.
    linear_method = None

    with _set_default_torch_dtype(model_config.dtype):
        # Create a model instance.
        # The weights will be initialized as empty tensors.
        with torch.device(device_config.device):
            if hasattr(model_config.hf_config, "model"):
                # Model class initialization for Harmony Speech Models and OpenVoice
                model = model_class(**model_config.hf_config.model)
            else:
                model = model_class(model_config.hf_config, linear_method)

        if model_config.load_format == "dummy":
            # NOTE: For accurate performance evaluation, we assign
            # random values to the weights.
            initialize_dummy_weights(model)
        else:
            # Load the weights from the cached or downloaded files.
            model.load_weights(model_config.model, model_config.download_dir,
                               model_config.load_format, model_config.revision)

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

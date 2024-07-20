"""Utilities for downloading and initializing model weights."""
import json
import os

import huggingface_hub.constants
import torch
from huggingface_hub import hf_hub_download

_DOWNLOAD_CKPT_URLS = {
    'EN': 'https://myshell-public-repo-hosting.s3.amazonaws.com/openvoice/basespeakers/EN/checkpoint.pth',
    'EN_V2': 'https://myshell-public-repo-hosting.s3.amazonaws.com/openvoice/basespeakers/EN_V2/checkpoint.pth',
    'FR': 'https://myshell-public-repo-hosting.s3.amazonaws.com/openvoice/basespeakers/FR/checkpoint.pth',
    'JP': 'https://myshell-public-repo-hosting.s3.amazonaws.com/openvoice/basespeakers/JP/checkpoint.pth',
    'ES': 'https://myshell-public-repo-hosting.s3.amazonaws.com/openvoice/basespeakers/ES/checkpoint.pth',
    'ZH': 'https://myshell-public-repo-hosting.s3.amazonaws.com/openvoice/basespeakers/ZH/checkpoint.pth',
    'KR': 'https://myshell-public-repo-hosting.s3.amazonaws.com/openvoice/basespeakers/KR/checkpoint.pth',
}

DOWNLOAD_CONFIG_URLS = {
    'EN': 'https://myshell-public-repo-hosting.s3.amazonaws.com/openvoice/basespeakers/EN/config.json',
    'EN_V2': 'https://myshell-public-repo-hosting.s3.amazonaws.com/openvoice/basespeakers/EN_V2/config.json',
    'FR': 'https://myshell-public-repo-hosting.s3.amazonaws.com/openvoice/basespeakers/FR/config.json',
    'JP': 'https://myshell-public-repo-hosting.s3.amazonaws.com/openvoice/basespeakers/JP/config.json',
    'ES': 'https://myshell-public-repo-hosting.s3.amazonaws.com/openvoice/basespeakers/ES/config.json',
    'ZH': 'https://myshell-public-repo-hosting.s3.amazonaws.com/openvoice/basespeakers/ZH/config.json',
    'KR': 'https://myshell-public-repo-hosting.s3.amazonaws.com/openvoice/basespeakers/KR/config.json',
}


_xdg_cache_home = os.getenv("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
_harmony_filelocks_path = os.path.join(_xdg_cache_home, "harmony/locks/")


def enable_hf_transfer():
    """automatically activates hf_transfer
    """
    if "HF_HUB_ENABLE_HF_TRANSFER" not in os.environ:
        try:
            # enable hf hub transfer if available
            import hf_transfer  # type: ignore # noqa
            huggingface_hub.constants.HF_HUB_ENABLE_HF_TRANSFER = True
        except ImportError:
            pass


enable_hf_transfer()



def initialize_dummy_weights(
    model: torch.nn.Module,
    low: float = -1e-3,
    high: float = 1e-3,
) -> None:
    """Initialize model weights with random values.

    The model weights must be randomly initialized for accurate performance
    measurements. Additionally, the model weights should not cause NaNs in the
    forward pass. We empirically found that initializing the weights with
    values between -1e-3 and 1e-3 works well for most models.
    """
    for param in model.state_dict().values():
        if torch.is_floating_point(param):
            param.data.uniform_(low, high)


class HParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()


def get_hparams_from_file(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        data = f.read()
    config = json.loads(data)

    hparams = HParams(**config)
    return hparams


def load_or_download_config(model_name_or_path: str, config_filename: str = "config.json", revision: str = None):
    config_base_path = model_name_or_path
    config_path = config_base_path + "/" + config_filename
    if not os.path.isfile(config_path):
        # Try via Huggingface if path is not pointing to a local file
        # assuming config_base_path is a huggingface repo URL
        config_path = hf_hub_download(
            repo_id=config_base_path,
            revision=revision,
            filename=config_filename
        )

    return get_hparams_from_file(config_path)


def load_or_download_model(model_name_or_path: str, device: str, ckpt_filename: str = "checkpoint.pth", revision: str = None):
    ckpt_base_path = model_name_or_path
    ckpt_path = ckpt_base_path + "/" + ckpt_filename
    if not os.path.isfile(ckpt_path):
        # Try via Huggingface if path is not pointing to a local file
        # assuming ckpt_base_path is a huggingface repo URL
        ckpt_path = hf_hub_download(
            repo_id=ckpt_base_path,
            revision=revision,
            filename=ckpt_filename
        )
    return torch.load(ckpt_path, map_location=device)

"""Utilities for downloading and initializing model weights."""

import json
import os

import huggingface_hub.constants
import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import LocalEntryNotFoundError
from loguru import logger

_xdg_cache_home = os.getenv("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
_harmony_filelocks_path = os.path.join(_xdg_cache_home, "harmony/locks/")

# In-memory caches for configs and static files loaded via HuggingFace.
# These are populated once (first request) and reused for every subsequent
# request, eliminating redundant HEAD/ETag checks against huggingface.co on
# every inference call.  Configs and speaker-embedding files are immutable
# for the lifetime of the process, so cache invalidation is not needed.
_config_cache: dict[tuple, object] = {}
_file_cache: dict[tuple, bytes] = {}


def _resolve_file_path(
    base_path: str, filename: str, revision: str | None = None
) -> str:
    """Resolve a file path using a three-tier strategy, minimising network use.

    Resolution order:
      1. **Local filesystem** — ``base_path/filename`` if it exists on disk
         (models unpacked to a local directory).
      2. **HuggingFace disk cache** — ``hf_hub_download(local_files_only=True)``
         checks the HF cache folder without any network request.
      3. **Network download** — full ``hf_hub_download()`` as a last-resort
         fallback for the very first fetch (logs a one-time INFO message).

    Args:
        base_path: Local directory path OR HuggingFace repo ID.
        filename: File name relative to *base_path*.
        revision: Optional HF model revision.

    Returns:
        Absolute path to the resolved file on disk.
    """
    # Tier 1: local filesystem
    local_path = os.path.join(base_path, filename)
    if os.path.isfile(local_path):
        return local_path

    # Tier 2: HuggingFace disk cache (no network)
    try:
        return hf_hub_download(
            repo_id=base_path, revision=revision, filename=filename,
            local_files_only=True,
        )
    except LocalEntryNotFoundError:
        pass

    # Tier 3: network download (first-ever fetch)
    logger.info(f"Downloading '{filename}' from HuggingFace repo '{base_path}' "
                f"(not found in local cache).")
    return hf_hub_download(
        repo_id=base_path, revision=revision, filename=filename,
    )


def enable_hf_transfer():
    """automatically activates hf_transfer"""
    if "HF_HUB_ENABLE_HF_TRANSFER" not in os.environ:
        try:
            # enable hf hub transfer if available
            import hf_transfer  # type: ignore # noqa

            huggingface_hub.constants.HF_HUB_ENABLE_HF_TRANSFER = True
        except ImportError:
            pass


enable_hf_transfer()


def initialize_dummy_weights(model: torch.nn.Module, low: float = -1e-3, high: float = 1e-3) -> None:
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
    with open(config_path, encoding="utf-8") as f:
        data = f.read()
    config = json.loads(data)

    hparams = HParams(**config)
    return hparams


def load_or_download_file(model_name_or_path: str, file_filename: str = "file.json", revision: str = None):
    cache_key = (model_name_or_path, file_filename, revision)
    if cache_key in _file_cache:
        return _file_cache[cache_key]

    file_path = _resolve_file_path(model_name_or_path, file_filename, revision)
    with open(file_path, "rb") as f:
        data = f.read()
    _file_cache[cache_key] = data
    return data


def load_or_download_config(model_name_or_path: str, config_filename: str = "config.json", revision: str = None):
    cache_key = (model_name_or_path, config_filename, revision)
    if cache_key in _config_cache:
        return _config_cache[cache_key]

    config_path = _resolve_file_path(model_name_or_path, config_filename, revision)
    hparams = get_hparams_from_file(config_path)
    _config_cache[cache_key] = hparams
    return hparams


def load_or_download_model(
    model_name_or_path: str, device: str, ckpt_filename: str = "checkpoint.pth", revision: str = None
):
    ckpt_path = _resolve_file_path(model_name_or_path, ckpt_filename, revision)
    return torch.load(ckpt_path, map_location=device, weights_only=False)

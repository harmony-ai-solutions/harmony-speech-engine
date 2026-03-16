"""Chatterbox model wrappers for Harmony Speech Engine.

This module provides wrapper classes for the Chatterbox TTS and VC models,
exposing a consistent interface for model loading and inference.
"""

import os
import types

import numpy as np
import torch

# Import Chatterbox models from the chatterbox library
from chatterbox import ChatterboxMultilingualTTS, ChatterboxTTS, ChatterboxVC
from chatterbox.mtl_tts import SUPPORTED_LANGUAGES as _CHATTERBOX_SUPPORTED_LANGUAGES
from chatterbox.tts_turbo import ChatterboxTurboTTS as _ChatterboxTurboTTS
from huggingface_hub import snapshot_download

# HuggingFace repo ID for ChatterboxTurboTTS (mirrors tts_turbo.py REPO_ID)
_CHATTERBOX_TURBO_REPO_ID = "ResembleAI/chatterbox-turbo"

# ---------------------------------------------------------------------------
# Monkey-patch for ChatterboxMultilingualTTS to support CPU loading
# ---------------------------------------------------------------------------
# The upstream ChatterboxMultilingualTTS.from_local doesn't pass map_location
# to torch.load, causing failures on CPU when the checkpoint was saved on CUDA.
# This patch adds proper CPU mapping when device="cpu" is specified.
import chatterbox.mtl_tts as _mtl_module

_original_torch_load = _mtl_module.torch.load


def _patched_torch_load(*args, map_location=None, **kwargs):
    """Patched torch.load that defaults to CPU map_location when device is cpu."""
    if map_location is None:
        # Check if we're being called from the multilingual model loading path
        # by inspecting the first argument (the file path)
        if args and isinstance(args[0], (str, object)) and hasattr(args[0], "__str__"):
            # Default to CPU if no map_location specified
            map_location = torch.device("cpu")
    kwargs["map_location"] = map_location
    return _original_torch_load(*args, **kwargs)


# Apply the patch to the chatterbox.mtl_tts module
_mtl_module.torch.load = _patched_torch_load

# ---------------------------------------------------------------------------
# Monkey-patch for perth package to handle missing PerthImplicitWatermarker
# ---------------------------------------------------------------------------
# The perth package may fail to import PerthImplicitWatermarker on some platforms
# (e.g., missing optional dependencies), causing it to be None. When chatterbox
# tries to instantiate it, we get "TypeError: 'NoneType' object is not callable".
# This patch provides a fallback that uses DummyWatermarker when the real one is unavailable.
import perth as _perth_module

if _perth_module.PerthImplicitWatermarker is None:

    class _FallbackPerthImplicitWatermarker:
        """Fallback watermarker that uses DummyWatermarker when PerthImplicitWatermarker is unavailable."""

        def __init__(self, *args, **kwargs):
            # Use DummyWatermarker as fallback
            self._dummy = _perth_module.DummyWatermarker(*args, **kwargs)

        def __getattr__(self, name):
            # Delegate all attribute access to the dummy
            return getattr(self._dummy, name)

        def apply_watermark(self, *args, **kwargs):
            return self._dummy.apply_watermark(*args, **kwargs)

        def detect_watermark(self, *args, **kwargs):
            return self._dummy.detect_watermark(*args, **kwargs)

    _perth_module.PerthImplicitWatermarker = _FallbackPerthImplicitWatermarker


class ChatterboxTTSModel:
    """Wrapper for ChatterboxTTS model.

    ChatterboxTTS provides text-to-speech synthesis with voice cloning support.
    """

    @classmethod
    def from_pretrained(cls, device: str | torch.device | None = None, **kwargs) -> "ChatterboxTTS":
        """Load ChatterboxTTS model from pretrained weights.

        Args:
            device: Device to load the model on (e.g., "cpu", "cuda").
            **kwargs: Additional arguments passed to ChatterboxTTS.from_pretrained.

        Returns:
            ChatterboxTTS model instance.
        """
        if device is None:
            device = "cpu"
        elif isinstance(device, torch.device):
            device = str(device)

        return ChatterboxTTS.from_pretrained(device=device, **kwargs)


class ChatterboxTurboTTSModel:
    """Wrapper for ChatterboxTurboTTS model.

    ChatterboxTurboTTS is a separate, faster TTS model from chatterbox.tts_turbo.
    It supports top_k and norm_loudness parameters and does NOT support
    exaggeration, cfg_weight, or min_p.
    """

    @classmethod
    def from_pretrained(cls, device: str | torch.device | None = None, **kwargs) -> "_ChatterboxTurboTTS":
        """Load ChatterboxTurboTTS model from pretrained weights.

        Workaround: The upstream `_ChatterboxTurboTTS.from_pretrained()` contains a bug
        where it calls `snapshot_download(token=os.getenv("HF_TOKEN") or True)`.
        When HF_TOKEN is not set, `None or True` evaluates to `True`, which instructs
        huggingface_hub to require a locally-cached token — causing a
        `LocalTokenNotFoundError` even though the model repo is public.

        We bypass this by calling `snapshot_download` ourselves with
        `token=os.getenv("HF_TOKEN") or None` (so unauthenticated downloads work for
        public repos), then delegating to `from_local`.

        Args:
            device: Device to load the model on (e.g., "cpu", "cuda").
            **kwargs: Additional keyword arguments (unused; kept for API compatibility).

        Returns:
            ChatterboxTurboTTS model instance.
        """
        if device is None:
            device = "cpu"
        elif isinstance(device, torch.device):
            device = str(device)

        # Use the caller-supplied token if available, otherwise None (allow anonymous
        # access to this public repo rather than forcing token=True).
        # Also treat empty string as None to avoid 'Illegal header value b'Bearer '' errors.
        hf_token = os.getenv("HF_TOKEN") or None
        if hf_token == "":
            hf_token = None

        local_path = snapshot_download(
            repo_id=_CHATTERBOX_TURBO_REPO_ID,
            token=hf_token,
            allow_patterns=["*.safetensors", "*.json", "*.txt", "*.pt", "*.model"],
        )

        model = _ChatterboxTurboTTS.from_local(local_path, device)

        # Workaround: librosa.load / librosa.resample inside prepare_conditionals
        # return float64 numpy arrays by default. Two downstream consumers both
        # crash when receiving float64 input:
        #
        # 1. s3tokenizer.log_mel_spectrogram — does torch.from_numpy(audio) preserving
        #    float64, then tries `_mel_filters (float32) @ magnitudes (float64)`:
        #      RuntimeError: expected scalar type Float but found Double
        #
        # 2. voice_encoder.forward (via ve.embeds_from_wavs) — LSTM rejects float64:
        #      ValueError: input must have the type torch.float32, got type torch.float64
        #
        # Fix: monkey-patch both consumers on the loaded model instance to cast
        # numpy arrays / tensors to float32 at entry.

        # --- Patch 1: s3tokenizer.log_mel_spectrogram ---
        _orig_log_mel = model.s3gen.tokenizer.log_mel_spectrogram.__func__

        def _log_mel_float32(self_tok, audio, padding=0):
            if isinstance(audio, np.ndarray):
                audio = audio.astype(np.float32)
            elif torch.is_tensor(audio) and audio.dtype != torch.float32:
                audio = audio.float()
            return _orig_log_mel(self_tok, audio, padding)

        model.s3gen.tokenizer.log_mel_spectrogram = types.MethodType(_log_mel_float32, model.s3gen.tokenizer)

        # --- Patch 2: voice_encoder.forward (mels input to LSTM) ---
        _orig_ve_forward = model.ve.forward

        def _ve_forward_float32(mels, *args, **kwargs):
            if torch.is_tensor(mels) and mels.dtype != torch.float32:
                mels = mels.float()
            return _orig_ve_forward(mels, *args, **kwargs)

        model.ve.forward = _ve_forward_float32

        return model


class ChatterboxMultilingualTTSModel:
    """Wrapper for ChatterboxMultilingualTTS model.

    ChatterboxMultilingualTTS provides TTS synthesis supporting multiple languages.
    """

    # 23 supported language codes mirroring upstream chatterbox library.
    # Used by serving_engine for LanguageOptions registration and API validation.
    SUPPORTED_LANGUAGES: dict = _CHATTERBOX_SUPPORTED_LANGUAGES.copy()

    @classmethod
    def from_pretrained(cls, device: str | torch.device | None = None, **kwargs) -> "ChatterboxMultilingualTTS":
        """Load ChatterboxMultilingualTTS model from pretrained weights.

        Args:
            device: Device to load the model on (e.g., "cpu", "cuda").
            **kwargs: Additional arguments passed to ChatterboxMultilingualTTS.from_pretrained.

        Returns:
            ChatterboxMultilingualTTS model instance.
        """
        if device is None:
            device = "cpu"
        elif isinstance(device, torch.device):
            device = str(device)

        model = ChatterboxMultilingualTTS.from_pretrained(device=device, **kwargs)

        # Workaround: newer transformers defaults to attn_implementation="sdpa" for LlamaModel.
        # The upstream AlignmentStreamAnalyzer (used only for multilingual inference) sets
        #   tfmr.config.output_attentions = True
        # which raises a ValueError when attn_implementation is "sdpa":
        #   "The `output_attentions` attribute is not supported when using the `attn_implementation` set to sdpa"
        # Fix: force _attn_implementation back to "eager" on the T3's LlamaModel config directly
        # so that the AlignmentStreamAnalyzer can enable output_attentions without crashing.
        if hasattr(model, "t3") and hasattr(model.t3, "tfmr") and hasattr(model.t3.tfmr, "config"):
            model.t3.tfmr.config._attn_implementation = "eager"

        return model


class ChatterboxVCModel:
    """Wrapper for ChatterboxVC (Voice Conversion) model.

    ChatterboxVC provides voice conversion functionality to transform
    speech from one voice to another while preserving content.
    """

    @classmethod
    def from_pretrained(cls, device: str | torch.device | None = None, **kwargs) -> "ChatterboxVC":
        """Load ChatterboxVC model from pretrained weights.

        Args:
            device: Device to load the model on (e.g., "cpu", "cuda").
            **kwargs: Additional arguments passed to ChatterboxVC.from_pretrained.

        Returns:
            ChatterboxVC model instance.
        """
        if device is None:
            device = "cpu"
        elif isinstance(device, torch.device):
            device = str(device)

        return ChatterboxVC.from_pretrained(device=device, **kwargs)


__all__ = ["ChatterboxTTSModel", "ChatterboxTurboTTSModel", "ChatterboxMultilingualTTSModel", "ChatterboxVCModel"]

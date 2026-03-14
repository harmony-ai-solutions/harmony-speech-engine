"""Chatterbox model wrappers for Harmony Speech Engine.

This module provides wrapper classes for the Chatterbox TTS and VC models,
exposing a consistent interface for model loading and inference.
"""
from typing import Optional, Union

import torch

# Import Chatterbox models from the chatterbox library
from chatterbox import ChatterboxMultilingualTTS, ChatterboxTTS, ChatterboxVC
from chatterbox.mtl_tts import SUPPORTED_LANGUAGES as _CHATTERBOX_SUPPORTED_LANGUAGES
from chatterbox.tts_turbo import ChatterboxTurboTTS as _ChatterboxTurboTTS


class ChatterboxTTSModel:
    """Wrapper for ChatterboxTTS model.
    
    ChatterboxTTS provides text-to-speech synthesis with voice cloning support.
    """

    @classmethod
    def from_pretrained(
        cls,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs
    ) -> "ChatterboxTTS":
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
    def from_pretrained(
        cls,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs
    ) -> "_ChatterboxTurboTTS":
        """Load ChatterboxTurboTTS model from pretrained weights.
        
        Args:
            device: Device to load the model on (e.g., "cpu", "cuda").
            **kwargs: Additional arguments passed to ChatterboxTurboTTS.from_pretrained.
        
        Returns:
            ChatterboxTurboTTS model instance.
        """
        if device is None:
            device = "cpu"
        elif isinstance(device, torch.device):
            device = str(device)
        
        return _ChatterboxTurboTTS.from_pretrained(device=device, **kwargs)


class ChatterboxMultilingualTTSModel:
    """Wrapper for ChatterboxMultilingualTTS model.
    
    ChatterboxMultilingualTTS provides TTS synthesis supporting multiple languages.
    """
    
    # 23 supported language codes mirroring upstream chatterbox library.
    # Used by serving_engine for LanguageOptions registration and API validation.
    SUPPORTED_LANGUAGES: dict = _CHATTERBOX_SUPPORTED_LANGUAGES.copy()

    @classmethod
    def from_pretrained(
        cls,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs
    ) -> "ChatterboxMultilingualTTS":
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
            
        return ChatterboxMultilingualTTS.from_pretrained(device=device, **kwargs)


class ChatterboxVCModel:
    """Wrapper for ChatterboxVC (Voice Conversion) model.
    
    ChatterboxVC provides voice conversion functionality to transform
    speech from one voice to another while preserving content.
    """

    @classmethod
    def from_pretrained(
        cls,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs
    ) -> "ChatterboxVC":
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


__all__ = [
    "ChatterboxTTSModel",
    "ChatterboxTurboTTSModel",
    "ChatterboxMultilingualTTSModel",
    "ChatterboxVCModel",
]
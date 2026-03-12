"""Chatterbox model wrappers for Harmony Speech Engine.

This module provides wrapper classes for the Chatterbox TTS and VC models,
exposing a consistent interface for model loading and inference.
"""
from typing import Optional, Union

import torch

# Import Chatterbox models from the chatterbox library
from chatterbox import ChatterboxMultilingualTTS, ChatterboxTTS, ChatterboxVC


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
    """Wrapper for ChatterboxTTS model in turbo mode.
    
    ChatterboxTurboTTS provides faster TTS synthesis with potentially
    reduced quality compared to the standard model.
    """

    @classmethod
    def from_pretrained(
        cls,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs
    ) -> "ChatterboxTTS":
        """Load ChatterboxTTS model in turbo mode from pretrained weights.
        
        Args:
            device: Device to load the model on (e.g., "cpu", "cuda").
            **kwargs: Additional arguments passed to ChatterboxTTS.from_pretrained.
                     May include turbo=True parameter.
            
        Returns:
            ChatterboxTTS model instance in turbo mode.
        """
        if device is None:
            device = "cpu"
        elif isinstance(device, torch.device):
            device = str(device)
            
        # Ensure turbo mode is enabled
        kwargs.setdefault("turbo", True)
            
        return ChatterboxTTS.from_pretrained(device=device, **kwargs)


class ChatterboxMultilingualTTSModel:
    """Wrapper for ChatterboxMultilingualTTS model.
    
    ChatterboxMultilingualTTS provides TTS synthesis supporting multiple languages.
    """

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
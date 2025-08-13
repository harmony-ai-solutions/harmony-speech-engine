"""
Utility functions adapted from VoiceFixer for Harmony Speech Engine integration.
"""

import torch
import torch.nn as nn
import numpy as np
import librosa
from typing import Optional, Tuple

EPS = 1e-8


def check_cuda_availability(cuda: bool = False):
    """Check if CUDA is available and requested."""
    if cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")


def try_tensor_cuda(tensor: torch.Tensor, cuda: bool = False) -> torch.Tensor:
    """Move tensor to CUDA if requested and available."""
    if cuda and torch.cuda.is_available():
        return tensor.cuda()
    return tensor


def tensor2numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor to numpy array."""
    return tensor.detach().cpu().numpy()


def from_log(x: torch.Tensor) -> torch.Tensor:
    """Convert from log scale."""
    return torch.exp(x)


def to_log(x: torch.Tensor) -> torch.Tensor:
    """Convert to log scale."""
    return torch.log(x + EPS)


def save_wave(wav: np.ndarray, fname: str, sample_rate: int = 44100):
    """Save waveform to file using librosa."""
    import soundfile as sf
    sf.write(fname, wav, sample_rate)


def read_wave(fname: str, sample_rate: int = 44100) -> Tuple[np.ndarray, int]:
    """Read waveform from file using librosa."""
    wav, sr = librosa.load(fname, sr=sample_rate)
    return wav, sr


class FDomainHelper(nn.Module):
    """Frequency domain helper for spectrogram operations."""
    
    def __init__(self, 
                 window_size: int = 2048,
                 hop_size: int = 441,
                 center: bool = True,
                 pad_mode: str = "reflect",
                 window: str = "hann",
                 freeze_parameters: bool = True):
        super().__init__()
        self.window_size = window_size
        self.hop_size = hop_size
        self.center = center
        self.pad_mode = pad_mode
        self.window = window
        self.freeze_parameters = freeze_parameters
        
        # Create window
        if window == "hann":
            self.register_buffer('window_tensor', torch.hann_window(window_size))
        else:
            raise NotImplementedError(f"Window {window} not implemented")
    
    def wav_to_spectrogram_phase(self, wav: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert waveform to spectrogram with phase information."""
        # STFT
        stft = torch.stft(
            wav.squeeze(1),
            n_fft=self.window_size,
            hop_length=self.hop_size,
            win_length=self.window_size,
            window=self.window_tensor,
            center=self.center,
            pad_mode=self.pad_mode,
            return_complex=True
        )
        
        # Get magnitude and phase
        magnitude = torch.abs(stft)
        phase = torch.angle(stft)
        
        # Add channel dimension and transpose to match expected format
        magnitude = magnitude.unsqueeze(1).transpose(-1, -2)
        phase = phase.unsqueeze(1).transpose(-1, -2)
        
        return magnitude, phase, stft


class MelScale(nn.Module):
    """Mel-scale transformation."""
    
    def __init__(self, n_mels: int = 128, sample_rate: int = 44100, n_stft: int = 1025):
        super().__init__()
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.n_stft = n_stft
        
        # Create mel filter bank
        mel_basis = librosa.filters.mel(
            sr=sample_rate,
            n_fft=(n_stft - 1) * 2,
            n_mels=n_mels
        )
        self.register_buffer('mel_basis', torch.from_numpy(mel_basis).float())
    
    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Convert spectrogram to mel-scale."""
        # spectrogram shape: [batch, channels, time, freq]
        batch_size, channels, time_steps, freq_bins = spectrogram.shape
        
        # Reshape for matrix multiplication
        spec_flat = spectrogram.view(-1, freq_bins)
        
        # Apply mel transformation
        mel = torch.matmul(spec_flat, self.mel_basis.T)
        
        # Reshape back
        mel = mel.view(batch_size, channels, time_steps, self.n_mels)
        
        return mel


def initialize_dummy_weights(model: nn.Module):
    """Initialize model with dummy weights for testing."""
    for param in model.parameters():
        if param.requires_grad:
            param.data.normal_(0, 0.02)

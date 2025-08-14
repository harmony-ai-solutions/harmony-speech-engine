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
    return wav, int(sr)


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
    
    def spectrogram_phase(self, input: torch.Tensor, eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute spectrogram with phase information for single channel."""
        # STFT - input should be [batch, samples]
        # Ensure we get the right number of frequency bins (n_fft//2 + 1 = 1025 for n_fft=2048)
        stft = torch.stft(
            input,
            n_fft=self.window_size,  # 2048
            hop_length=self.hop_size,  # 441
            win_length=self.window_size,  # 2048
            window=self.window_tensor,
            center=self.center,
            pad_mode=self.pad_mode,
            return_complex=True
        )
        
        # stft shape should be [batch, freq_bins, time_steps] where freq_bins = n_fft//2 + 1 = 1025
        # Get magnitude and phase
        real = stft.real
        imag = stft.imag
        mag = torch.clamp(real**2 + imag**2, eps, np.inf) ** 0.5
        cos = real / (mag + eps)
        sin = imag / (mag + eps)
        
        return mag, cos, sin
    
    def wav_to_spectrogram_phase(self, input: torch.Tensor, eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Waveform to spectrogram with phase information.
        
        Args:
            input: (batch_size, channels_num, segment_samples)
        
        Outputs:
            sps: (batch_size, channels_num, time_steps, freq_bins)
            coss: (batch_size, channels_num, time_steps, freq_bins) 
            sins: (batch_size, channels_num, time_steps, freq_bins)
        """
        sp_list = []
        cos_list = []
        sin_list = []
        channels_num = input.shape[1]
        
        for channel in range(channels_num):
            mag, cos, sin = self.spectrogram_phase(input[:, channel, :], eps=eps)
            # Add channel dimension: [batch, freq, time] -> [batch, 1, freq, time]
            sp_list.append(mag.unsqueeze(1))
            cos_list.append(cos.unsqueeze(1))
            sin_list.append(sin.unsqueeze(1))

        # Concatenate along channel dimension: [batch, channels, freq, time]
        sps = torch.cat(sp_list, dim=1)
        coss = torch.cat(cos_list, dim=1)
        sins = torch.cat(sin_list, dim=1)
        
        # Transpose to get [batch, channels, time, freq] format
        sps = sps.transpose(-1, -2)
        coss = coss.transpose(-1, -2)
        sins = sins.transpose(-1, -2)
        
        return sps, coss, sins


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
        """Convert spectrogram to mel-scale.
        
        Args:
            spectrogram: Input spectrogram [batch, channels, freq, time] (after permute in VoiceFixer)
        Returns:
            mel: Mel spectrogram [batch, channels, freq, time] -> [batch, channels, n_mels, time]
        """
        # spectrogram shape: [batch, channels, freq, time] (after permute operation)
        batch_size, channels, freq_bins, time_steps = spectrogram.shape
        
        # Ensure we have the expected number of frequency bins
        if freq_bins != self.n_stft:
            raise ValueError(f"Expected {self.n_stft} frequency bins, got {freq_bins}")
        
        # Reshape for matrix multiplication: [batch*channels*time, freq]
        # Need to transpose to get [batch, channels, time, freq] first
        spec_transposed = spectrogram.transpose(-1, -2)  # [batch, channels, time, freq]
        spec_flat = spec_transposed.contiguous().view(-1, freq_bins)
        
        # Apply mel transformation: [batch*channels*time, n_mels]
        mel = torch.matmul(spec_flat, self.mel_basis.T)
        
        # Reshape back to [batch, channels, time, n_mels] then transpose to [batch, channels, n_mels, time]
        mel = mel.view(batch_size, channels, time_steps, self.n_mels)
        mel = mel.transpose(-1, -2)  # [batch, channels, n_mels, time]
        
        return mel


def initialize_dummy_weights(model: nn.Module):
    """Initialize model with dummy weights for testing."""
    for param in model.parameters():
        if param.requires_grad:
            param.data.normal_(0, 0.02)

"""
Utility functions adapted from VoiceFixer for Harmony Speech Engine integration.
"""
import math
import warnings

from torchlibrosa.stft import STFT, ISTFT, magphase
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
    """Frequency domain helper for spectrogram operations. Significantly cut down in scope for HSE."""
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

        self.stft = STFT(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=freeze_parameters,
        )

    def spectrogram_phase(self, input, eps=0.0):
        (real, imag) = self.stft(input.float())
        mag = torch.clamp(real**2 + imag**2, eps, np.inf) ** 0.5
        cos = real / mag
        sin = imag / mag
        return mag, cos, sin

    def wav_to_spectrogram_phase(self, input, eps=1e-8):
        """Waveform to spectrogram.

        Args:
          input: (batch_size, channels_num, segment_samples)

        Outputs:
          output: (batch_size, channels_num, time_steps, freq_bins)
        """
        sp_list = []
        cos_list = []
        sin_list = []
        channels_num = input.shape[1]
        for channel in range(channels_num):
            mag, cos, sin = self.spectrogram_phase(input[:, channel, :], eps=eps)
            sp_list.append(mag)
            cos_list.append(cos)
            sin_list.append(sin)

        sps = torch.cat(sp_list, dim=1)
        coss = torch.cat(cos_list, dim=1)
        sins = torch.cat(sin_list, dim=1)
        return sps, coss, sins


class MelScale(nn.Module):
    r"""Turn a normal STFT into a mel frequency STFT, using a conversion
    matrix.  This uses triangular filter banks.

    User can control which device the filter bank (`fb`) is (e.g. fb.to(spec_f.device)).

    Args:
        n_mels (int, optional): Number of mel filterbanks. (Default: ``128``)
        sample_rate (int, optional): Sample rate of audio signal. (Default: ``16000``)
        f_min (float, optional): Minimum frequency. (Default: ``0.``)
        f_max (float or None, optional): Maximum frequency. (Default: ``sample_rate // 2``)
        n_stft (int, optional): Number of bins in STFT. See ``n_fft`` in :class:`Spectrogram`. (Default: ``201``)
        norm (str or None, optional): If 'slaney', divide the triangular mel weights by the width of the mel band
            (area normalization). (Default: ``None``)
        mel_scale (str, optional): Scale to use: ``htk`` or ``slaney``. (Default: ``htk``)

    See also:
        :py:func:`torchaudio.functional.melscale_fbanks` - The function used to
        generate the filter banks.
    """
    __constants__ = ["n_mels", "sample_rate", "f_min", "f_max"]
    
    def __init__(
        self,
        n_mels: int = 128,
        sample_rate: int = 16000,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        n_stft: int = 201,
        norm: Optional[str] = None,
        mel_scale: str = "htk",
    ):
        super().__init__()
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.f_max = f_max if f_max is not None else float(sample_rate // 2)
        self.f_min = f_min
        self.n_stft = n_stft
        self.norm = norm
        self.mel_scale = mel_scale
        
        # Create mel filter bank
        fb = self._melscale_fbanks()
        self.register_buffer("fb", fb)
    
    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            specgram (Tensor): A spectrogram STFT of dimension (..., freq, time).

        Returns:
            Tensor: Mel frequency spectrogram of size (..., ``n_mels``, time).
        """

        # (..., time, freq) dot (freq, n_mels) -> (..., n_mels, time)
        mel_specgram = torch.matmul(spectrogram.transpose(-1, -2), self.fb).transpose(
            -1, -2
        )

        return mel_specgram

    def _melscale_fbanks(self) -> torch.Tensor:
        r"""Create a frequency bin conversion matrix.

        Note:
            For the sake of the numerical compatibility with librosa, not all the coefficients
            in the resulting filter bank has magnitude of 1.

            .. image:: https://download.pytorch.org/torchaudio/doc-assets/mel_fbanks.png
               :alt: Visualization of generated filter bank

        Args:
            n_freqs (int): Number of frequencies to highlight/apply
            f_min (float): Minimum frequency (Hz)
            f_max (float): Maximum frequency (Hz)
            n_mels (int): Number of mel filterbanks
            sample_rate (int): Sample rate of the audio waveform
            norm (str or None, optional): If 'slaney', divide the triangular mel weights by the width of the mel band
                (area normalization). (Default: ``None``)
            mel_scale (str, optional): Scale to use: ``htk`` or ``slaney``. (Default: ``htk``)

        Returns:
            Tensor: Triangular filter banks (fb matrix) of size (``n_stft``, ``n_mels``)
            meaning number of frequencies to highlight/apply to x the number of filterbanks.
            Each column is a filterbank so that assuming there is a matrix A of
            size (..., ``n_stft``), the applied result would be
            ``A * melscale_fbanks(A.size(-1), ...)``.

        """

        # freq bins
        all_freqs = torch.linspace(0, self.sample_rate // 2, self.n_stft)

        # calculate mel freq bins
        m_min = 2595.0 * math.log10(1.0 + (self.f_min / 700.0))
        m_max = 2595.0 * math.log10(1.0 + (self.f_max / 700.0))

        m_pts = torch.linspace(m_min, m_max, self.n_mels + 2)
        f_pts = 700.0 * (10.0 ** (m_pts / 2595.0) - 1.0)

        # create filterbank
        fb = self._create_triangular_filterbank(all_freqs, f_pts)

        if (fb.max(dim=0).values == 0.0).any():
            warnings.warn(
                "At least one mel filterbank has all zero values. "
                f"The value for `n_mels` ({self.n_mels}) may be set too high. "
                f"Or, the value for `n_stft` ({self.n_stft}) may be set too low."
            )

        return fb

    def _create_triangular_filterbank(
        self,
        all_freqs: torch.Tensor,
        f_pts: torch.Tensor,
    ) -> torch.Tensor:
        """Create a triangular filter bank.

        Args:
            all_freqs (Tensor): STFT freq points of size (`n_freqs`).
            f_pts (Tensor): Filter mid points of size (`n_filter`).

        Returns:
            fb (Tensor): The filter bank of size (`n_freqs`, `n_filter`).
        """
        # Adopted from Librosa
        # calculate the difference between each filter mid point and each stft freq point in hertz
        f_diff = f_pts[1:] - f_pts[:-1]  # (n_filter + 1)
        slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)  # (n_freqs, n_filter + 2)
        # create overlapping triangles
        zero = torch.zeros(1)
        down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1]  # (n_freqs, n_filter)
        up_slopes = slopes[:, 2:] / f_diff[1:]  # (n_freqs, n_filter)
        fb = torch.max(zero, torch.min(down_slopes, up_slopes))

        return fb


def initialize_dummy_weights(model: nn.Module):
    """Initialize model with dummy weights for testing."""
    for param in model.parameters():
        if param.requires_grad:
            param.data.normal_(0, 0.02)

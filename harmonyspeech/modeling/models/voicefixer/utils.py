"""
Utility functions adapted from VoiceFixer for Harmony Speech Engine integration.
"""
import math
import warnings

from torchlibrosa.stft import STFT
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


def tr_normalize(S: torch.Tensor) -> torch.Tensor:
    """Tensor version of normalize with proper clipping."""
    # VoiceFixer Config values for 44.1kHz
    allow_clipping_in_normalization = True
    symmetric_mels = True
    max_abs_value = 4.0
    min_db = -115
    
    if allow_clipping_in_normalization:
        if symmetric_mels:
            return torch.clip(
                (2 * max_abs_value) * ((S - min_db) / (-min_db))
                - max_abs_value,
                -max_abs_value,
                max_abs_value,
            )
        else:
            return torch.clip(
                max_abs_value * ((S - min_db) / (-min_db)),
                0,
                max_abs_value,
            )

    assert S.max() <= 0 and S.min() - min_db >= 0
    if symmetric_mels:
        return (2 * max_abs_value) * (
            (S - min_db) / (-min_db)
        ) - max_abs_value
    else:
        return max_abs_value * ((S - min_db) / (-min_db))


def tr_amp_to_db(x: torch.Tensor) -> torch.Tensor:
    """Tensor version of amp_to_db conversion."""
    min_level_db = -100
    min_level = torch.exp(min_level_db / 20 * torch.log(torch.tensor(10.0)))
    min_level = min_level.type_as(x)
    return 20 * torch.log10(torch.maximum(min_level, x))


def tr_pre(npy: torch.Tensor) -> torch.Tensor:
    """Critical preprocessing with padding and transpose."""
    num_mels = 128  # VoiceFixer Config value

    conditions = npy.transpose(1, 2)
    l = conditions.size(-1)
    pad_tail = l % 2 + 4
    zeros = (
        torch.zeros([conditions.size()[0], num_mels, pad_tail]).type_as(
            conditions
        )
        + -4.0
    )
    return torch.cat([conditions, zeros], dim=-1)


def get_mel_weight_torch(percent=1, a=18.8927416350036, b=0.0269863588184314):
    mel_weight_torch = torch.tensor(
        [
            19.40951426,
            19.94047336,
            20.4859038,
            21.04629067,
            21.62194148,
            22.21335214,
            22.8210215,
            23.44529231,
            24.08660962,
            24.74541882,
            25.42234287,
            26.11770576,
            26.83212784,
            27.56615283,
            28.32007747,
            29.0947679,
            29.89060111,
            30.70832636,
            31.54828121,
            32.41121487,
            33.29780773,
            34.20865341,
            35.14437675,
            36.1056621,
            37.09332763,
            38.10795802,
            39.15039691,
            40.22119881,
            41.32154931,
            42.45172373,
            43.61293329,
            44.80609379,
            46.031602,
            47.29070223,
            48.58427549,
            49.91327905,
            51.27863232,
            52.68119708,
            54.1222372,
            55.60274206,
            57.12364703,
            58.68617876,
            60.29148652,
            61.94081306,
            63.63501986,
            65.37562658,
            67.16408954,
            69.00109084,
            70.88850318,
            72.82736101,
            74.81985537,
            76.86654792,
            78.96885475,
            81.12900906,
            83.34840929,
            85.62810662,
            87.97005418,
            90.37689804,
            92.84887686,
            95.38872881,
            97.99777002,
            100.67862715,
            103.43232942,
            106.26140638,
            109.16827015,
            112.15470471,
            115.22184756,
            118.37439245,
            121.6122689,
            124.93877158,
            128.35661454,
            131.86761321,
            135.47417938,
            139.18059494,
            142.98713744,
            146.89771854,
            150.91684347,
            155.0446638,
            159.28614648,
            163.64270198,
            168.12035831,
            172.71749158,
            177.44220154,
            182.29556933,
            187.28286676,
            192.40502126,
            197.6682721,
            203.07516896,
            208.63088733,
            214.33770931,
            220.19910108,
            226.22363072,
            232.41087124,
            238.76803591,
            245.30079083,
            252.01064464,
            258.90261676,
            265.98474,
            273.26010248,
            280.73496362,
            288.41440094,
            296.30489752,
            304.41180337,
            312.7377183,
            321.28877878,
            330.07870237,
            339.10812951,
            348.38276173,
            357.91393924,
            367.70513992,
            377.76413924,
            388.09467408,
            398.70920178,
            409.61813793,
            420.81980127,
            432.33215467,
            444.16083117,
            456.30919947,
            468.78589276,
            481.61325588,
            494.78824596,
            508.31969844,
            522.2238331,
            536.51163441,
            551.18859414,
            566.26142988,
            581.75006061,
            597.66210737,
        ]
    )

    x_orig_torch = torch.linspace(1, mel_weight_torch.shape[0], steps=mel_weight_torch.shape[0])
    b = percent * b

    def func(a, b, x):
        return a * torch.exp(b * x)

    return func(a, b, x_orig_torch)

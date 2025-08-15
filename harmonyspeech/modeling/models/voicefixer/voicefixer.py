"""
VoiceFixer models adapted for Harmony Speech Engine.
Based on the original VoiceFixer implementation by Haohe Liu.
"""
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Dict, Any

from .utils import (
    from_log, to_log, FDomainHelper, MelScale,
    tr_normalize, tr_amp_to_db, tr_pre, get_mel_weight_torch
)
from .modules import UNetResComplex_100Mb, BN_GRU, UpsampleNet, ResStack, PQMF


class VoiceFixerGenerator(nn.Module):
    """VoiceFixer generator combining denoiser and UNet."""

    def __init__(self, n_mel: int = 128, hidden: int = 1025, channels: int = 2):
        super().__init__()

        # Denoiser network
        self.denoiser = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Linear(n_mel, n_mel * 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(1),
            nn.Linear(n_mel * 2, n_mel * 4),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            BN_GRU(
                input_dim=n_mel * 4,
                hidden_dim=n_mel * 2,
                bidirectional=True,
                layer=2,
                batchnorm=True,
            ),
            BN_GRU(
                input_dim=n_mel * 4,
                hidden_dim=n_mel * 2,
                bidirectional=True,
                layer=2,
                batchnorm=True,
            ),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Linear(n_mel * 4, n_mel * 4),
            nn.Dropout(0.5),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Linear(n_mel * 4, n_mel),
            nn.Sigmoid(),
        )

        # UNet for enhancement - use the proper original architecture
        self.unet = UNetResComplex_100Mb(channels=channels)

    def forward(self, sp: torch.Tensor, mel_orig: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Denoising
        noisy = mel_orig.clone()
        clean = self.denoiser(noisy) * noisy
        x = to_log(clean.detach())
        unet_in = torch.cat([to_log(mel_orig), x], dim=1)

        # UNet enhancement
        unet_out = self.unet(unet_in)["mel"]

        # Final mel spectrogram
        mel = unet_out + x

        return {
            "mel": mel,
            "lstm_out": unet_out,
            "unet_out": unet_out,
            "noisy": noisy,
            "clean": clean,
        }


class VoiceFixerRestorer(nn.Module):
    """VoiceFixer restoration model for audio enhancement."""

    def __init__(self):
        super().__init__()
        # Use VoiceFixer's pretrained defaults
        self.channels = 2
        self.sample_rate = 44100
        self.window_size = 2048
        self.hop_size = 441
        self.n_mel = 128

        # Initialize components
        self.f_helper = FDomainHelper(
            window_size=self.window_size,
            hop_size=self.hop_size,
            center=True,
            pad_mode="reflect",
            window="hann",
            freeze_parameters=True,
        )

        self.mel = MelScale(
            n_mels=self.n_mel,
            sample_rate=self.sample_rate,
            n_stft=self.window_size // 2 + 1
        )

        # Main generator model
        self.generator = VoiceFixerGenerator(
            n_mel=self.n_mel,
            hidden=self.window_size // 2 + 1,
            channels=self.channels
        )

        # Mel weight for 44kHz 128-bin setup
        self.mel_weight_44k_128 = (
            torch.tensor(
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
            / 19.40951426
        )
        self.register_buffer('mel_weight_tensor', self.mel_weight_44k_128[None, None, None, ...])

    def _pre(self, input_wav: torch.Tensor) -> tuple:
        """
        Preprocessing: convert audio to mel spectrogram.
        Uses the original VoiceFixer preprocessing logic with strategic permute operations.
        """
        # Get spectrogram - this returns [batch, channels, time, freq] format from the corrected FDomainHelper
        sp, _, _ = self.f_helper.wav_to_spectrogram_phase(input_wav)
        mel_orig = self.mel(sp.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)

        return sp, mel_orig

    def _trim_center(self, est: torch.Tensor, ref: torch.Tensor) -> tuple:
        """Trim tensors to same length."""
        diff = abs(est.shape[-1] - ref.shape[-1])
        if est.shape[-1] == ref.shape[-1]:
            return est, ref
        elif est.shape[-1] > ref.shape[-1]:
            min_len = min(est.shape[-1], ref.shape[-1])
            est = est[..., int(diff // 2): -int(diff // 2)] if diff > 0 else est
            est, ref = est[..., :min_len], ref[..., :min_len]
            return est, ref
        else:
            min_len = min(est.shape[-1], ref.shape[-1])
            ref = ref[..., int(diff // 2): -int(diff // 2)] if diff > 0 else ref
            est, ref = est[..., :min_len], ref[..., :min_len]
            return est, ref

    def forward(self, audio_tensor: torch.Tensor) -> torch.Tensor:
        """
        Restore audio quality by removing noise and artifacts.
        
        Args:
            audio_tensor: Input audio tensor [batch, channels, samples]
        Returns:
            Enhanced mel spectrogram tensor [batch, 1, time, 128] - EXACTLY what vocoder expects
        """
        # Process audio in segments to handle memory constraints
        seg_length = self.sample_rate * 30  # 30 second segments
        results = []

        for i in range(0, audio_tensor.shape[-1], seg_length):
            segment = audio_tensor[..., i:i + seg_length]

            # Preprocessing - both sp and mel_noisy are [batch, channels, freq, time]
            sp, mel_noisy = self._pre(segment)

            # Generate enhanced mel spectrogram
            output = self.generator(sp, mel_noisy)
            denoised_mel = from_log(output["mel"])

            # Return the enhanced mel spectrogram
            results.append(denoised_mel)

        # Concatenate results along time dimension (dimension 3 for [batch, channels, freq, time])
        if len(results) > 1:
            output_mel = torch.cat(results, dim=3)  # Concatenate along time dimension
        else:
            output_mel = results[0]

        return output_mel

    def load_weights(self, checkpoint: Dict[str, Any], hf_config: Optional[Dict] = None):
        """Load weights from VoiceFixer checkpoint."""
        if isinstance(checkpoint, dict):
            # Handle different checkpoint formats
            if "model_state" in checkpoint:
                state_dict = checkpoint["model_state"]
            elif "generator" in checkpoint:
                state_dict = checkpoint["generator"]
            else:
                state_dict = checkpoint

            # Load state dict with error handling
            try:
                # Create a mapping for our model structure
                model_state_dict = self.state_dict()
                new_state_dict = {}

                for key, value in state_dict.items():
                    # Map original VoiceFixer keys to our structure
                    if key.startswith("generator."):
                        new_key = key.replace("generator.", "generator.")
                        if new_key in model_state_dict:
                            new_state_dict[new_key] = value
                    elif key in model_state_dict:
                        new_state_dict[key] = value

                # Load the mapped weights
                self.load_state_dict(new_state_dict, strict=False)

            except Exception as e:
                print(f"Warning: Could not load some weights: {e}")
                # Initialize with random weights if loading fails
                self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights for modules."""
        for m in module.modules():
            if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
                for name, param in m.named_parameters():
                    if "weight_ih" in name:
                        torch.nn.init.xavier_uniform_(param.data)
                    elif "weight_hh" in name:
                        torch.nn.init.orthogonal_(param.data)
                    elif "bias" in name:
                        param.data.fill_(0)


class VoiceFixerVocoderGenerator(nn.Module):
    """Original VoiceFixer Generator with conditional network and sophisticated upsampling."""

    def __init__(self, in_channels=128, use_elu=False, hp=None):
        super(VoiceFixerVocoderGenerator, self).__init__()
        self.hp = hp

        # Configuration parameters (from original VoiceFixer Config for 44.1kHz)
        self.channels = 1024
        self.upsample_scales = [7, 7, 3, 3]
        self.use_condnet = True
        self.out_channels = 1
        self.resstack_depth = [8, 8, 8, 8]
        self.use_postnet = False
        self.use_cond_rnn = False

        # Conditional network for feature processing
        if self.use_condnet:
            cond_channels = 512  # From original config
            self.condnet = nn.Sequential(
                nn.utils.parametrizations.weight_norm(
                    nn.Conv1d(in_channels, cond_channels, kernel_size=3, padding=1)
                ),
                nn.ELU(),
                nn.utils.parametrizations.weight_norm(
                    nn.Conv1d(cond_channels, cond_channels, kernel_size=3, padding=1)
                ),
                nn.ELU(),
                nn.utils.parametrizations.weight_norm(
                    nn.Conv1d(cond_channels, cond_channels, kernel_size=3, padding=1)
                ),
                nn.ELU(),
                nn.utils.parametrizations.weight_norm(
                    nn.Conv1d(cond_channels, cond_channels, kernel_size=3, padding=1)
                ),
                nn.ELU(),
                nn.utils.parametrizations.weight_norm(
                    nn.Conv1d(cond_channels, cond_channels, kernel_size=3, padding=1)
                ),
                nn.ELU(),
            )
            in_channels = cond_channels

        # Activation function
        if use_elu:
            act = nn.ELU()
        else:
            act = nn.LeakyReLU(0.2, True)

        # Kernel sizes for each layer
        kernel_size = [3, 3, 3, 3]

        # Main generator network
        if self.out_channels == 1:
            self.generator = nn.Sequential(
                nn.ReflectionPad1d(3),
                nn.utils.parametrizations.weight_norm(nn.Conv1d(in_channels, self.channels, kernel_size=7)),
                act,
                UpsampleNet(self.channels, self.channels // 2, self.upsample_scales[0], hp, 0),
                ResStack(self.channels // 2, kernel_size[0], self.resstack_depth[0], hp),
                act,
                UpsampleNet(
                    self.channels // 2, self.channels // 4, self.upsample_scales[1], hp, 1
                ),
                ResStack(self.channels // 4, kernel_size[1], self.resstack_depth[1], hp),
                act,
                UpsampleNet(
                    self.channels // 4, self.channels // 8, self.upsample_scales[2], hp, 2
                ),
                ResStack(self.channels // 8, kernel_size[2], self.resstack_depth[2], hp),
                act,
                UpsampleNet(
                    self.channels // 8, self.channels // 16, self.upsample_scales[3], hp, 3
                ),
                ResStack(self.channels // 16, kernel_size[3], self.resstack_depth[3], hp),
                act,
                nn.ReflectionPad1d(3),
                nn.utils.parametrizations.weight_norm(
                    nn.Conv1d(self.channels // 16, self.out_channels, kernel_size=7)
                ),
                nn.Tanh(),
            )
        else:
            # Multi-channel output with PQMF
            m_channels = 768  # From original config
            self.generator = nn.Sequential(
                nn.ReflectionPad1d(3),
                nn.utils.parametrizations.weight_norm(nn.Conv1d(in_channels, m_channels, kernel_size=7)),
                act,
                UpsampleNet(m_channels, m_channels // 2, self.upsample_scales[0], hp),
                ResStack(m_channels // 2, kernel_size[0], self.resstack_depth[0], hp),
                act,
                UpsampleNet(m_channels // 2, m_channels // 4, self.upsample_scales[1], hp),
                ResStack(m_channels // 4, kernel_size[1], self.resstack_depth[1], hp),
                act,
                UpsampleNet(m_channels // 4, m_channels // 8, self.upsample_scales[2], hp),
                ResStack(m_channels // 8, kernel_size[2], self.resstack_depth[2], hp),
                act,
                nn.ReflectionPad1d(3),
                nn.utils.parametrizations.weight_norm(
                    nn.Conv1d(m_channels // 8, self.out_channels, kernel_size=7)
                ),
                nn.Tanh(),
            )

        # PQMF for multi-channel output
        if self.out_channels > 1:
            self.pqmf = PQMF(4, 64)

    def forward(self, conditions, use_res=False, f0=None):
        """Forward pass through the generator."""
        # Apply conditional network
        if self.use_condnet:
            conditions = self.condnet(conditions)

        # Apply conditional RNN if enabled
        if self.use_cond_rnn:
            conditions, _ = self.rnn(conditions.transpose(1, 2))
            conditions = conditions.transpose(1, 2)

        # Generate audio
        wav = self.generator(conditions)

        # Multi-channel processing with PQMF
        if self.out_channels > 1:
            B = wav.size(0)
            f_wav = (
                self.pqmf.synthesis(wav)
                .transpose(1, 2)
                .reshape(B, 1, -1)
                .clamp(-0.99, 0.99)
            )
            return f_wav, wav

        return wav

    def remove_weight_norm(self):
        """Remove weight normalization from all layers."""

        def _remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)


class VoiceFixerVocoder(nn.Module):
    """VoiceFixer vocoder for converting mel-spectrograms to audio using original architecture."""

    def __init__(self):
        super().__init__()
        # Use VoiceFixer's pretrained defaults for 44.1kHz
        self.sample_rate = 44100
        self.mel_bins = 128

        # Initialize with original Generator architecture
        self.model = VoiceFixerVocoderGenerator(in_channels=self.mel_bins)

        # Mel weight for normalization
        self.register_buffer('weight_torch', get_mel_weight_torch(percent=1.0)[None, None, None, ...])

    def forward(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Convert mel-spectrogram to audio waveform using original VoiceFixer logic.
        
        Args:
            mel_spectrogram: Mel-spectrogram tensor [batchsize, 1, t-steps, n_mel]
        Returns:
            Audio waveform tensor [batch, 1, samples]
        """
        # fail hard if wrong format
        assert mel_spectrogram.size()[-1] == 128, f"Expected 128 mel bins, got {mel_spectrogram.size()[-1]}"

        # Ensure tensors are of the same type
        self.weight_torch = self.weight_torch.type_as(mel_spectrogram)

        # 1. Apply mel weight normalization
        mel = mel_spectrogram / self.weight_torch

        # 2. Convert to dB and normalize (tr_amp_to_db + tr_normalize)
        mel = tr_normalize(tr_amp_to_db(torch.abs(mel)) - 20.0)

        # 3. Apply preprocessing with padding and transpose (tr_pre)
        # Original: mel[:, 0, ...] extracts first channel, then tr_pre
        mel = tr_pre(mel[:, 0, ...])

        # 4. Generate audio through the original Generator
        audio = self.model(mel)
        return audio

    def load_weights(self, checkpoint: Dict[str, Any], hf_config: Optional[Dict] = None):
        """Load weights from VoiceFixer vocoder checkpoint."""
        if isinstance(checkpoint, dict):
            # Handle different checkpoint formats
            if "generator" in checkpoint:
                state_dict = checkpoint["generator"]
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint

            # Load state dict with error handling
            try:
                # Create a mapping for our model structure
                model_state_dict = self.state_dict()
                new_state_dict = {}

                for key, value in state_dict.items():
                    # Map original VoiceFixer vocoder keys to our structure
                    new_key = key.replace("melgan.", "model.")
                    if new_key in model_state_dict:
                        new_state_dict[new_key] = value
                    elif key in model_state_dict:
                        new_state_dict[key] = value

                # Load the mapped weights
                self.load_state_dict(new_state_dict, strict=False)

            except Exception as e:
                print(f"Warning: Could not load some vocoder weights: {e}")
                # Initialize with random weights if loading fails
                self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights for modules."""
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d, nn.ConvTranspose2d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

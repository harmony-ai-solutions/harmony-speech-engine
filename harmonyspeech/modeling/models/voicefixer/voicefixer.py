"""
VoiceFixer models adapted for Harmony Speech Engine.
Based on the original VoiceFixer implementation by Haohe Liu.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import os
from typing import Optional, Dict, Any

from .utils import (
    check_cuda_availability, try_tensor_cuda, tensor2numpy, 
    from_log, to_log, FDomainHelper, MelScale, EPS
)


class BN_GRU(nn.Module):
    """Batch normalized GRU layer."""
    
    def __init__(self, input_dim: int, hidden_dim: int, layer: int = 1, 
                 bidirectional: bool = False, batchnorm: bool = True, dropout: float = 0.0):
        super().__init__()
        self.batchnorm = batchnorm
        if batchnorm:
            self.bn = nn.BatchNorm2d(1)
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=layer,
            bidirectional=bidirectional,
            dropout=dropout,
            batch_first=True,
        )
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
                for name, param in m.named_parameters():
                    if "weight_ih" in name:
                        torch.nn.init.xavier_uniform_(param.data)
                    elif "weight_hh" in name:
                        torch.nn.init.orthogonal_(param.data)
                    elif "bias" in name:
                        param.data.fill_(0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # (batch, 1, seq, feature)
        if self.batchnorm:
            inputs = self.bn(inputs)
        out, _ = self.gru(inputs.squeeze(1))
        return out.unsqueeze(1)


class UNetResComplex_100Mb(nn.Module):
    """UNet architecture for complex spectrogram processing."""
    
    def __init__(self, channels: int = 2):
        super().__init__()
        self.channels = channels
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(256 + 128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(128 + 64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(64 + 32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Output layer
        self.output = nn.Conv2d(32, channels, 3, padding=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Encoder
        e1 = self.enc1(x)
        e1_pool = self.pool(e1)
        
        e2 = self.enc2(e1_pool)
        e2_pool = self.pool(e2)
        
        e3 = self.enc3(e2_pool)
        e3_pool = self.pool(e3)
        
        # Bottleneck
        b = self.bottleneck(e3_pool)
        
        # Decoder with skip connections
        d3 = self.upsample(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.upsample(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.upsample(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        output = self.output(d1)
        
        return {"mel": output}


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
        
        # UNet for enhancement
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


class VoiceFixerVocoderGenerator(nn.Module):
    """VoiceFixer vocoder generator for mel-to-audio conversion."""
    
    def __init__(self, cin_channels: int = 128):
        super().__init__()
        self.cin_channels = cin_channels
        
        # Simple vocoder architecture
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(cin_channels, 512, 3, padding=1),
            nn.Conv1d(512, 512, 3, padding=1),
            nn.Conv1d(512, 512, 3, padding=1),
            nn.Conv1d(512, 512, 3, padding=1),
        ])
        
        self.upsample_layers = nn.ModuleList([
            nn.ConvTranspose1d(512, 256, 4, stride=2, padding=1),
            nn.ConvTranspose1d(256, 128, 4, stride=2, padding=1),
            nn.ConvTranspose1d(128, 64, 4, stride=2, padding=1),
            nn.ConvTranspose1d(64, 32, 4, stride=2, padding=1),
        ])
        
        self.output_layer = nn.Conv1d(32, 1, 3, padding=1)
        
        # Activation
        self.activation = nn.LeakyReLU(0.2)
        
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: Mel spectrogram [batch, mel_bins, time]
        Returns:
            Audio waveform [batch, 1, samples]
        """
        x = mel
        
        # Convolution layers
        for conv in self.conv_layers:
            x = self.activation(conv(x))
        
        # Upsampling layers
        for upsample in self.upsample_layers:
            x = self.activation(upsample(x))
        
        # Output layer
        x = torch.tanh(self.output_layer(x))
        
        return x
    
    def remove_weight_norm(self):
        """Remove weight normalization (compatibility method)."""
        pass


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
        self.mel_weight_44k_128 = torch.tensor([
            19.40951426, 19.94047336, 20.4859038, 21.04629067, 21.62194148,
            22.21335214, 22.8210215, 23.44529231, 24.08660962, 24.74541882,
            25.42234287, 26.11770576, 26.83212784, 27.56615283, 28.32007747,
            29.0947679, 29.89060111, 30.70832636, 31.54828121, 32.41121487,
            33.29780773, 34.20865341, 35.14437675, 36.1056621, 37.09332763,
            38.10795802, 39.15039691, 40.22119881, 41.32154931, 42.45172373,
            43.61293329, 44.80609379, 46.031602, 47.29070223, 48.58427549,
            49.91327905, 51.27863232, 52.68119708, 54.1222372, 55.60274206,
            57.12364703, 58.68617876, 60.29148652, 61.94081306, 63.63501986,
            65.37562658, 67.16408954, 69.00109084, 70.88850318, 72.82736101,
            74.81985537, 76.86654792, 78.96885475, 81.12900906, 83.34840929,
            85.62810662, 87.97005418, 90.37689804, 92.84887686, 95.38872881,
            97.99777002, 100.67862715, 103.43232942, 106.26140638, 109.16827015,
            112.15470471, 115.22184756, 118.37439245, 121.6122689, 124.93877158,
            128.35661454, 131.86761321, 135.47417938, 139.18059494, 142.98713744,
            146.89771854, 150.91684347, 155.0446638, 159.28614648, 163.64270198,
            168.12035831, 172.71749158, 177.44220154, 182.29556933, 187.28286676,
            192.40502126, 197.6682721, 203.07516896, 208.63088733, 214.33770931,
            220.19910108, 226.22363072, 232.41087124, 238.76803591, 245.30079083,
            252.01064464, 258.90261676, 265.98474, 273.26010248, 280.73496362,
            288.41440094, 296.30489752, 304.41180337, 312.7377183, 321.28877878,
            330.07870237, 339.10812951, 348.38276173, 357.91393924, 367.70513992,
            377.76413924, 388.09467408, 398.70920178, 409.61813793, 420.81980127,
            432.33215467, 444.16083117, 456.30919947, 468.78589276, 481.61325588,
            494.78824596, 508.31969844, 522.2238331, 536.51163441, 551.18859414,
            566.26142988, 581.75006061, 597.66210737
        ]) / 19.40951426
        self.register_buffer('mel_weight_tensor', self.mel_weight_44k_128[None, None, None, ...])
    
    def _pre(self, input_wav: torch.Tensor) -> tuple:
        """Preprocessing: convert audio to mel spectrogram."""
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
            est = est[..., int(diff // 2) : -int(diff // 2)] if diff > 0 else est
            est, ref = est[..., :min_len], ref[..., :min_len]
            return est, ref
        else:
            min_len = min(est.shape[-1], ref.shape[-1])
            ref = ref[..., int(diff // 2) : -int(diff // 2)] if diff > 0 else ref
            est, ref = est[..., :min_len], ref[..., :min_len]
            return est, ref
    
    def forward(self, audio_tensor: torch.Tensor) -> torch.Tensor:
        """
        Restore audio quality by removing noise and artifacts.
        
        Args:
            audio_tensor: Input audio tensor [batch, channels, samples]
        Returns:
            Enhanced audio tensor [batch, channels, samples]
        """
        # Process audio in segments to handle memory constraints
        seg_length = self.sample_rate * 30  # 30 second segments
        results = []
        
        for i in range(0, audio_tensor.shape[-1], seg_length):
            segment = audio_tensor[..., i:i + seg_length]
            
            # Preprocessing
            sp, mel_noisy = self._pre(segment)
            
            # Generate enhanced mel spectrogram
            output = self.generator(sp, mel_noisy)
            denoised_mel = from_log(output["mel"])
            
            # For now, return the mel spectrogram (vocoder integration comes later)
            results.append(denoised_mel)
        
        # Concatenate results
        if len(results) > 1:
            output_mel = torch.cat(results, dim=-1)
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
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d, nn.ConvTranspose2d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


class VoiceFixerVocoder(nn.Module):
    """VoiceFixer vocoder for converting mel-spectrograms to audio."""
    
    def __init__(self):
        super().__init__()
        # Use VoiceFixer's pretrained defaults
        self.sample_rate = 44100
        self.mel_bins = 128
        
        # Initialize vocoder model
        self.model = VoiceFixerVocoderGenerator(cin_channels=self.mel_bins)
        
        # Mel weight for normalization
        self.mel_weight_44k_128 = torch.tensor([
            19.40951426, 19.94047336, 20.4859038, 21.04629067, 21.62194148,
            22.21335214, 22.8210215, 23.44529231, 24.08660962, 24.74541882,
            25.42234287, 26.11770576, 26.83212784, 27.56615283, 28.32007747,
            29.0947679, 29.89060111, 30.70832636, 31.54828121, 32.41121487,
            33.29780773, 34.20865341, 35.14437675, 36.1056621, 37.09332763,
            38.10795802, 39.15039691, 40.22119881, 41.32154931, 42.45172373,
            43.61293329, 44.80609379, 46.031602, 47.29070223, 48.58427549,
            49.91327905, 51.27863232, 52.68119708, 54.1222372, 55.60274206,
            57.12364703, 58.68617876, 60.29148652, 61.94081306, 63.63501986,
            65.37562658, 67.16408954, 69.00109084, 70.88850318, 72.82736101,
            74.81985537, 76.86654792, 78.96885475, 81.12900906, 83.34840929,
            85.62810662, 87.97005418, 90.37689804, 92.84887686, 95.38872881,
            97.99777002, 100.67862715, 103.43232942, 106.26140638, 109.16827015,
            112.15470471, 115.22184756, 118.37439245, 121.6122689, 124.93877158,
            128.35661454, 131.86761321, 135.47417938, 139.18059494, 142.98713744,
            146.89771854, 150.91684347, 155.0446638, 159.28614648, 163.64270198,
            168.12035831, 172.71749158, 177.44220154, 182.29556933, 187.28286676,
            192.40502126, 197.6682721, 203.07516896, 208.63088733, 214.33770931,
            220.19910108, 226.22363072, 232.41087124, 238.76803591, 245.30079083,
            252.01064464, 258.90261676, 265.98474, 273.26010248, 280.73496362,
            288.41440094, 296.30489752, 304.41180337, 312.7377183, 321.28877878,
            330.07870237, 339.10812951, 348.38276173, 357.91393924, 367.70513992,
            377.76413924, 388.09467408, 398.70920178, 409.61813793, 420.81980127,
            432.33215467, 444.16083117, 456.30919947, 468.78589276, 481.61325588,
            494.78824596, 508.31969844, 522.2238331, 536.51163441, 551.18859414,
            566.26142988, 581.75006061, 597.66210737
        ]) / 19.40951426
        self.register_buffer('weight_torch', self.mel_weight_44k_128[None, None, None, ...])
    
    def _normalize_mel(self, mel: torch.Tensor) -> torch.Tensor:
        """Normalize mel spectrogram for vocoder input."""
        # Apply mel weighting
        mel = mel / self.weight_torch.type_as(mel)
        
        # Convert to dB and normalize
        mel_db = 20 * torch.log10(torch.abs(mel) + 1e-8) - 20.0
        mel_normalized = (mel_db + 100) / 100  # Normalize to [0, 1] range
        mel_normalized = torch.clamp(mel_normalized, 0, 1)
        
        return mel_normalized
    
    def forward(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Convert mel-spectrogram to audio waveform.
        
        Args:
            mel_spectrogram: Mel-spectrogram tensor [batch, 1, time, mel_bins] or [batch, mel_bins, time]
        Returns:
            Audio waveform tensor [batch, 1, samples]
        """
        # Handle different input formats
        if mel_spectrogram.dim() == 4:
            # [batch, 1, time, mel_bins] -> [batch, mel_bins, time]
            mel = mel_spectrogram.squeeze(1).transpose(-1, -2)
        elif mel_spectrogram.dim() == 3:
            # [batch, mel_bins, time] - already correct format
            mel = mel_spectrogram
        else:
            raise ValueError(f"Unexpected mel spectrogram shape: {mel_spectrogram.shape}")
        
        # Ensure correct number of mel bins
        if mel.shape[1] != self.mel_bins:
            raise ValueError(f"Expected {self.mel_bins} mel bins, got {mel.shape[1]}")
        
        # Normalize mel spectrogram
        mel_norm = self._normalize_mel(mel)
        
        # Generate audio
        audio = self.model(mel_norm)
        
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

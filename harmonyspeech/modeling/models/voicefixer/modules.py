"""
VoiceFixer modules with proper dimension handling and original vocoder architecture.
Based on the original VoiceFixer implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


def init_layer(layer):
    """Initialize a Linear or Convolutional layer."""
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.0)


def init_bn(bn):
    """Initialize a Batchnorm layer."""
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)


def init_gru(rnn):
    """Initialize a GRU layer."""
    def _concat_init(tensor, init_funcs):
        (length, fan_out) = tensor.shape
        fan_in = length // len(init_funcs)
        for (i, init_func) in enumerate(init_funcs):
            init_func(tensor[i * fan_in : (i + 1) * fan_in, :])

    def _inner_uniform(tensor):
        fan_in = nn.init._calculate_correct_fan(tensor, "fan_in")
        nn.init.uniform_(tensor, -math.sqrt(3 / fan_in), math.sqrt(3 / fan_in))

    for i in range(rnn.num_layers):
        _concat_init(
            getattr(rnn, "weight_ih_l{}".format(i)),
            [_inner_uniform, _inner_uniform, _inner_uniform],
        )
        torch.nn.init.constant_(getattr(rnn, "bias_ih_l{}".format(i)), 0)

        _concat_init(
            getattr(rnn, "weight_hh_l{}".format(i)),
            [_inner_uniform, _inner_uniform, nn.init.orthogonal_],
        )
        torch.nn.init.constant_(getattr(rnn, "bias_hh_l{}".format(i)), 0)


def act(x, activation):
    """Apply activation function."""
    if activation == "relu":
        return F.relu_(x)
    elif activation == "leaky_relu":
        return F.leaky_relu_(x, negative_slope=0.2)
    elif activation == "swish":
        return x * torch.sigmoid(x)
    else:
        raise Exception("Incorrect activation!")


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


class ConvBlockRes(nn.Module):
    """Residual convolution block."""
    
    def __init__(self, in_channels, out_channels, size, activation, momentum):
        super(ConvBlockRes, self).__init__()

        self.activation = activation
        if type(size) == type((3, 4)):
            pad = size[0] // 2
            size = size[0]
        else:
            pad = size // 2
            size = size

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(size, size),
            stride=(1, 1),
            dilation=(1, 1),
            padding=(pad, pad),
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(in_channels, momentum=momentum)

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(size, size),
            stride=(1, 1),
            dilation=(1, 1),
            padding=(pad, pad),
            bias=False,
        )

        self.bn2 = nn.BatchNorm2d(out_channels, momentum=momentum)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
            )
            self.is_shortcut = True
        else:
            self.is_shortcut = False

        self.init_weights()

    def init_weights(self):
        init_bn(self.bn1)
        init_layer(self.conv1)
        init_layer(self.conv2)
        if self.is_shortcut:
            init_layer(self.shortcut)

    def forward(self, x):
        origin = x
        x = self.conv1(F.leaky_relu_(self.bn1(x), negative_slope=0.01))
        x = self.conv2(F.leaky_relu_(self.bn2(x), negative_slope=0.01))

        if self.is_shortcut:
            return self.shortcut(origin) + x
        else:
            return origin + x


class EncoderBlockRes(nn.Module):
    """Encoder block with residual connections."""
    
    def __init__(self, in_channels, out_channels, downsample, activation, momentum):
        super(EncoderBlockRes, self).__init__()
        size = 3

        self.conv_block1 = ConvBlockRes(
            in_channels, out_channels, size, activation, momentum
        )
        self.conv_block2 = ConvBlockRes(
            out_channels, out_channels, size, activation, momentum
        )
        self.conv_block3 = ConvBlockRes(
            out_channels, out_channels, size, activation, momentum
        )
        self.conv_block4 = ConvBlockRes(
            out_channels, out_channels, size, activation, momentum
        )
        self.downsample = downsample

    def forward(self, x):
        encoder = self.conv_block1(x)
        encoder = self.conv_block2(encoder)
        encoder = self.conv_block3(encoder)
        encoder = self.conv_block4(encoder)
        encoder_pool = F.avg_pool2d(encoder, kernel_size=self.downsample)
        return encoder_pool, encoder


class DecoderBlockRes(nn.Module):
    """Decoder block with residual connections and dimension management."""
    
    def __init__(self, in_channels, out_channels, stride, activation, momentum):
        super(DecoderBlockRes, self).__init__()
        size = 3
        self.activation = activation

        self.conv1 = torch.nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(size, size),
            stride=stride,
            padding=(0, 0),
            output_padding=(0, 0),
            bias=False,
            dilation=(1, 1),
        )

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv_block2 = ConvBlockRes(
            out_channels * 2, out_channels, size, activation, momentum
        )
        self.conv_block3 = ConvBlockRes(
            out_channels, out_channels, size, activation, momentum
        )
        self.conv_block4 = ConvBlockRes(
            out_channels, out_channels, size, activation, momentum
        )
        self.conv_block5 = ConvBlockRes(
            out_channels, out_channels, size, activation, momentum
        )

        self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)

    def prune(self, x, both=False):
        """Prune the shape of x after transpose convolution."""
        if both:
            x = x[:, :, 0:-1, 0:-1]
        else:
            x = x[:, :, 0:-1, :]
        return x

    def forward(self, input_tensor, concat_tensor, both=False):
        x = self.conv1(F.relu_(self.bn1(input_tensor)))
        x = self.prune(x, both=both)
        x = torch.cat((x, concat_tensor), dim=1)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        return x


class UNetResComplex_100Mb(nn.Module):
    """
    Original VoiceFixer UNet architecture with proper dimension handling.
    """
    
    def __init__(self, channels, nsrc=1):
        super(UNetResComplex_100Mb, self).__init__()
        activation = "relu"
        momentum = 0.01

        self.nsrc = nsrc
        self.channels = channels
        self.downsample_ratio = 2**6  # This number equals 2^{#encoder_blocks}

        self.encoder_block1 = EncoderBlockRes(
            in_channels=channels * nsrc,
            out_channels=32,
            downsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.encoder_block2 = EncoderBlockRes(
            in_channels=32,
            out_channels=64,
            downsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.encoder_block3 = EncoderBlockRes(
            in_channels=64,
            out_channels=128,
            downsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.encoder_block4 = EncoderBlockRes(
            in_channels=128,
            out_channels=256,
            downsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.encoder_block5 = EncoderBlockRes(
            in_channels=256,
            out_channels=384,
            downsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.encoder_block6 = EncoderBlockRes(
            in_channels=384,
            out_channels=384,
            downsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.conv_block7 = ConvBlockRes(
            in_channels=384,
            out_channels=384,
            size=3,
            activation=activation,
            momentum=momentum,
        )
        self.decoder_block1 = DecoderBlockRes(
            in_channels=384,
            out_channels=384,
            stride=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.decoder_block2 = DecoderBlockRes(
            in_channels=384,
            out_channels=384,
            stride=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.decoder_block3 = DecoderBlockRes(
            in_channels=384,
            out_channels=256,
            stride=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.decoder_block4 = DecoderBlockRes(
            in_channels=256,
            out_channels=128,
            stride=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.decoder_block5 = DecoderBlockRes(
            in_channels=128,
            out_channels=64,
            stride=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.decoder_block6 = DecoderBlockRes(
            in_channels=64,
            out_channels=32,
            stride=(2, 2),
            activation=activation,
            momentum=momentum,
        )

        self.after_conv_block1 = ConvBlockRes(
            in_channels=32,
            out_channels=32,
            size=3,
            activation=activation,
            momentum=momentum,
        )

        self.after_conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=1,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True,
        )

        self.init_weights()

    def init_weights(self):
        init_layer(self.after_conv2)

    def forward(self, sp):
        """
        Forward pass with proper dimension management.
        
        Args:
            sp: Input spectrogram [batch, channels, time, freq]
        Returns:
            output_dict: {"mel": enhanced_spectrogram}
        """
        # Batch normalization
        x = sp

        # Pad spectrogram to be evenly divided by downsample ratio.
        origin_len = x.shape[2]  # time_steps
        pad_len = (
            int(np.ceil(x.shape[2] / self.downsample_ratio)) * self.downsample_ratio
            - origin_len
        )
        x = F.pad(x, pad=(0, 0, 0, pad_len))
        x = x[..., 0 : x.shape[-1] - 1]  # (bs, channels, T, F)

        # UNet encoder
        (x1_pool, x1) = self.encoder_block1(x)  # x1_pool: (bs, 32, T / 2, F / 2)
        (x2_pool, x2) = self.encoder_block2(x1_pool)  # x2_pool: (bs, 64, T / 4, F / 4)
        (x3_pool, x3) = self.encoder_block3(x2_pool)  # x3_pool: (bs, 128, T / 8, F / 8)
        (x4_pool, x4) = self.encoder_block4(x3_pool)  # x4_pool: (bs, 256, T / 16, F / 16)
        (x5_pool, x5) = self.encoder_block5(x4_pool)  # x5_pool: (bs, 384, T / 32, F / 32)
        (x6_pool, x6) = self.encoder_block6(x5_pool)  # x6_pool: (bs, 384, T / 64, F / 64)
        
        # Bottleneck
        x_center = self.conv_block7(x6_pool)  # (bs, 384, T / 64, F / 64)
        
        # UNet decoder with skip connections
        x7 = self.decoder_block1(x_center, x6)  # (bs, 384, T / 32, F / 32)
        x8 = self.decoder_block2(x7, x5)  # (bs, 384, T / 16, F / 16)
        x9 = self.decoder_block3(x8, x4)  # (bs, 256, T / 8, F / 8)
        x10 = self.decoder_block4(x9, x3)  # (bs, 128, T / 4, F / 4)
        x11 = self.decoder_block5(x10, x2)  # (bs, 64, T / 2, F / 2)
        x12 = self.decoder_block6(x11, x1)  # (bs, 32, T, F)
        
        # Final processing
        x = self.after_conv_block1(x12)  # (bs, 32, T, F)
        x = self.after_conv2(x)  # (bs, 1, T, F)

        # Recover original shape
        x = F.pad(x, pad=(0, 1))
        x = x[:, :, 0:origin_len, :]

        output_dict = {"mel": x}
        return output_dict


class UpsampleNet(nn.Module):
    """Original VoiceFixer upsampling network with skip connections and smoothing."""
    
    def __init__(self, input_size, output_size, upsample_factor, hp=None, index=0):
        super(UpsampleNet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.upsample_factor = upsample_factor
        self.skip_conv = nn.Conv1d(input_size, output_size, kernel_size=1)
        self.index = index
        
        # Configuration parameters (from original VoiceFixer Config)
        self.up_type = "transpose"  # Default upsampling type
        self.use_smooth = False     # Smoothing disabled by default
        self.use_drop = False       # Dropout disabled by default
        self.no_skip = False        # Skip connections enabled
        self.org = False            # Original mode disabled
        
        # Main upsampling layer
        layer = nn.ConvTranspose1d(
            input_size,
            output_size,
            upsample_factor * 2,
            upsample_factor,
            padding=upsample_factor // 2 + upsample_factor % 2,
            output_padding=upsample_factor % 2,
        )
        self.layer = nn.utils.parametrizations.weight_norm(layer)

    def forward(self, inputs):
        if not self.org:
            inputs = inputs + torch.sin(inputs)
            B, C, T = inputs.size()
            res = inputs.repeat(1, self.upsample_factor, 1).view(B, C, -1)
            skip = self.skip_conv(res)

        outputs = self.layer(inputs)

        if self.no_skip:
            return outputs

        if not self.org:
            outputs = outputs + skip

        if self.use_drop:
            outputs = F.dropout(outputs, p=0.05)

        return outputs


class ResStack(nn.Module):
    """Original VoiceFixer residual stack with dilated convolutions."""
    
    def __init__(self, channel, kernel_size=3, resstack_depth=4, hp=None):
        super(ResStack, self).__init__()
        
        self.use_wn = False          # Weight normalization disabled by default
        self.use_shift_scale = False # Shift-scale disabled by default
        self.channel = channel

        def get_padding(kernel_size, dilation=1):
            return int((kernel_size * dilation - dilation) / 2)

        # Create residual layers with dilated convolutions
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.LeakyReLU(),
                nn.utils.parametrizations.weight_norm(
                    nn.Conv1d(
                        channel,
                        channel,
                        kernel_size=kernel_size,
                        dilation=3 ** (i % 10),
                        padding=get_padding(kernel_size, 3 ** (i % 10)),
                    )
                ),
                nn.LeakyReLU(),
                nn.utils.parametrizations.weight_norm(
                    nn.Conv1d(
                        channel,
                        channel,
                        kernel_size=kernel_size,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
            )
            for i in range(resstack_depth)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x


class PQMF(nn.Module):
    """Pseudo Quadrature Mirror Filter for multi-band processing."""
    
    def __init__(self, subbands=4, taps=64, cutoff_ratio=0.142, beta=9.0):
        super(PQMF, self).__init__()
        self.subbands = subbands
        self.taps = taps
        self.cutoff_ratio = cutoff_ratio
        self.beta = beta

        # Create analysis and synthesis filters
        QMF = torch.tensor(self._get_qmf_filter(), dtype=torch.float32)
        
        # Analysis filter bank
        self.register_buffer("analysis_filter", QMF)
        
        # Synthesis filter bank  
        self.register_buffer("synthesis_filter", QMF)

    def _get_qmf_filter(self):
        """Generate QMF filter coefficients."""
        from scipy.signal import kaiser
        import numpy as np
        
        # Generate prototype filter
        h = kaiser(self.taps + 1, self.beta)
        h = h / np.sum(h)
        
        # Create QMF filter bank
        filters = []
        for k in range(self.subbands):
            # Modulate prototype filter
            n = np.arange(self.taps + 1)
            h_k = h * np.cos((2 * k + 1) * np.pi * n / (2 * self.subbands) + (-1)**k * np.pi / 4)
            filters.append(h_k)
        
        return np.array(filters)

    def analysis(self, x):
        """Analysis filter bank - split into subbands."""
        # x: [batch, 1, time]
        batch_size = x.size(0)
        
        # Apply analysis filters
        subbands = []
        for k in range(self.subbands):
            # Convolve with k-th analysis filter
            h_k = self.analysis_filter[k:k+1].unsqueeze(0)  # [1, 1, taps]
            y_k = F.conv1d(x, h_k, padding=self.taps//2)
            
            # Downsample by subbands factor
            y_k = y_k[:, :, ::self.subbands]
            subbands.append(y_k)
        
        # Stack subbands: [batch, subbands, time//subbands]
        return torch.stack(subbands, dim=1)

    def synthesis(self, x):
        """Synthesis filter bank - reconstruct from subbands."""
        # x: [batch, subbands, time//subbands]
        batch_size, subbands, subband_length = x.size()
        
        # Upsample and filter each subband
        y = torch.zeros(batch_size, 1, subband_length * self.subbands, 
                       device=x.device, dtype=x.dtype)
        
        for k in range(self.subbands):
            # Upsample by inserting zeros
            x_k = x[:, k:k+1, :]  # [batch, 1, time//subbands]
            x_up = torch.zeros(batch_size, 1, subband_length * self.subbands,
                              device=x.device, dtype=x.dtype)
            x_up[:, :, ::self.subbands] = x_k
            
            # Apply synthesis filter
            h_k = self.synthesis_filter[k:k+1].unsqueeze(0)  # [1, 1, taps]
            y_k = F.conv1d(x_up, h_k, padding=self.taps//2)
            
            y += y_k
        
        return y

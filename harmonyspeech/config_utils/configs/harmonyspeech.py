from typing import List, Dict

from harmonyspeech.modeling.models.harmonyspeech.synthesizer.symbols import symbols


class HarmonySpeechEncoderConfig:

    def __init__(
        self,
        mel_n_channels: int = 40,
        model_hidden_size: int = 768,
        model_num_layers: int = 3,
        model_embedding_size: int = 768
    ):
        self.mel_n_channels = mel_n_channels
        self.model_hidden_size = model_hidden_size
        self.model_num_layers = model_num_layers
        self.model_embedding_size = model_embedding_size


class HarmonySpeechSynthesizerConfig:

    def __init__(
        self,
        embed_dims: int = 256,
        series_embed_dims: int = 64,
        num_chars: int = len(symbols),
        durpred_conv_dims: int = 256,
        durpred_rnn_dims: int = 64,
        durpred_dropout: float = 0.5,
        pitch_conv_dims: int = 256,
        pitch_rnn_dims: int = 128,
        pitch_dropout: float = 0.5,
        pitch_strength: float = 1.,
        energy_conv_dims: int = 256,
        energy_rnn_dims: int = 64,
        energy_dropout: float = 0.5,
        energy_strength: float = 1.,
        rnn_dims: int = 512,
        prenet_dims: int = 256,
        prenet_k: int = 16,
        postnet_num_highways: int = 4,
        prenet_dropout: float = 0.5,
        postnet_dims: int = 256,
        postnet_k: int = 8,
        prenet_num_highways: int = 4,
        postnet_dropout: float = 0.5,
        n_mels: int = 80,
        speaker_embed_dims: int = 768,
        padding_value: float = -11.5129
    ):
        self.embed_dims = embed_dims
        self.series_embed_dims = series_embed_dims
        self.num_chars = num_chars
        self.durpred_conv_dims = durpred_conv_dims
        self.durpred_rnn_dims = durpred_rnn_dims
        self.durpred_dropout = durpred_dropout
        self.pitch_conv_dims = pitch_conv_dims
        self.pitch_rnn_dims = pitch_rnn_dims
        self.pitch_dropout = pitch_dropout
        self.pitch_strength = pitch_strength
        self.energy_conv_dims = energy_conv_dims
        self.energy_rnn_dims = energy_rnn_dims
        self.energy_dropout = energy_dropout
        self.energy_strength = energy_strength
        self.rnn_dims = rnn_dims
        self.prenet_dims = prenet_dims
        self.prenet_k = prenet_k
        self.postnet_num_highways = postnet_num_highways
        self.prenet_dropout = prenet_dropout
        self.postnet_dims = postnet_dims
        self.postnet_k = postnet_k
        self.prenet_num_highways = prenet_num_highways
        self.postnet_dropout = postnet_dropout
        self.n_mels = n_mels
        self.speaker_embed_dims = speaker_embed_dims
        self.padding_value = padding_value


class HarmonySpeechVocoderConfig:

    def __init__(
        self,
        in_channels: int = 80,
        out_channels: int = 1,
        kernel_size: int = 7,
        channels: int = 512,
        bias: bool = True,
        upsample_scales: List[int] = [8, 8, 2, 2],
        stack_kernel_size: int = 3,
        stacks: int = 3,
        nonlinear_activation: str = "LeakyReLU",
        nonlinear_activation_params: Dict[str, float] = {"negative_slope": 0.2},
        pad: str = "ReflectionPad1d",
        pad_params: Dict = {},
        use_final_nonlinear_activation: bool = True,
        use_weight_norm: bool = True,
        use_causal_conv: bool = False,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.channels = channels
        self.bias = bias
        self.upsample_scales = upsample_scales
        self.stack_kernel_size = stack_kernel_size
        self.stacks = stacks
        self.nonlinear_activation = nonlinear_activation
        self.nonlinear_activation_params = nonlinear_activation_params
        self.pad = pad
        self.pad_params = pad_params
        self.use_final_nonlinear_activation = use_final_nonlinear_activation
        self.use_weight_norm = use_weight_norm
        self.use_causal_conv = use_causal_conv


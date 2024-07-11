import logging
from typing import Callable, Optional

from torch.nn import Embedding
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch import nn
import torch.nn.functional as F
import numpy as np
import torch

from harmonyspeech.modeling.hf_downloader import hf_model_weights_iterator, default_weight_loader
from harmonyspeech.modeling.models.harmonyspeech.vocoder.parallel_wavegan.layers.causal_conv import (
    CausalConv1d,
    CausalConvTranspose1d
)
from harmonyspeech.modeling.models.harmonyspeech.vocoder.parallel_wavegan.layers.residual_stack import ResidualStack
from harmonyspeech.modeling.models.harmonyspeech.vocoder.parallel_wavegan.utils.utils import read_hdf5


class LengthRegulator(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, dur: torch.Tensor) -> torch.Tensor:
        dur[dur < 0] = 0.
        x_expanded = []
        for i in range(x.size(0)):
            x_exp = torch.repeat_interleave(x[i], (dur[i] + 0.5).long(), dim=0)
            x_expanded.append(x_exp)
        x_expanded = pad_sequence(x_expanded, padding_value=0., batch_first=True)
        return x_expanded


class HighwayNetwork(nn.Module):

    def __init__(self, size: int) -> None:
        super().__init__()
        self.W1 = nn.Linear(size, size)
        self.W2 = nn.Linear(size, size)
        self.W1.bias.data.fill_(0.)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.W1(x)
        x2 = self.W2(x)
        g = torch.sigmoid(x2)
        y = g * F.relu(x1) + (1. - g) * x
        return y


class BatchNormConv(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel: int, relu=True) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel, stride=1, padding=kernel // 2, bias=False)
        self.bnorm = nn.BatchNorm1d(out_channels)
        self.relu = relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = F.relu(x) if self.relu is True else x
        return self.bnorm(x)


class CBHG(nn.Module):

    def __init__(self,
                 K: int,
                 in_channels: int,
                 channels: int,
                 proj_channels: list,
                 num_highways: int,
                 dropout: float = 0.5) -> None:
        super().__init__()

        self.dropout = dropout
        self.bank_kernels = [i for i in range(1, K + 1)]
        self.conv1d_bank = nn.ModuleList()
        for k in self.bank_kernels:
            conv = BatchNormConv(in_channels, channels, k)
            self.conv1d_bank.append(conv)

        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)

        self.conv_project1 = BatchNormConv(len(self.bank_kernels) * channels, proj_channels[0], 3)
        self.conv_project2 = BatchNormConv(proj_channels[0], proj_channels[1], 3, relu=False)

        self.pre_highway = nn.Linear(proj_channels[-1], channels, bias=False)
        self.highways = nn.ModuleList()
        for i in range(num_highways):
            hn = HighwayNetwork(channels)
            self.highways.append(hn)

        self.rnn = nn.GRU(channels, channels, batch_first=True, bidirectional=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        seq_len = x.size(-1)
        conv_bank = []

        # Convolution Bank
        for conv in self.conv1d_bank:
            c = conv(x) # Convolution
            conv_bank.append(c[:, :, :seq_len])

        # Stack along the channel axis
        conv_bank = torch.cat(conv_bank, dim=1)

        # dump the last padding to fit residual
        x = self.maxpool(conv_bank)[:, :, :seq_len]
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Conv1d projections
        x = self.conv_project1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv_project2(x)

        # Residual Connect
        x = x + residual

        # Through the highways
        x = x.transpose(1, 2)
        x = self.pre_highway(x)
        for h in self.highways:
            x = h(x)

        # And then the RNN
        x, _ = self.rnn(x)
        return x


class SpeakerEncoder(nn.Module):
    def __init__(
        self,
        mel_n_channels: int,
        model_hidden_size: int,
        model_num_layers: int,
        model_embedding_size: int
    ):
        super().__init__()

        # Network defition
        self.lstm = nn.LSTM(input_size=mel_n_channels,
                            hidden_size=model_hidden_size,
                            num_layers=model_num_layers,
                            batch_first=True)
        self.linear = nn.Linear(in_features=model_hidden_size,
                                out_features=model_embedding_size)
        self.relu = torch.nn.ReLU()

        # Cosine similarity scaling (with fixed initial parameter values)
        self.similarity_weight = nn.Parameter(torch.tensor([10.]))
        self.similarity_bias = nn.Parameter(torch.tensor([-5.]))

        # Loss
        self.loss_fn = nn.CrossEntropyLoss()

    def do_gradient_ops(self):
        # Gradient scale
        self.similarity_weight.grad *= 0.01
        self.similarity_bias.grad *= 0.01

        # Gradient clipping
        clip_grad_norm_(self.parameters(), 3, norm_type=2)

    def forward(self, utterances, hidden_init=None):
        """
        Computes the embeddings of a batch of utterance spectrograms.

        :param utterances: batch of mel-scale filterbanks of same duration as a tensor of shape
        (batch_size, n_frames, n_channels)
        :param hidden_init: initial hidden state of the LSTM as a tensor of shape (num_layers,
        batch_size, hidden_size). Will default to a tensor of zeros if None.
        :return: the embeddings as a tensor of shape (batch_size, embedding_size)
        """
        # Pass the input through the LSTM layers and retrieve all outputs, the final hidden state
        # and the final cell state.
        out, (hidden, cell) = self.lstm(utterances, hidden_init)

        # We take only the hidden state of the last layer
        embeds_raw = self.relu(self.linear(hidden[-1]))

        # L2-normalize it
        embeds = embeds_raw / torch.norm(embeds_raw, dim=1, keepdim=True)

        return embeds
    
    def load_weights(
        self,
        model_name_or_path: str,
        cache_dir: Optional[str] = None,
        load_format: str = "auto",
        revision: Optional[str] = None,
    ):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path, cache_dir, load_format, revision):
            if "rotary_emb.inv_freq" in name:
                continue
            if ("rotary_emb.cos_cached" in name
                    or "rotary_emb.sin_cached" in name):
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                try:
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
                except KeyError as e:
                    logging.warning(f"Param {name} not found in params_dict, ignoring and attempting to continue")


class SeriesPredictor(nn.Module):

    def __init__(self, num_chars, emb_dim=64, spk_emb_dims=768, conv_dims=256, rnn_dims=64, dropout=0.5):
        super().__init__()
        self.embedding = Embedding(num_chars, emb_dim)
        self.convs = torch.nn.ModuleList([
            BatchNormConv(emb_dim+spk_emb_dims, conv_dims, 5, relu=True),
            BatchNormConv(conv_dims, conv_dims, 5, relu=True),
            BatchNormConv(conv_dims, conv_dims, 5, relu=True),
        ])
        self.rnn = nn.GRU(conv_dims, rnn_dims, batch_first=True, bidirectional=True)
        self.lin = nn.Linear(2 * rnn_dims, 1)
        self.dropout = dropout

    def forward(self,
                x: torch.Tensor,
                spk_emb: torch.Tensor,
                alpha: float = 1.0) -> torch.Tensor:
        x = self.embedding(x)
        speaker_embedding = spk_emb[:, None, :]
        speaker_embedding = speaker_embedding.repeat(1, x.shape[1], 1)
        x = torch.cat([x, speaker_embedding], dim=2)
        x = x.transpose(1, 2)
        for conv in self.convs:
            x = conv(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.transpose(1, 2)
        x, _ = self.rnn(x)
        x = self.lin(x)
        return x / alpha


class ForwardTacotron(nn.Module):

    def __init__(self,
                 embed_dims: int,
                 series_embed_dims: int,
                 num_chars: int,
                 durpred_conv_dims: int,
                 durpred_rnn_dims: int,
                 durpred_dropout: float,
                 pitch_conv_dims: int,
                 pitch_rnn_dims: int,
                 pitch_dropout: float,
                 pitch_strength: float,
                 energy_conv_dims: int,
                 energy_rnn_dims: int,
                 energy_dropout: float,
                 energy_strength: float,
                 rnn_dims: int,
                 prenet_dims: int,
                 prenet_k: int,
                 postnet_num_highways: int,
                 prenet_dropout: float,
                 postnet_dims: int,
                 postnet_k: int,
                 prenet_num_highways: int,
                 postnet_dropout: float,
                 n_mels: int,
                 speaker_embed_dims: int,
                 padding_value=-11.5129):
        super().__init__()
        self.rnn_dims = rnn_dims
        self.padding_value = padding_value
        self.embedding = nn.Embedding(num_chars, embed_dims)
        self.lr = LengthRegulator()
        self.dur_pred = SeriesPredictor(num_chars=num_chars,
                                        emb_dim=series_embed_dims,
                                        spk_emb_dims=speaker_embed_dims,
                                        conv_dims=durpred_conv_dims,
                                        rnn_dims=durpred_rnn_dims,
                                        dropout=durpred_dropout)
        self.pitch_pred = SeriesPredictor(num_chars=num_chars,
                                          emb_dim=series_embed_dims,
                                          spk_emb_dims=speaker_embed_dims,
                                          conv_dims=pitch_conv_dims,
                                          rnn_dims=pitch_rnn_dims,
                                          dropout=pitch_dropout)
        self.energy_pred = SeriesPredictor(num_chars=num_chars,
                                           emb_dim=series_embed_dims,
                                           spk_emb_dims=speaker_embed_dims,
                                           conv_dims=energy_conv_dims,
                                           rnn_dims=energy_rnn_dims,
                                           dropout=energy_dropout)
        self.prenet = CBHG(K=prenet_k,
                           in_channels=embed_dims,
                           channels=prenet_dims,
                           proj_channels=[prenet_dims, embed_dims],
                           num_highways=prenet_num_highways,
                           dropout=prenet_dropout)
        self.lstm = nn.LSTM(2 * prenet_dims + speaker_embed_dims,
                            rnn_dims,
                            batch_first=True,
                            bidirectional=True)
        self.lin = torch.nn.Linear(2 * rnn_dims, n_mels)
        self.register_buffer('step', torch.zeros(1, dtype=torch.long))
        self.postnet = CBHG(K=postnet_k,
                            in_channels=n_mels,
                            channels=postnet_dims,
                            proj_channels=[postnet_dims, n_mels],
                            num_highways=postnet_num_highways,
                            dropout=postnet_dropout)
        self.post_proj = nn.Linear(2 * postnet_dims, n_mels, bias=False)
        self.pitch_strength = pitch_strength
        self.energy_strength = energy_strength
        self.pitch_proj = nn.Conv1d(1, 2 * prenet_dims, kernel_size=3, padding=1)
        self.energy_proj = nn.Conv1d(1, 2 * prenet_dims, kernel_size=3, padding=1)

        self.init_model()
        self.num_params()

        self.register_buffer("step", torch.zeros(1, dtype=torch.long))

    def __repr__(self):
        num_params = sum([np.prod(p.size()) for p in self.parameters()])
        return f'ForwardTacotron, num params: {num_params}'

    def forward(self, x, mel, dur, spk_emb, mel_lens, pitch, energy):
        pitch = pitch.unsqueeze(1)
        energy = energy.unsqueeze(1)

        if self.training:
            self.step += 1

        dur_hat = self.dur_pred(x, spk_emb).squeeze(-1)
        pitch_hat = self.pitch_pred(x, spk_emb).transpose(1, 2)
        energy_hat = self.energy_pred(x, spk_emb).transpose(1, 2)

        x = self.embedding(x)
        # speaker_embedding = spk_emb[:, None, :]
        # speaker_embedding = speaker_embedding.repeat(1, x.shape[1], 1)
        # x = torch.cat([x, speaker_embedding], dim=2)
        x = x.transpose(1, 2)
        x = self.prenet(x)

        pitch_proj = self.pitch_proj(pitch)
        pitch_proj = pitch_proj.transpose(1, 2)
        x = x + pitch_proj * self.pitch_strength

        energy_proj = self.energy_proj(energy)
        energy_proj = energy_proj.transpose(1, 2)
        x = x + energy_proj * self.energy_strength

        x = self.lr(x, dur)

        speaker_embedding = spk_emb[:, None, :]
        speaker_embedding = speaker_embedding.repeat(1, x.shape[1], 1)
        x = torch.cat([x, speaker_embedding], dim=2)

        x = pack_padded_sequence(x, lengths=mel_lens.cpu(), enforce_sorted=False,
                                 batch_first=True)

        x, _ = self.lstm(x)

        x, _ = pad_packed_sequence(x, padding_value=self.padding_value, batch_first=True)

        x = self.lin(x)
        x = x.transpose(1, 2)

        x_post = self.postnet(x)
        x_post = self.post_proj(x_post)
        x_post = x_post.transpose(1, 2)

        x_post = self._pad(x_post, mel.size(2))
        x = self._pad(x, mel.size(2))

        return x, x_post, dur_hat, pitch_hat, energy_hat
        #return {'mel': x, 'mel_post': x_post,
        #        'dur': dur_hat, 'pitch': pitch_hat, 'energy': energy_hat}

    def generate(self,
                 x: torch.Tensor,
                 spk_emb: torch.Tensor,
                 alpha=1.0,
                 pitch_function: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
                 energy_function: Callable[[torch.Tensor], torch.Tensor] = lambda x: x):
        self.eval()
        with torch.no_grad():
            dur_hat = self.dur_pred(x, spk_emb, alpha=alpha)
            dur_hat = dur_hat.squeeze(2)
            if torch.sum(dur_hat.long()) <= 0:
                torch.fill_(dur_hat, value=2.)
            pitch_hat = self.pitch_pred(x, spk_emb).transpose(1, 2)
            pitch_hat = pitch_function(pitch_hat)
            energy_hat = self.energy_pred(x, spk_emb).transpose(1, 2)
            energy_hat = energy_function(energy_hat)
            return self._generate_mel(x=x, spk_emb=spk_emb,
                                      dur_hat=dur_hat,
                                      pitch_hat=pitch_hat,
                                      energy_hat=energy_hat)

    @torch.jit.export
    def generate_jit(self,
                     x: torch.Tensor,
                     alpha: float = 1.0,
                     beta: float = 1.0):
        with torch.no_grad():
            dur_hat = self.dur_pred(x, alpha=alpha)
            dur_hat = dur_hat.squeeze(2)
            if torch.sum(dur_hat.long()) <= 0:
                torch.fill_(dur_hat, value=2.)
            pitch_hat = self.pitch_pred(x).transpose(1, 2) * beta
            energy_hat = self.energy_pred(x).transpose(1, 2)
            return self._generate_mel(x=x, dur_hat=dur_hat,
                                      pitch_hat=pitch_hat,
                                      energy_hat=energy_hat)

    def get_step(self) -> int:
        return self.step.data.item()

    def _generate_mel(self,
                      x: torch.Tensor,
                      spk_emb: torch.Tensor,
                      dur_hat: torch.Tensor,
                      pitch_hat: torch.Tensor,
                      energy_hat: torch.Tensor):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x = self.prenet(x)

        pitch_proj = self.pitch_proj(pitch_hat)
        pitch_proj = pitch_proj.transpose(1, 2)
        x = x + pitch_proj * self.pitch_strength

        energy_proj = self.energy_proj(energy_hat)
        energy_proj = energy_proj.transpose(1, 2)
        x = x + energy_proj * self.energy_strength

        x = self.lr(x, dur_hat)

        speaker_embedding = spk_emb[:, None, :]
        speaker_embedding = speaker_embedding.repeat(1, x.shape[1], 1)
        x = torch.cat([x, speaker_embedding], dim=2)

        x, _ = self.lstm(x)

        x = self.lin(x)
        x = x.transpose(1, 2)

        x_post = self.postnet(x)
        x_post = self.post_proj(x_post)
        x_post = x_post.transpose(1, 2)

        return x, x_post, dur_hat, pitch_hat, energy_hat
        #return {'mel': x, 'mel_post': x_post, 'dur': dur_hat,
        #        'pitch': pitch_hat, 'energy': energy_hat}

    def _pad(self, x: torch.Tensor, max_len: int) -> torch.Tensor:
        x = x[:, :, :max_len]
        x = F.pad(x, [0, max_len - x.size(2), 0, 0], 'constant', self.padding_value)
        return x

    def load(self, path, optimizer=None, checkpoint=None):
        # Use device of model params as location for loaded state
        if checkpoint is None:
            device = next(self.parameters()).device
            checkpoint = torch.load(str(path), map_location=device)

        # Load weights
        self.load_state_dict(checkpoint["model_state"])
        if "optimizer_state" in checkpoint and optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state"])

    def save(self, path, optimizer=None):
        if optimizer is not None:
            torch.save({
                "model_state": self.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }, str(path))
        else:
            torch.save({
                "model_state": self.state_dict(),
            }, str(path))

    def init_model(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def num_params(self, print_out=True):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        if print_out:
            print("Trainable Parameters: %.3fM" % parameters)
        return parameters

    # @classmethod
    # def from_config(cls, config: Dict[str, Any]) -> 'ForwardTacotron':
    #     model_config = config['forward_tacotron']['model']
    #     model_config['num_chars'] = len(symbols)
    #     model_config['n_mels'] = config['dsp']['num_mels']
    #     return ForwardTacotron(**model_config)
    #
    # @classmethod
    # def from_checkpoint(cls, path: Union[Path, str]) -> 'ForwardTacotron':
    #     checkpoint = torch.load(path, map_location=torch.device('cpu'))
    #     model = ForwardTacotron.from_config(checkpoint['config'])
    #     model.load_state_dict(checkpoint['model'])
    #     return model


class MelGANGenerator(torch.nn.Module):
    """MelGAN generator module."""

    def __init__(
        self,
        in_channels=80,
        out_channels=1,
        kernel_size=7,
        channels=512,
        bias=True,
        upsample_scales=[8, 8, 2, 2],
        stack_kernel_size=3,
        stacks=3,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.2},
        pad="ReflectionPad1d",
        pad_params={},
        use_final_nonlinear_activation=True,
        use_weight_norm=True,
        use_causal_conv=False,
    ):
        """Initialize MelGANGenerator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Kernel size of initial and final conv layer.
            channels (int): Initial number of channels for conv layer.
            bias (bool): Whether to add bias parameter in convolution layers.
            upsample_scales (list): List of upsampling scales.
            stack_kernel_size (int): Kernel size of dilated conv layers in residual stack.
            stacks (int): Number of stacks in a single residual stack.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
            pad (str): Padding function module name before dilated convolution layer.
            pad_params (dict): Hyperparameters for padding function.
            use_final_nonlinear_activation (torch.nn.Module): Activation function for the final layer.
            use_weight_norm (bool): Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.
            use_causal_conv (bool): Whether to use causal convolution.

        """
        super(MelGANGenerator, self).__init__()

        # check hyper parameters is valid
        assert channels >= np.prod(upsample_scales)
        assert channels % (2 ** len(upsample_scales)) == 0
        if not use_causal_conv:
            assert (kernel_size - 1) % 2 == 0, "Not support even number kernel size."

        # add initial layer
        layers = []
        if not use_causal_conv:
            layers += [
                getattr(torch.nn, pad)((kernel_size - 1) // 2, **pad_params),
                torch.nn.Conv1d(in_channels, channels, kernel_size, bias=bias),
            ]
        else:
            layers += [
                CausalConv1d(
                    in_channels,
                    channels,
                    kernel_size,
                    bias=bias,
                    pad=pad,
                    pad_params=pad_params,
                ),
            ]

        for i, upsample_scale in enumerate(upsample_scales):
            # add upsampling layer
            layers += [
                getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params)
            ]
            if not use_causal_conv:
                layers += [
                    torch.nn.ConvTranspose1d(
                        channels // (2**i),
                        channels // (2 ** (i + 1)),
                        upsample_scale * 2,
                        stride=upsample_scale,
                        padding=upsample_scale // 2 + upsample_scale % 2,
                        output_padding=upsample_scale % 2,
                        bias=bias,
                    )
                ]
            else:
                layers += [
                    CausalConvTranspose1d(
                        channels // (2**i),
                        channels // (2 ** (i + 1)),
                        upsample_scale * 2,
                        stride=upsample_scale,
                        bias=bias,
                    )
                ]

            # add residual stack
            for j in range(stacks):
                layers += [
                    ResidualStack(
                        kernel_size=stack_kernel_size,
                        channels=channels // (2 ** (i + 1)),
                        dilation=stack_kernel_size**j,
                        bias=bias,
                        nonlinear_activation=nonlinear_activation,
                        nonlinear_activation_params=nonlinear_activation_params,
                        pad=pad,
                        pad_params=pad_params,
                        use_causal_conv=use_causal_conv,
                    )
                ]

        # add final layer
        layers += [
            getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params)
        ]
        if not use_causal_conv:
            layers += [
                getattr(torch.nn, pad)((kernel_size - 1) // 2, **pad_params),
                torch.nn.Conv1d(
                    channels // (2 ** (i + 1)), out_channels, kernel_size, bias=bias
                ),
            ]
        else:
            layers += [
                CausalConv1d(
                    channels // (2 ** (i + 1)),
                    out_channels,
                    kernel_size,
                    bias=bias,
                    pad=pad,
                    pad_params=pad_params,
                ),
            ]
        if use_final_nonlinear_activation:
            layers += [torch.nn.Tanh()]

        # define the model as a single function
        self.melgan = torch.nn.Sequential(*layers)

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

        # reset parameters
        self.reset_parameters()

        # initialize pqmf for inference
        self.pqmf = None

    def forward(self, c):
        """Calculate forward propagation.

        Args:
            c (Tensor): Input tensor (B, channels, T).

        Returns:
            Tensor: Output tensor (B, 1, T ** prod(upsample_scales)).

        """
        return self.melgan(c)

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                # logging.debug(f"Weight norm is removed from {m}.")
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(
                m, torch.nn.ConvTranspose1d
            ):
                torch.nn.utils.weight_norm(m)
                # logging.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def reset_parameters(self):
        """Reset parameters.

        This initialization follows official implementation manner.
        https://github.com/descriptinc/melgan-neurips/blob/master/mel2wav/modules.py

        """

        def _reset_parameters(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(
                m, torch.nn.ConvTranspose1d
            ):
                m.weight.data.normal_(0.0, 0.02)
                # logging.debug(f"Reset parameters in {m}.")

        self.apply(_reset_parameters)

    def register_stats(self, stats):
        """Register stats for de-normalization as buffer.

        Args:
            stats (str): Path of statistics file (".npy" or ".h5").

        """
        assert stats.endswith(".h5") or stats.endswith(".npy")
        if stats.endswith(".h5"):
            mean = read_hdf5(stats, "mean").reshape(-1)
            scale = read_hdf5(stats, "scale").reshape(-1)
        else:
            mean = np.load(stats)[0].reshape(-1)
            scale = np.load(stats)[1].reshape(-1)
        self.register_buffer("mean", torch.from_numpy(mean).float())
        self.register_buffer("scale", torch.from_numpy(scale).float())
        # logging.info("Successfully registered stats as buffer.")

    def inference(self, c, normalize_before=False):
        """Perform inference.

        Args:
            c (Union[Tensor, ndarray]): Input tensor (T, in_channels).
            normalize_before (bool): Whether to perform normalization.

        Returns:
            Tensor: Output tensor (T ** prod(upsample_scales), out_channels).

        """
        if not isinstance(c, torch.Tensor):
            c = torch.tensor(c, dtype=torch.float).to(next(self.parameters()).device)
        if normalize_before:
            c = (c - self.mean) / self.scale
        c = self.melgan(c.transpose(1, 0).unsqueeze(0))
        if self.pqmf is not None:
            c = self.pqmf.synthesis(c)
        return c.squeeze(0).transpose(1, 0)

    def num_params(self, print_out=True):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        if print_out:
            print("Trainable Parameters: %.3fM" % parameters)
        return parameters

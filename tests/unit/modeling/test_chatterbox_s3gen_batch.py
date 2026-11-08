"""S3Gen padded-batch gates (Phase 9-5).

No model weights: every gate runs on a tiny, randomly initialised S3Gen built
in-process from the **real upstream classes** (the 9-1/9-3 precedent) —
``CausalMaskedDiffWithXvec`` (a 512-dim ``UpsampleConformerEncoder`` + a tiny
``ConditionalDecoder``/``CausalConditionalCFM``) and a tiny ``HiFTGenerator``.

What is pinned:

* ``B=1`` with a seed is bit-identical to upstream ``S3Gen.inference``;
* mixed-length rows are equal to their solo runs (≤1e-6, the measured floor is
  ~2e-7 on this tiny graph);
* a row's seeded audio is identical alone or in a different batch composition
  (per-row ``torch.Generator`` for all four upstream RNG draws);
* padding-independence: poisoning the internal pad token/value leaves every
  output bit-identical;
* the audit's leak numbers: the encoder leaks 0.4 peak without masking, and the
  meanflow graph is refused (the documented hop skip).
"""

from __future__ import annotations

import pytest
import torch

from chatterbox.models.s3gen.configs import CFM_PARAMS
from chatterbox.models.s3gen.decoder import ConditionalDecoder
from chatterbox.models.s3gen.f0_predictor import ConvRNNF0Predictor
from chatterbox.models.s3gen.flow import CausalMaskedDiffWithXvec, make_pad_mask
from chatterbox.models.s3gen.flow_matching import CausalConditionalCFM
from chatterbox.models.s3gen.hifigan import HiFTGenerator
from chatterbox.models.s3gen.s3gen import S3Token2Wav
from chatterbox.models.s3gen.transformer.upsample_encoder import UpsampleConformerEncoder
from chatterbox.models.s3tokenizer import S3_SR, S3Tokenizer
from s3tokenizer.model_v2 import ModelConfig, S3TokenizerV2

from harmonyspeech.modeling.models.chatterbox.s3gen_batch import (
    S3GenBatchError,
    S3GenRow,
    inference_batched,
    supports_batched_s3gen,
    supports_batched_tokenizer,
    tokenize_batched,
)

_S3GEN_SR = 24000


def _build_encoder() -> UpsampleConformerEncoder:
    return UpsampleConformerEncoder(
        input_size=512,
        output_size=512,
        attention_heads=2,
        linear_units=64,
        num_blocks=1,
        dropout_rate=0.0,
        positional_dropout_rate=0.0,
        attention_dropout_rate=0.0,
        normalize_before=True,
        input_layer="linear",
        pos_enc_layer_type="rel_pos_espnet",
        selfattention_layer_type="rel_selfattn",
        use_cnn_module=False,
        macaron_style=False,
    )


def _build_flow() -> CausalMaskedDiffWithXvec:
    estimator = ConditionalDecoder(
        in_channels=320,
        out_channels=80,
        causal=True,
        channels=[32],
        dropout=0.0,
        attention_head_dim=16,
        n_blocks=1,
        num_mid_blocks=1,
        num_heads=2,
        act_fn="gelu",
        meanflow=False,
    )
    decoder = CausalConditionalCFM(spk_emb_dim=80, cfm_params=CFM_PARAMS, estimator=estimator)
    return CausalMaskedDiffWithXvec(encoder=_build_encoder(), decoder=decoder)


def _build_hift() -> HiFTGenerator:
    return HiFTGenerator(
        in_channels=80,
        base_channels=8,
        nb_harmonics=2,
        sampling_rate=_S3GEN_SR,
        upsample_rates=[2, 2],
        upsample_kernel_sizes=[4, 4],
        istft_params={"n_fft": 8, "hop_len": 2},
        resblock_kernel_sizes=[3],
        resblock_dilation_sizes=[[1, 3, 5]],
        source_resblock_kernel_sizes=[3, 3],
        source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5]],
        f0_predictor=ConvRNNF0Predictor(num_class=1, in_channels=80, cond_channels=16),
    )


class TinyS3Gen(S3Token2Wav):
    """A real ``S3Token2Wav`` graph without the (checkpoint-backed) tokenizer."""

    def __init__(self, *, meanflow: bool = False):
        torch.nn.Module.__init__(self)
        self.flow = _build_flow()
        self.mel2wav = _build_hift()
        # Short enough for the tiny waveforms (upstream's real buffer is 960).
        n_trim = 4
        trim_fade = torch.zeros(2 * n_trim)
        trim_fade[n_trim:] = (torch.cos(torch.linspace(torch.pi, 0, n_trim)) + 1) / 2
        self.register_buffer("trim_fade", trim_fade, persistent=False)
        self.estimator_dtype = "fp32"
        self.meanflow = meanflow
        self.resamplers = {}

    @property
    def device(self):
        return next(self.flow.parameters()).device

    @property
    def dtype(self):
        return next(self.flow.parameters()).dtype


def tiny_s3gen(*, seed: int = 20261107) -> TinyS3Gen:
    torch.manual_seed(seed)
    model = TinyS3Gen().eval()
    model.requires_grad_(False)
    return model


def make_row(tokens, prompt_len: int, seed: int | None, generator: torch.Generator) -> S3GenRow:
    ref_dict = {
        "prompt_token": torch.randint(0, 100, (1, prompt_len), generator=generator),
        "prompt_token_len": torch.tensor([prompt_len]),
        "prompt_feat": torch.randn(1, 2 * prompt_len, 80, generator=generator),
        "prompt_feat_len": None,
        "embedding": torch.randn(1, 192, generator=generator),
    }
    return S3GenRow(tokens=tokens, ref_dict=ref_dict, seed=seed)


def _rows(spec, seed_base: int = 500) -> list[S3GenRow]:
    generator = torch.Generator().manual_seed(7)
    rows = []
    for index, (token_len, prompt_len) in enumerate(spec):
        tokens = torch.randint(0, 100, (token_len,), generator=generator)
        rows.append(make_row(tokens, prompt_len, seed_base + index, generator))
    return rows


# ---------------------------------------------------------------------------
# Bit-identity + mixed lengths + reproducibility
# ---------------------------------------------------------------------------
def test_b1_with_a_seed_is_bit_identical_to_upstream():
    s3gen = tiny_s3gen()
    rows = _rows([(6, 4), (3, 2), (9, 3)])
    for row in rows:
        produced = inference_batched(s3gen, [row])[0]
        torch.manual_seed(row.seed)
        upstream, _ = s3gen.inference(
            speech_tokens=row.tokens.view(1, -1), ref_dict=dict(row.ref_dict)
        )
        upstream = upstream[0]
        assert produced.shape == upstream.shape
        assert torch.equal(produced, upstream), row.seed


def test_mixed_length_rows_match_their_solo_runs():
    s3gen = tiny_s3gen()
    rows = _rows([(6, 4), (3, 2), (9, 3)])
    batched = inference_batched(s3gen, rows)
    assert [waveform.numel() for waveform in batched] == [
        row._token_len * 2 * 4 * 2 for row in rows
    ]
    for row, waveform in zip(rows, batched, strict=True):
        solo = inference_batched(s3gen, [row])[0]
        assert torch.allclose(waveform, solo, atol=1e-6), row.seed


def test_a_row_is_identical_alone_or_in_a_different_batch_composition():
    s3gen = tiny_s3gen()
    rows = _rows([(6, 4), (3, 2), (9, 3)])
    reference = inference_batched(s3gen, [rows[1]])[0]
    for composition, position in (
        ([rows[1], rows[0]], 0),
        ([rows[2], rows[1], rows[0]], 1),
        ([rows[1], rows[2]], 0),
    ):
        produced = inference_batched(s3gen, composition)
        assert torch.allclose(produced[position], reference, atol=1e-6)


def test_per_row_noise_is_seeded_not_global():
    s3gen = tiny_s3gen()
    rows = _rows([(5, 3), (5, 3)])
    first = inference_batched(s3gen, rows)
    # Consume the global RNG; the seeded rows must not move.
    torch.manual_seed(999)
    torch.randn(1000)
    second = inference_batched(s3gen, rows)
    for left, right in zip(first, second, strict=True):
        assert torch.equal(left, right)


# ---------------------------------------------------------------------------
# Padding independence (the audit's poison test)
# ---------------------------------------------------------------------------
def test_poisoned_padding_never_reaches_the_output(monkeypatch):
    import harmonyspeech.modeling.models.chatterbox.s3gen_batch as module

    s3gen = tiny_s3gen()
    rows = _rows([(6, 4), (3, 2), (9, 3)])
    clean = inference_batched(s3gen, rows)
    monkeypatch.setattr(module, "PAD_TOKEN_ID", 4321)
    monkeypatch.setattr(module, "PAD_VALUE", 1e3)
    dirty = inference_batched(s3gen, rows)
    for left, right in zip(clean, dirty, strict=True):
        assert torch.equal(left, right)


# ---------------------------------------------------------------------------
# Audit bounds
# ---------------------------------------------------------------------------
def test_encoder_leaks_when_padding_is_not_masked():
    """The audit's 0.4075 peak: embed bias + lookahead read a short row's padding."""
    torch.manual_seed(11)
    encoder = _build_encoder().eval()
    encoder.requires_grad_(False)
    lengths = torch.tensor([7, 4])
    width = int(lengths.max())
    inputs = torch.randn(2, width, 512, generator=torch.Generator().manual_seed(5))
    mask = (~make_pad_mask(lengths, width)).unsqueeze(-1).float()
    inputs = inputs * mask

    with torch.no_grad():
        padded, _ = encoder(inputs, lengths)
        solo, _ = encoder(inputs[1:2, :4], lengths[1:2])
    valid = 4 * 2
    leak = (padded[1:2, :valid] - solo[:, :valid]).abs().max().item()
    assert leak > 0.1  # measured 0.4075 on this instance


def test_meanflow_graph_is_refused():
    s3gen = tiny_s3gen()
    s3gen.meanflow = True
    assert supports_batched_s3gen(s3gen) is False
    rows = _rows([(4, 2)])
    try:
        inference_batched(s3gen, rows)
    except S3GenBatchError:
        pass
    else:  # pragma: no cover - the guard must fire
        raise AssertionError("meanflow S3Gen must be refused")


def test_graph_without_mel2wav_is_refused():
    s3gen = tiny_s3gen()
    del s3gen.mel2wav
    assert supports_batched_s3gen(s3gen) is False


def test_prompt_feat_one_frame_longer_than_2x_tokens_is_accepted():
    """Real ``embed_ref`` shape: a prompt mel can be ``2·tokens + 1`` frames.

    Upstream trims the reference tokens to ``mel_frames // 2`` and warns, so the
    real VC target measured 181 frames for 90 tokens. The batched hop must accept
    it (the strict 2× check made every real VC target fall back to the loop).
    """
    s3gen = tiny_s3gen()
    generator = torch.Generator().manual_seed(7)
    row = make_row(torch.randint(0, 100, (4,), generator=generator), 3, 1, generator)
    row.ref_dict["prompt_feat"] = torch.randn(1, 2 * 3 + 1, 80, generator=generator)
    produced = inference_batched(s3gen, [row])[0]
    torch.manual_seed(row.seed)
    upstream, _ = s3gen.inference(
        speech_tokens=row.tokens.view(1, -1), ref_dict=dict(row.ref_dict)
    )
    assert produced.shape == upstream[0].shape
    assert torch.equal(produced, upstream[0])


def test_prompt_longer_than_the_encoded_row_is_refused_before_any_forward():
    s3gen = tiny_s3gen()
    generator = torch.Generator().manual_seed(7)
    row = make_row(torch.randint(0, 100, (4,), generator=generator), 3, 1, generator)
    # 3 prompt + 4 speech tokens = 7 tokens -> 14 encoded frames; a 14-frame
    # prompt leaves no generated span.
    row.ref_dict["prompt_feat"] = torch.randn(1, 2 * (3 + 4), 80, generator=generator)
    try:
        inference_batched(s3gen, [row])
    except S3GenBatchError:
        pass
    else:  # pragma: no cover
        raise AssertionError("a prompt that leaves no generated span must be refused")


# ---------------------------------------------------------------------------
# Source tokenizer padding audit (Phase 11-1)
# ---------------------------------------------------------------------------
class TinyS3Tokenizer(S3Tokenizer):
    """A real ``S3Tokenizer`` with small geometry (rotary dim ⇒ state/head = 64)."""

    def __init__(self, config: ModelConfig):
        # Mirror S3Tokenizer.__init__ but pass the config through (upstream drops
        # it for the encoder geometry).
        S3TokenizerV2.__init__(self, "speech_tokenizer_v2_25hz", config)
        self.n_fft = 400
        import librosa

        mel_filters = librosa.filters.mel(sr=S3_SR, n_fft=self.n_fft, n_mels=config.n_mels)
        self.register_buffer("_mel_filters", torch.FloatTensor(mel_filters))
        self.register_buffer("window", torch.hann_window(self.n_fft))


def tiny_tokenizer(seed: int = 20261108) -> TinyS3Tokenizer:
    torch.manual_seed(seed)
    config = ModelConfig(n_mels=128, n_audio_state=128, n_audio_head=2, n_audio_layer=2)
    model = TinyS3Tokenizer(config).eval()
    model.requires_grad_(False)
    return model


def _clip(samples: int, seed: int) -> torch.Tensor:
    """Speech-like 16 kHz clip: harmonics plus a seeded noise floor."""
    generator = torch.Generator().manual_seed(seed)
    t = torch.arange(samples, dtype=torch.float32)
    signal = 0.3 * torch.sin(2 * torch.pi * 220.0 * t / S3_SR)
    signal += 0.15 * torch.sin(2 * torch.pi * 440.0 * t / S3_SR)
    signal += 0.05 * torch.randn(samples, generator=generator)
    return signal


def _solo_tokens(tokenizer, clip: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        tokens, _ = tokenizer(clip[None, :])
    return tokens[0]


def test_tokenize_batched_b1_is_bit_identical_to_upstream():
    tokenizer = tiny_tokenizer()
    for samples, seed in ((16000, 1), (32000, 2), (9600, 3)):
        clip = _clip(samples, seed)
        upstream = _solo_tokens(tokenizer, clip)
        produced = tokenize_batched(tokenizer, [clip])[0]
        assert torch.equal(produced, upstream), (samples, seed)


def test_tokenize_batched_mixed_lengths_match_their_solo_runs():
    tokenizer = tiny_tokenizer()
    specs = ((16000, 1), (32000, 2), (9600, 3))
    clips = [_clip(samples, seed) for samples, seed in specs]
    batched = tokenize_batched(tokenizer, clips)
    assert [token.numel() for token in batched] == [25, 50, 15]
    for clip, produced in zip(clips, batched, strict=True):
        assert torch.equal(produced, _solo_tokens(tokenizer, clip))


def test_tokenize_batched_long_audio_mixed_path_matches_solo():
    """A >30 s row and a short row share one quantize (the mixed path)."""
    tokenizer = tiny_tokenizer()
    clips = [_clip(16000, 1), _clip(S3_SR * 31, 7)]
    batched = tokenize_batched(tokenizer, clips)
    assert batched[1].numel() == 775
    for clip, produced in zip(clips, batched, strict=True):
        assert torch.equal(produced, _solo_tokens(tokenizer, clip))


def test_poisoned_mel_padding_never_reaches_the_tokens(monkeypatch):
    import harmonyspeech.modeling.models.chatterbox.s3gen_batch as module

    tokenizer = tiny_tokenizer()
    clips = [_clip(16000, 1), _clip(32000, 2), _clip(9600, 3)]
    clean = tokenize_batched(tokenizer, clips)
    monkeypatch.setattr(module, "PAD_MEL_VALUE", 1e3)
    dirty = tokenize_batched(tokenizer, clips)
    for left, right in zip(clean, dirty, strict=True):
        assert torch.equal(left, right)


def test_batched_stft_leaks_at_the_row_boundary_so_the_stft_is_per_row():
    """The audit's 0.21+ mel leak: a padded batch's last valid frame drifts.

    This is why ``tokenize_batched`` runs the STFT per row and only batches the
    encoder/quantizer.
    """
    tokenizer = tiny_tokenizer()
    clips = [_clip(16000, 1), _clip(32000, 2)]
    solo_mels = []
    with torch.no_grad():
        for clip in clips:
            solo_mels.append(tokenizer.log_mel_spectrogram(clip[None, :]).squeeze(0))

    width = max(int(clip.numel()) for clip in clips)
    padded = torch.zeros(len(clips), width)
    for row, clip in enumerate(clips):
        padded[row, : clip.numel()] = clip
    with torch.no_grad():
        stft = torch.stft(
            padded, tokenizer.n_fft, 160, window=tokenizer.window, return_complex=True
        )
        magnitudes = stft[..., :-1].abs() ** 2
        mel_spec = tokenizer._mel_filters @ magnitudes
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max(dim=-1, keepdim=True).values - 8.0)
        log_spec = (log_spec + 4.0) / 4.0

    valid = 16000 // 160
    leak = (log_spec[0, :, :valid] - solo_mels[0][:, :valid]).abs().max().item()
    assert leak > 0.1  # measured 0.2133 on this instance; the long row is exact


def test_supports_batched_tokenizer_rejects_a_plain_callable():
    assert supports_batched_tokenizer(None) is False
    assert supports_batched_tokenizer(lambda audio: None) is False
    assert supports_batched_tokenizer(tiny_tokenizer()) is True


def test_tokenizer_without_the_audited_split_is_refused():
    with pytest.raises(S3GenBatchError):
        tokenize_batched(lambda audio: None, [_clip(1600, 1)])

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

import torch

from chatterbox.models.s3gen.configs import CFM_PARAMS
from chatterbox.models.s3gen.decoder import ConditionalDecoder
from chatterbox.models.s3gen.f0_predictor import ConvRNNF0Predictor
from chatterbox.models.s3gen.flow import CausalMaskedDiffWithXvec, make_pad_mask
from chatterbox.models.s3gen.flow_matching import CausalConditionalCFM
from chatterbox.models.s3gen.hifigan import HiFTGenerator
from chatterbox.models.s3gen.s3gen import S3Token2Wav
from chatterbox.models.s3gen.transformer.upsample_encoder import UpsampleConformerEncoder

from harmonyspeech.modeling.models.chatterbox.s3gen_batch import (
    S3GenBatchError,
    S3GenRow,
    inference_batched,
    supports_batched_s3gen,
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


def test_inconsistent_prompt_length_is_refused_before_any_forward():
    s3gen = tiny_s3gen()
    generator = torch.Generator().manual_seed(7)
    row = make_row(torch.randint(0, 100, (4,), generator=generator), 3, 1, generator)
    row.ref_dict["prompt_feat"] = row.ref_dict["prompt_feat"][:, :4]  # not 2 × 3
    try:
        inference_batched(s3gen, [row])
    except S3GenBatchError:
        pass
    else:  # pragma: no cover
        raise AssertionError("a mismatched prompt length must be refused")

"""Unit tests for padded-batch tone conversion (Phase 8-3).

Covers the vendored ``SynthesizerTrn.voice_conversion_batched`` entry point that
serves both OpenVoice V1 and V2 tone converters. The equivalence proof runs on a
tiny randomly-initialised instance of the real vendored class on CPU:

* with ``B == 1`` the batched entry point must be **bit-identical** to the
  single-sample one (same ops, same RNG draws);
* with conversion noise disabled (``tau=0``) a row inside a padded batch must
  equal its solo run up to batch-kernel float accumulation — this is the
  padding-leak proof, including the decoder stages the tone converter's waveform
  comes from;
* the caller trims in the sample domain with the model's own frame counts
  (``y_lengths * prod(upsample_rates)``), never with a padded-shape assumption.

Real weights are exercised by the ``batching:1-vs-N`` harness axis instead (GPU,
see docs/benchmarks/throughput.md).
"""

import pytest
import torch

from harmonyspeech.modeling.models.openvoice.openvoice import SynthesizerTrn
from harmonyspeech.task_handler.batching import trim_decoder_output

#: prod(upsample_rates) of :func:`_tiny_tone_converter`: one frame == 4 samples.
SPF = 4


def _tiny_tone_converter() -> SynthesizerTrn:
    """A converter-shaped ``SynthesizerTrn``: no speaker table (``n_speakers=0``)."""
    torch.manual_seed(0)
    return SynthesizerTrn(
        n_vocab=64,
        spec_channels=8,
        inter_channels=8,
        hidden_channels=8,
        filter_channels=8,
        n_heads=2,
        n_layers=1,
        kernel_size=3,
        p_dropout=0.0,
        resblock="1",
        resblock_kernel_sizes=[3],
        resblock_dilation_sizes=[[1, 3, 1]],
        upsample_rates=[2, 2],
        upsample_initial_channel=8,
        upsample_kernel_sizes=[4, 4],
        n_speakers=0,
        gin_channels=4,
    ).eval()


def _batch(lengths, seed: int = 3):
    """A padded ``[B, 8, T_max]`` mel batch plus per-row source/target embeddings."""
    generator = torch.Generator().manual_seed(seed)
    y = torch.randn(len(lengths), 8, max(lengths), generator=generator)
    for row, length in enumerate(lengths):
        y[row, :, length:] = 0.0
    sid_src = torch.randn(len(lengths), 4, 1, generator=generator)
    sid_tgt = torch.randn(len(lengths), 4, 1, generator=generator)
    return y, torch.tensor(lengths), sid_src, sid_tgt


# ---------------------------------------------------------------------------
# B=1 equivalence: the batched entry point must not change a single bit
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_voice_conversion_batched_b1_is_bit_identical_to_single_path():
    model = _tiny_tone_converter()
    y, lengths, sid_src, sid_tgt = _batch([19])

    with torch.no_grad():
        torch.manual_seed(11)
        single, single_mask, _ = model.voice_conversion(y, lengths, sid_src=sid_src, sid_tgt=sid_tgt, tau=0.3)
        torch.manual_seed(11)
        batched, batched_mask, _ = model.voice_conversion_batched(
            y, lengths, sid_src=sid_src, sid_tgt=sid_tgt, tau=0.3
        )

    assert torch.equal(single, batched)
    assert torch.equal(single_mask, batched_mask)


# ---------------------------------------------------------------------------
# Mixed lengths: correct per-row output, no padding leak
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_voice_conversion_batched_mixed_lengths_match_each_solo_run():
    """With noise disabled, every row (padding included) is its solo run.

    ``tau=0`` makes the posterior sample deterministic, so any difference is
    padding leaking through the encoder, the flow or the decoder — not RNG.
    """
    model = _tiny_tone_converter()
    lengths = [23, 17, 11]
    y, lengths_t, sid_src, sid_tgt = _batch(lengths)

    with torch.no_grad():
        output, mask, _ = model.voice_conversion_batched(
            y, lengths_t, sid_src=sid_src, sid_tgt=sid_tgt, tau=0.0
        )

    assert output.shape == (len(lengths), 1, max(lengths) * SPF)
    assert mask.shape == (len(lengths), 1, max(lengths))
    assert [int(value) for value in mask.sum(-1).squeeze(1)] == lengths

    for row, length in enumerate(lengths):
        with torch.no_grad():
            solo, _, _ = model.voice_conversion(
                y[row : row + 1, :, :length],
                torch.tensor([length]),
                sid_src=sid_src[row : row + 1],
                sid_tgt=sid_tgt[row : row + 1],
                tau=0.0,
            )
        assert solo.shape == (1, 1, length * SPF)
        torch.testing.assert_close(
            output[row : row + 1, :, : solo.shape[-1]], solo, rtol=1e-5, atol=1e-6
        )


@pytest.mark.unit
def test_voice_conversion_batched_unmasked_decoder_padding_would_leak():
    """Documents the trap: without per-stage decoder masking a padded row drifts.

    A solo decoder simply has no frames past a row's length; a padded batch does,
    and the transposed convs spread their conv-bias-filled padding back into the
    valid tail (and, through the resblocks, much further). The masked entry point
    removes that; this test pins the contrast so a future refactor cannot drop
    the masking silently.
    """
    model = _tiny_tone_converter()
    lengths = [23, 17, 11]
    y, lengths_t, sid_src, sid_tgt = _batch(lengths)

    with torch.no_grad():
        masked, _, _ = model.voice_conversion_batched(
            y, lengths_t, sid_src=sid_src, sid_tgt=sid_tgt, tau=0.0
        )
        unmasked, _, _ = model._voice_conversion_impl(
            y, lengths_t, sid_src, sid_tgt, tau=0.0, mask_decoder_padding=False
        )
        solo, _, _ = model.voice_conversion(
            y[2:3, :, :11], torch.tensor([11]), sid_src=sid_src[2:3], sid_tgt=sid_tgt[2:3], tau=0.0
        )

    masked_diff = (masked[2:3, :, : solo.shape[-1]] - solo).abs().max()
    unmasked_diff = (unmasked[2:3, :, : solo.shape[-1]] - solo).abs().max()
    assert float(masked_diff) < 1e-6
    assert float(unmasked_diff) > 1e-3


@pytest.mark.unit
def test_voice_conversion_batched_unpads_on_the_model_frame_axis():
    """Trimming uses ``y_lengths * prod(upsample_rates)``, the solo output length."""
    model = _tiny_tone_converter()
    lengths = [13, 7, 5]
    y, lengths_t, sid_src, sid_tgt = _batch(lengths, seed=5)

    with torch.no_grad():
        output, _, _ = model.voice_conversion_batched(
            y, lengths_t, sid_src=sid_src, sid_tgt=sid_tgt, tau=0.0
        )
    trimmed = trim_decoder_output(output, lengths_t, SPF)

    for row, length in enumerate(lengths):
        assert trimmed[row].shape == (1, length * SPF)
        assert torch.equal(trimmed[row][0], output[row, 0, : length * SPF])


# ---------------------------------------------------------------------------
# Per-sample parameters and speaker embeddings
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_voice_conversion_batched_accepts_per_sample_tau():
    """A length-B ``tau`` vector is broadcast to ``[B, 1, 1]``."""
    model = _tiny_tone_converter()
    y, lengths_t, sid_src, sid_tgt = _batch([11, 11], seed=7)

    with torch.no_grad():
        torch.manual_seed(5)
        scalar, _, _ = model.voice_conversion_batched(
            y, lengths_t, sid_src=sid_src, sid_tgt=sid_tgt, tau=0.3
        )
        torch.manual_seed(5)
        vector, _, _ = model.voice_conversion_batched(
            y, lengths_t, sid_src=sid_src, sid_tgt=sid_tgt, tau=[0.3, 0.3]
        )

    assert torch.equal(scalar, vector)


@pytest.mark.unit
def test_voice_conversion_batched_rejects_a_mismatched_tau_vector():
    model = _tiny_tone_converter()
    y, lengths_t, sid_src, sid_tgt = _batch([11, 9], seed=7)

    with torch.no_grad(), pytest.raises(ValueError, match="1 values for a batch of 2"):
        model.voice_conversion_batched(y, lengths_t, sid_src=sid_src, sid_tgt=sid_tgt, tau=[0.3])


@pytest.mark.unit
def test_voice_conversion_batched_pairs_distinct_speaker_embeddings():
    """Row ``i`` is converted with ``sid_src[i]``/``sid_tgt[i]`` (no cross-talk)."""
    model = _tiny_tone_converter()
    lengths = [17, 17]
    y, lengths_t, sid_src, sid_tgt = _batch(lengths, seed=9)

    with torch.no_grad():
        output, _, _ = model.voice_conversion_batched(
            y, lengths_t, sid_src=sid_src, sid_tgt=sid_tgt, tau=0.0
        )

    swapped_src = sid_src.flip(0)
    swapped_tgt = sid_tgt.flip(0)
    with torch.no_grad():
        swapped, _, _ = model.voice_conversion_batched(
            y, lengths_t, sid_src=swapped_src, sid_tgt=swapped_tgt, tau=0.0
        )

    first_solo_src = model.voice_conversion(
        y[0:1], torch.tensor([17]), sid_src=sid_src[0:1], sid_tgt=sid_tgt[0:1], tau=0.0
    )[0]
    torch.testing.assert_close(output[0:1], first_solo_src, rtol=1e-5, atol=1e-6)
    assert not torch.allclose(output[0:1], swapped[0:1], atol=1e-3)


@pytest.mark.unit
def test_voice_conversion_zero_g_speakerless_variant_still_batches():
    """``zero_g`` converters ignore the embeddings on both paths, identically."""
    model = _tiny_tone_converter()
    model.zero_g = True
    lengths = [13, 9]
    y, lengths_t, sid_src, sid_tgt = _batch(lengths, seed=13)

    with torch.no_grad():
        output, _, _ = model.voice_conversion_batched(
            y, lengths_t, sid_src=sid_src, sid_tgt=sid_tgt, tau=0.0
        )
        other, _, _ = model.voice_conversion_batched(
            y, lengths_t, sid_src=torch.randn_like(sid_src), sid_tgt=torch.randn_like(sid_tgt), tau=0.0
        )
        solo, _, _ = model.voice_conversion(
            y[1:2, :, :9], torch.tensor([9]), sid_src=sid_src[1:2], sid_tgt=sid_tgt[1:2], tau=0.0
        )

    assert torch.equal(output, other)
    torch.testing.assert_close(
        output[1:2, :, : solo.shape[-1]], solo, rtol=1e-5, atol=1e-6
    )

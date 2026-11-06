"""Unit tests for the batched embedder forwards (Phase 8-2).

Covers the two vendored speaker-encoder entry points the embed handlers batch:

* OpenVoice ``ReferenceEncoder`` (the tone-converter encoder used by the V1/V2
  clone pipelines) — mel frames in, one embedding per segment out;
* HarmonySpeech ``SpeakerEncoder`` — fixed-length partial utterances in, one
  embedding per partial out.

The equivalence proofs run on **tiny randomly-initialised instances of the real
vendored classes** on CPU (the Phase 5-2 pattern): with ``B == 1`` the batched
entry point must be bit-identical to the single-sample one, and a row inside a
length-mixed padded batch must match its solo run to float32 accumulation noise
(``atol=1e-6``; the observed delta is ~1e-8). Real weights are exercised by the
``batching:1-vs-N`` harness axis instead (GPU, see docs/benchmarks/throughput.md).
"""

import pytest
import torch

from harmonyspeech.modeling.models.harmonyspeech.harmonyspeech import SpeakerEncoder
from harmonyspeech.modeling.models.openvoice.openvoice import ReferenceEncoder

SPEC_CHANNELS = 8
GIN_CHANNELS = 6
MEL_CHANNELS = 6


def _reference_encoder(seed: int = 0) -> ReferenceEncoder:
    torch.manual_seed(seed)
    return ReferenceEncoder(spec_channels=SPEC_CHANNELS, gin_channels=GIN_CHANNELS).eval()


def _speaker_encoder(seed: int = 0, layers: int = 2) -> SpeakerEncoder:
    torch.manual_seed(seed)
    return SpeakerEncoder(
        mel_n_channels=MEL_CHANNELS,
        model_hidden_size=5,
        model_num_layers=layers,
        model_embedding_size=4,
    ).eval()


def _mel_rows(lengths, seed: int = 1, channels: int = SPEC_CHANNELS):
    generator = torch.Generator().manual_seed(seed)
    return [torch.randn(1, length, channels, generator=generator) for length in lengths]


def _pad_rows(rows, channels: int) -> torch.Tensor:
    padded = torch.zeros(len(rows), max(row.shape[1] for row in rows), channels)
    for index, row in enumerate(rows):
        padded[index, : row.shape[1]] = row[0]
    return padded


# ---------------------------------------------------------------------------
# OpenVoice ReferenceEncoder (OpenVoiceV1ToneConverterEncoder / V2)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("frames", [65, 100, 128])
def test_openvoice_reference_encoder_b1_is_bit_identical(frames):
    encoder = _reference_encoder()
    inputs = torch.randn(1, frames, SPEC_CHANNELS)

    with torch.no_grad():
        single = encoder(inputs)
        batched = encoder.forward_batched(inputs, torch.tensor([frames]))

    assert single.shape == (1, GIN_CHANNELS)
    assert torch.equal(batched, single)


@pytest.mark.unit
def test_openvoice_reference_encoder_b1_equals_single_with_explicit_full_length():
    """A full-length row takes the plain path, so no packing changes a value."""
    encoder = _reference_encoder()
    inputs = torch.randn(1, 96, SPEC_CHANNELS)

    with torch.no_grad():
        single = encoder(inputs)
        batched = encoder.forward_batched(inputs, [96])
        plain = encoder.forward_batched(inputs, None)

    assert torch.equal(batched, single)
    assert torch.equal(plain, single)


@pytest.mark.unit
def test_openvoice_reference_encoder_mixed_lengths_match_solo_runs():
    lengths = [128, 64, 100]
    encoder = _reference_encoder()
    rows = _mel_rows(lengths)
    padded = _pad_rows(rows, SPEC_CHANNELS)

    with torch.no_grad():
        solo = [encoder(row) for row in rows]
        batched = encoder.forward_batched(padded, torch.tensor(lengths))

    assert batched.shape == (len(lengths), GIN_CHANNELS)
    for index, expected in enumerate(solo):
        assert torch.allclose(batched[index], expected[0], atol=1e-6)


@pytest.mark.unit
def test_openvoice_reference_encoder_padding_does_not_reach_the_embedding():
    """A row's embedding must not depend on what the batch pads it with."""
    lengths = [100, 70]
    encoder = _reference_encoder()
    rows = _mel_rows(lengths)
    padded = _pad_rows(rows, SPEC_CHANNELS)
    noisy = padded.clone()
    noisy[1, lengths[1] :] = 1e3  # garbage past the short row's end

    with torch.no_grad():
        clean = encoder.forward_batched(padded, torch.tensor(lengths))
        dirty = encoder.forward_batched(noisy, torch.tensor(lengths))

    assert torch.allclose(clean, dirty, atol=1e-6)


@pytest.mark.unit
def test_openvoice_reference_encoder_rejects_more_frames_than_the_batch_holds():
    encoder = _reference_encoder()
    padded = torch.zeros(2, 64, SPEC_CHANNELS)

    with pytest.raises(ValueError, match="batch holds"):
        encoder.forward_batched(padded, torch.tensor([64, 256]))


@pytest.mark.unit
def test_openvoice_reference_encoder_global_pooling_shapes():
    """The embedding is one row per segment: [N, gin_channels]. The handler
    unsqueezes it to stack the per-segment vectors of a request."""
    encoder = _reference_encoder()
    padded = _pad_rows(_mel_rows([128, 64, 96]), SPEC_CHANNELS)

    with torch.no_grad():
        batched = encoder.forward_batched(padded, torch.tensor([128, 64, 96]))

    assert batched.shape == (3, GIN_CHANNELS)


@pytest.mark.unit
def test_openvoice_segment_stack_mean_matches_the_loop_arithmetic():
    """N segments computed solo vs inside one padded batch → same stored value.

    This mirrors the handler's aggregation: each segment is stored as
    ``[1, gin, 1]`` and the request embedding is ``torch.stack(gs).mean(0)``.
    """
    lengths = [120, 90, 151, 64]
    encoder = _reference_encoder()
    rows = _mel_rows(lengths, seed=7)
    padded = _pad_rows(rows, SPEC_CHANNELS)

    with torch.no_grad():
        loop_gs = [encoder(row).unsqueeze(-1) for row in rows]
        loop_embedding = torch.stack([g.detach() for g in loop_gs]).mean(0)

        batched = encoder.forward_batched(padded, torch.tensor(lengths))
        batch_gs = [batched[index].detach().unsqueeze(-1).unsqueeze(0) for index in range(len(lengths))]
        batch_embedding = torch.stack(batch_gs).mean(0)

    assert loop_embedding.shape == (1, GIN_CHANNELS, 1)
    # Not bitwise on purpose: the conv/GRU kernels block differently for a
    # different batch shape, so a row's floats can differ in the last bits even
    # though the arithmetic (valid frames only, same order) is the same. Measured
    # delta ≤1e-8; the stacking/mean order itself is identical.
    assert torch.allclose(batch_embedding, loop_embedding, atol=1e-6)


# ---------------------------------------------------------------------------
# HarmonySpeech SpeakerEncoder
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("frames", [20, 50, 160])
@pytest.mark.parametrize("layers", [1, 2])
def test_harmonyspeech_speaker_encoder_b1_is_bit_identical(frames, layers):
    encoder = _speaker_encoder(layers=layers)
    utterances = torch.randn(1, frames, MEL_CHANNELS)

    with torch.no_grad():
        single = encoder(utterances)
        batched = encoder.forward_batched(utterances, torch.tensor([frames]))

    assert single.shape == (1, 4)
    assert torch.equal(batched, single)


@pytest.mark.unit
def test_harmonyspeech_speaker_encoder_equal_length_batch_equals_forward():
    """The encoder's fixed-length partials take the plain batched path."""
    encoder = _speaker_encoder()
    utterances = torch.randn(4, 160, MEL_CHANNELS)

    with torch.no_grad():
        expected = encoder(utterances)
        batched = encoder.forward_batched(utterances, torch.tensor([160] * 4))

    assert torch.equal(batched, expected)


@pytest.mark.unit
def test_harmonyspeech_speaker_encoder_mixed_lengths_match_solo_runs():
    lengths = [160, 40, 111]
    encoder = _speaker_encoder()
    rows = [torch.randn(1, length, MEL_CHANNELS) for length in lengths]
    padded = _pad_rows(rows, MEL_CHANNELS)

    with torch.no_grad():
        solo = [encoder(row) for row in rows]
        batched = encoder.forward_batched(padded, torch.tensor(lengths))

    assert batched.shape == (len(lengths), 4)
    for index, expected in enumerate(solo):
        assert torch.allclose(batched[index], expected[0], atol=1e-6)


@pytest.mark.unit
def test_harmonyspeech_speaker_encoder_padding_does_not_reach_the_embedding():
    lengths = [160, 60]
    encoder = _speaker_encoder()
    rows = [torch.randn(1, length, MEL_CHANNELS) for length in lengths]
    padded = _pad_rows(rows, MEL_CHANNELS)
    noisy = padded.clone()
    noisy[1, lengths[1] :] = 1e3

    with torch.no_grad():
        clean = encoder.forward_batched(padded, torch.tensor(lengths))
        dirty = encoder.forward_batched(noisy, torch.tensor(lengths))

    assert torch.allclose(clean, dirty, atol=1e-6)


@pytest.mark.unit
def test_harmonyspeech_speaker_encoder_rejects_more_frames_than_the_batch_holds():
    encoder = _speaker_encoder()
    padded = torch.zeros(2, 40, MEL_CHANNELS)

    with pytest.raises(ValueError, match="batch holds"):
        encoder.forward_batched(padded, torch.tensor([40, 400]))

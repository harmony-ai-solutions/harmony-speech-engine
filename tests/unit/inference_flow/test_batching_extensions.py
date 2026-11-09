"""Unit tests for the axis-general padded-batch helpers (Phase 8-1).

Phase 5-2 batched the phone axis only. These helpers generalize the same
pad → forward → trim pattern to the two axes the remaining steps need: mel /
feature frames (tone converters, restorers, vocoders, the HarmonySpeech
encoder) and audio samples (speaker embedders).

Every test runs on CPU with pure torch and asserts the property the handlers
depend on: lengths are explicit, the padded batch is recoverable bitwise, and
``plan_batches`` / the budget work in whatever unit the step passes in.
"""

import pytest
import torch

from harmonyspeech.task_handler.batching import (
    DEFAULT_MEL_FRAME_BATCH_BUDGET,
    DEFAULT_PHONE_BATCH_BUDGET,
    LENGTH_SIMILARITY_RATIO,
    default_token_budget,
    pad_feature_batch,
    pad_waveform_batch,
    plan_batches,
    trim_feature_output,
)


def _features(lengths, bins: int = 3, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    return [torch.randn(bins, frames, generator=generator) for frames in lengths]


def _assert_grouping(groups, lengths, max_batch_size: int, budget: int | None = None) -> None:
    """Assert the invariants every ``plan_batches`` result must satisfy."""
    flat = [index for group in groups for index in group]
    assert sorted(flat) == list(range(len(lengths))), "every index exactly once"
    for group in groups:
        assert group == sorted(group), "groups keep request order"
        assert len(group) <= max_batch_size
        group_lengths = [int(lengths[index]) for index in group]
        assert max(group_lengths) / max(1, min(group_lengths)) <= LENGTH_SIMILARITY_RATIO
        if budget is not None:
            assert len(group) * max(group_lengths) <= budget


# ---------------------------------------------------------------------------
# Feature axis (mel frames)
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_pad_feature_batch_returns_frame_lengths_on_request():
    features = _features([5, 2, 7])

    padded, frame_lengths = pad_feature_batch(features, return_lengths=True)

    assert padded.shape == (3, 3, 7)
    assert frame_lengths.tolist() == [5, 2, 7]
    for row, feature in enumerate(features):
        assert torch.equal(padded[row, :, : feature.shape[-1]], feature)
        assert torch.count_nonzero(padded[row, :, feature.shape[-1] :]) == 0


@pytest.mark.unit
def test_pad_feature_batch_default_stays_tensor_only():
    """The Phase 5-2 call sites destructure nothing: they must keep working."""
    features = _features([4, 6])

    padded = pad_feature_batch(features)

    assert isinstance(padded, torch.Tensor)
    assert padded.shape == (2, 3, 6)


@pytest.mark.unit
def test_feature_padding_trim_round_trip_recovers_every_slice_exactly():
    lengths = [9, 1, 4, 9, 2]
    features = _features(lengths)

    padded, frame_lengths = pad_feature_batch(features, return_lengths=True)
    trimmed = trim_feature_output(padded, frame_lengths)

    assert len(trimmed) == len(features)
    for recovered, original in zip(trimmed, features, strict=True):
        assert recovered.shape == original.shape
        assert torch.equal(recovered, original)


@pytest.mark.unit
def test_trim_feature_output_handles_a_2d_batch():
    padded = torch.arange(3 * 6, dtype=torch.float32).reshape(3, 6)

    trimmed = trim_feature_output(padded, torch.tensor([2, 6, 0]))

    assert [tuple(item.shape) for item in trimmed] == [(2,), (6,), (0,)]
    assert torch.equal(trimmed[0], padded[0, :2])
    assert torch.equal(trimmed[1], padded[1])


@pytest.mark.unit
def test_trim_feature_output_supports_another_time_axis():
    """``dim`` is a real parameter: ``[B, T, D]`` trims its frame axis at 1."""
    padded = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)

    trimmed = trim_feature_output(padded, torch.tensor([5, 3]), dim=1)

    assert [tuple(item.shape) for item in trimmed] == [(5, 3), (3, 3)]
    assert torch.equal(trimmed[1], padded[1, :3, :])


@pytest.mark.unit
def test_trim_feature_output_requires_one_length_per_batch_row():
    padded = torch.zeros(2, 4, 8)

    with pytest.raises(ValueError, match="lengths for a batch of 2"):
        trim_feature_output(padded, torch.tensor([8]))


@pytest.mark.unit
def test_trim_feature_output_rejects_a_length_beyond_the_padded_axis():
    padded = torch.zeros(1, 4, 8)

    with pytest.raises(ValueError, match="needs 9 frames"):
        trim_feature_output(padded, torch.tensor([9]))


@pytest.mark.unit
def test_trim_feature_output_rejects_a_batch_axis_time_dim():
    with pytest.raises(ValueError, match="not a non-batch axis"):
        trim_feature_output(torch.zeros(2, 4), torch.tensor([1, 1]), dim=0)


# ---------------------------------------------------------------------------
# Sample axis (waveforms for embedders)
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_pad_waveform_batch_pads_to_b_one_t_and_returns_sample_lengths():
    waveforms = [torch.ones(6), torch.ones(3), torch.ones(6)]

    padded, sample_lengths = pad_waveform_batch(waveforms, sample_rates=[24000, 24000, 24000])

    assert padded.shape == (3, 1, 6)
    assert sample_lengths.tolist() == [6, 3, 6]
    assert torch.equal(padded[1, 0, :3], waveforms[1])
    assert torch.equal(padded[1, 0, 3:], torch.zeros(3))


@pytest.mark.unit
def test_pad_waveform_batch_accepts_artifact_style_rows():
    waveforms = [torch.ones(4), torch.ones(2)]

    padded, sample_lengths = pad_waveform_batch([(waveforms[0], 16000), (waveforms[1], 16000)])

    assert padded.shape == (2, 1, 4)
    assert sample_lengths.tolist() == [4, 2]


@pytest.mark.unit
def test_waveform_padding_trim_round_trip_recovers_every_slice_exactly():
    generator = torch.Generator().manual_seed(7)
    waveforms = [torch.randn(length, generator=generator) for length in (11, 3, 8)]

    padded, sample_lengths = pad_waveform_batch([(wave, 22050) for wave in waveforms])
    trimmed = trim_feature_output(padded, sample_lengths)

    assert [tuple(item.shape) for item in trimmed] == [(1, 11), (1, 3), (1, 8)]
    for recovered, original in zip(trimmed, waveforms, strict=True):
        assert torch.equal(recovered[0], original)


@pytest.mark.unit
def test_pad_waveform_batch_rejects_mixed_sample_rates():
    with pytest.raises(ValueError, match="mixed sample rates"):
        pad_waveform_batch([(torch.ones(4), 24000), (torch.ones(4), 16000)])


@pytest.mark.unit
def test_pad_waveform_batch_rejects_a_non_1d_row():
    with pytest.raises(ValueError, match="expected a 1-D waveform"):
        pad_waveform_batch([(torch.ones(2, 3), 24000)])


@pytest.mark.unit
def test_pad_waveform_batch_rejects_a_rate_count_mismatch():
    with pytest.raises(ValueError, match="sample rates for"):
        pad_waveform_batch([torch.ones(4), torch.ones(4)], sample_rates=[24000])


# ---------------------------------------------------------------------------
# plan_batches / budget on every axis
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_plan_batches_splits_a_pathological_frame_length_mix():
    lengths = [4000, 3900, 100, 90]

    groups = plan_batches(lengths, max_batch_size=4)

    assert groups == [[0, 1], [2, 3]]
    _assert_grouping(groups, lengths, max_batch_size=4)


@pytest.mark.unit
def test_plan_batches_groups_the_sample_axis_the_same_way():
    lengths = [44100, 44000, 8000, 7000]

    groups = plan_batches(lengths, max_batch_size=4)

    assert groups == [[0, 1], [2, 3]]
    _assert_grouping(groups, lengths, max_batch_size=4)


@pytest.mark.unit
@pytest.mark.parametrize(
    "lengths,max_batch_size,budget",
    [
        ([5, 5, 5, 5], 4, 15),  # phones (token axis)
        ([100, 100, 100, 100], 4, 250),  # mel frames
        ([24000, 24000, 24000, 24000], 4, 50000),  # audio samples
    ],
    ids=["token-axis", "frame-axis", "sample-axis"],
)
def test_budget_is_enforced_on_every_axis(lengths, max_batch_size, budget):
    groups = plan_batches(lengths, max_batch_size, budget)

    assert len(groups) > 1, "the budget must actually split this batch"
    _assert_grouping(groups, lengths, max_batch_size, budget=budget)


@pytest.mark.unit
def test_default_token_budget_applies_on_the_frame_axis():
    lengths = [100, 100, 100]
    budget = default_token_budget(lengths, max_batch_size=2)

    groups = plan_batches(lengths, max_batch_size=2, token_budget=budget)

    assert budget == 200
    assert groups == [[0, 1], [2]]
    _assert_grouping(groups, lengths, max_batch_size=2, budget=budget)


@pytest.mark.unit
def test_default_token_budget_caps_the_frame_area():
    lengths = [3660] * 8
    assert default_token_budget(lengths, max_batch_size=8) == 8 * 3660
    budget = default_token_budget(
        lengths, max_batch_size=8, max_budget=DEFAULT_MEL_FRAME_BATCH_BUDGET
    )

    assert budget == DEFAULT_MEL_FRAME_BATCH_BUDGET == 7500
    groups = plan_batches(lengths, max_batch_size=8, token_budget=budget)
    assert groups == [[0, 1], [2, 3], [4, 5], [6, 7]]
    _assert_grouping(groups, lengths, max_batch_size=8, budget=budget)


@pytest.mark.unit
def test_default_token_budget_phone_cap_keeps_short_chunk_width():
    lengths = [128] * 8
    budget = default_token_budget(
        lengths, max_batch_size=8, max_budget=DEFAULT_PHONE_BATCH_BUDGET
    )

    assert budget == 8 * 128 <= DEFAULT_PHONE_BATCH_BUDGET
    groups = plan_batches(lengths, max_batch_size=8, token_budget=budget)
    assert groups == [list(range(8))]
    _assert_grouping(groups, lengths, max_batch_size=8, budget=budget)

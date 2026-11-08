"""Unit tests for padded-batch inference primitives (Phase 5-2).

Covers the vendored ``infer_batched`` entry points on the OpenVoice V1 and
MeloTTS synthesizers, the padding/unpadding helpers, the length-similar
sub-batcher, and the per-model ``batched_inference`` config flag.

The equivalence proof runs on **tiny randomly-initialised instances of the real
vendored classes** on CPU: with ``B == 1`` the batched entry point must be
bit-identical to the single-sample one (same ops, same RNG draws), and with
sampling noise disabled a sample inside a padded batch must equal its solo run
up to floating-point accumulation order. Real weights are exercised by the
``batching:1-vs-N`` harness axis instead (GPU, see docs/benchmarks/throughput.md).
"""

import pytest
import torch

from harmonyspeech.common.config import DeviceConfig, ModelConfig
from harmonyspeech.modeling.models.melo.melo import SynthesizerTrn as MeloSynthesizerTrn
from harmonyspeech.modeling.models.openvoice.openvoice import SynthesizerTrn as OpenVoiceSynthesizerTrn
from harmonyspeech.task_handler.batching import (
    LENGTH_SIMILARITY_RATIO,
    batched_inference_enabled,
    collapse_per_sample,
    default_token_budget,
    pad_feature_batch,
    pad_phone_batch,
    plan_batches,
    supports_batched_inference,
    trim_decoder_output,
)

HOP = 4  # tiny decoder: prod(upsample_rates) = 2 * 2


def _tiny_openvoice() -> OpenVoiceSynthesizerTrn:
    torch.manual_seed(0)
    return OpenVoiceSynthesizerTrn(
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
        n_speakers=4,
        gin_channels=4,
    ).eval()


def _tiny_melo() -> MeloSynthesizerTrn:
    torch.manual_seed(0)
    return MeloSynthesizerTrn(
        n_vocab=64,
        spec_channels=8,
        segment_size=4,
        inter_channels=8,
        hidden_channels=8,
        filter_channels=8,
        n_heads=2,
        # TextEncoder with speaker conditioning injects at layer 2.
        n_layers=3,
        kernel_size=3,
        p_dropout=0.0,
        resblock="1",
        resblock_kernel_sizes=[3],
        resblock_dilation_sizes=[[1, 3, 1]],
        upsample_rates=[2, 2],
        upsample_initial_channel=8,
        upsample_kernel_sizes=[4, 4],
        n_speakers=2,
        gin_channels=4,
        use_sdp=False,
        n_flow_layer=1,
        n_layers_trans_flow=1,
        use_transformer_flow=False,
        num_languages=4,
        num_tones=4,
    ).eval()


def _ov1_inputs(seed: int = 1):
    generator = torch.Generator().manual_seed(seed)
    phones = torch.randint(1, 64, (4,), generator=generator)
    sid = torch.tensor([2])
    return phones, sid


def _melo_batch_inputs(lengths=(7, 5)):
    phones = torch.randint(1, 64, (len(lengths), max(lengths)))
    lengths_t = torch.tensor(list(lengths))
    mask = torch.arange(max(lengths)).unsqueeze(0) < lengths_t.unsqueeze(1)
    phones = torch.where(mask, phones, torch.zeros_like(phones))
    batch = len(lengths)
    tones = torch.zeros(batch, max(lengths), dtype=torch.long)
    langs = torch.zeros(batch, max(lengths), dtype=torch.long)
    bert = torch.randn(batch, 1024, max(lengths))
    ja_bert = torch.randn(batch, 768, max(lengths))
    sid = torch.zeros(batch, dtype=torch.long)
    return phones, lengths_t, sid, tones, langs, bert, ja_bert


# ---------------------------------------------------------------------------
# B=1 equivalence: the batched entry point must not change a single bit
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_openvoice_infer_batched_b1_is_bit_identical_to_single_path():
    model = _tiny_openvoice()
    phones, sid = _ov1_inputs()
    x = phones.unsqueeze(0)
    lengths = torch.tensor([phones.numel()])

    torch.manual_seed(11)
    single = model.infer(x, lengths, sid=sid)[0]
    torch.manual_seed(11)
    batched = model.infer_batched(x, lengths, sid=sid)[0]

    assert torch.equal(single, batched)


@pytest.mark.unit
def test_melo_infer_batched_b1_is_bit_identical_to_single_path():
    model = _tiny_melo()
    phones, lengths, sid, tones, langs, bert, ja_bert = _melo_batch_inputs(lengths=(7,))
    kwargs = {"sdp_ratio": 0.2, "noise_scale": 0.6, "noise_scale_w": 0.8, "length_scale": 1.0}

    torch.manual_seed(11)
    single = model.infer(phones, lengths, sid, tones, langs, bert, ja_bert, **kwargs)[0]
    torch.manual_seed(11)
    batched = model.infer_batched(phones, lengths, sid, tones, langs, bert, ja_bert, **kwargs)[0]

    assert torch.equal(single, batched)


@pytest.mark.unit
def test_openvoice_batched_durations_and_prior_match_each_solo_run():
    """Padding must not change the predicted frames nor the post-attention prior.

    ``y_lengths`` come from the duration predictor + ``generate_path`` and
    ``m_p``/``logs_p`` from the text encoder + path expansion, so comparing them
    per sample proves the padded region is genuinely masked and not merely
    trimmed away afterwards.
    """
    model = _tiny_openvoice()
    lengths = [7, 5, 3]
    phones = torch.randint(1, 64, (len(lengths), max(lengths)))
    mask = torch.arange(max(lengths)).unsqueeze(0) < torch.tensor(lengths).unsqueeze(1)
    phones = torch.where(mask, phones, torch.zeros_like(phones))
    lengths_t = torch.tensor(lengths)
    sid = torch.tensor([1, 2, 3])

    output, y_lengths, _, _, (_, z_p, m_p, logs_p) = model.infer_batched(
        phones, lengths_t, sid=sid, noise_scale=0.0, noise_scale_w=0.0
    )
    assert output.shape[0] == len(lengths)

    for index in range(len(lengths)):
        _, _, solo_mask, (_, solo_zp, solo_mp, solo_lp) = model.infer(
            phones[index : index + 1],
            lengths_t[index : index + 1],
            sid=sid[index : index + 1],
            noise_scale=0.0,
            noise_scale_w=0.0,
        )
        frames = int(solo_mask.sum())
        assert int(y_lengths[index]) == frames
        torch.testing.assert_close(z_p[index : index + 1, :, :frames], solo_zp, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(m_p[index : index + 1, :, :frames], solo_mp, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(logs_p[index : index + 1, :, :frames], solo_lp, rtol=1e-5, atol=1e-5)


@pytest.mark.unit
def test_openvoice_batched_waveform_matches_solo_interior():
    """Same length and the same waveform away from the vocoder's right edge.

    The decoder is a convolutional vocoder: beyond a sample's last frame the
    padded batch carries explicit zeros while the solo run has the conv's own
    implicit zero padding, so the final frame (and a short
    receptive-field-dependent tail) legitimately differs. With real trained
    weights that tail is ≤1e-4 peak (the ``batching:1-vs-N`` gate bounds it);
    this tiny untrained model amplifies it, so the interior is what is asserted
    exactly here.
    """
    model = _tiny_openvoice()
    lengths = [7, 5, 3]
    phones = torch.randint(1, 64, (len(lengths), max(lengths)))
    mask = torch.arange(max(lengths)).unsqueeze(0) < torch.tensor(lengths).unsqueeze(1)
    phones = torch.where(mask, phones, torch.zeros_like(phones))
    lengths_t = torch.tensor(lengths)
    sid = torch.tensor([1, 2, 3])

    output, y_lengths, _, _, _ = model.infer_batched(
        phones, lengths_t, sid=sid, noise_scale=0.0, noise_scale_w=0.0
    )
    trimmed = trim_decoder_output(output, y_lengths, HOP)

    for index in range(len(lengths)):
        solo = model.infer(
            phones[index : index + 1],
            lengths_t[index : index + 1],
            sid=sid[index : index + 1],
            noise_scale=0.0,
            noise_scale_w=0.0,
        )[0][0][0]
        batched = trimmed[index][0]
        assert batched.shape == solo.shape
        interior = max(0, solo.shape[-1] - 2 * HOP)
        torch.testing.assert_close(batched[:interior], solo[:interior], rtol=5e-3, atol=5e-3)


@pytest.mark.unit
def test_melo_batched_durations_and_prior_match_each_solo_run():
    model = _tiny_melo()
    phones, lengths, sid, tones, langs, bert, ja_bert = _melo_batch_inputs(lengths=(7, 5))
    kwargs = {"sdp_ratio": 0.2, "noise_scale": 0.0, "noise_scale_w": 0.0, "length_scale": 1.0}

    _, y_lengths, _, _, (_, z_p, m_p, logs_p) = model.infer_batched(
        phones, lengths, sid, tones, langs, bert, ja_bert, **kwargs
    )

    for index in range(phones.shape[0]):
        _, _, solo_mask, (_, solo_zp, solo_mp, solo_lp) = model.infer(
            phones[index : index + 1],
            lengths[index : index + 1],
            sid[index : index + 1],
            tones[index : index + 1],
            langs[index : index + 1],
            bert[index : index + 1],
            ja_bert[index : index + 1],
            **kwargs,
        )
        frames = int(solo_mask.sum())
        assert int(y_lengths[index]) == frames
        torch.testing.assert_close(z_p[index : index + 1, :, :frames], solo_zp, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(m_p[index : index + 1, :, :frames], solo_mp, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(logs_p[index : index + 1, :, :frames], solo_lp, rtol=1e-5, atol=1e-5)


@pytest.mark.unit
def test_melo_batched_waveform_matches_solo_interior():
    model = _tiny_melo()
    phones, lengths, sid, tones, langs, bert, ja_bert = _melo_batch_inputs(lengths=(7, 5))
    kwargs = {"sdp_ratio": 0.2, "noise_scale": 0.0, "noise_scale_w": 0.0, "length_scale": 1.0}

    output, y_lengths, _, _, _ = model.infer_batched(
        phones, lengths, sid, tones, langs, bert, ja_bert, **kwargs
    )
    trimmed = trim_decoder_output(output, y_lengths, HOP)

    for index in range(phones.shape[0]):
        solo = model.infer(
            phones[index : index + 1],
            lengths[index : index + 1],
            sid[index : index + 1],
            tones[index : index + 1],
            langs[index : index + 1],
            bert[index : index + 1],
            ja_bert[index : index + 1],
            **kwargs,
        )[0][0][0]
        batched = trimmed[index][0]
        assert batched.shape == solo.shape
        interior = max(0, solo.shape[-1] - 2 * HOP)
        torch.testing.assert_close(batched[:interior], solo[:interior], rtol=5e-3, atol=5e-3)


@pytest.mark.unit
def test_infer_batched_accepts_per_sample_parameters():
    """A length-B parameter vector is broadcast to [B, 1, 1] (per-request speed)."""
    model = _tiny_openvoice()
    lengths = [5, 5]
    phones = torch.randint(1, 64, (2, 5))
    lengths_t = torch.tensor(lengths)
    sid = torch.tensor([1, 1])

    output, y_lengths, _, _, _ = model.infer_batched(
        phones, lengths_t, sid=sid, noise_scale=0.0, noise_scale_w=0.0, length_scale=[1.0, 2.0]
    )

    assert output.shape[0] == 2
    # Same token count, doubled length scale -> roughly twice as many frames.
    assert int(y_lengths[1]) > int(y_lengths[0])


# ---------------------------------------------------------------------------
# Padding / unpadding primitives
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_pad_phone_batch_pads_with_pad_id_and_reports_lengths():
    sequences = [torch.tensor([1, 2, 3]), torch.tensor([4])]
    padded, lengths = pad_phone_batch(sequences, pad_id=0)

    assert padded.tolist() == [[1, 2, 3], [4, 0, 0]]
    assert lengths.tolist() == [3, 1]
    assert padded.dtype == torch.long


@pytest.mark.unit
def test_pad_feature_batch_pads_feature_axis():
    features = [torch.ones(2, 3), torch.full((2, 1), 5.0)]
    padded = pad_feature_batch(features)

    assert padded.shape == (2, 2, 3)
    assert padded[1].tolist() == [[5.0, 0.0, 0.0], [5.0, 0.0, 0.0]]


@pytest.mark.unit
def test_pad_feature_batch_rejects_inconsistent_channels():
    with pytest.raises(ValueError, match="inconsistent channel count"):
        pad_feature_batch([torch.ones(2, 3), torch.ones(4, 1)])


@pytest.mark.unit
def test_trim_decoder_output_uses_per_sample_frame_lengths():
    output = torch.arange(3 * 1 * 100, dtype=torch.float32).reshape(3, 1, 100)
    trimmed = trim_decoder_output(output, torch.tensor([1, 5, 10]), hop_length=4)

    assert [t.shape for t in trimmed] == [(1, 4), (1, 20), (1, 40)]
    # Trimming is a slice of the same padded tensor, not a re-computation.
    assert torch.equal(trimmed[1][0], output[1, 0, :20])


@pytest.mark.unit
def test_trim_decoder_output_rejects_lengths_beyond_the_decoder_output():
    output = torch.zeros(1, 1, 8)
    with pytest.raises(ValueError, match="needs 40 samples"):
        trim_decoder_output(output, torch.tensor([10]), hop_length=4)


@pytest.mark.unit
def test_trim_decoder_output_rejects_length_count_mismatch():
    with pytest.raises(ValueError, match="2 lengths for a batch of 1"):
        trim_decoder_output(torch.zeros(1, 1, 8), torch.tensor([1, 2]), hop_length=4)


# ---------------------------------------------------------------------------
# Length-similar sub-batching
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_plan_batches_respects_max_batch_size():
    groups = plan_batches([10, 10, 10, 10, 10], max_batch_size=2)
    assert [len(group) for group in groups] == [2, 2, 1]
    assert sorted(index for group in groups for index in group) == [0, 1, 2, 3, 4]


@pytest.mark.unit
def test_plan_batches_splits_a_pathological_length_mix():
    """One long + many short texts must not pad everything to the long length."""
    lengths = [100, 5, 5, 5]
    groups = plan_batches(lengths, max_batch_size=4)

    assert len(groups) >= 2
    for group in groups:
        values = [lengths[index] for index in group]
        assert max(values) / max(min(values), 1) <= LENGTH_SIMILARITY_RATIO


@pytest.mark.unit
def test_plan_batches_enforces_the_token_budget():
    # 4 x 10 tokens exceeds a budget of 25 even though max_batch_size allows 4.
    groups = plan_batches([10, 10, 10, 10], max_batch_size=4, token_budget=25)
    assert all(len(group) * 10 <= 25 for group in groups)
    assert sum(len(group) for group in groups) == 4


@pytest.mark.unit
def test_plan_batches_keeps_length_similar_items_together():
    groups = plan_batches([20, 21, 19, 3, 3], max_batch_size=4)
    by_index = {index: position for position, group in enumerate(groups) for index in group}
    assert by_index[0] == by_index[1] == by_index[2]
    assert by_index[3] == by_index[4]


@pytest.mark.unit
def test_plan_batches_handles_empty_and_single_inputs():
    assert plan_batches([], max_batch_size=4) == []
    assert plan_batches([7], max_batch_size=4) == [[0]]


@pytest.mark.unit
def test_default_token_budget_scales_with_the_step_median():
    assert default_token_budget([10, 20, 30], max_batch_size=4) == 4 * 20
    assert default_token_budget([], max_batch_size=4) is None


@pytest.mark.unit
def test_collapse_per_sample_returns_a_scalar_for_uniform_values():
    assert collapse_per_sample([1.0, 1.0, 1.0]) == 1.0
    assert collapse_per_sample([1.0, 2.0]) == [1.0, 2.0]


# ---------------------------------------------------------------------------
# Per-model config flag
# ---------------------------------------------------------------------------


def _config(model_type: str, **kwargs) -> ModelConfig:
    return ModelConfig(
        name="cfg",
        model="dummy/model",
        model_type=model_type,
        max_batch_size=kwargs.pop("max_batch_size", 4),
        device_config=DeviceConfig("cpu"),
        dtype="float32",
        **kwargs,
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "model_type",
    [
        "OpenVoiceV1Synthesizer",
        "MeloTTSSynthesizer",
        # Phase 8-2 embedders
        "OpenVoiceV1ToneConverterEncoder",
        "OpenVoiceV2ToneConverterEncoder",
        "HarmonySpeechEncoder",
        # Phase 8-3 tone converters
        "OpenVoiceV1ToneConverter",
        "OpenVoiceV2ToneConverter",
        # Phase 8-4 vocoders and restorer
        "HarmonySpeechVocoder",
        "VoiceFixerRestorer",
        "VoiceFixerVocoder",
        # Phase 11-1: VC — batched source tokenize + S3Gen vocode
        "ChatterboxVC",
    ],
)
def test_batched_inference_defaults_on_for_the_batched_families(model_type):
    config = _config(model_type)
    assert supports_batched_inference(config) is True
    assert config.batched_inference is None
    assert batched_inference_enabled(config) is True


@pytest.mark.unit
@pytest.mark.parametrize(
    "model_type",
    [
        # Documented loop-only families: the HarmonySpeech synthesizer skipped the
        # padded batch on evidence (8-4 audit), the rest run per-item by design.
        "HarmonySpeechSynthesizer",
        "ChatterboxTTS",
        "FasterWhisper",
        "SileroVAD",
        "KittenTTSSynthesizer",
    ],
)
def test_batched_inference_is_off_for_loop_only_families(model_type):
    assert batched_inference_enabled(_config(model_type)) is False


@pytest.mark.unit
def test_batched_inference_can_be_disabled_per_model():
    config = _config("MeloTTSSynthesizer", batched_inference=False)
    assert batched_inference_enabled(config) is False


@pytest.mark.unit
def test_batched_inference_true_on_a_loop_only_family_still_uses_the_loop():
    config = _config("HarmonySpeechSynthesizer", batched_inference=True)
    assert batched_inference_enabled(config) is False


@pytest.mark.unit
def test_batched_inference_flag_is_validated():
    with pytest.raises(ValueError, match="batched_inference must be a bool"):
        _config("MeloTTSSynthesizer", batched_inference="yes")


@pytest.mark.unit
def test_batch_token_budget_is_validated():
    with pytest.raises(ValueError, match="batch_token_budget"):
        _config("MeloTTSSynthesizer", batch_token_budget=0)

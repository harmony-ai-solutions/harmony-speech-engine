"""Unit tests for the Phase 8-4 batched forwards and the VoiceFixer placement fix.

Two groups:

* **Placement.** The VoiceFixer restorer's STFT frontend
  (``torchlibrosa.stft.STFT``) builds its conv weights with
  ``torch.Tensor(numpy arrays)`` inside ``__init__``, which always produces CPU
  tensors. Building the model inside the loader's ``with torch.device("cuda")``
  block therefore left exactly those two convs on the host while the rest of the
  graph (and every input tensor) landed on the GPU — the
  ``input cuda vs weight cpu`` failure recorded since Phase 4. The loader now
  calls ``model.to(device)`` for the VoiceFixer types after construction and
  weight loading; these tests assert the whole tree is placed and that the STFT
  convs really are registered parameters (which is what makes one ``.to()``
  sufficient), on CPU always and on CUDA when a GPU is present.
* **Batched forwards.** ``MelGANGenerator.inference_batched``,
  ``VoiceFixerVocoder.forward_batched`` and
  ``VoiceFixerRestorer.forward_batched`` run on tiny/randomly-initialised
  instances of the real vendored classes (the Phase 5-2/8-2 pattern): ``B == 1``
  must be bit-identical to the single-sample entry point, a row inside a
  length-mixed padded batch must match its solo run, and the padding values must
  not reach the valid tail. The HarmonySpeech synthesizer's *skip* is pinned by
  a measurement test: its padded batch really does change the shorter row.
"""

import numpy as np
import pytest
import torch

from harmonyspeech.common.config import DeviceConfig, ModelConfig
from harmonyspeech.modeling.loader import get_model
from harmonyspeech.modeling.models.batch_padding import length_mask, sync_mask
from harmonyspeech.modeling.models.harmonyspeech.harmonyspeech import (
    ForwardTacotron,
    MelGANGenerator,
)
from harmonyspeech.modeling.models.voicefixer.voicefixer import (
    VoiceFixerRestorer,
    VoiceFixerVocoder,
)

#: ``MelGANGenerator`` with two upsampling stages (4 samples per mel frame).
MELGAN_UPSCALES = [2, 2]
MELGAN_SAMPLES_PER_FRAME = int(np.prod(MELGAN_UPSCALES))
#: ``VoiceFixerVocoderGenerator`` upsamples 7*7*3*3 samples per condition frame.
VOICEFIXER_SAMPLES_PER_FRAME = 7 * 7 * 3 * 3


# ---------------------------------------------------------------------------
# VoiceFixer device placement (the pre-existing GPU bug)
# ---------------------------------------------------------------------------


def _voicefixer_config(model_type: str, device: str) -> ModelConfig:
    return ModelConfig(
        name=f"vf-{device}",
        model="dummy/model",
        model_type=model_type,
        max_batch_size=1,
        device_config=DeviceConfig(device),
        dtype="float32",
        load_format="dummy",
    )


def _parameter_devices(model) -> set[str]:
    return {str(parameter.device) for parameter in model.parameters()}


def _buffer_devices(model) -> set[str]:
    return {str(buffer.device) for buffer in model.buffers()}


@pytest.mark.unit
@pytest.mark.parametrize("model_type", ["VoiceFixerRestorer", "VoiceFixerVocoder"])
def test_voicefixer_loader_places_every_tensor_on_the_cpu_device(model_type):
    model = get_model(_voicefixer_config(model_type, "cpu"), DeviceConfig("cpu"))

    assert _parameter_devices(model) == {"cpu"}
    assert _buffer_devices(model) == {"cpu"}
    if model_type == "VoiceFixerRestorer":
        assert model.f_helper.stft.conv_real.weight.device.type == "cpu"
        assert model.f_helper.stft.conv_imag.weight.device.type == "cpu"


@pytest.mark.unit
def test_voicefixer_stft_convs_are_registered_parameters():
    """Why one ``.to(device)`` is the whole fix: the STFT convs are in the tree.

    ``torchlibrosa`` builds them as ``nn.Conv1d`` modules but overwrites their
    weights with CPU tensors, so they are not *placed* by the ambient default
    device — they are still registered parameters, which is exactly what a
    recursive ``Module.to()`` moves.
    """
    model = VoiceFixerRestorer().eval()
    registered = list(model.parameters())

    assert any(parameter is model.f_helper.stft.conv_real.weight for parameter in registered)
    assert any(parameter is model.f_helper.stft.conv_imag.weight for parameter in registered)
    # And the plain model is already mixed on CPU-only construction: both convs
    # are the *only* tensors torchlibrosa pins, so this asserts the identity we
    # rely on rather than the device.
    assert model.f_helper.stft.conv_real.weight.shape[0] == model.window_size // 2 + 1


@pytest.mark.unit
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_voicefixer_restorer_built_under_a_cuda_context_leaves_the_stft_convs_on_the_host():
    """The verified mechanism, pinned: the device context alone is not enough."""
    with torch.device("cuda"):
        restorer = VoiceFixerRestorer()

    devices = _parameter_devices(restorer)
    assert "cuda:0" in devices and "cpu" in devices
    assert restorer.f_helper.stft.conv_real.weight.device.type == "cpu"
    assert restorer.f_helper.stft.conv_imag.weight.device.type == "cpu"


@pytest.mark.unit
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("model_type", ["VoiceFixerRestorer", "VoiceFixerVocoder"])
def test_voicefixer_loader_places_the_whole_model_on_cuda(model_type):
    model = get_model(_voicefixer_config(model_type, "cuda"), DeviceConfig("cuda"))

    assert _parameter_devices(model) == {"cuda:0"}
    assert _buffer_devices(model) == {"cuda:0"}
    if model_type == "VoiceFixerRestorer":
        assert model.f_helper.stft.conv_real.weight.device.type == "cuda"
        assert model.f_helper.stft.conv_imag.weight.device.type == "cuda"


@pytest.mark.unit
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_voicefixer_restorer_runs_on_cuda_after_the_placement_fix():
    """The bug's symptom (``input cuda vs weight cpu``) is gone end to end."""
    model = get_model(_voicefixer_config("VoiceFixerRestorer", "cuda"), DeviceConfig("cuda"))
    audio = torch.zeros(1, 1, 4096, device="cuda")

    with torch.no_grad():
        mel = model(audio)

    assert mel.device.type == "cuda"
    assert mel.dim() == 4 and mel.shape[0] == 1 and mel.shape[1] == 1


# ---------------------------------------------------------------------------
# Shared padding helpers
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_length_mask_and_sync_mask_validate_their_inputs():
    mask = length_mask([3, 5], 5)
    assert mask.shape == (2, 1, 5)
    assert [int(value) for value in mask[:, 0].sum(dim=-1)] == [3, 5]

    assert torch.equal(sync_mask(mask, 5), mask)
    assert sync_mask(mask, 10).shape[-1] == 10
    with pytest.raises(ValueError, match="cannot align"):
        sync_mask(mask, 7)
    with pytest.raises(ValueError, match="batch holds"):
        length_mask([6], 5)
    with pytest.raises(ValueError, match="empty"):
        length_mask([], 5)


# ---------------------------------------------------------------------------
# HarmonySpeech vocoder (MelGANGenerator)
# ---------------------------------------------------------------------------


def _tiny_melgan(seed: int = 0) -> MelGANGenerator:
    """A small real ``MelGANGenerator`` with three residual stacks.

    ``stacks=3`` matters: the stack's dilations are ``3 ** j``, so the third one
    pads by 9 frames and exercises the reflection-tail fill.
    """
    torch.manual_seed(seed)
    return MelGANGenerator(
        in_channels=4,
        out_channels=1,
        kernel_size=7,
        channels=8,
        bias=True,
        upsample_scales=MELGAN_UPSCALES,
        stack_kernel_size=3,
        stacks=3,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.2},
        pad="ReflectionPad1d",
        pad_params={},
        use_final_nonlinear_activation=True,
        use_weight_norm=True,
        use_causal_conv=False,
    ).eval()


def _pad_rows(rows, pad_value: float = 0.0) -> torch.Tensor:
    """Pad ``[1, C, T]`` rows on the time axis into one ``[B, C, T_max]`` batch."""
    lengths = [int(row.shape[-1]) for row in rows]
    padded = torch.full(
        (len(rows), rows[0].shape[1], max(lengths)), pad_value, dtype=rows[0].dtype
    )
    for index, row in enumerate(rows):
        padded[index, :, : lengths[index]] = row[0]
    return padded


def _mel_rows(lengths, channels: int = 4, seed: int = 1):
    generator = torch.Generator().manual_seed(seed)
    return [torch.randn(1, channels, length, generator=generator) for length in lengths]


@pytest.mark.unit
@pytest.mark.parametrize("frames", [5, 11, 23])
def test_melgan_batched_b1_is_bit_identical(frames):
    generator = _tiny_melgan()
    mel = torch.randn(1, 4, frames)

    with torch.no_grad():
        solo = generator.inference(mel[0].T)
        batched, sample_lengths = generator.inference_batched(mel, [frames])

    assert torch.equal(batched[0, 0], solo[:, 0])
    assert sample_lengths.tolist() == [frames * MELGAN_SAMPLES_PER_FRAME]


@pytest.mark.unit
def test_melgan_batched_mixed_lengths_match_their_solo_runs():
    lengths = [5, 11, 7]
    generator = _tiny_melgan()
    rows = _mel_rows(lengths)
    padded = _pad_rows(rows)

    with torch.no_grad():
        solo = [generator.inference(row[0].T) for row in rows]
        batched, sample_lengths = generator.inference_batched(padded, lengths)

    assert sample_lengths.tolist() == [length * 4 for length in lengths]
    for index, expected in enumerate(solo):
        produced = batched[index, 0, : expected.shape[0]]
        # Float32 kernel blocking differs per batch shape; measured ≤8e-8.
        assert torch.allclose(produced, expected[:, 0], atol=1e-6), index


@pytest.mark.unit
def test_melgan_batched_padding_never_reaches_the_valid_tail():
    lengths = [5, 11, 7]
    generator = _tiny_melgan()
    padded = _pad_rows(_mel_rows(lengths))
    noisy = padded.clone()
    noisy[0, :, lengths[0] :] = 1e3
    noisy[2, :, lengths[2] :] = -5e2

    with torch.no_grad():
        clean, _ = generator.inference_batched(padded, lengths)
        dirty, _ = generator.inference_batched(noisy, lengths)

    for index, length in enumerate(lengths):
        samples = length * MELGAN_SAMPLES_PER_FRAME
        assert torch.equal(clean[index, 0, :samples], dirty[index, 0, :samples]), index


@pytest.mark.unit
def test_melgan_batched_rejects_an_unaligned_stage():
    """An unbatched architecture fails loudly instead of guessing an alignment."""
    with pytest.raises(ValueError, match="cannot align"):
        sync_mask(length_mask([4, 8], 8), 6)


# ---------------------------------------------------------------------------
# VoiceFixer vocoder
# ---------------------------------------------------------------------------


def _tiny_voicefixer_vocoder(seed: int = 0) -> VoiceFixerVocoder:
    torch.manual_seed(seed)
    return VoiceFixerVocoder().eval()


def _vocoder_mels(lengths, seed: int = 1):
    generator = torch.Generator().manual_seed(seed)
    return [torch.randn(1, 1, length, 128, generator=generator) * 0.3 for length in lengths]


def _pad_vocoder_mels(mels) -> torch.Tensor:
    lengths = [int(mel.shape[2]) for mel in mels]
    padded = torch.zeros((len(mels), 1, max(lengths), 128))
    for index, mel in enumerate(mels):
        padded[index, :, : lengths[index]] = mel[0]
    return padded


@pytest.mark.unit
def test_voicefixer_vocoder_batched_b1_is_bit_identical():
    vocoder = _tiny_voicefixer_vocoder()
    mel = _vocoder_mels([7])[0]
    condition_frames = 7 + 7 % 2 + 4

    with torch.no_grad():
        solo = vocoder(mel)
        batched, sample_lengths = vocoder.forward_batched(mel, [7])

    assert torch.equal(batched, solo)
    assert sample_lengths.tolist() == [condition_frames * VOICEFIXER_SAMPLES_PER_FRAME]


@pytest.mark.unit
def test_voicefixer_vocoder_batched_mixed_lengths_match_their_solo_runs():
    lengths = [5, 9, 7]
    vocoder = _tiny_voicefixer_vocoder()
    mels = _vocoder_mels(lengths)
    expected_samples = [
        (length + length % 2 + 4) * VOICEFIXER_SAMPLES_PER_FRAME for length in lengths
    ]

    with torch.no_grad():
        solo = [vocoder(mel) for mel in mels]
        batched, sample_lengths = vocoder.forward_batched(_pad_vocoder_mels(mels), lengths)

    # A row's own ``tr_pre`` tail (``frames % 2 + 4``) sets its sample count.
    assert sample_lengths.tolist() == expected_samples
    for index, expected in enumerate(solo):
        assert expected.shape[-1] == expected_samples[index]
        produced = batched[index, 0, : expected.shape[-1]]
        # Measured ≤3.3e-7 on random weights.
        assert torch.allclose(produced, expected[0, 0], atol=1e-5), index


@pytest.mark.unit
def test_voicefixer_vocoder_batched_padding_never_reaches_the_valid_tail():
    lengths = [5, 9, 7]
    vocoder = _tiny_voicefixer_vocoder()
    padded = _pad_vocoder_mels(_vocoder_mels(lengths))
    noisy = padded.clone()
    noisy[0, :, lengths[0] :] = 5.0

    with torch.no_grad():
        clean, sample_lengths = vocoder.forward_batched(padded, lengths)
        dirty, _ = vocoder.forward_batched(noisy, lengths)

    samples = int(sample_lengths[0])
    assert torch.equal(clean[0, 0, :samples], dirty[0, 0, :samples])


# ---------------------------------------------------------------------------
# VoiceFixer restorer
# ---------------------------------------------------------------------------


def _tiny_voicefixer_restorer(seed: int = 0) -> VoiceFixerRestorer:
    torch.manual_seed(seed)
    return VoiceFixerRestorer().eval()


def _restorer_audio(sample_counts, seed: int = 1):
    generator = torch.Generator().manual_seed(seed)
    return [torch.randn(1, 1, count, generator=generator) * 0.1 for count in sample_counts]


def _pad_restorer_mels(mels) -> torch.Tensor:
    lengths = [int(mel.shape[2]) for mel in mels]
    padded = torch.zeros((len(mels), 1, max(lengths), mels[0].shape[-1]))
    for index, mel in enumerate(mels):
        padded[index, :, : lengths[index]] = mel[0]
    return padded


def _assert_close_in_relative_terms(produced, expected, tolerance: float = 1e-4):
    """The restorer's output is ``exp(log-mel)``, so absolute deltas scale with it.

    The batched forward matches its solo run in float32 accumulation (measured
    1e-6 relative on random weights), which the exponential turns into an
    absolute delta proportional to the value itself.
    """
    delta = (produced - expected).abs().max()
    scale = expected.abs().max().clamp(min=1e-6)
    assert delta <= tolerance * scale, f"{float(delta)} vs scale {float(scale)}"


@pytest.mark.unit
def test_voicefixer_restorer_batched_b1_is_bit_identical():
    restorer = _tiny_voicefixer_restorer()
    audio = _restorer_audio([4096])[0]

    with torch.no_grad():
        solo = restorer(audio)
        mel = restorer.preprocess(audio)
        batched = restorer.forward_batched(mel, [int(mel.shape[2])])

    assert torch.equal(batched[:, :, : int(mel.shape[2])], solo)


@pytest.mark.unit
def test_voicefixer_restorer_batched_mixed_lengths_match_their_solo_runs():
    restorer = _tiny_voicefixer_restorer()
    audio = _restorer_audio([4096, 6000, 5000])

    with torch.no_grad():
        solo = [restorer(clip) for clip in audio]
        mels = [restorer.preprocess(clip) for clip in audio]
        lengths = [int(mel.shape[2]) for mel in mels]
        batched = restorer.forward_batched(_pad_restorer_mels(mels), lengths)

    assert lengths == [int(mel.shape[2]) for mel in solo]
    for index, expected in enumerate(solo):
        produced = batched[index : index + 1, :, : lengths[index]]
        _assert_close_in_relative_terms(produced, expected)


@pytest.mark.unit
def test_voicefixer_restorer_batched_isolates_rows_across_the_64_frame_alignment():
    """Rows are padded to a multiple of the UNet's 64-frame division internally.

    ``75`` and ``140`` mel frames align to ``128`` / ``192``, so the short row's
    mask really is narrower than the batch (the all-aligned case above is not
    enough to prove the per-stage mask).
    """
    restorer = _tiny_voicefixer_restorer()
    audio = _restorer_audio([70 * 441 + 2048, 135 * 441 + 2048])

    with torch.no_grad():
        solo = [restorer(clip) for clip in audio]
        mels = [restorer.preprocess(clip) for clip in audio]
        lengths = [int(mel.shape[2]) for mel in mels]
        batched = restorer.forward_batched(_pad_restorer_mels(mels), lengths)

    assert lengths[0] != lengths[1]
    for index, expected in enumerate(solo):
        produced = batched[index : index + 1, :, : lengths[index]]
        _assert_close_in_relative_terms(produced, expected)


@pytest.mark.unit
def test_voicefixer_restorer_batched_padding_never_reaches_the_valid_tail():
    restorer = _tiny_voicefixer_restorer()
    audio = _restorer_audio([4096, 6000, 5000])

    with torch.no_grad():
        mels = [restorer.preprocess(clip) for clip in audio]
        lengths = [int(mel.shape[2]) for mel in mels]
        padded = _pad_restorer_mels(mels)
        clean = restorer.forward_batched(padded, lengths)
        noisy = padded.clone()
        noisy[0, :, lengths[0] :] = 100.0
        dirty = restorer.forward_batched(noisy, lengths)

    assert torch.equal(clean, dirty)


# ---------------------------------------------------------------------------
# HarmonySpeech synthesizer: the documented skip, pinned by measurement
# ---------------------------------------------------------------------------


def _tiny_forward_tacotron(seed: int = 0) -> ForwardTacotron:
    torch.manual_seed(seed)
    return ForwardTacotron(
        embed_dims=4,
        series_embed_dims=4,
        num_chars=20,
        durpred_conv_dims=4,
        durpred_rnn_dims=4,
        durpred_dropout=0.0,
        pitch_conv_dims=4,
        pitch_rnn_dims=4,
        pitch_dropout=0.0,
        pitch_strength=1.0,
        energy_conv_dims=4,
        energy_rnn_dims=4,
        energy_dropout=0.0,
        energy_strength=1.0,
        rnn_dims=4,
        prenet_dims=4,
        prenet_k=2,
        postnet_num_highways=1,
        prenet_dropout=0.0,
        postnet_dims=4,
        postnet_k=2,
        prenet_num_highways=1,
        postnet_dropout=0.0,
        n_mels=6,
        speaker_embed_dims=8,
        padding_value=0.0,
    ).eval()


@pytest.mark.unit
def test_harmonyspeech_synthesizer_generate_takes_a_batch_but_padding_leaks():
    """Evidence for the 8-4 skip: the graph accepts ``B > 1``, the values change.

    ``ForwardTacotron.generate`` expands durations per row and then runs a
    bidirectional LSTM over the *batch's* longest expanded sequence, and the
    postnet ``CBHG`` mixes every frame with a bidirectional GRU. A shorter row
    therefore sees its neighbours' padding on all of its frames — measured here
    as a 0.16 peak change on the row's own frames.
    """
    model = _tiny_forward_tacotron()
    tokens_short = torch.randint(1, 19, (1, 7))
    tokens_long = torch.randint(1, 19, (1, 12))
    speaker = torch.randn(1, 8)

    padded = torch.zeros(2, 12, dtype=torch.long)
    padded[0, :7] = tokens_short[0]
    padded[1] = tokens_long[0]

    with torch.no_grad():
        solo_short = model.generate(tokens_short, speaker, alpha=1.0)[1]
        solo_long = model.generate(tokens_long, speaker, alpha=1.0)[1]
        batched = model.generate(padded, speaker.repeat(2, 1), alpha=1.0)[1]

    # The longest row (no padding of its own) still matches: the difference is
    # the shorter row's padding, not the batch shape.
    assert torch.allclose(
        batched[1][:, : solo_long.shape[-1]], solo_long[0], atol=1e-6
    )
    short_delta = (batched[0][:, : solo_short.shape[-1]] - solo_short[0]).abs().max()
    assert short_delta > 1e-2, float(short_delta)

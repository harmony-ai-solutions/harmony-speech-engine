"""Unit tests for the batched HarmonySpeech / VoiceFixer handlers (Phase 8-4).

The three batched handlers flatten their work into rows — one row per request for
the two vocoders, one row per ``(request, 30 s slice)`` for the restorer — pad
them on the mel-frame axis, forward one masked batch per length-similar
sub-batch, and reassemble per request. What is proven here:

* four requests share exactly one batched forward, and the batched outputs match
  the per-item loop;
* the HarmonySpeech vocoder's per-request ``breaks`` pairing survives batching
  (the silence insertion still uses the request's own frame counts);
* the VoiceFixer vocoder normalizes **per request**, not by the batch maximum;
* a single request keeps the loop fast path and a poisoned request errors alone.

The vendored math itself is covered by
``tests/unit/modeling/test_batched_hs_voicefixer.py``; the models here are real
tiny instances (or deterministic stubs where the property under test is the
handler's arithmetic).
"""

import numpy as np
import pytest
import torch

from harmonyspeech.common.artifacts import AudioArtifact, MelArtifact
from harmonyspeech.common.config import DeviceConfig, ModelConfig
from harmonyspeech.common.inputs import AudioConversionRequestInput
from harmonyspeech.common.request import EngineRequest
from harmonyspeech.modeling.models.harmonyspeech.harmonyspeech import MelGANGenerator
from harmonyspeech.modeling.models.voicefixer.voicefixer import (
    VoiceFixerRestorer,
    VoiceFixerVocoder,
)
from harmonyspeech.task_handler.model_runner_base import ModelRunnerBase

#: Big enough that length similarity, not the padded-area budget, decides groups.
ROOMY_BUDGET = 1_000_000
HARMONYSPEECH_SR = 16000
#: The vocoder re-inserts ``0.15 s`` of silence per break (handler constant).
BREAK_SILENCE = int(0.15 * HARMONYSPEECH_SR)


def _runner(model, model_type: str, **config) -> ModelRunnerBase:
    runner = ModelRunnerBase.__new__(ModelRunnerBase)
    runner.device = "cpu"
    runner.device_config = DeviceConfig("cpu")
    runner._baseline_state = None
    runner.model = model
    runner.embedding_cache = None
    runner.model_identity = None
    runner.model_config = ModelConfig(
        name="stub",
        model="dummy/model",
        model_type=model_type,
        max_batch_size=config.pop("max_batch_size", 4),
        device_config=DeviceConfig("cpu"),
        dtype="float32",
        **config,
    )
    return runner


def _request(request_id: str) -> EngineRequest:
    request_data = AudioConversionRequestInput(
        request_id=request_id, requested_model="stub", model="stub", source_audio=""
    )
    return EngineRequest(request_id=request_id, request_data=request_data, arrival_time=0.0)


def _mel(frames: int, bins: int = 4, seed: int = 1, value: float | None = None) -> MelArtifact:
    if value is not None:
        array = np.full((1, bins, frames), value, dtype=np.float32)
    else:
        t = np.arange(bins * frames, dtype=np.float32) / 10.0 + float(seed)
        array = (0.1 * np.sin(t)).reshape(1, bins, frames).astype(np.float32)
    return MelArtifact.from_numpy(array, sample_rate=HARMONYSPEECH_SR, hop_length=200)


def _clip(samples: int, freq: float = 220.0, amplitude: float = 0.1) -> AudioArtifact:
    t = np.arange(samples) / 44100.0
    return AudioArtifact.from_numpy(
        (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32), 44100
    )


# ---------------------------------------------------------------------------
# Recording wrappers around the real models (no stub math)
# ---------------------------------------------------------------------------


class _RecordingMelGan:
    """A real ``MelGANGenerator`` whose entry points are recorded."""

    def __init__(self, model: MelGANGenerator):
        self.model = model
        self.batched_calls: list[list[int]] = []
        self.loop_calls: list[int] = []

    def inference(self, c, **kwargs):
        self.loop_calls.append(int(c.shape[0]))
        return self.model.inference(c, **kwargs)

    def inference_batched(self, c, frame_lengths=None, **kwargs):
        self.batched_calls.append([int(value) for value in frame_lengths])
        return self.model.inference_batched(c, frame_lengths, **kwargs)


class _RecordingRestorer:
    """A real ``VoiceFixerRestorer`` whose entry points are recorded."""

    def __init__(self, model: VoiceFixerRestorer):
        self.model = model
        self.sample_rate = model.sample_rate
        self.batched_calls: list[list[int]] = []
        self.loop_calls: list[int] = []

    def __call__(self, audio_tensor):
        self.loop_calls.append(int(audio_tensor.shape[-1]))
        return self.model(audio_tensor)

    def preprocess(self, segment):
        return self.model.preprocess(segment)

    def forward_batched(self, mels, frame_lengths=None):
        self.batched_calls.append([int(value) for value in frame_lengths])
        return self.model.forward_batched(mels, frame_lengths)


class _RecordingVocoder:
    """A real ``VoiceFixerVocoder`` whose entry points are recorded."""

    def __init__(self, model: VoiceFixerVocoder):
        self.model = model
        self.batched_calls: list[list[int]] = []
        self.loop_calls: list[int] = []

    def __call__(self, mel_spectrogram):
        self.loop_calls.append(int(mel_spectrogram.shape[2]))
        return self.model(mel_spectrogram)

    def forward_batched(self, mel_spectrogram, frame_lengths=None):
        self.batched_calls.append([int(value) for value in frame_lengths])
        return self.model.forward_batched(mel_spectrogram, frame_lengths)


def _tiny_melgan(seed: int = 0) -> MelGANGenerator:
    torch.manual_seed(seed)
    return MelGANGenerator(
        in_channels=4,
        out_channels=1,
        kernel_size=7,
        channels=8,
        bias=True,
        upsample_scales=[2, 2],
        stack_kernel_size=3,
        stacks=2,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.2},
        pad="ReflectionPad1d",
        pad_params={},
        use_final_nonlinear_activation=True,
        use_weight_norm=True,
        use_causal_conv=False,
    ).eval()


# ---------------------------------------------------------------------------
# HarmonySpeech vocoder
# ---------------------------------------------------------------------------


def _run_harmonyspeech_vocoder(runner, inputs, requests):
    return runner._execute_harmonyspeech_vocoder(inputs, requests)


def _vocoder_inputs(frames, with_breaks=None):
    """``(MelArtifact, breaks)`` pairs; ``with_breaks`` maps index → breaks list."""
    with_breaks = with_breaks or {}
    return [(_mel(length, seed=index), with_breaks.get(index)) for index, length in enumerate(frames)]


@pytest.mark.unit
def test_harmonyspeech_vocoder_forwards_a_batch_of_four_in_one_call():
    model = _RecordingMelGan(_tiny_melgan())
    runner = _runner(model, "HarmonySpeechVocoder", max_batch_size=8, batch_token_budget=ROOMY_BUDGET)
    frames = [5, 9, 6, 11]
    requests = [_request(f"hs-voc-{index}") for index in range(len(frames))]

    results = _run_harmonyspeech_vocoder(runner, _vocoder_inputs(frames), requests)

    assert [result.request_id for result in results] == [f"hs-voc-{index}" for index in range(4)]
    assert all(result.result_data.finish_reason == "stop" for result in results)
    assert model.batched_calls == [frames]
    assert model.loop_calls == []


@pytest.mark.unit
def test_harmonyspeech_vocoder_batched_matches_the_per_item_loop():
    shared = _RecordingMelGan(_tiny_melgan())
    batched_runner = _runner(shared, "HarmonySpeechVocoder", batch_token_budget=ROOMY_BUDGET)
    loop_runner = _runner(shared, "HarmonySpeechVocoder", batched_inference=False)
    frames = [5, 9, 6, 11]

    batched = _run_harmonyspeech_vocoder(
        batched_runner, _vocoder_inputs(frames), [_request(f"b-{i}") for i in range(4)]
    )
    looped = _run_harmonyspeech_vocoder(
        loop_runner, _vocoder_inputs(frames), [_request(f"l-{i}") for i in range(4)]
    )

    for index in range(4):
        produced = batched[index].result_data.audio_artifact
        expected = looped[index].result_data.audio_artifact
        assert produced.num_samples == expected.num_samples
        # Same decode modulo float32 kernel blocking; measured ≤1e-7.
        assert np.allclose(produced.numpy, expected.numpy, atol=1e-5), index


@pytest.mark.unit
def test_harmonyspeech_vocoder_keeps_breaks_per_request_under_batching():
    """A request with breaks gets its own silence insertion, in request order."""
    model = _RecordingMelGan(_tiny_melgan())
    batched_runner = _runner(model, "HarmonySpeechVocoder", batch_token_budget=ROOMY_BUDGET)
    loop_runner = _runner(model, "HarmonySpeechVocoder", batched_inference=False)
    frames = [5, 4]
    inputs = _vocoder_inputs(frames, with_breaks={0: [1]})

    batched = _run_harmonyspeech_vocoder(
        batched_runner, inputs, [_request("with-breaks"), _request("plain")]
    )
    looped = _run_harmonyspeech_vocoder(
        loop_runner, _vocoder_inputs(frames, with_breaks={0: [1]}), [_request("solo-0"), _request("solo-1")]
    )

    # 5 frames × 4 samples/frame + one 0.15 s silence block; the plain request
    # keeps its own (shorter) length, so a swapped pairing would fail here.
    with_breaks = batched[0].result_data.audio_artifact
    plain = batched[1].result_data.audio_artifact
    assert with_breaks.num_samples == 5 * 4 + BREAK_SILENCE
    assert plain.num_samples == 4 * 4
    for index in range(2):
        assert np.allclose(
            batched[index].result_data.audio_artifact.numpy,
            looped[index].result_data.audio_artifact.numpy,
            atol=1e-6,
        ), index


@pytest.mark.unit
def test_harmonyspeech_vocoder_single_request_keeps_the_loop_path():
    model = _RecordingMelGan(_tiny_melgan())
    runner = _runner(model, "HarmonySpeechVocoder")

    results = _run_harmonyspeech_vocoder(runner, _vocoder_inputs([5]), [_request("solo")])

    assert model.batched_calls == []
    assert model.loop_calls == [5]
    assert results[0].result_data.finish_reason == "stop"


@pytest.mark.unit
def test_harmonyspeech_vocoder_poisoned_request_errors_alone():
    model = _RecordingMelGan(_tiny_melgan())
    runner = _runner(model, "HarmonySpeechVocoder", batch_token_budget=ROOMY_BUDGET)
    inputs = [
        (_mel(5, seed=1), None),
        (_mel(4, bins=3, seed=2), None),  # inconsistent bin count breaks the batch
        (_mel(6, seed=3), None),
    ]
    requests = [_request(f"voc-{index}") for index in range(3)]

    results = _run_harmonyspeech_vocoder(runner, inputs, requests)

    assert results[0].result_data.finish_reason == "stop"
    assert results[1].result_data.finish_reason == "error"
    assert results[2].result_data.finish_reason == "stop"
    # The poisoned row forced the loop, so no batched forward was recorded.
    assert model.batched_calls == []


# ---------------------------------------------------------------------------
# VoiceFixer restorer
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_voicefixer_restorer_forwards_a_batch_of_four_in_one_call():
    restorer = _RecordingRestorer(VoiceFixerRestorer().eval())
    runner = _runner(restorer, "VoiceFixerRestorer", max_batch_size=8, batch_token_budget=ROOMY_BUDGET)
    clips = [_clip(4096), _clip(6000, freq=330.0), _clip(5000, freq=440.0), _clip(7000, freq=550.0)]
    requests = [_request(f"vf-rest-{index}") for index in range(4)]

    results = runner._execute_voicefixer_restorer(clips, requests)

    assert [result.request_id for result in results] == [f"vf-rest-{index}" for index in range(4)]
    assert all(result.result_data.finish_reason == "stop" for result in results)
    assert len(restorer.batched_calls) == 1
    assert len(restorer.batched_calls[0]) == 4
    assert restorer.loop_calls == []


@pytest.mark.unit
def test_voicefixer_restorer_batched_matches_the_per_item_loop():
    shared = _RecordingRestorer(VoiceFixerRestorer().eval())
    batched_runner = _runner(shared, "VoiceFixerRestorer", batch_token_budget=ROOMY_BUDGET)
    loop_runner = _runner(shared, "VoiceFixerRestorer", batched_inference=False)
    clips = [_clip(4096), _clip(6000, freq=330.0), _clip(5000, freq=440.0), _clip(7000, freq=550.0)]

    batched = batched_runner._execute_voicefixer_restorer(
        clips, [_request(f"b-{index}") for index in range(4)]
    )
    looped = loop_runner._execute_voicefixer_restorer(
        clips, [_request(f"l-{index}") for index in range(4)]
    )

    for index in range(4):
        produced = batched[index].result_data.mel_artifact
        expected = looped[index].result_data.mel_artifact
        assert produced.mel.shape == expected.mel.shape, index
        # The restorer emits ``exp(log-mel)``; compare relative to its own scale
        # (float32 accumulation is ~1e-6 relative, see the model test).
        delta = (produced.mel - expected.mel).abs().max()
        assert delta <= 1e-4 * expected.mel.abs().max().clamp(min=1e-6), index


@pytest.mark.unit
def test_voicefixer_restorer_single_request_keeps_the_loop_path():
    restorer = _RecordingRestorer(VoiceFixerRestorer().eval())
    runner = _runner(restorer, "VoiceFixerRestorer")

    results = runner._execute_voicefixer_restorer([_clip(4096)], [_request("solo")])

    assert restorer.batched_calls == []
    assert restorer.loop_calls == [4096]
    assert results[0].result_data.finish_reason == "stop"


@pytest.mark.unit
def test_voicefixer_restorer_poisoned_request_errors_alone():
    restorer = _RecordingRestorer(VoiceFixerRestorer().eval())
    runner = _runner(restorer, "VoiceFixerRestorer", batch_token_budget=ROOMY_BUDGET)
    clips = [_clip(4096), _clip(64), _clip(5000, freq=440.0)]  # 64 samples cannot be STFT'd
    requests = [_request(f"rest-{index}") for index in range(3)]

    results = runner._execute_voicefixer_restorer(clips, requests)

    assert results[0].result_data.finish_reason == "stop"
    assert results[1].result_data.finish_reason == "error"
    assert results[2].result_data.finish_reason == "stop"
    # The poisoned row never entered the batch, so the other two still batched.
    assert len(restorer.batched_calls) == 1 and len(restorer.batched_calls[0]) == 2


# ---------------------------------------------------------------------------
# VoiceFixer vocoder
# ---------------------------------------------------------------------------


class _AmplitudeStubVocoder:
    """Deterministic vocoder: the mel's first value sets the waveform amplitude.

    Used for the normalization test only — a per-batch maximum would rescale the
    quiet request by its neighbour's loudest sample, which this stub makes
    visible as a different peak.
    """

    def __init__(self):
        self.batched_calls: list[list[int]] = []
        self.loop_calls: list[int] = []

    @staticmethod
    def _render(frames: int, marker: float) -> torch.Tensor:
        return torch.full((1, 1, int(frames) * 4), float(marker))

    def __call__(self, mel_spectrogram):
        frames = int(mel_spectrogram.shape[2])
        marker = float(mel_spectrogram[0, 0, 0, 0])
        self.loop_calls.append(frames)
        return self._render(frames, marker)

    def forward_batched(self, mel_spectrogram, frame_lengths=None):
        lengths = [int(value) for value in frame_lengths]
        self.batched_calls.append(lengths)
        audio = torch.zeros(len(lengths), 1, max(lengths) * 4)
        for row, frames in enumerate(lengths):
            audio[row, 0, : frames * 4] = float(mel_spectrogram[row, 0, 0, 0])
        return audio, torch.tensor([frames * 4 for frames in lengths], dtype=torch.long)


@pytest.mark.unit
def test_voicefixer_vocoder_normalizes_per_request_not_per_batch():
    model = _AmplitudeStubVocoder()
    runner = _runner(model, "VoiceFixerVocoder", batch_token_budget=ROOMY_BUDGET)
    inputs = [_mel(6, bins=128, value=1.0), _mel(9, bins=128, value=0.1)]
    requests = [_request("loud"), _request("quiet")]

    results = runner._execute_voicefixer_vocoder(inputs, requests)

    assert model.batched_calls == [[6, 9]]
    for result in results:
        waveform = result.result_data.audio_artifact.numpy
        # Each request is normalized by *its own* peak; a batch maximum would
        # leave the quiet one at 0.097.
        assert np.isclose(np.abs(waveform).max(), 0.97, atol=1e-6)


@pytest.mark.unit
def test_voicefixer_vocoder_batched_matches_the_per_item_loop():
    shared = _RecordingVocoder(VoiceFixerVocoder().eval())
    batched_runner = _runner(shared, "VoiceFixerVocoder", batch_token_budget=ROOMY_BUDGET)
    loop_runner = _runner(shared, "VoiceFixerVocoder", batched_inference=False)
    inputs = [_mel(5, bins=128, seed=1), _mel(9, bins=128, seed=2), _mel(7, bins=128, seed=3)]

    batched = batched_runner._execute_voicefixer_vocoder(
        inputs, [_request(f"b-{index}") for index in range(3)]
    )
    looped = loop_runner._execute_voicefixer_vocoder(
        inputs, [_request(f"l-{index}") for index in range(3)]
    )

    assert len(shared.batched_calls) == 1
    for index in range(3):
        produced = batched[index].result_data.audio_artifact
        expected = looped[index].result_data.audio_artifact
        assert produced.num_samples == expected.num_samples, index
        assert np.allclose(produced.numpy, expected.numpy, atol=1e-5), index


@pytest.mark.unit
def test_voicefixer_vocoder_single_request_keeps_the_loop_path():
    model = _AmplitudeStubVocoder()
    runner = _runner(model, "VoiceFixerVocoder")

    results = runner._execute_voicefixer_vocoder(
        [_mel(6, bins=128, value=1.0)], [_request("solo")]
    )

    assert model.batched_calls == []
    assert model.loop_calls == [6]
    assert results[0].result_data.finish_reason == "stop"


@pytest.mark.unit
def test_voicefixer_vocoder_poisoned_request_errors_alone():
    vocoder = _RecordingVocoder(VoiceFixerVocoder().eval())
    runner = _runner(vocoder, "VoiceFixerVocoder", batch_token_budget=ROOMY_BUDGET)
    inputs = [_mel(5, bins=128, seed=1), _mel(6, bins=64, seed=2), _mel(7, bins=128, seed=3)]
    requests = [_request(f"vf-voc-{index}") for index in range(3)]

    results = runner._execute_voicefixer_vocoder(inputs, requests)

    assert results[0].result_data.finish_reason == "stop"
    assert results[1].result_data.finish_reason == "error"
    assert results[2].result_data.finish_reason == "stop"
    # The inconsistent bin count fails the whole sub-batch, so every request in
    # it retried through the loop (and only the poisoned one errored).
    assert vocoder.batched_calls == []

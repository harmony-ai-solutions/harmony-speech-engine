"""Engine-level mixed-length batched synthesis (Phase 5-2), embedding (8-2)
and tone conversion (8-3).

Four requests with very different text lengths go through the real engine,
scheduler and OpenVoice-family handler in as few engine steps as the batch
configuration allows; the assertions cover the end-to-end contract of the padded
path: every request gets exactly one result, each result carries its own
(audio length consistent) waveform, and a pathological length mix is sub-batched
instead of padding everything to the longest text.

Phase 8-2 adds the embed step of the clone pipeline: four clone requests with
distinct references must come back with their own embeddings, equal to the
per-item loop's values.

Phase 8-3 adds the tone-conversion step: four clone requests with distinct source
audio and target speakers must come back with their own converted audio, in one
batched forward, with the per-request output length derived from the model's own
frame counts.
"""

import types
from unittest.mock import patch

import numpy as np
import pytest
import torch

from harmonyspeech.common.artifacts import AudioArtifact, SpeakerArtifact
from harmonyspeech.common.config import DeviceConfig, ModelConfig
from harmonyspeech.common.inputs import (
    SpeechEmbeddingRequestInput,
    TextToSpeechRequestInput,
    VoiceConversionRequestInput,
)
from harmonyspeech.common.outputs import SpeechSynthesisRequestOutput
from harmonyspeech.common.request import EngineRequest
from harmonyspeech.engine.harmonyspeech_engine import HarmonySpeechEngine
from harmonyspeech.modeling.models.openvoice.openvoice import ReferenceEncoder
from harmonyspeech.task_handler.model_runner_base import ModelRunnerBase

HOP = 8
SAMPLE_RATE = 200
GAP = int(SAMPLE_RATE * 0.05)
SAMPLING = {"noise_scale": 0.667, "noise_scale_w": 0.6, "sdp_ratio": 0.2}
FRAMES_PER_TOKEN = 2


class LengthStubModel(torch.nn.Module):
    """One-shot synthesizer stand-in: frames are a function of the input length."""

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Linear(1, 1)
        self.batched_calls: list[list[int]] = []
        self.loop_calls: list[int] = []

    def infer(self, x, x_lengths, sid=None, **kwargs):
        tokens = int(x_lengths[0])
        self.loop_calls.append(tokens)
        return (torch.zeros(1, 1, tokens * FRAMES_PER_TOKEN * HOP), None, None, None)

    def infer_batched(self, x, x_lengths, sid=None, **kwargs):
        lengths = [int(value) for value in x_lengths]
        self.batched_calls.append(lengths)
        y_lengths = x_lengths * FRAMES_PER_TOKEN
        output = torch.zeros(len(lengths), 1, max(lengths) * FRAMES_PER_TOKEN * HOP)
        return output, y_lengths, None, None, None


class StubExecutor:
    def __init__(self, runner):
        self.runner = runner

    def execute_model(self, requests_to_batch):
        return self.runner.execute_model(requests_to_batch)

    async def execute_model_async(self, requests_to_batch):
        return self.runner.execute_model(requests_to_batch)

    def check_health(self):
        return {"status": "ok"}


def _fake_hf_config():
    import types

    return types.SimpleNamespace(
        data=types.SimpleNamespace(sampling_rate=SAMPLE_RATE, hop_length=HOP),
        speakers={"default": 0},
    )


def _request(request_id: str, text: str) -> EngineRequest:
    request_data = TextToSpeechRequestInput(
        request_id=request_id,
        requested_model="ov1",
        model="ov1",
        input_text=text,
        mode="single_speaker_tts",
    )
    return EngineRequest(request_id=request_id, request_data=request_data, arrival_time=0.0)


def _prepare_for_lengths(model_config, requests_to_batch):
    """Prepare stand-in inputs whose token count is the request's text length."""

    def tokens(text: str) -> int:
        return len(text) * 5

    return [
        ([torch.arange(1, tokens(request.request_data.input_text) + 1, dtype=torch.long)], 0, 1.0, dict(SAMPLING))
        for request in requests_to_batch
    ]


@pytest.mark.integration
def test_four_mixed_length_requests_produce_four_correct_length_outputs():
    model = LengthStubModel()
    runner = ModelRunnerBase.__new__(ModelRunnerBase)
    runner.model = model
    runner.device = "cpu"
    runner._baseline_state = None
    runner.model_config = ModelConfig(
        name="ov1",
        model="dummy/model",
        model_type="OpenVoiceV1Synthesizer",
        max_batch_size=4,
        device_config=DeviceConfig("cpu"),
        dtype="float32",
    )

    with patch("harmonyspeech.executor.cpu_executor.CPUExecutor._init_worker"):
        engine = HarmonySpeechEngine(model_configs=[runner.model_config], log_stats=False)
    engine.model_executors = {"ov1": StubExecutor(runner)}

    texts = {
        "mix-long": "a" * 40,  # 200 tokens
        "mix-medium": "b" * 20,  # 100 tokens
        "mix-short": "c" * 8,  # 40 tokens
        "mix-tiny": "d" * 4,  # 20 tokens
    }
    for request_id, text in texts.items():
        engine.add_request(request_id, _request(request_id, text).request_data)

    with patch(
        "harmonyspeech.task_handler.model_runner_base.prepare_inputs", side_effect=_prepare_for_lengths
    ), patch(
        "harmonyspeech.task_handler.model_runner_base.get_model_config", return_value=_fake_hf_config()
    ), patch("harmonyspeech.task_handler.model_runner_base.get_model_flavour", return_value="default"):
        collected: dict[str, list] = {}
        for _ in range(8):
            if not engine.has_unfinished_requests():
                break
            outputs, _forwarded = engine.step()
            for output in outputs:
                collected.setdefault(output.request_id, []).append(output)

    assert sorted(collected) == sorted(texts)
    for request_id, text in texts.items():
        assert len(collected[request_id]) == 1
        result = collected[request_id][0]
        assert isinstance(result, SpeechSynthesisRequestOutput)
        assert result.finish_reason == "stop"
        expected_samples = len(text) * 5 * FRAMES_PER_TOKEN * HOP + GAP
        assert result.audio_artifact.waveform.shape[-1] == expected_samples

    # The long text cannot share a padded batch with the tiny one
    # (T_max / T_min = 10 > 4), so the step is split into length-similar groups;
    # every request still went through the batched forward exactly once.
    assert sum(len(call) for call in model.batched_calls) == 4
    assert len(model.batched_calls) >= 2
    for call in model.batched_calls:
        assert max(call) / min(call) <= 4
    assert model.loop_calls == []


@pytest.mark.integration
def test_two_similar_length_requests_share_one_padded_forward():
    model = LengthStubModel()
    runner = ModelRunnerBase.__new__(ModelRunnerBase)
    runner.model = model
    runner.device = "cpu"
    runner._baseline_state = None
    runner.model_config = ModelConfig(
        name="ov1",
        model="dummy/model",
        model_type="OpenVoiceV1Synthesizer",
        max_batch_size=4,
        device_config=DeviceConfig("cpu"),
        dtype="float32",
    )

    with patch("harmonyspeech.executor.cpu_executor.CPUExecutor._init_worker"):
        engine = HarmonySpeechEngine(model_configs=[runner.model_config], log_stats=False)
    engine.model_executors = {"ov1": StubExecutor(runner)}

    for request_id, text in (("pa", "a" * 10), ("pb", "b" * 11)):
        engine.add_request(request_id, _request(request_id, text).request_data)

    with patch(
        "harmonyspeech.task_handler.model_runner_base.prepare_inputs", side_effect=_prepare_for_lengths
    ), patch(
        "harmonyspeech.task_handler.model_runner_base.get_model_config", return_value=_fake_hf_config()
    ), patch("harmonyspeech.task_handler.model_runner_base.get_model_flavour", return_value="default"):
        outputs = []
        for _ in range(4):
            if not engine.has_unfinished_requests():
                break
            step_outputs, _ = engine.step()
            outputs.extend(step_outputs)

    assert len(outputs) == 2
    assert model.batched_calls == [[50, 55]]


# ---------------------------------------------------------------------------
# Phase 8-2: the clone pipeline's embed step (OpenVoice V2 tone-converter
# encoder) batches a flood of distinct references.
# ---------------------------------------------------------------------------

ENC_SAMPLE_RATE = 16000
ENC_FILTER_LENGTH = 16
ENC_HOP_LENGTH = 4
ENC_WIN_LENGTH = 16
ENC_GIN_CHANNELS = 6


def _fake_encoder_hf_config():
    return types.SimpleNamespace(
        data=types.SimpleNamespace(
            sampling_rate=ENC_SAMPLE_RATE,
            filter_length=ENC_FILTER_LENGTH,
            hop_length=ENC_HOP_LENGTH,
            win_length=ENC_WIN_LENGTH,
        )
    )


def _tiny_reference_encoder():
    torch.manual_seed(0)
    return ReferenceEncoder(
        spec_channels=ENC_FILTER_LENGTH // 2 + 1, gin_channels=ENC_GIN_CHANNELS
    ).eval()


def _reference_clip(samples: int, freq: float) -> AudioArtifact:
    t = np.arange(samples) / ENC_SAMPLE_RATE
    return AudioArtifact.from_numpy((0.1 * np.sin(2 * np.pi * freq * t)).astype(np.float32), ENC_SAMPLE_RATE)


def _segments_from_request(request) -> list:
    """Stand-in for the real VAD-sliced prepare: one or two segments per request.

    The distinct reference clips make every request's segment set unique, so a
    pairing regression shows up as a swapped embedding rather than a tie.
    """
    artifact = request.artifacts["input_audio"]
    if artifact.num_samples >= 2800:
        return [artifact, artifact]
    return [artifact]


def _run_encoder_engine(references: dict[str, object], batched: bool) -> dict[str, object]:
    runner = ModelRunnerBase.__new__(ModelRunnerBase)
    runner.model = types.SimpleNamespace(ref_enc=_tiny_reference_encoder())
    runner.device = "cpu"
    runner._baseline_state = None
    runner.model_config = ModelConfig(
        name="ov2-encoder",
        model="dummy/model",
        model_type="OpenVoiceV2ToneConverterEncoder",
        max_batch_size=4,
        device_config=DeviceConfig("cpu"),
        dtype="float32",
        batched_inference=batched,
        batch_token_budget=1_000_000,
    )

    with patch("harmonyspeech.executor.cpu_executor.CPUExecutor._init_worker"):
        engine = HarmonySpeechEngine(model_configs=[runner.model_config], log_stats=False)
    engine.model_executors = {"ov2-encoder": StubExecutor(runner)}

    for request_id, artifact in references.items():
        request_data = SpeechEmbeddingRequestInput(
            request_id=request_id, requested_model="ov2-encoder", model="ov2-encoder"
        )
        engine.add_request(request_id, request_data, artifacts={"input_audio": artifact})

    collected: dict[str, object] = {}
    with patch(
        "harmonyspeech.task_handler.model_runner_base.prepare_inputs", side_effect=_prepare_clone_segments
    ), patch(
        "harmonyspeech.task_handler.model_runner_base.get_model_config",
        return_value=_fake_encoder_hf_config(),
    ), patch("harmonyspeech.task_handler.model_runner_base.get_model_flavour", return_value="default"):
        for _ in range(8):
            if not engine.has_unfinished_requests():
                break
            outputs, _forwarded = engine.step()
            for output in outputs:
                collected[output.request_id] = output
    return collected


def _prepare_clone_segments(model_config, requests_to_batch):
    return [_segments_from_request(request) for request in requests_to_batch]


@pytest.mark.integration
def test_four_clone_requests_with_distinct_references_pair_their_embeddings():
    """A four-reference flood: one batched embed step must pair every embedding.

    The batched engine run and the per-item loop run use the same tiny
    ``ReferenceEncoder`` weights, so the embeddings must agree per request id;
    distinct references keep a swapped pairing from passing by accident.
    """
    references = {
        "clone-a": _reference_clip(2000, freq=220.0),
        "clone-b": _reference_clip(2400, freq=330.0),
        "clone-c": _reference_clip(2800, freq=440.0),
        "clone-d": _reference_clip(3200, freq=550.0),
    }

    batched = _run_encoder_engine(references, batched=True)
    looped = _run_encoder_engine(references, batched=False)

    assert sorted(batched) == sorted(looped) == sorted(references)
    for request_id in references:
        assert batched[request_id].finish_reason == "stop", request_id
        assert looped[request_id].finish_reason == "stop", request_id
        batched_embedding = batched[request_id].embedding_artifact.embedding
        loop_embedding = looped[request_id].embedding_artifact.embedding
        assert batched_embedding.shape == loop_embedding.shape == (1, ENC_GIN_CHANNELS, 1)
        assert torch.allclose(batched_embedding, loop_embedding, atol=1e-6), request_id

    embeddings = [batched[request_id].embedding_artifact.embedding for request_id in references]
    for left in range(len(embeddings)):
        for right in range(left + 1, len(embeddings)):
            assert not torch.allclose(embeddings[left], embeddings[right]), (left, right)


# ---------------------------------------------------------------------------
# Phase 8-3: the clone pipeline's tone-conversion step (OpenVoice V2 tone
# converter) batches a flood of distinct source clips + target speakers.
# ---------------------------------------------------------------------------

TC_SAMPLE_RATE = 16000
TC_FILTER_LENGTH = 16
TC_HOP_LENGTH = 4
TC_WIN_LENGTH = 16
TC_GIN_CHANNELS = 4
#: prod(upsample_rates) of the stub converter's decoder.
TC_SAMPLES_PER_FRAME = 4


def _fake_tc_hf_config():
    return types.SimpleNamespace(
        data=types.SimpleNamespace(
            sampling_rate=TC_SAMPLE_RATE,
            filter_length=TC_FILTER_LENGTH,
            hop_length=TC_HOP_LENGTH,
            win_length=TC_WIN_LENGTH,
        )
    )


def _tc_clip(samples: int, freq: float) -> AudioArtifact:
    t = np.arange(samples) / TC_SAMPLE_RATE
    return AudioArtifact.from_numpy((0.1 * np.sin(2 * np.pi * freq * t)).astype(np.float32), TC_SAMPLE_RATE)


class ToneConverterStubModel:
    """One-shot converter stand-in: markers prove pairing, lengths prove unpadding.

    The output's sample value is the request's source-speaker marker and its
    length is the decoder's ``frames * samples_per_frame``, so a swapped row or a
    padded-shape unpad shows up immediately. Using a stub (rather than the real
    stochastic converter) keeps the batched and loop runs exactly comparable at
    the engine level; the vendored math is covered by
    ``tests/unit/modeling/test_batched_tone_converter.py``.
    """

    def __init__(self):
        self.dec = types.SimpleNamespace(upsample_rates=[TC_SAMPLES_PER_FRAME])
        self.batched_calls: list[list[int]] = []
        self.loop_calls: list[int] = []

    @staticmethod
    def _render(frames: int, marker: float) -> torch.Tensor:
        return torch.full((1, 1, int(frames) * TC_SAMPLES_PER_FRAME), marker)

    def voice_conversion(self, y, y_lengths, sid_src, sid_tgt, tau=1.0):
        frames = int(y_lengths[0])
        self.loop_calls.append(frames)
        return self._render(frames, float(sid_src[0, 0, 0])), torch.ones(1, 1, frames), None

    def voice_conversion_batched(self, y, y_lengths, sid_src, sid_tgt, tau=1.0):
        frames = [int(value) for value in y_lengths]
        self.batched_calls.append(frames)
        output = torch.zeros(len(frames), 1, max(frames) * TC_SAMPLES_PER_FRAME)
        for row, count in enumerate(frames):
            output[row, 0, : count * TC_SAMPLES_PER_FRAME] = float(sid_src[row, 0, 0])
        return output, torch.ones(len(frames), 1, max(frames)), None


def _tc_request(request_id: str, model: str = "ov2-tone-converter") -> EngineRequest:
    request_data = VoiceConversionRequestInput(
        request_id=request_id,
        requested_model=model,
        model=model,
        source_audio="",
        target_audio=None,
        target_embedding="",
        generation_options=None,
        output_options=None,
    )
    return EngineRequest(request_id=request_id, request_data=request_data, arrival_time=0.0)


def _prepare_tc_triples(clips: dict[str, AudioArtifact]):
    """Stand-in for ``prepare_openvoice_tone_converter_inputs``: artifact triples.

    The source speaker's marker doubles as the expected output value, and the
    target speaker is distinct so a target/source mix-up also fails.
    """

    def prepare(model_config, requests_to_batch):
        triples = []
        for index, request in enumerate(requests_to_batch):
            clip = clips[request.request_id]
            triples.append(
                (
                    clip,
                    SpeakerArtifact(
                        embedding=torch.full((1, TC_GIN_CHANNELS, 1), 10.0 + index), backend="openvoice"
                    ),
                    SpeakerArtifact(
                        embedding=torch.full((1, TC_GIN_CHANNELS, 1), float(index + 1)), backend="openvoice"
                    ),
                )
            )
        return triples

    return prepare


def _run_tone_converter_engine(clips: dict[str, AudioArtifact], batched: bool) -> tuple[dict, object]:
    model = ToneConverterStubModel()
    runner = ModelRunnerBase.__new__(ModelRunnerBase)
    runner.model = model
    runner.device = "cpu"
    runner._baseline_state = None
    runner.model_config = ModelConfig(
        name="ov2-tone-converter",
        model="dummy/model",
        model_type="OpenVoiceV2ToneConverter",
        max_batch_size=4,
        device_config=DeviceConfig("cpu"),
        dtype="float32",
        batched_inference=batched,
        batch_token_budget=1_000_000,
    )

    with patch("harmonyspeech.executor.cpu_executor.CPUExecutor._init_worker"):
        engine = HarmonySpeechEngine(model_configs=[runner.model_config], log_stats=False)
    engine.model_executors = {"ov2-tone-converter": StubExecutor(runner)}

    for request_id in clips:
        engine.add_request(request_id, _tc_request(request_id).request_data)

    collected: dict[str, object] = {}
    with patch(
        "harmonyspeech.task_handler.model_runner_base.prepare_inputs",
        side_effect=_prepare_tc_triples(clips),
    ), patch(
        "harmonyspeech.task_handler.model_runner_base.get_model_config",
        return_value=_fake_tc_hf_config(),
    ), patch("harmonyspeech.task_handler.model_runner_base.get_model_flavour", return_value="default"):
        for _ in range(8):
            if not engine.has_unfinished_requests():
                break
            outputs, _forwarded = engine.step()
            for output in outputs:
                collected[output.request_id] = output
    return collected, model


@pytest.mark.integration
def test_four_clone_requests_flow_through_one_batched_tone_conversion():
    """A four-request flood: one batched TC forward, per-request audio and lengths.

    Mel frames at this geometry (``filter_length=16``, ``hop_length=4``) are
    ``samples // 4``, so with a ``4``-samples-per-frame decoder each converted
    clip is exactly as long as its source clip.
    """
    clips = {
        "clone-a": _tc_clip(2000, freq=220.0),
        "clone-b": _tc_clip(2400, freq=330.0),
        "clone-c": _tc_clip(2800, freq=440.0),
        "clone-d": _tc_clip(3200, freq=550.0),
    }
    expected_frames = {"clone-a": 500, "clone-b": 600, "clone-c": 700, "clone-d": 800}

    batched, batched_model = _run_tone_converter_engine(clips, batched=True)
    looped, looped_model = _run_tone_converter_engine(clips, batched=False)

    assert sorted(batched) == sorted(looped) == sorted(clips)
    assert batched_model.batched_calls == [[500, 600, 700, 800]]
    assert batched_model.loop_calls == []
    assert looped_model.batched_calls == []
    assert looped_model.loop_calls == [500, 600, 700, 800]

    markers = {request_id: float(index + 1) for index, request_id in enumerate(clips)}
    for request_id, frames in expected_frames.items():
        assert batched[request_id].finish_reason == "stop", request_id
        assert looped[request_id].finish_reason == "stop", request_id
        batched_audio = batched[request_id].audio_artifact
        loop_audio = looped[request_id].audio_artifact
        # Frame-domain unpad: the model's own frame count * samples per frame.
        assert batched_audio.num_samples == frames * TC_SAMPLES_PER_FRAME, request_id
        assert loop_audio.num_samples == frames * TC_SAMPLES_PER_FRAME, request_id
        # Pairing: the converted clip carries this request's source-speaker marker.
        assert np.allclose(batched_audio.numpy, markers[request_id]), request_id
        assert np.array_equal(batched_audio.numpy, loop_audio.numpy), request_id
        assert batched_audio.sample_rate == loop_audio.sample_rate == TC_SAMPLE_RATE


# ---------------------------------------------------------------------------
# Phase 8-4: the HarmonySpeech vocoder step and the VoiceFixer restorer step
# batch a four-request flood (the synthesizer stays a documented per-item loop).
# ---------------------------------------------------------------------------

HS_VOC_SAMPLES_PER_FRAME = 4
HS_VOC_BREAK_SILENCE = int(0.15 * 16000)


def _tiny_harmonyspeech_vocoder():
    from harmonyspeech.modeling.models.harmonyspeech.harmonyspeech import MelGANGenerator

    torch.manual_seed(0)
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


class _RecordingVocoderModel:
    """Real vocoder plus per-entry-point call log (proves the batch happened)."""

    def __init__(self, model):
        self.model = model
        self.batched_calls: list[list[int]] = []
        self.loop_calls: list[int] = []

    def inference(self, c, **kwargs):
        self.loop_calls.append(int(c.shape[0]))
        return self.model.inference(c, **kwargs)

    def inference_batched(self, c, frame_lengths=None, **kwargs):
        self.batched_calls.append([int(value) for value in frame_lengths])
        return self.model.inference_batched(c, frame_lengths, **kwargs)


def _hs_vocoder_mel(request_id: str, frames: int) -> "object":
    from harmonyspeech.common.artifacts import MelArtifact

    t = np.arange(4 * frames, dtype=np.float32) / 7.0 + float(len(request_id))
    return MelArtifact.from_numpy(
        (0.1 * np.sin(t)).reshape(1, 4, frames).astype(np.float32),
        sample_rate=16000,
        hop_length=200,
    )


def _run_hs_vocoder_engine(mels: dict[str, object], breaks: dict[str, list[int] | None], batched: bool):
    model = _RecordingVocoderModel(_tiny_harmonyspeech_vocoder())
    runner = ModelRunnerBase.__new__(ModelRunnerBase)
    runner.model = model
    runner.device = "cpu"
    runner._baseline_state = None
    runner.embedding_cache = None
    runner.model_identity = None
    runner.model_config = ModelConfig(
        name="hs1-vocoder",
        model="dummy/model",
        model_type="HarmonySpeechVocoder",
        max_batch_size=4,
        device_config=DeviceConfig("cpu"),
        dtype="float32",
        batched_inference=batched,
        batch_token_budget=1_000_000,
    )

    with patch("harmonyspeech.executor.cpu_executor.CPUExecutor._init_worker"):
        engine = HarmonySpeechEngine(model_configs=[runner.model_config], log_stats=False)
    engine.model_executors = {"hs1-vocoder": StubExecutor(runner)}

    for request_id in mels:
        request_data = TextToSpeechRequestInput(
            request_id=request_id,
            requested_model="hs1-vocoder",
            model="hs1-vocoder",
            input_text="hello",
            mode="single_speaker_tts",
        )
        engine.add_request(request_id, request_data)

    def prepare(model_config, requests_to_batch):
        return [(mels[request.request_id], breaks.get(request.request_id)) for request in requests_to_batch]

    collected: dict[str, object] = {}
    with patch("harmonyspeech.task_handler.model_runner_base.prepare_inputs", side_effect=prepare):
        for _ in range(8):
            if not engine.has_unfinished_requests():
                break
            outputs, _forwarded = engine.step()
            for output in outputs:
                collected[output.request_id] = output
    return collected, model


@pytest.mark.integration
def test_four_mixed_length_vocoder_requests_pair_their_breaks_and_audio():
    """hs1 vocoder: four padded rows, per-request breaks, loop-equal audio."""
    frames = {"voc-a": 5, "voc-b": 9, "voc-c": 6, "voc-d": 11}
    mels = {request_id: _hs_vocoder_mel(request_id, length) for request_id, length in frames.items()}
    breaks = {"voc-a": None, "voc-b": [1], "voc-c": None, "voc-d": None}

    batched, batched_model = _run_hs_vocoder_engine(mels, breaks, batched=True)
    looped, looped_model = _run_hs_vocoder_engine(mels, breaks, batched=False)

    assert sorted(batched) == sorted(looped) == sorted(frames)
    assert batched_model.batched_calls == [list(frames.values())]
    assert batched_model.loop_calls == []
    assert looped_model.batched_calls == []
    assert sorted(looped_model.loop_calls) == sorted(frames.values())

    for request_id, length in frames.items():
        assert batched[request_id].finish_reason == "stop", request_id
        assert looped[request_id].finish_reason == "stop", request_id
        produced = batched[request_id].audio_artifact
        expected = looped[request_id].audio_artifact
        silence = HS_VOC_BREAK_SILENCE if breaks[request_id] else 0
        assert produced.num_samples == length * HS_VOC_SAMPLES_PER_FRAME + silence, request_id
        assert expected.num_samples == produced.num_samples, request_id
        assert np.allclose(produced.numpy, expected.numpy, atol=1e-6), request_id
        assert produced.sample_rate == expected.sample_rate == 16000


class _RecordingRestorerModel:
    def __init__(self, model):
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


def _run_voicefixer_restorer_engine(clips: dict[str, "object"], batched: bool):
    from harmonyspeech.modeling.models.voicefixer.voicefixer import VoiceFixerRestorer

    # Same random weights on both sides so the loop is a valid reference.
    torch.manual_seed(0)
    model = _RecordingRestorerModel(VoiceFixerRestorer().eval())
    runner = ModelRunnerBase.__new__(ModelRunnerBase)
    runner.model = model
    runner.device = "cpu"
    runner._baseline_state = None
    runner.embedding_cache = None
    runner.model_identity = None
    runner.model_config = ModelConfig(
        name="vf-restorer",
        model="dummy/model",
        model_type="VoiceFixerRestorer",
        max_batch_size=4,
        device_config=DeviceConfig("cpu"),
        dtype="float32",
        batched_inference=batched,
        batch_token_budget=1_000_000,
    )

    with patch("harmonyspeech.executor.cpu_executor.CPUExecutor._init_worker"):
        engine = HarmonySpeechEngine(model_configs=[runner.model_config], log_stats=False)
    engine.model_executors = {"vf-restorer": StubExecutor(runner)}

    for request_id in clips:
        from harmonyspeech.common.inputs import AudioConversionRequestInput

        request_data = AudioConversionRequestInput(
            request_id=request_id, requested_model="vf-restorer", model="vf-restorer", source_audio=""
        )
        engine.add_request(request_id, request_data)

    def prepare(model_config, requests_to_batch):
        return [clips[request.request_id] for request in requests_to_batch]

    collected: dict[str, object] = {}
    with patch("harmonyspeech.task_handler.model_runner_base.prepare_inputs", side_effect=prepare):
        for _ in range(8):
            if not engine.has_unfinished_requests():
                break
            outputs, _forwarded = engine.step()
            for output in outputs:
                collected[output.request_id] = output
    return collected, model


@pytest.mark.integration
def test_four_voicefixer_restoration_requests_pair_their_own_mels():
    """VoiceFixer restorer: four padded rows, one per request, own mel lengths."""
    sample_counts = {"restore-a": 4096, "restore-b": 6000, "restore-c": 5000, "restore-d": 7000}
    clips = {}
    for index, (request_id, samples) in enumerate(sample_counts.items()):
        t = np.arange(samples) / 44100.0
        clips[request_id] = AudioArtifact.from_numpy(
            (0.1 * np.sin(2 * np.pi * (220.0 + 110.0 * index) * t)).astype(np.float32), 44100
        )

    batched, batched_model = _run_voicefixer_restorer_engine(clips, batched=True)
    looped, looped_model = _run_voicefixer_restorer_engine(clips, batched=False)

    assert sorted(batched) == sorted(looped) == sorted(clips)
    assert len(batched_model.batched_calls) == 1 and len(batched_model.batched_calls[0]) == 4
    assert batched_model.loop_calls == []
    assert looped_model.batched_calls == []

    shapes = set()
    for request_id in clips:
        assert batched[request_id].finish_reason == "stop", request_id
        assert looped[request_id].finish_reason == "stop", request_id
        produced = batched[request_id].mel_artifact
        expected = looped[request_id].mel_artifact
        # [1, bins, frames] with the request's own frame count.
        assert produced.mel.shape == expected.mel.shape, request_id
        assert produced.mel.shape[0] == 1 and produced.mel.shape[1] == 128, request_id
        shapes.add(produced.mel.shape[-1])
        delta = (produced.mel - expected.mel).abs().max()
        assert delta <= 1e-4 * expected.mel.abs().max().clamp(min=1e-6), request_id
        assert produced.sample_rate == expected.sample_rate == 44100

    # Distinct clip lengths keep a swapped pairing from passing by accident.
    assert len(shapes) == 4


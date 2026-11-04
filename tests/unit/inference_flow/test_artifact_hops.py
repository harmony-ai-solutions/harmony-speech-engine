"""Unit tests for artifact hops through the executor/forwarding layer (Phase 2-3)."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from harmonyspeech.common.artifacts import AudioArtifact, MelArtifact, SpeakerArtifact, VadArtifact
from harmonyspeech.common.config import DeviceConfig, ModelConfig
from harmonyspeech.common.inputs import (
    AudioConversionRequestInput,
    SpeechEmbeddingRequestInput,
    TextToSpeechRequestInput,
)
from harmonyspeech.common.outputs import (
    AudioConversionRequestOutput,
    SpeechEmbeddingRequestOutput,
    SpeechSynthesisRequestOutput,
    SpeechTranscriptionRequestOutput,
    TextToSpeechRequestOutput,
    VocodeRequestOutput,
)
from harmonyspeech.common.request import EngineRequest, ExecutorResult, RequestStatus
from harmonyspeech.engine.harmonyspeech_engine import HarmonySpeechEngine
from harmonyspeech.task_handler import model_runner_base as runner_module
from harmonyspeech.task_handler.model_runner_base import ModelRunnerBase, _bind_to_device


def _artifact(seconds: float = 0.1, sample_rate: int = 16000) -> AudioArtifact:
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    return AudioArtifact.from_numpy((0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sample_rate)


def _cfg(name: str, model_type: str) -> SimpleNamespace:
    return SimpleNamespace(name=name, model_type=model_type, language=None)


class _FakeScheduler:
    def __init__(self, requests=()):
        self.waiting = []
        self.running = list(requests)
        self.updates = []

    def update_request_status(self, request_id, status):
        self.updates.append((request_id, status))


def _engine_shell(*configs, requests=()):
    engine = HarmonySpeechEngine.__new__(HarmonySpeechEngine)
    engine.model_configs = list(configs)
    engine.scheduler = _FakeScheduler(requests)
    engine.add_request = MagicMock()
    return engine


def _engine_request(request_data, artifacts=None) -> EngineRequest:
    return EngineRequest(
        request_id="hop-1",
        request_data=request_data,
        arrival_time=0.0,
        artifacts=artifacts,
    )


# ---------------------------------------------------------------------------
# check_forward_processing: artifacts travel, no re-encoding
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_embedding_forward_carries_speaker_artifact_by_reference():
    conditionals = SimpleNamespace(gen={"prompt_token": torch.zeros(1, 2, dtype=torch.long)})
    speaker = SpeakerArtifact(conditionals=conditionals, backend="chatterbox")
    input_data = TextToSpeechRequestInput(
        request_id="hop-1",
        requested_model="chatterbox",
        model="cb-embed",
        input_text="hello",
        mode="voice_cloning",
        input_audio="legacy-audio",
    )
    engine_request = _engine_request(input_data, {"input_audio": _artifact()})
    engine = _engine_shell(_cfg("cb-embed", "ChatterboxEmbedding"), _cfg("cb-tts", "ChatterboxTTS"), requests=[engine_request])

    result = ExecutorResult(
        request_id="hop-1",
        input_data=input_data,
        result_data=SpeechEmbeddingRequestOutput(
            request_id="hop-1", output="legacy-embedding", embedding_artifact=speaker
        ),
    )

    status, forwarding = engine.check_forward_processing(result)

    assert status == RequestStatus.FINISHED_FORWARDED
    assert forwarding is input_data
    _, kwargs = engine.add_request.call_args
    forwarded = kwargs["artifacts"]
    assert forwarded["input_embedding"] is speaker
    assert "input_audio" not in forwarded  # cleared for the synth step
    # No legacy base64 field is written on the internal hop.
    assert input_data.input_embedding is None
    assert input_data.input_audio is None


@pytest.mark.unit
def test_embedding_forward_falls_back_to_legacy_string():
    input_data = TextToSpeechRequestInput(
        request_id="hop-2",
        requested_model="chatterbox",
        model="cb-embed",
        input_text="hello",
        mode="voice_cloning",
        input_audio="legacy-audio",
    )
    engine_request = _engine_request(input_data)
    engine = _engine_shell(_cfg("cb-embed", "ChatterboxEmbedding"), _cfg("cb-tts", "ChatterboxTTS"), requests=[engine_request])

    result = ExecutorResult(
        request_id="hop-2",
        input_data=input_data,
        result_data=SpeechEmbeddingRequestOutput(request_id="hop-2", output="legacy-embedding"),
    )

    engine.check_forward_processing(result)

    assert input_data.input_embedding == "legacy-embedding"


@pytest.mark.unit
def test_vad_forward_carries_vad_artifact():
    vad = VadArtifact(segments=[{"start": 0.0, "end": 1.0, "text": "hi"}], info={"language": "en"})
    payload = json.dumps({"text": "hi", "segments": vad.segments, "info": vad.info})
    input_data = SpeechEmbeddingRequestInput(
        request_id="hop-3",
        requested_model="openvoice_v2",
        model="whisper",
        input_audio="legacy-audio",
    )
    engine_request = _engine_request(input_data, {"input_audio": _artifact()})
    engine = _engine_shell(
        _cfg("whisper", "FasterWhisper"),
        _cfg("tce", "OpenVoiceV2ToneConverterEncoder"),
        requests=[engine_request],
    )

    result = ExecutorResult(
        request_id="hop-3",
        input_data=input_data,
        result_data=SpeechTranscriptionRequestOutput(
            request_id="hop-3", output=payload, vad_artifact=vad
        ),
    )

    status, _ = engine.check_forward_processing(result)

    assert status == RequestStatus.FINISHED_FORWARDED
    _, kwargs = engine.add_request.call_args
    assert kwargs["artifacts"]["input_vad_data"] is vad
    # JSON text stays the routing signal until Phase 3.
    assert input_data.input_vad_data == payload


@pytest.mark.unit
def test_synthesis_forward_carries_audio_artifact():
    audio = _artifact(seconds=0.2, sample_rate=22050)
    input_data = TextToSpeechRequestInput(
        request_id="hop-4",
        requested_model="openvoice_v2",
        model="ov2-synth",
        input_text="hello",
        mode="voice_cloning",
        input_embedding="legacy-embedding",
    )
    engine_request = _engine_request(input_data)
    engine = _engine_shell(
        _cfg("ov2-synth", "MeloTTSSynthesizer"),
        _cfg("ov2-tc", "OpenVoiceV2ToneConverter"),
        requests=[engine_request],
    )

    result = ExecutorResult(
        request_id="hop-4",
        input_data=input_data,
        result_data=SpeechSynthesisRequestOutput(request_id="hop-4", output="", audio_artifact=audio),
    )

    engine.check_forward_processing(result)

    _, kwargs = engine.add_request.call_args
    assert kwargs["artifacts"]["input_audio"] is audio
    assert input_data.input_audio is None


@pytest.mark.unit
def test_harmonyspeech_synthesis_forward_carries_mel_artifact():
    mel = MelArtifact.from_numpy(np.zeros((80, 5), dtype=np.float32), 16000, 200, breaks=[5])
    input_data = TextToSpeechRequestInput(
        request_id="hop-5",
        requested_model="harmonyspeech",
        model="hs-synth",
        input_text="hello",
        mode="voice_cloning",
        input_embedding="legacy-embedding",
    )
    engine_request = _engine_request(input_data)
    engine = _engine_shell(
        _cfg("hs-synth", "HarmonySpeechSynthesizer"),
        _cfg("hs-voc", "HarmonySpeechVocoder"),
        requests=[engine_request],
    )

    result = ExecutorResult(
        request_id="hop-5",
        input_data=input_data,
        result_data=SpeechSynthesisRequestOutput(request_id="hop-5", output="", mel_artifact=mel),
    )

    engine.check_forward_processing(result)

    _, kwargs = engine.add_request.call_args
    assert kwargs["artifacts"]["input_audio"] is mel
    assert input_data.input_audio is None


@pytest.mark.unit
def test_voicefixer_forward_carries_mel_artifact():
    mel = MelArtifact.from_numpy(np.zeros((128, 9), dtype=np.float32), 44100, 441)
    input_data = AudioConversionRequestInput(
        request_id="hop-6",
        requested_model="voicefixer",
        model="vf-restore",
        source_audio="legacy-audio",
    )
    engine_request = _engine_request(input_data)
    engine = _engine_shell(
        _cfg("vf-restore", "VoiceFixerRestorer"),
        _cfg("vf-voc", "VoiceFixerVocoder"),
        requests=[engine_request],
    )

    result = ExecutorResult(
        request_id="hop-6",
        input_data=input_data,
        result_data=AudioConversionRequestOutput(request_id="hop-6", output="", mel_artifact=mel),
    )

    engine.check_forward_processing(result)

    _, kwargs = engine.add_request.call_args
    assert kwargs["artifacts"]["input_mel_spectrogram"] is mel
    assert input_data.input_mel_spectrogram is None


@pytest.mark.unit
def test_voicefixer_vocoder_completion_does_not_forward():
    audio = _artifact(seconds=0.1, sample_rate=44100)
    input_data = AudioConversionRequestInput(
        request_id="hop-7",
        requested_model="voicefixer",
        model="vf-voc",
        source_audio="legacy-audio",
    )
    engine = _engine_shell(
        _cfg("vf-restore", "VoiceFixerRestorer"),
        _cfg("vf-voc", "VoiceFixerVocoder"),
        requests=[_engine_request(input_data)],
    )

    result = ExecutorResult(
        request_id="hop-7",
        input_data=input_data,
        result_data=AudioConversionRequestOutput(request_id="hop-7", output="", audio_artifact=audio),
    )

    status, forwarding = engine.check_forward_processing(result)

    assert status == RequestStatus.FINISHED_STOPPED
    # The vocoder step is terminal; the request object is returned for counting
    # only (matching the pre-Phase-2 behaviour) and is never re-submitted.
    assert forwarding is input_data
    engine.add_request.assert_not_called()
    assert engine.scheduler.updates[-1][1] == RequestStatus.FINISHED_STOPPED


@pytest.mark.unit
def test_final_tts_result_keeps_audio_artifact():
    audio = _artifact()
    input_data = TextToSpeechRequestInput(
        request_id="hop-8",
        requested_model="harmonyspeech",
        model="hs-voc",
        input_text="hello",
        mode="voice_cloning",
    )
    engine = _engine_shell(
        _cfg("hs-voc", "HarmonySpeechVocoder"),
        _cfg("hs-synth", "HarmonySpeechSynthesizer"),
        requests=[_engine_request(input_data)],
    )

    result = ExecutorResult(
        request_id="hop-8",
        input_data=input_data,
        result_data=VocodeRequestOutput(request_id="hop-8", output="", audio_artifact=audio),
    )

    status, _ = engine.check_forward_processing(result)

    assert status == RequestStatus.FINISHED_STOPPED
    assert isinstance(result.result_data, TextToSpeechRequestOutput)
    assert result.result_data.audio_artifact is audio


# ---------------------------------------------------------------------------
# Handler hops: typed outputs, no JSON/base64 on the mel path
# ---------------------------------------------------------------------------


def _runner(device="cpu") -> ModelRunnerBase:
    runner = ModelRunnerBase.__new__(ModelRunnerBase)
    runner.device = device
    runner._baseline_state = None
    runner.model_config = ModelConfig(
        name="m",
        model="dummy",
        model_type="VoiceFixerRestorer",
        max_batch_size=1,
        device_config=DeviceConfig("cpu"),
        dtype="float32",
    )
    return runner


@pytest.mark.unit
def test_encoder_handler_produces_speaker_artifact(monkeypatch):
    runner = _runner()
    runner.model = lambda utterances: torch.ones(utterances.shape[0], 4)
    monkeypatch.setattr(runner_module, "preprocess_wav", lambda wav, source_sr=None: wav)
    monkeypatch.setattr(runner_module, "get_input_frames", lambda wav: np.zeros((2, 3, 40), dtype=np.float32))
    request = TextToSpeechRequestInput(
        request_id="enc-1",
        requested_model="harmonyspeech",
        model="m",
        input_text="hello",
        mode="voice_cloning",
    )
    engine_request = _engine_request(request)

    (result,) = runner._execute_harmonyspeech_encoder([_artifact()], [engine_request])

    output = result.result_data
    assert output.finish_reason == "stop"
    assert output.embedding_artifact is not None
    assert output.embedding_artifact.backend == "harmonyspeech"
    assert output.embedding_artifact.embedding.shape == (4,)
    assert output.output == ""  # no wire encoding on the internal hop


@pytest.mark.unit
def test_voicefixer_hop_never_serializes_mel(monkeypatch):
    runner = _runner()

    def restorer(audio_tensor):
        assert audio_tensor.shape == (1, 1, 4410)
        return torch.ones(1, 1, 50, 128)

    runner.model = restorer
    monkeypatch.setattr(runner_module.json, "dumps", MagicMock(side_effect=AssertionError("json used")))
    assert not hasattr(runner_module, "base64"), "model runner must not serialize audio"

    request = AudioConversionRequestInput(
        request_id="vf-1", requested_model="voicefixer", model="m", source_audio="x"
    )
    engine_request = _engine_request(request)

    (restored,) = runner._execute_voicefixer_restorer([_artifact(seconds=0.1, sample_rate=44100)], [engine_request])
    mel_artifact = restored.result_data.mel_artifact
    assert isinstance(mel_artifact, MelArtifact)
    assert mel_artifact.mel.shape == (1, 128, 50)

    def vocoder(mel_tensor):
        assert mel_tensor.shape == (1, 1, 50, 128)
        return torch.zeros(1, 1, 4410)

    runner.model = vocoder
    (vocoded,) = runner._execute_voicefixer_vocoder([mel_artifact], [engine_request])
    assert isinstance(vocoded.result_data.audio_artifact, AudioArtifact)
    assert vocoded.result_data.audio_artifact.sample_rate == 44100


@pytest.mark.unit
def test_chatterbox_tts_consumes_conditionals_by_reference():
    class Model:
        sr = 24000

        def __init__(self):
            self.conds = None
            self.conds_at_generate = []

        def generate(self, text=None, **kwargs):
            self.conds_at_generate.append(self.conds)
            return torch.zeros(1, 32)

    conditionals = SimpleNamespace(to=lambda device: conditionals)
    runner = _runner()
    runner.model = Model()
    text = "hello"
    inputs = [(text, SpeakerArtifact(conditionals=conditionals, backend="chatterbox"), 0.5, 0.5, 0.8, 1.2, 1.0, 0.05)]
    request = TextToSpeechRequestInput(
        request_id="cb-1",
        requested_model="chatterbox",
        model="m",
        input_text=text,
        mode="single_speaker_tts",
    )

    (result,) = runner._execute_chatterbox_tts(inputs, [_engine_request(request)])

    assert result.result_data.finish_reason == "stop"
    assert result.result_data.audio_artifact is not None
    # The exact conditionals object (not a pickle copy) was active during generate.
    assert runner.model.conds_at_generate == [conditionals]
    # ...and the per-request baseline is restored afterwards.
    assert runner.model.conds is None


# ---------------------------------------------------------------------------
# Device discipline
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_bind_to_device_requires_device_tag():
    with pytest.raises(AssertionError, match="missing its device tag"):
        _bind_to_device(SimpleNamespace(device=None), "cpu")


@pytest.mark.unit
def test_bind_to_device_is_noop_on_same_device():
    artifact = _artifact()
    assert _bind_to_device(artifact, "cpu") is artifact


@pytest.mark.unit
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cpu_artifact_moves_to_gpu_exactly_once():
    artifact = _artifact()
    moved = _bind_to_device(artifact, "cuda")

    assert moved.device.type == "cuda"
    assert artifact.device.type == "cpu"  # source artifact untouched
    assert _bind_to_device(moved, "cuda") is moved  # second bind is a no-op

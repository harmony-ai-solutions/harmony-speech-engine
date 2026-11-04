"""Unit tests for per-request Chatterbox state isolation (Phase 0-2).

The upstream library reads voice conditionals from ``model.conds`` /
``model.ref_dict``, so the engine assigns them per request. These tests prove
that every handler restores the baseline snapshot afterwards and that a request
without conditioning fails loudly instead of inheriting the previous voice.
"""

import base64
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from harmonyspeech.common import audio_codec
from harmonyspeech.common.artifacts import AudioArtifact, SpeakerArtifact
from harmonyspeech.common.config import DeviceConfig, ModelConfig
from harmonyspeech.common.inputs import TextToSpeechRequestInput, VoiceConversionRequestInput
from harmonyspeech.common.request import EngineRequest
from harmonyspeech.task_handler.model_runner_base import ModelRunnerBase
from tests.audio_fixtures import tiny_wav_bytes

#: Real WAV payload: the boundary decoder normalises reference audio.
CLONING_AUDIO = tiny_wav_bytes(16000, 0.10)
CLONING_AUDIO_B64 = base64.b64encode(CLONING_AUDIO).decode()


class StubChatterboxModel:
    """Minimal Chatterbox stand-in that records the state seen by generate()."""

    def __init__(self):
        self.conds = None
        self.ref_dict = None
        self.sr = 24000
        self.prepared_audio = []
        self.ref_audio = []
        self.source_audio = []
        self.generate_calls = []
        self.conds_at_generate = []
        self.ref_dict_at_generate = []

    def prepare_conditionals(self, audio_buf, **kwargs):
        audio, _ = audio_codec.decode_audio(audio_buf.read())
        self.prepared_audio.append(audio)
        # Opaque sentinel: what matters is that it differs from the baseline.
        self.conds = {"prepared_from": audio}

    def set_target_voice(self, audio_buf):
        audio, _ = audio_codec.decode_audio(audio_buf.read())
        self.ref_audio.append(audio)
        self.ref_dict = {"prompt_token": torch.zeros(1, 2, dtype=torch.long)}

    def generate(self, text=None, audio=None, **kwargs):
        if audio is not None:
            self.source_audio.append(audio_codec.decode_audio(audio.read())[0])
        self.generate_calls.append({"text": text, "audio": audio, "kwargs": kwargs})
        self.conds_at_generate.append(self.conds)
        self.ref_dict_at_generate.append(self.ref_dict)
        return torch.zeros(1, 64)


class FakeConditionals:
    """Stands in for ``chatterbox.tts.Conditionals`` in prepared inputs."""

    def __init__(self, marker="precomputed"):
        self.marker = marker
        self.gen = {"prompt_token": torch.zeros(1, 2, dtype=torch.long)}

    def to(self, device):
        return self


def _make_runner(model=None, model_type="ChatterboxTTS") -> ModelRunnerBase:
    runner = ModelRunnerBase.__new__(ModelRunnerBase)
    runner.device = "cpu"
    runner._baseline_state = None
    runner.model = model if model is not None else StubChatterboxModel()
    runner.model_config = ModelConfig(
        name="cb",
        model="dummy/chatterbox",
        model_type=model_type,
        max_batch_size=1,
        device_config=DeviceConfig("cpu"),
        dtype="float32",
    )
    return runner


def _tts_request(request_id="state-tts-001", input_audio=None, input_embedding=None) -> EngineRequest:
    request_data = TextToSpeechRequestInput(
        request_id=request_id,
        requested_model="chatterbox",
        model="cb",
        input_text="hello",
        mode="voice_cloning" if (input_audio or input_embedding) else "single_speaker_tts",
        input_audio=input_audio,
        input_embedding=input_embedding,
    )
    return EngineRequest(request_id=request_id, request_data=request_data, arrival_time=0.0)


def _speaker_for_audio(audio: bytes) -> SpeakerArtifact:
    waveform, sample_rate = audio_codec.decode_audio(audio)
    return SpeakerArtifact(
        reference_audio=AudioArtifact.from_numpy(waveform, sample_rate), backend="chatterbox"
    )


def _tts_params(speaker=None):
    return ("hello", speaker, 0.5, 0.5, 0.8, 1.2, 1.0, 0.05)


# ---------------------------------------------------------------------------
# TTS handlers restore the baseline
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_tts_restores_none_baseline_after_audio_cloning_request():
    runner = _make_runner()
    runner._capture_baseline_state()
    assert runner.model.conds is None

    results = runner._execute_chatterbox_tts(
        [_tts_params(_speaker_for_audio(CLONING_AUDIO))], [_tts_request(input_audio=CLONING_AUDIO_B64)]
    )

    assert results[0].result_data.finish_reason == "stop"
    # The request's own audio was prepared...
    np.testing.assert_allclose(runner.model.prepared_audio[0], audio_codec.decode_audio(CLONING_AUDIO)[0])
    # ...and the baseline (no conditionals) was restored afterwards.
    assert runner.model.conds is None


@pytest.mark.unit
def test_tts_restores_non_none_baseline_after_request():
    baseline = {"default_voice": True}
    model = StubChatterboxModel()
    model.conds = baseline
    runner = _make_runner(model)
    runner._capture_baseline_state()

    runner._execute_chatterbox_tts(
        [_tts_params(_speaker_for_audio(CLONING_AUDIO))], [_tts_request(input_audio=CLONING_AUDIO_B64)]
    )

    # The baseline is a deep copy, so it is restored by value (not identity).
    assert runner.model.conds == baseline
    assert runner.model.conds is not baseline


@pytest.mark.unit
def test_tts_precomputed_conditionals_used_then_restored():
    runner = _make_runner()
    runner._capture_baseline_state()
    conditionals = FakeConditionals()

    runner._execute_chatterbox_tts(
        [_tts_params(SpeakerArtifact(conditionals=conditionals, backend="chatterbox"))], [_tts_request()]
    )

    assert runner.model.conds_at_generate == [conditionals]
    assert runner.model.conds is None


@pytest.mark.unit
def test_tts_without_conditioning_fails_loudly_and_restores_state():
    runner = _make_runner()
    runner._capture_baseline_state()

    results = runner._execute_chatterbox_tts([_tts_params(None)], [_tts_request()])

    output = results[0].result_data
    assert output.finish_reason == "error"
    assert "voice cloning requires input_audio or a precomputed embedding/conditionals" in output.error
    # Nothing was generated and the shared model state is still the baseline.
    assert runner.model.generate_calls == []
    assert runner.model.conds is None


@pytest.mark.unit
@pytest.mark.parametrize(
    "handler_name, model_type, params_builder",
    [
        (
            "_execute_chatterbox_turbo_tts",
            "ChatterboxTurboTTS",
            lambda speaker: ("hello", speaker, 0.8, 1.2, 0.95, 1000, True),
        ),
        (
            "_execute_chatterbox_multilingual_tts",
            "ChatterboxMultilingualTTS",
            lambda speaker: ("hello", "en", speaker, 0.5, 0.5, 0.8, 1.2, 1.0, 0.05),
        ),
    ],
)
def test_turbo_and_multilingual_restore_state(handler_name, model_type, params_builder):
    runner = _make_runner(model_type=model_type)
    runner._capture_baseline_state()
    handler = getattr(runner, handler_name)

    results = handler(
        [params_builder(_speaker_for_audio(CLONING_AUDIO))], [_tts_request(input_audio=CLONING_AUDIO_B64)]
    )

    assert results[0].result_data.finish_reason == "stop"
    assert runner.model.conds is None


# ---------------------------------------------------------------------------
# VC handler restores ref_dict
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_vc_restores_ref_dict_after_conversion():
    model = StubChatterboxModel()
    baseline_ref = {"prompt_token": torch.zeros(1, 1, dtype=torch.long)}
    model.ref_dict = baseline_ref
    runner = _make_runner(model, model_type="ChatterboxVC")
    runner._capture_baseline_state()

    request_data = VoiceConversionRequestInput(
        request_id="state-vc-001",
        requested_model="chatterbox_vc",
        model="cb-vc",
        source_audio="c291cmNl",
        target_audio="dGFyZ2V0",
        target_embedding=None,
        generation_options=None,
        output_options=None,
    )
    request = EngineRequest(request_id="state-vc-001", request_data=request_data, arrival_time=0.0)

    def artifact(audio: bytes) -> AudioArtifact:
        waveform, sample_rate = audio_codec.decode_audio(audio)
        return AudioArtifact.from_numpy(waveform, sample_rate)

    results = runner._execute_chatterbox_vc([(artifact(CLONING_AUDIO), None, artifact(CLONING_AUDIO))], [request])

    assert results[0].result_data.finish_reason == "stop"
    np.testing.assert_allclose(runner.model.ref_audio[0], audio_codec.decode_audio(CLONING_AUDIO)[0])
    np.testing.assert_allclose(runner.model.source_audio[0], audio_codec.decode_audio(CLONING_AUDIO)[0])
    # Baseline restored by value (deep-copied snapshot), never the request's dict.
    assert runner.model.ref_dict == baseline_ref
    assert runner.model.ref_dict is not baseline_ref


@pytest.mark.unit
def test_vc_without_target_fails_loudly_and_restores_state():
    runner = _make_runner(model_type="ChatterboxVC")
    runner._capture_baseline_state()

    request_data = VoiceConversionRequestInput(
        request_id="state-vc-002",
        requested_model="chatterbox_vc",
        model="cb-vc",
        source_audio="c291cmNl",
        target_audio=None,
        target_embedding=None,
        generation_options=None,
        output_options=None,
    )
    request = EngineRequest(request_id="state-vc-002", request_data=request_data, arrival_time=0.0)

    source = AudioArtifact.from_numpy(audio_codec.decode_audio(CLONING_AUDIO)[0], 16000)
    results = runner._execute_chatterbox_vc([(source, None, None)], [request])

    assert results[0].result_data.finish_reason == "error"
    assert "voice conversion requires target_audio or target_embedding" in results[0].result_data.error
    assert runner.model.ref_dict is None


@pytest.mark.unit
def test_tts_uses_configured_default_voice_when_unconditioned():
    """Chatterbox bundles a default voice: no per-request conditioning is valid."""
    baseline = {"default_voice": True}
    model = StubChatterboxModel()
    model.conds = baseline
    runner = _make_runner(model)
    runner._capture_baseline_state()

    results = runner._execute_chatterbox_tts([_tts_params(None)], [_tts_request()])

    assert results[0].result_data.finish_reason == "stop"
    # The default voice was used (a copy, so upstream mutation cannot taint it)...
    assert runner.model.conds_at_generate == [baseline]
    # ...and the stored baseline object was restored untouched.
    assert runner.model.conds == baseline


@pytest.mark.unit
def test_restore_is_noop_for_non_chatterbox_models():
    """Restoring must not invent conds/ref_dict attributes on other models."""
    plain_model = MagicMock(spec=["generate", "sr"])
    runner = _make_runner(plain_model, model_type="KittenTTSSynthesizer")
    runner._capture_baseline_state()

    runner._restore_chatterbox_state()

    assert not hasattr(plain_model, "conds")

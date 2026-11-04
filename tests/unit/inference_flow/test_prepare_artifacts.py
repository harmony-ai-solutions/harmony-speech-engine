"""Unit tests for artifact-backed prepare functions and precedence (Phase 2-2)."""

import base64
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from chatterbox.models.t3.modules.cond_enc import T3Cond
from chatterbox.tts import Conditionals

from harmonyspeech.common import audio_codec
from harmonyspeech.common.artifacts import AudioArtifact, MelArtifact, SpeakerArtifact, VadArtifact
from harmonyspeech.common.inputs import (
    AudioConversionRequestInput,
    SpeechEmbeddingRequestInput,
    TextToSpeechRequestInput,
    get_audio_artifact,
    prefill_audio_artifacts,
)
from harmonyspeech.common.request import EngineRequest
from harmonyspeech.common.serialization import conditionals_to_payload
from harmonyspeech.task_handler import inputs as prepare_module
from harmonyspeech.task_handler.inputs import (
    prepare_chatterbox_tts_inputs,
    prepare_faster_whisper_inputs,
    prepare_harmonyspeech_encoder_inputs,
    prepare_harmonyspeech_synthesizer_inputs,
    prepare_harmonyspeech_vocoder_inputs,
    prepare_openvoice_tone_converter_encoder_inputs,
    prepare_voicefixer_vocoder_inputs,
)
from tests.audio_fixtures import tiny_wav_b64, tiny_wav_bytes


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def _engine_request(request_data, artifacts=None) -> EngineRequest:
    return EngineRequest(
        request_id="prep-1",
        request_data=request_data,
        arrival_time=0.0,
        artifacts=artifacts,
    )


def _tts_request(**overrides) -> TextToSpeechRequestInput:
    kwargs = {
        "request_id": "tts-1",
        "requested_model": "harmonyspeech",
        "model": "hs",
        "input_text": "hello",
        "mode": "voice_cloning",
    }
    kwargs.update(overrides)
    return TextToSpeechRequestInput(**kwargs)


# ---------------------------------------------------------------------------
# Precedence + decode-once
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_artifact_wins_over_base64_field():
    cached = AudioArtifact.from_numpy(np.zeros(800, dtype=np.float32), 16000)
    request = _tts_request(input_audio=tiny_wav_b64())
    engine_request = _engine_request(request, {"input_audio": cached})

    # Even a corrupt base64 field cannot win while the artifact is cached.
    request.input_audio = "!!!not-base64!!!"
    assert get_audio_artifact(request, engine_request, "input_audio") is cached


@pytest.mark.unit
def test_audio_artifact_is_decoded_once_and_cached(monkeypatch):
    request = _tts_request(input_audio=tiny_wav_b64())
    engine_request = _engine_request(request)
    calls = []
    real_decode = audio_codec.decode_audio

    def counting_decode(*args, **kwargs):
        calls.append(1)
        return real_decode(*args, **kwargs)

    monkeypatch.setattr("harmonyspeech.common.inputs.decode_audio", counting_decode)

    first = get_audio_artifact(request, engine_request, "input_audio")
    second = get_audio_artifact(request, engine_request, "input_audio")

    assert first is second
    assert engine_request.artifacts["input_audio"] is first
    assert len(calls) == 1


@pytest.mark.unit
def test_two_prepare_steps_share_one_decode(monkeypatch):
    """Whisper (16 kHz) then the encoder (native) decode the field exactly once."""
    request = _tts_request(input_audio=tiny_wav_b64())
    engine_request = _engine_request(request)
    calls = []
    real_decode = audio_codec.decode_audio

    def counting_decode(*args, **kwargs):
        calls.append(1)
        return real_decode(*args, **kwargs)

    monkeypatch.setattr("harmonyspeech.common.inputs.decode_audio", counting_decode)

    (whisper_artifact,) = prepare_faster_whisper_inputs([engine_request])
    (encoder_artifact,) = prepare_harmonyspeech_encoder_inputs([engine_request])

    assert whisper_artifact.sample_rate == 16000
    assert encoder_artifact is engine_request.artifacts["input_audio"]
    assert len(calls) == 1


@pytest.mark.unit
def test_admission_prefill_decodes_once(monkeypatch):
    request = _tts_request(input_audio=tiny_wav_b64())
    calls = []
    real_decode = audio_codec.decode_audio

    def counting_decode(*args, **kwargs):
        calls.append(1)
        return real_decode(*args, **kwargs)

    monkeypatch.setattr("harmonyspeech.common.inputs.decode_audio", counting_decode)

    artifacts = prefill_audio_artifacts(request)
    assert "input_audio" in artifacts
    assert len(calls) == 1

    # Idempotent: re-prefilling the same store does not decode again.
    prefill_audio_artifacts(request, artifacts)
    assert len(calls) == 1


@pytest.mark.unit
def test_admission_prefill_decodes_source_and_target_audio():
    request = AudioConversionRequestInput(
        request_id="ac-1",
        requested_model="voicefixer",
        model="vf",
        source_audio=tiny_wav_b64(),
    )
    artifacts = prefill_audio_artifacts(request)
    assert isinstance(artifacts["source_audio"], AudioArtifact)


@pytest.mark.unit
def test_admission_skips_non_audio_input_audio():
    """The HarmonySpeech vocoder's ``input_audio`` is JSON, not audio: leave it."""
    envelope = json.dumps({"mel": "abc"})
    request = TextToSpeechRequestInput(
        request_id="voc-1",
        requested_model="harmonyspeech",
        model="hs-voc",
        input_text="",
        mode="voice_cloning",
        input_audio=envelope,
    )
    artifacts = prefill_audio_artifacts(request)
    assert "input_audio" not in artifacts


# ---------------------------------------------------------------------------
# Prepare functions return artifacts
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_voicefixer_vocoder_legacy_mel_becomes_artifact():
    mel = np.linspace(0, 1, 80 * 12, dtype=np.float32).reshape(80, 12)
    payload = _b64(json.dumps(mel.tolist()).encode("utf-8"))
    request = AudioConversionRequestInput(
        request_id="ac-2",
        requested_model="voicefixer",
        model="vf-voc",
        source_audio="",
        input_mel_spectrogram=payload,
    )
    engine_request = _engine_request(request)

    (artifact,) = prepare_voicefixer_vocoder_inputs([engine_request])

    assert isinstance(artifact, MelArtifact)
    assert artifact.mel.shape == (1, 80, 12)
    assert artifact.sample_rate == 44100
    assert artifact.hop_length == 441
    np.testing.assert_allclose(artifact.numpy[0], mel, rtol=1e-6)


@pytest.mark.unit
def test_harmonyspeech_vocoder_prepare_legacy_envelope():
    mel = np.linspace(0, 1, 80 * 6, dtype=np.float32).reshape(80, 6)
    envelope = json.dumps(
        {
            "mel": _b64(json.dumps(mel.tolist()).encode("utf-8")),
            "breaks": _b64(json.dumps([3, 3]).encode("utf-8")),
        }
    )
    request = TextToSpeechRequestInput(
        request_id="voc-2",
        requested_model="harmonyspeech",
        model="hs-voc",
        input_text="",
        mode="voice_cloning",
        input_audio=envelope,
    )
    (artifact, breaks) = prepare_harmonyspeech_vocoder_inputs([_engine_request(request)])[0]

    assert isinstance(artifact, MelArtifact)
    assert artifact.mel.shape == (1, 80, 6)
    assert breaks == [3, 3]
    assert artifact.breaks == [3, 3]


@pytest.mark.unit
def test_harmonyspeech_vocoder_prepare_uses_forwarded_artifact():
    mel_artifact = MelArtifact.from_numpy(
        np.zeros((80, 6), dtype=np.float32), sample_rate=16000, hop_length=200, breaks=[6]
    )
    request = TextToSpeechRequestInput(
        request_id="voc-3",
        requested_model="harmonyspeech",
        model="hs-voc",
        input_text="",
        mode="voice_cloning",
    )
    engine_request = _engine_request(request, {"input_audio": mel_artifact})

    artifact, breaks = prepare_harmonyspeech_vocoder_inputs([engine_request])[0]

    assert artifact is mel_artifact
    assert breaks == [6]


@pytest.mark.unit
def test_true_speaker_embedding_artifact_is_preferred(monkeypatch):
    embedding = np.arange(8, dtype=np.float32)
    request = _tts_request(input_embedding=_b64(embedding.tobytes()))
    engine_request = _engine_request(request)

    captured = {}

    def fake_prepare_synthesis_inputs(text, array):
        captured["embedding"] = np.asarray(array)
        return np.array([[1, 2, 3]]), np.asarray(array)[None, :]

    monkeypatch.setattr(prepare_module, "prepare_synthesis_inputs", fake_prepare_synthesis_inputs)

    inputs = prepare_harmonyspeech_synthesizer_inputs([engine_request])
    assert np.array_equal(captured["embedding"], embedding)
    assert isinstance(engine_request.artifacts["input_embedding"], SpeakerArtifact)
    assert inputs[0][1].shape == (1, embedding.size)


# ---------------------------------------------------------------------------
# OpenVoice tone-converter encoder segment extraction (no tempfile double decode)
# ---------------------------------------------------------------------------


def _ov_config(sample_rate: int) -> SimpleNamespace:
    data = SimpleNamespace(
        sampling_rate=sample_rate,
        filter_length=1024,
        hop_length=256,
        win_length=1024,
    )
    return SimpleNamespace(data=data)


@pytest.mark.unit
def test_tone_converter_encoder_prepare_reuses_artifacts(monkeypatch):
    sample_rate = 22050
    seconds = 3.0
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    waveform = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    audio_artifact = AudioArtifact.from_numpy(waveform, sample_rate)
    vad_artifact = VadArtifact(
        segments=[{"start": 0.0, "end": 3.0, "text": "hello world"}],
        info={"language": "en"},
    )
    request = SpeechEmbeddingRequestInput(
        request_id="emb-1", requested_model="openvoice_v2", model="ov2-tce"
    )
    engine_request = _engine_request(
        request, {"input_audio": audio_artifact, "input_vad_data": vad_artifact}
    )

    monkeypatch.setattr(prepare_module, "get_model_flavour", lambda config: "default")
    monkeypatch.setattr(
        prepare_module, "get_model_config", lambda *args, **kwargs: _ov_config(sample_rate)
    )
    calls = []
    real_decode = audio_codec.decode_audio
    monkeypatch.setattr(
        "harmonyspeech.common.inputs.decode_audio",
        lambda *a, **k: calls.append(1) or real_decode(*a, **k),
    )

    (segments,) = prepare_openvoice_tone_converter_encoder_inputs(
        MagicMock(model="ov2", model_type="OpenVoiceV2ToneConverterEncoder"), [engine_request]
    )

    assert calls == []  # nothing decoded again
    assert len(segments) == 1
    assert isinstance(segments[0], AudioArtifact)
    assert segments[0].sample_rate == sample_rate
    assert segments[0].duration_s == pytest.approx(3.0, rel=0.05)


# ---------------------------------------------------------------------------
# Chatterbox prepare: conditionals by reference / reference audio artifact
# ---------------------------------------------------------------------------


def _conditionals_payload() -> str:
    conditionals = Conditionals(
        t3=T3Cond(speaker_emb=torch.zeros(1, 4)),
        gen={"prompt_token": torch.zeros(1, 2, dtype=torch.long)},
    )
    import io

    buffer = io.BytesIO()
    torch.save(conditionals_to_payload(conditionals), buffer)
    return base64.b64encode(buffer.getvalue()).decode()


@pytest.mark.unit
def test_chatterbox_tts_prepare_returns_reference_audio_artifact():
    request = TextToSpeechRequestInput(
        request_id="cb-1",
        requested_model="chatterbox",
        model="cb",
        input_text="hello",
        mode="voice_cloning",
        input_audio=tiny_wav_b64(),
    )
    (params,) = prepare_chatterbox_tts_inputs([_engine_request(request)])

    speaker = params[1]
    assert isinstance(speaker, SpeakerArtifact)
    assert speaker.backend == "chatterbox"
    assert speaker.conditionals is None
    assert isinstance(speaker.reference_audio, AudioArtifact)
    np.testing.assert_array_equal(speaker.reference_audio.numpy, audio_codec.decode_audio(tiny_wav_bytes())[0])


@pytest.mark.unit
def test_chatterbox_tts_prepare_loads_conditionals_safely():
    request = TextToSpeechRequestInput(
        request_id="cb-2",
        requested_model="chatterbox",
        model="cb",
        input_text="hello",
        mode="single_speaker_tts",
        input_embedding=_conditionals_payload(),
    )
    (params,) = prepare_chatterbox_tts_inputs([_engine_request(request)])

    speaker = params[1]
    assert isinstance(speaker, SpeakerArtifact)
    assert speaker.conditionals is not None
    assert speaker.conditionals.t3.speaker_emb.shape == (1, 4)


@pytest.mark.unit
def test_chatterbox_tts_prepare_without_conditioning_returns_none():
    request = TextToSpeechRequestInput(
        request_id="cb-3",
        requested_model="chatterbox",
        model="cb",
        input_text="hello",
        mode="single_speaker_tts",
    )
    (params,) = prepare_chatterbox_tts_inputs([_engine_request(request)])
    assert params[1] is None

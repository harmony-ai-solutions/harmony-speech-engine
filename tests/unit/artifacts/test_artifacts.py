"""Unit tests for the typed in-process artifacts (Phase 2-1)."""

import numpy as np
import pytest
import torch

from harmonyspeech.common.artifacts import AudioArtifact, MelArtifact, SpeakerArtifact, VadArtifact
from harmonyspeech.common.config import DeviceConfig, ModelConfig
from harmonyspeech.common.request import EngineRequest, RequestStatus
from harmonyspeech.processing.scheduler import Scheduler


def _waveform(seconds: float = 0.1, sample_rate: int = 16000) -> np.ndarray:
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    return (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


# ---------------------------------------------------------------------------
# AudioArtifact
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_audio_artifact_roundtrip_numpy_device_numpy():
    waveform = _waveform()
    artifact = AudioArtifact.from_numpy(waveform, 16000)

    moved = artifact.to("cpu")
    assert moved is artifact  # no-op move on the same device
    np.testing.assert_array_equal(moved.numpy, waveform)


@pytest.mark.unit
def test_audio_artifact_standardizes_on_batched_float32():
    artifact = AudioArtifact.from_numpy(_waveform(), 16000)
    assert artifact.waveform.shape[0] == 1
    assert artifact.waveform.dtype == torch.float32
    assert artifact.waveform.device.type == "cpu"
    assert artifact.device == torch.device("cpu")


@pytest.mark.unit
def test_audio_artifact_duration_and_len():
    artifact = AudioArtifact.from_numpy(_waveform(seconds=0.5), 16000)
    assert artifact.duration_s == pytest.approx(0.5, rel=1e-6)
    assert len(artifact) == 8000
    assert artifact.num_samples == 8000


@pytest.mark.unit
def test_audio_artifact_resample_is_transient():
    artifact = AudioArtifact.from_numpy(_waveform(seconds=1.0, sample_rate=16000), 16000)
    downsampled = artifact.resampled(8000)

    assert downsampled is not artifact
    assert downsampled.sample_rate == 8000
    assert downsampled.duration_s == pytest.approx(1.0, rel=0.02)
    # The cached artifact keeps its native rate.
    assert artifact.sample_rate == 16000
    assert artifact.resampled(16000) is artifact


@pytest.mark.unit
def test_audio_artifact_device_mismatch_fails_loudly():
    with pytest.raises(ValueError, match="device mismatch"):
        AudioArtifact(
            waveform=torch.zeros(1, 10),
            sample_rate=16000,
            device=torch.device("meta"),
        )


@pytest.mark.unit
def test_audio_artifact_to_wav_bytes_is_pcm16():
    import io

    import soundfile as sf

    artifact = AudioArtifact.from_numpy(_waveform(), 16000)
    with sf.SoundFile(io.BytesIO(artifact.to_wav_bytes())) as handle:
        assert handle.samplerate == 16000
        assert handle.subtype == "PCM_16"
        assert len(handle) == artifact.num_samples


# ---------------------------------------------------------------------------
# MelArtifact
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_mel_artifact_preserves_shape_and_breaks():
    mel = np.random.default_rng(0).random((80, 20), dtype=np.float32)
    artifact = MelArtifact.from_numpy(mel, sample_rate=22050, hop_length=200, breaks=[5, 15])

    assert artifact.mel.shape == (1, 80, 20)
    assert artifact.mel_bins == 80
    assert artifact.frames == 20
    assert artifact.breaks == [5, 15]
    np.testing.assert_array_equal(artifact.numpy[0], mel)


@pytest.mark.unit
def test_mel_artifact_rejects_non_3d():
    with pytest.raises(ValueError, match=r"\[1, mel_bins, frames\]"):
        MelArtifact(mel=torch.zeros(4, 5, 6, 7), sample_rate=22050, hop_length=200)


@pytest.mark.unit
def test_mel_artifact_device_move_is_explicit():
    artifact = MelArtifact(mel=torch.zeros(1, 4, 5), sample_rate=22050, hop_length=200)
    assert artifact.to("cpu") is artifact


# ---------------------------------------------------------------------------
# SpeakerArtifact
# ---------------------------------------------------------------------------


class _Conditionals:
    """Stand-in for chatterbox Conditionals that tracks device moves."""

    def __init__(self):
        self.moves = []

    def to(self, device):
        self.moves.append(device)
        return self


@pytest.mark.unit
def test_speaker_artifact_moves_embedding_not_conditionals():
    conditionals = _Conditionals()
    artifact = SpeakerArtifact(
        embedding=torch.zeros(1, 8), conditionals=conditionals, backend="chatterbox"
    )

    moved = artifact.to("cpu")

    # Conditionals stay a by-reference object; the consumer moves them itself.
    assert moved.conditionals is conditionals
    assert conditionals.moves == []


@pytest.mark.unit
def test_speaker_artifact_keeps_conditionals_identity():
    conditionals = _Conditionals()
    artifact = SpeakerArtifact(conditionals=conditionals, backend="chatterbox")
    assert artifact.conditionals is conditionals
    assert artifact.embedding is None


# ---------------------------------------------------------------------------
# VadArtifact
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_vad_artifact_from_payload():
    payload = {
        "text": "hello",
        "segments": [{"start": 0.0, "end": 1.0, "text": "hello"}],
        "info": {"language": "en"},
    }
    artifact = VadArtifact.from_payload(payload)

    assert artifact.text == "hello"
    assert artifact.info == {"language": "en"}
    assert artifact.segments[0]["start"] == 0.0
    assert artifact.has_speech
    assert artifact.to("cpu") is artifact


# ---------------------------------------------------------------------------
# Ownership: EngineRequest store + deterministic release
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_engine_request_artifact_store_and_clear():
    artifact = AudioArtifact.from_numpy(_waveform(), 16000)
    request = EngineRequest(
        request_id="req-1",
        request_data=object(),
        arrival_time=0.0,
        artifacts={"input_audio": artifact},
    )

    assert request.get_artifact("input_audio") is artifact
    request.set_artifact("input_embedding", "sentinel")
    assert request.artifacts["input_embedding"] == "sentinel"

    request.clear_artifacts()
    assert request.artifacts is None
    # Idempotent.
    request.clear_artifacts()
    assert request.artifacts is None


@pytest.mark.unit
def test_scheduler_releases_artifacts_on_finish():
    config = ModelConfig(
        name="cb",
        model="dummy",
        model_type="ChatterboxTTS",
        max_batch_size=1,
        device_config=DeviceConfig("cpu"),
        dtype="float32",
    )
    scheduler = Scheduler(model_configs=[config])
    artifact = AudioArtifact.from_numpy(_waveform(), 16000)
    request = EngineRequest(
        request_id="req-finish",
        request_data=object(),
        arrival_time=0.0,
        artifacts={"input_audio": artifact},
    )
    scheduler.add_request(request)

    scheduler.update_request_status("req-finish", RequestStatus.FINISHED_STOPPED)

    assert request.artifacts is None
    assert not scheduler.has_unfinished_requests()


@pytest.mark.unit
def test_forwarded_requests_keep_their_artifact_copy():
    """Forwarding snapshots the dict before the finished request releases it."""
    config = ModelConfig(
        name="cb",
        model="dummy",
        model_type="ChatterboxTTS",
        max_batch_size=1,
        device_config=DeviceConfig("cpu"),
        dtype="float32",
    )
    scheduler = Scheduler(model_configs=[config])
    artifact = AudioArtifact.from_numpy(_waveform(), 16000)
    request = EngineRequest(
        request_id="req-forward",
        request_data=object(),
        arrival_time=0.0,
        artifacts={"input_audio": artifact},
    )
    scheduler.add_request(request)

    forwarded = dict(request.artifacts)
    scheduler.update_request_status("req-forward", RequestStatus.FINISHED_FORWARDED)

    assert request.artifacts is None
    assert forwarded["input_audio"] is artifact

"""Unit tests for boundary decode sites in task_handler/inputs.py (Phase 1-1)."""

import base64

import pytest
import torch

from harmonyspeech.common import audio_codec
from harmonyspeech.common.inputs import (
    AudioConversionRequestInput,
    DetectVoiceActivityRequestInput,
    SpeechEmbeddingRequestInput,
    SpeechTranscribeRequestInput,
    TextToSpeechRequestInput,
    VoiceConversionRequestInput,
)
from harmonyspeech.task_handler.inputs import (
    _decode_audio_field,
    prepare_chatterbox_embedding_inputs,
    prepare_chatterbox_vc_inputs,
    prepare_faster_whisper_inputs,
    prepare_harmonyspeech_encoder_inputs,
    prepare_silero_vad_inputs,
    prepare_voicefixer_restorer_inputs,
)

FFMPEG_ONLY = pytest.mark.skipif(not audio_codec.FFMPEG_AVAILABLE, reason="ffmpeg not available on PATH")


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def _bytes(test_data_dir, name: str) -> bytes:
    return (test_data_dir / "samples" / name).read_bytes()


def _transcribe(payload: str) -> SpeechTranscribeRequestInput:
    return SpeechTranscribeRequestInput(
        request_id="stt-1", requested_model="stt", model="stt", input_audio=payload
    )


# ---------------------------------------------------------------------------
# Error conversion carries the field name
# ---------------------------------------------------------------------------


def test_decode_helper_names_field():
    with pytest.raises(ValueError, match="input_audio"):
        _decode_audio_field(_b64(b"not audio"), "input_audio")


def test_decode_helper_missing_payload():
    with pytest.raises(ValueError, match="source_audio"):
        _decode_audio_field(None, "source_audio")


def test_decode_helper_invalid_base64():
    with pytest.raises(ValueError, match="input_audio"):
        _decode_audio_field("!!!", "input_audio")


def test_faster_whisper_prepare_converts_decode_error():
    with pytest.raises(ValueError, match="input_audio"):
        prepare_faster_whisper_inputs([_transcribe(_b64(b"garbage bytes"))])


def test_silero_vad_prepare_converts_decode_error():
    request = DetectVoiceActivityRequestInput(
        request_id="vad-1", requested_model="vad", model="vad", input_audio=_b64(b"garbage bytes")
    )
    with pytest.raises(ValueError, match="input_audio"):
        prepare_silero_vad_inputs([request])


def test_chatterbox_vc_names_source_field():
    request = VoiceConversionRequestInput(
        request_id="vc-1",
        requested_model="vc",
        model="vc",
        source_audio=_b64(b"garbage"),
        target_audio=_b64(b"garbage"),
        target_embedding=None,
        generation_options=None,
        output_options=None,
    )
    with pytest.raises(ValueError, match="source_audio"):
        prepare_chatterbox_vc_inputs([request])


def test_voicefixer_prepare_names_source_field():
    request = AudioConversionRequestInput(
        request_id="ac-1", requested_model="ac", model="ac", source_audio=_b64(b"garbage")
    )
    with pytest.raises(ValueError, match="source_audio"):
        prepare_voicefixer_restorer_inputs([request])


# ---------------------------------------------------------------------------
# Successful decodes keep the historical shapes
# ---------------------------------------------------------------------------


def test_faster_whisper_prepare_resamples_to_16k(test_data_dir):
    from harmonyspeech.common.artifacts import AudioArtifact

    inputs = prepare_faster_whisper_inputs([_transcribe(_b64(_bytes(test_data_dir, "codec_tone.mp3")))])
    artifact = inputs[0]
    assert isinstance(artifact, AudioArtifact)
    assert artifact.sample_rate == 16000
    assert abs(artifact.duration_s - 1.0) < 0.05


def test_silero_vad_prepare_returns_artifact_and_params(test_data_dir):
    from harmonyspeech.common.artifacts import AudioArtifact

    request = DetectVoiceActivityRequestInput(
        request_id="vad-1",
        requested_model="vad",
        model="vad",
        input_audio=_b64(_bytes(test_data_dir, "codec_tone.wav")),
    )
    (artifact, params), = prepare_silero_vad_inputs([request])
    assert isinstance(artifact, AudioArtifact)
    assert artifact.sample_rate == 16000
    assert artifact.waveform.dtype == torch.float32
    assert abs(artifact.duration_s - 1.0) < 0.05
    assert params["threshold"] == 0.5


def test_voicefixer_prepare_returns_artifact(test_data_dir):
    from harmonyspeech.common.artifacts import AudioArtifact

    request = AudioConversionRequestInput(
        request_id="ac-1",
        requested_model="ac",
        model="ac",
        source_audio=_b64(_bytes(test_data_dir, "codec_tone.ogg")),
    )
    inputs = prepare_voicefixer_restorer_inputs([request])
    artifact = inputs[0]
    assert isinstance(artifact, AudioArtifact)
    assert artifact.sample_rate == 44100
    assert artifact.waveform.shape[0] == 1  # mono batch axis
    assert abs(artifact.duration_s - 1.0) < 0.05


def test_encoder_prepare_returns_audio_artifact(test_data_dir):
    from harmonyspeech.common.artifacts import AudioArtifact

    request = TextToSpeechRequestInput(
        request_id="tts-1",
        requested_model="hs",
        model="hs",
        input_text="hello",
        mode="voice_cloning",
        input_audio=_b64(_bytes(test_data_dir, "wanda4.wav")),
    )
    inputs = prepare_harmonyspeech_encoder_inputs([request])
    assert len(inputs) == 1
    assert isinstance(inputs[0], AudioArtifact)
    assert inputs[0].num_samples > 0


# ---------------------------------------------------------------------------
# Chatterbox reference audio accepts mp3/aac through the same path
# ---------------------------------------------------------------------------


def test_chatterbox_embedding_passes_known_container_through(test_data_dir):
    wav = _bytes(test_data_dir, "codec_tone.wav")
    request = SpeechEmbeddingRequestInput(
        request_id="emb-1", requested_model="emb", model="emb", input_audio=_b64(wav)
    )
    (audio_artifact,) = prepare_chatterbox_embedding_inputs([request])
    # Decoded exactly once at admission; re-encoding as WAV keeps the samples.
    assert audio_artifact.to_wav_bytes()[:4] == b"RIFF"
    assert audio_artifact.num_samples > 0


def test_chatterbox_embedding_accepts_mp3(test_data_dir):
    mp3 = _bytes(test_data_dir, "codec_tone.mp3")
    request = SpeechEmbeddingRequestInput(
        request_id="emb-1", requested_model="emb", model="emb", input_audio=_b64(mp3)
    )
    (audio_artifact,) = prepare_chatterbox_embedding_inputs([request])
    # The decoded artifact is re-wrapped as a WAV buffer for the upstream loader.
    import io

    import soundfile as sf

    with sf.SoundFile(io.BytesIO(audio_artifact.to_wav_bytes())) as handle:
        assert handle.samplerate == audio_artifact.sample_rate
        assert len(handle) == audio_artifact.num_samples


@FFMPEG_ONLY
def test_chatterbox_embedding_accepts_aac(test_data_dir):
    aac = _bytes(test_data_dir, "codec_tone.aac")
    request = SpeechEmbeddingRequestInput(
        request_id="emb-1", requested_model="emb", model="emb", input_audio=_b64(aac)
    )
    (audio_artifact,) = prepare_chatterbox_embedding_inputs([request])
    assert audio_artifact.to_wav_bytes()[:4] == b"RIFF"
    assert audio_artifact.num_samples > 0


def test_chatterbox_vc_target_audio_accepts_mp3(test_data_dir):
    from harmonyspeech.common.artifacts import AudioArtifact

    mp3 = _bytes(test_data_dir, "codec_tone.mp3")
    request = VoiceConversionRequestInput(
        request_id="vc-1",
        requested_model="vc",
        model="vc",
        source_audio=_b64(mp3),
        target_audio=_b64(mp3),
        target_embedding=None,
        generation_options=None,
        output_options=None,
    )
    source_artifact, conditionals, target_artifact = prepare_chatterbox_vc_inputs([request])[0]
    assert isinstance(source_artifact, AudioArtifact)
    assert isinstance(target_artifact, AudioArtifact)
    assert conditionals is None
    assert source_artifact.sample_rate > 0
    assert source_artifact.num_samples == target_artifact.num_samples

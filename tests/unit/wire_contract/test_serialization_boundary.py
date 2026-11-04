"""Unit tests for the boundary serialization seam and the grep invariant (Phase 2-4)."""

import base64
import io
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from harmonyspeech.common.artifacts import AudioArtifact, SpeakerArtifact
from harmonyspeech.common.inputs import TextToSpeechRequestInput
from harmonyspeech.common.outputs import (
    SpeechEmbeddingRequestOutput,
    TextToSpeechRequestOutput,
)
from harmonyspeech.common.request import EngineRequest
from harmonyspeech.common.serialization import conditionals_to_payload
from harmonyspeech.endpoints.openai.protocol import AudioOutputOptions
from harmonyspeech.endpoints.openai.wire import from_wire_audio, to_wire_audio, to_wire_embedding
from tests.audio_fixtures import tiny_wav_b64

REPO_ROOT = Path(__file__).resolve().parents[3]

#: The only places allowed to know about wire text encoding.
ALLOWED_ENCODING_FILES = {
    "common/audio_codec.py",
    "common/inputs.py",
    "common/outputs.py",
}
ALLOWED_ENCODING_PREFIX = "endpoints/"


def _waveform(seconds: float = 0.25, sample_rate: int = 24000) -> np.ndarray:
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    return (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def _legacy_wav_b64(waveform: np.ndarray, sample_rate: int) -> str:
    """Reproduce the pre-Phase-2 executor encoding exactly."""
    buffer = io.BytesIO()
    sf.write(buffer, waveform, samplerate=sample_rate, format="WAV")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# Grep invariant: serialization only at the boundary
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_engine_core_contains_no_wire_encoding():
    """``harmonyspeech/`` may only mention base64 in the boundary/allowed files."""
    offenders = []
    for path in (REPO_ROOT / "harmonyspeech").rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "base64" not in text:
            continue
        relative = path.relative_to(REPO_ROOT / "harmonyspeech").as_posix()
        if relative in ALLOWED_ENCODING_FILES or relative.startswith(ALLOWED_ENCODING_PREFIX):
            continue
        offenders.append(relative)

    assert offenders == [], (
        "Wire encoding leaked into the engine core: "
        f"{offenders}. Allowed: {sorted(ALLOWED_ENCODING_FILES)} and {ALLOWED_ENCODING_PREFIX}*"
    )


@pytest.mark.unit
def test_all_serving_modules_use_the_wire_seam():
    serving_dir = REPO_ROOT / "harmonyspeech" / "endpoints" / "openai"
    for name in [
        "serving_text_to_speech.py",
        "serving_voice_conversion.py",
        "serving_audio_conversion.py",
        "serving_speech_to_text.py",
        "serving_voice_embed.py",
        "serving_voice_activity_detection.py",
    ]:
        source = (serving_dir / name).read_text(encoding="utf-8")
        assert "from_wire_audio" in source, f"{name} must decode inputs via wire.from_wire_audio"

    assert "to_wire_audio" in (serving_dir / "serving_text_to_speech.py").read_text(encoding="utf-8")
    assert "to_wire_audio" in (serving_dir / "serving_voice_conversion.py").read_text(encoding="utf-8")
    assert "to_wire_audio" in (serving_dir / "serving_audio_conversion.py").read_text(encoding="utf-8")
    assert "to_wire_embedding" in (serving_dir / "serving_voice_embed.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# from_wire_audio
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_from_wire_audio_decodes_once_and_keeps_the_field():
    request = TextToSpeechRequestInput(
        request_id="wire-1",
        requested_model="harmonyspeech",
        model="hs",
        input_text="hello",
        mode="voice_cloning",
        input_audio=tiny_wav_b64(),
    )
    original = request.input_audio

    artifacts = from_wire_audio(request)

    assert isinstance(artifacts["input_audio"], AudioArtifact)
    # Backward compatibility/observability: the base64 field is untouched.
    assert request.input_audio == original
    assert from_wire_audio(request, artifacts)["input_audio"] is artifacts["input_audio"]


@pytest.mark.unit
def test_from_wire_audio_fills_an_engine_request_store():
    request = TextToSpeechRequestInput(
        request_id="wire-2",
        requested_model="harmonyspeech",
        model="hs",
        input_text="hello",
        mode="voice_cloning",
        input_audio=tiny_wav_b64(),
    )
    engine_request = EngineRequest(request_id="wire-2", request_data=request, arrival_time=0.0)

    from_wire_audio(request, engine_request)

    assert isinstance(engine_request.artifacts["input_audio"], AudioArtifact)


# ---------------------------------------------------------------------------
# to_wire_audio: artifact path == legacy path (byte-identical default)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_to_wire_audio_from_artifact_matches_legacy_payload():
    waveform = _waveform()
    legacy = _legacy_wav_b64(waveform, 24000)

    with_artifact = TextToSpeechRequestOutput(
        request_id="tts-1",
        text="hello",
        output="",
        audio_artifact=AudioArtifact.from_numpy(waveform, 24000),
    )
    without_artifact = TextToSpeechRequestOutput(request_id="tts-2", text="hello", output=legacy)

    from_artifact = to_wire_audio(with_artifact, None)
    from_legacy = to_wire_audio(without_artifact, None)

    assert from_artifact.data == legacy
    assert from_artifact.data == from_legacy.data
    assert from_artifact.format == "wav"
    assert from_artifact.sample_rate is None


@pytest.mark.unit
def test_to_wire_audio_honors_output_options():
    waveform = _waveform(seconds=0.25, sample_rate=24000)
    result = TextToSpeechRequestOutput(
        request_id="tts-3",
        text="hello",
        output="",
        audio_artifact=AudioArtifact.from_numpy(waveform, 24000),
    )

    encoded = to_wire_audio(result, AudioOutputOptions(format="pcm"))
    assert encoded.format == "pcm"
    assert encoded.sample_rate == 24000
    pcm = np.frombuffer(base64.b64decode(encoded.data), dtype="<i2")
    assert pcm.size == waveform.size

    resampled = to_wire_audio(result, AudioOutputOptions(format="wav", sample_rate=16000))
    decoded, sr = sf.read(io.BytesIO(base64.b64decode(resampled.data)))
    assert sr == 16000
    assert resampled.sample_rate == 16000


@pytest.mark.unit
def test_to_wire_audio_falls_back_to_legacy_string_with_options():
    waveform = _waveform()
    legacy = _legacy_wav_b64(waveform, 24000)
    result = TextToSpeechRequestOutput(request_id="tts-4", text="hello", output=legacy)

    encoded = to_wire_audio(result, AudioOutputOptions(format="flac"))
    assert encoded.format == "flac"


# ---------------------------------------------------------------------------
# to_wire_embedding: exact pre-Phase-2 payloads
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_to_wire_embedding_harmonyspeech_is_raw_float32():
    embedding = torch.arange(8, dtype=torch.float32)
    result = SpeechEmbeddingRequestOutput(
        request_id="emb-1",
        output="",
        embedding_artifact=SpeakerArtifact(embedding=embedding, backend="harmonyspeech"),
    )

    expected = base64.b64encode(np.ascontiguousarray(embedding.numpy(), dtype=np.float32).tobytes()).decode()
    assert to_wire_embedding(result) == expected


@pytest.mark.unit
def test_to_wire_embedding_openvoice_is_torch_save():
    embedding = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4)
    result = SpeechEmbeddingRequestOutput(
        request_id="emb-2",
        output="",
        embedding_artifact=SpeakerArtifact(embedding=embedding, backend="openvoice"),
    )

    buffer = io.BytesIO()
    torch.save(embedding, buffer)
    assert to_wire_embedding(result) == base64.b64encode(buffer.getvalue()).decode()


@pytest.mark.unit
def test_to_wire_embedding_conditionals_uses_safe_payload():
    from chatterbox.models.t3.modules.cond_enc import T3Cond
    from chatterbox.tts import Conditionals

    conditionals = Conditionals(
        t3=T3Cond(speaker_emb=torch.zeros(1, 4)),
        gen={"prompt_token": torch.zeros(1, 2, dtype=torch.long)},
    )
    result = SpeechEmbeddingRequestOutput(
        request_id="emb-3",
        output="",
        embedding_artifact=SpeakerArtifact(conditionals=conditionals, backend="chatterbox"),
    )

    buffer = io.BytesIO()
    torch.save(conditionals_to_payload(conditionals), buffer)
    assert to_wire_embedding(result) == base64.b64encode(buffer.getvalue()).decode()


@pytest.mark.unit
def test_to_wire_embedding_falls_back_to_legacy_output():
    result = SpeechEmbeddingRequestOutput(request_id="emb-4", output="bGVnYWN5")
    assert to_wire_embedding(result) == "bGVnYWN5"

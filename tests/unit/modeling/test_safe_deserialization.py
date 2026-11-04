"""Unit tests for hardened deserialization (Phase 0-1).

Covers:
* hostile pickle payloads in request fields are rejected with ``ValueError``
  and never executed (OpenVoice tone-converter and Chatterbox conditionals),
* encoder-produced embedding blobs still round-trip under ``weights_only=True``,
* Chatterbox conditionals serialize to a tensor/primitive-only payload that
  reloads through the real ``Conditionals.load`` implementation,
* checkpoint loading uses ``weights_only=True`` with an explicit escape hatch.
"""

import base64
import io
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from chatterbox.models.t3.modules.cond_enc import T3Cond
from chatterbox.tts import Conditionals

from harmonyspeech.common.config import DeviceConfig, ModelConfig
from harmonyspeech.common.inputs import (
    SpeechEmbeddingRequestInput,
    TextToSpeechRequestInput,
    VoiceConversionRequestInput,
)
from harmonyspeech.common.request import EngineRequest
from harmonyspeech.modeling.hf_downloader import ALLOW_UNSAFE_CHECKPOINT_LOAD_ENV, load_or_download_model
from harmonyspeech.task_handler.inputs import (
    _load_conditionals_payload,
    prepare_chatterbox_tts_inputs,
    prepare_chatterbox_vc_inputs,
)
from harmonyspeech.task_handler.model_runner_base import (
    ModelRunnerBase,
    _conditionals_to_payload,
    _load_embedding_tensor,
)
from tests.audio_fixtures import tiny_wav_bytes

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TrustedOnlyCheckpointState:
    """Custom class in a checkpoint — only loadable with full unpickling."""


def _hostile_pickle(marker_path) -> bytes:
    """Return a pickled ``__reduce__`` gadget that would create ``marker_path``.

    The gadget is harmless if it ever ran (it only creates an empty marker file)
    but is enough to prove that pickle code execution is disabled.
    """

    class _Gadget:
        def __reduce__(self):
            return (open, (str(marker_path), "w"))

    buf = io.BytesIO()
    torch.save(_Gadget(), buf)
    return buf.getvalue()


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def _engine_request(request_data, request_id="req-safe-001") -> EngineRequest:
    return EngineRequest(request_id=request_id, request_data=request_data, arrival_time=0.0)


def _make_tone_converter_runner() -> ModelRunnerBase:
    runner = ModelRunnerBase.__new__(ModelRunnerBase)
    runner.model = MagicMock()
    runner.device = "cpu"
    runner._baseline_state = None
    runner.model_config = ModelConfig(
        name="ov2-tone-converter",
        model="dummy/openvoice",
        model_type="OpenVoiceV2ToneConverter",
        max_batch_size=1,
        device_config=DeviceConfig("cpu"),
        dtype="float32",
    )
    return runner


_OV_CONFIG = SimpleNamespace(
    data=SimpleNamespace(
        filter_length=1024,
        sampling_rate=22050,
        hop_length=256,
        win_length=1024,
    )
)


def _tensor_bytes(value: float = 1.0) -> bytes:
    buf = io.BytesIO()
    torch.save(torch.full((1, 4, 8), value), buf)
    return buf.getvalue()


def _make_conditionals() -> Conditionals:
    t3 = T3Cond(
        speaker_emb=torch.zeros(1, 10),
        clap_emb=torch.zeros(1, 5),
        cond_prompt_speech_tokens=torch.zeros(1, 3, dtype=torch.long),
    )
    gen = {
        "prompt_token": torch.zeros(1, 4, dtype=torch.long),
        "prompt_token_len": torch.tensor([4]),
        "prompt_feat": torch.zeros(1, 8, 6),
        "prompt_feat_len": torch.tensor([8]),
        "embedding": torch.zeros(1, 7),
    }
    return Conditionals(t3=t3, gen=gen)


# ---------------------------------------------------------------------------
# Hostile payloads: OpenVoice tone converter
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_hostile_embedding_payload_rejected_by_tone_converter(tmp_path):
    """A pickle gadget in a speaker embedding field must not run; it fails closed."""
    from harmonyspeech.common.inputs import get_speaker_artifact

    marker = tmp_path / "tone_converter_pwned.txt"

    request = TextToSpeechRequestInput(
        request_id="tc-hostile",
        requested_model="openvoice_v2",
        model="ov2-tone-converter",
        input_text="hello",
        mode="voice_cloning",
        input_audio=_b64(b"audio"),
        input_embedding=_b64(_hostile_pickle(marker)),
    )
    engine_request = _engine_request(request)

    with pytest.raises(ValueError, match="Unsupported input_embedding payload"):
        get_speaker_artifact(request, engine_request, "input_embedding", backend="openvoice")

    assert not marker.exists(), "hostile pickle payload was executed!"


@pytest.mark.unit
def test_encoder_embedding_roundtrips_through_safe_loader():
    """A tensor saved by the tone-converter encoder loads with weights_only=True."""
    tensor = torch.arange(24, dtype=torch.float32).reshape(1, 3, 8)
    buf = io.BytesIO()
    torch.save(tensor.cpu(), buf)
    buf.seek(0)

    loaded = _load_embedding_tensor(buf)

    assert torch.is_tensor(loaded)
    assert torch.equal(loaded, tensor)


@pytest.mark.unit
def test_safe_loader_rejects_non_tensor_payload():
    """A pickle that deserializes to a non-tensor is rejected as a typed error."""
    buf = io.BytesIO()
    torch.save({"not": "a tensor"}, buf)
    buf.seek(0)

    with pytest.raises(ValueError, match="expected a single tensor"):
        _load_embedding_tensor(buf)


# ---------------------------------------------------------------------------
# Hostile payloads: Chatterbox conditionals (prepare functions)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_hostile_conditionals_rejected_for_chatterbox_tts(tmp_path):
    """A pickle gadget in input_embedding is rejected, not executed (TTS prepare)."""
    marker = tmp_path / "chatterbox_tts_pwned.txt"
    request = TextToSpeechRequestInput(
        request_id="cb-hostile-tts",
        requested_model="chatterbox",
        model="chatterbox",
        input_text="hello",
        mode="voice_cloning",
        input_embedding=_b64(_hostile_pickle(marker)),
    )

    with pytest.raises(ValueError, match="Unsupported input_embedding payload"):
        prepare_chatterbox_tts_inputs([request])

    assert not marker.exists(), "hostile pickle payload was executed!"


@pytest.mark.unit
def test_hostile_conditionals_rejected_for_chatterbox_vc(tmp_path):
    """A pickle gadget in target_embedding is rejected, not executed (VC prepare)."""
    marker = tmp_path / "chatterbox_vc_pwned.txt"
    request = VoiceConversionRequestInput(
        request_id="cb-hostile-vc",
        requested_model="chatterbox_vc",
        model="chatterbox_vc",
        source_audio=_b64(tiny_wav_bytes(16000, 0.10)),
        target_audio=None,
        target_embedding=_b64(_hostile_pickle(marker)),
        generation_options=None,
        output_options=None,
    )

    with pytest.raises(ValueError, match="Unsupported target_embedding payload"):
        prepare_chatterbox_vc_inputs([request])

    assert not marker.exists(), "hostile pickle payload was executed!"


# ---------------------------------------------------------------------------
# Chatterbox conditionals: tensor-only payload round trip
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_conditionals_payload_is_tensor_and_primitive_only():
    """Every value in the serialized payload is a tensor or a primitive."""
    payload = _conditionals_to_payload(_make_conditionals())

    assert set(payload) == {"t3", "gen"}
    for section in payload.values():
        for value in section.values():
            assert torch.is_tensor(value) or isinstance(value, (str, int, float, bool, bytes, type(None)))


@pytest.mark.unit
def test_conditionals_payload_roundtrips_with_weights_only():
    """Serialized conditionals reload through the upstream Conditionals.load."""
    buf = io.BytesIO()
    torch.save(_conditionals_to_payload(_make_conditionals()), buf)
    buf.seek(0)

    loaded = Conditionals.load(buf, map_location="cpu")

    assert loaded.t3.speaker_emb.shape == (1, 10)
    assert loaded.gen["prompt_token"].shape == (1, 4)


@pytest.mark.unit
def test_embedding_handler_payload_reloads_via_conditionals_loader():
    """End-to-end serialization seam: embedding handler artifact -> wire -> loader."""
    from harmonyspeech.common.artifacts import AudioArtifact
    from harmonyspeech.endpoints.openai.wire import to_wire_embedding

    runner = ModelRunnerBase.__new__(ModelRunnerBase)
    runner.device = "cpu"
    runner._baseline_state = None
    conditionals = _make_conditionals()

    model = MagicMock()
    model.conds = None
    model.prepare_conditionals.side_effect = lambda buf, **kwargs: setattr(model, "conds", conditionals)
    runner.model = model

    request = SpeechEmbeddingRequestInput(
        request_id="cb-embed-001",
        requested_model="chatterbox",
        model="chattebox-embed",
        input_audio=_b64(b"reference audio"),
    )

    inputs = [AudioArtifact.from_numpy(np.zeros(16000, dtype=np.float32), 16000)]
    results = runner._execute_chatterbox_embedding(inputs, [_engine_request(request)])

    assert len(results) == 1
    output = results[0].result_data
    assert output.finish_reason == "stop"
    # Internal hop: the conditionals object itself is attached (no serialization).
    assert output.embedding_artifact is not None
    assert output.embedding_artifact.conditionals is conditionals

    # Wire boundary: the artifact serializes to the legacy 0-1-safe payload.
    reloaded = _load_conditionals_payload(to_wire_embedding(output), "input_embedding")
    assert reloaded.t3.speaker_emb.shape == (1, 10)
    # Per-request state is restored to the baseline (None) after the handler.
    assert model.conds is None


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_checkpoint_loads_with_weights_only(tmp_path):
    """A plain state dict checkpoint loads through the hardened loader."""
    checkpoint = {"model": {"weight": torch.zeros(2, 3)}, "iteration": 1}
    torch.save(checkpoint, tmp_path / "checkpoint.pth")

    loaded = load_or_download_model(str(tmp_path), "cpu")

    assert loaded["iteration"] == 1
    assert torch.equal(loaded["model"]["weight"], torch.zeros(2, 3))


@pytest.mark.unit
def test_hostile_checkpoint_rejected_by_default(tmp_path, monkeypatch):
    """A checkpoint that needs full unpickling is rejected unless explicitly allowed."""
    monkeypatch.delenv(ALLOW_UNSAFE_CHECKPOINT_LOAD_ENV, raising=False)

    torch.save({"model": TrustedOnlyCheckpointState()}, tmp_path / "checkpoint.pth")

    with pytest.raises(ValueError, match="weights_only=True"):
        load_or_download_model(str(tmp_path), "cpu")


@pytest.mark.unit
def test_checkpoint_escape_hatch_allows_full_pickle(tmp_path, monkeypatch):
    """HSE_ALLOW_UNSAFE_CHECKPOINT_LOAD=1 opts a trusted checkpoint back in."""
    monkeypatch.setenv(ALLOW_UNSAFE_CHECKPOINT_LOAD_ENV, "1")

    torch.save({"model": TrustedOnlyCheckpointState()}, tmp_path / "checkpoint.pth")

    loaded = load_or_download_model(str(tmp_path), "cpu")

    assert isinstance(loaded["model"], TrustedOnlyCheckpointState)

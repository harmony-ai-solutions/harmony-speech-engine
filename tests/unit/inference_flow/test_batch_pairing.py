"""Unit tests for batch pairing correctness (Phase 0-3).

Covers the three defects fixed in Phase 0-3:
* worker helpers receive their request explicitly, so the audio used for a
  request always belongs to that request (per-iteration pairing),
* mixed-type batches are partitioned before input preparation, so embedding
  and TTS requests in one batch both get their type-correct handler,
* handler entry asserts ``len(inputs) == len(requests_to_batch)``; the engine
  path converts a violation into per-request typed error results.
"""

import base64
from unittest.mock import patch

import numpy as np
import pytest
import torch
from chatterbox.models.t3.modules.cond_enc import T3Cond
from chatterbox.tts import Conditionals

from harmonyspeech.common import audio_codec
from harmonyspeech.common.artifacts import AudioArtifact, SpeakerArtifact
from harmonyspeech.common.config import DeviceConfig, ModelConfig
from harmonyspeech.common.inputs import SpeechEmbeddingRequestInput, TextToSpeechRequestInput
from harmonyspeech.common.outputs import SpeechEmbeddingRequestOutput, TextToSpeechRequestOutput
from harmonyspeech.common.request import EngineRequest
from harmonyspeech.task_handler.model_runner_base import ModelRunnerBase
from tests.audio_fixtures import tiny_wav_bytes

# Real (distinct) audio containers: the boundary decoder normalises every
# reference-audio field, so placeholder bytes would be rejected.
AUDIO_A = tiny_wav_bytes(16000, 0.10)
AUDIO_B = tiny_wav_bytes(16000, 0.20)


class PairingModel:
    """Chatterbox stand-in recording (text, reference-waveform) per generate call."""

    sr = 24000

    def __init__(self):
        self.conds = None
        self.ref_dict = None
        self.calls = []
        self.prepared_audio = []

    def prepare_conditionals(self, audio_buf, **kwargs):
        audio, _ = audio_codec.decode_audio(audio_buf.read())
        self.prepared_audio.append(audio)
        self.conds = Conditionals(
            t3=T3Cond(speaker_emb=torch.zeros(1, 4)),
            gen={"prompt_token": torch.zeros(1, 2, dtype=torch.long)},
        )

    def generate(self, text=None, audio=None, **kwargs):
        self.calls.append((text, self.prepared_audio[-1] if self.prepared_audio else None))
        return torch.zeros(1, 32)


def _make_runner(model=None, model_type="ChatterboxTTS") -> ModelRunnerBase:
    runner = ModelRunnerBase.__new__(ModelRunnerBase)
    runner.device = "cpu"
    runner._baseline_state = None
    runner.model = model if model is not None else PairingModel()
    runner.model_config = ModelConfig(
        name="cb",
        model="dummy/chatterbox",
        model_type=model_type,
        max_batch_size=4,
        device_config=DeviceConfig("cpu"),
        dtype="float32",
    )
    return runner


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def _tts_request(request_id: str, text: str, audio: bytes) -> EngineRequest:
    request_data = TextToSpeechRequestInput(
        request_id=request_id,
        requested_model="chatterbox",
        model="cb",
        input_text=text,
        mode="voice_cloning",
        input_audio=_b64(audio),
    )
    return EngineRequest(request_id=request_id, request_data=request_data, arrival_time=0.0)


def _embed_request(request_id: str, audio: bytes) -> EngineRequest:
    request_data = SpeechEmbeddingRequestInput(
        request_id=request_id,
        requested_model="chatterbox",
        model="cb",
        input_audio=_b64(audio),
    )
    return EngineRequest(request_id=request_id, request_data=request_data, arrival_time=0.0)


def _tts_params(request: EngineRequest) -> tuple:
    audio = base64.b64decode(request.request_data.input_audio)
    waveform, sample_rate = audio_codec.decode_audio(audio)
    speaker = SpeakerArtifact(reference_audio=AudioArtifact.from_numpy(waveform, sample_rate), backend="chatterbox")
    return (request.request_data.input_text, speaker, 0.5, 0.5, 0.8, 1.2, 1.0, 0.05)


def _expected(audio: bytes) -> np.ndarray:
    return audio_codec.decode_audio(audio)[0]


# ---------------------------------------------------------------------------
# Pairing: each request's own payload drives its own generation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_pairs_each_request_with_its_own_audio():
    runner = _make_runner()
    requests = [
        _tts_request("pair-1", "text-A", AUDIO_A),
        _tts_request("pair-2", "text-B", AUDIO_B),
    ]
    inputs = [_tts_params(request) for request in requests]

    results = runner._execute_chatterbox_tts(inputs, requests)

    assert [r.request_id for r in results] == ["pair-1", "pair-2"]
    assert all(r.result_data.finish_reason == "stop" for r in results)
    # The text from the prepared input and the audio from the request must
    # belong to the same iteration/request.
    assert [call[0] for call in runner.model.calls] == ["text-A", "text-B"]
    np.testing.assert_allclose(runner.model.calls[0][1], _expected(AUDIO_A))
    np.testing.assert_allclose(runner.model.calls[1][1], _expected(AUDIO_B))


@pytest.mark.unit
def test_pairs_each_request_with_its_own_audio_in_shuffled_order():
    runner = _make_runner()
    requests = [
        _tts_request("pair-2", "text-B", AUDIO_B),
        _tts_request("pair-1", "text-A", AUDIO_A),
    ]
    inputs = [_tts_params(request) for request in requests]

    results = runner._execute_chatterbox_tts(inputs, requests)

    assert [r.request_id for r in results] == ["pair-2", "pair-1"]
    assert [call[0] for call in runner.model.calls] == ["text-B", "text-A"]
    np.testing.assert_allclose(runner.model.calls[0][1], _expected(AUDIO_B))
    np.testing.assert_allclose(runner.model.calls[1][1], _expected(AUDIO_A))


# ---------------------------------------------------------------------------
# Mixed-type batches
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_mixed_embedding_and_tts_batch_dispatches_both_types():
    runner = _make_runner()
    embed_request = _embed_request("mixed-embed", AUDIO_A)
    tts_request = _tts_request("mixed-tts", "hello", AUDIO_A)

    results = runner.execute_model([embed_request, tts_request])

    assert len(results) == 2
    by_id = {result.request_id: result.result_data for result in results}
    assert set(by_id) == {"mixed-embed", "mixed-tts"}
    assert isinstance(by_id["mixed-embed"], SpeechEmbeddingRequestOutput)
    assert isinstance(by_id["mixed-tts"], TextToSpeechRequestOutput)
    assert all(output.finish_reason == "stop" for output in by_id.values())
    # The embedding result carries the conditionals by reference (no pickle hop).
    assert by_id["mixed-embed"].embedding_artifact is not None
    assert by_id["mixed-embed"].embedding_artifact.conditionals is not None


@pytest.mark.unit
def test_mixed_batch_dispatch_order_is_type_grouped_but_complete():
    """Reversed input order still yields every request exactly once."""
    runner = _make_runner()
    embed_request = _embed_request("mixed-embed", AUDIO_A)
    tts_request = _tts_request("mixed-tts", "hello", AUDIO_A)

    results = runner.execute_model([tts_request, embed_request])

    assert sorted(result.request_id for result in results) == ["mixed-embed", "mixed-tts"]


# ---------------------------------------------------------------------------
# Alignment invariants
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_handler_rejects_misaligned_inputs():
    runner = _make_runner()
    request = _tts_request("align-1", "hello", AUDIO_A)

    with pytest.raises(AssertionError, match="2 inputs vs 1 requests"):
        runner._execute_chatterbox_tts([_tts_params(request), _tts_params(request)], [request])


@pytest.mark.unit
def test_execute_model_converts_misalignment_to_typed_errors():
    """A prepare stage that returns the wrong number of inputs fails closed."""
    runner = _make_runner()
    requests = [
        _tts_request("align-2", "text-A", AUDIO_A),
        _tts_request("align-3", "text-B", AUDIO_B),
    ]

    with patch(
        "harmonyspeech.task_handler.model_runner_base.prepare_inputs",
        return_value=[_tts_params(requests[0])],
    ):
        results = runner.execute_model(requests)

    assert len(results) == 2
    for result in results:
        assert result.result_data.finish_reason == "error"
        assert "inputs vs" in result.result_data.error

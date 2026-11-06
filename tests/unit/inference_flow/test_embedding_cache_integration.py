"""Unit tests for the embedding cache wired into the handlers (Phase 7-2).

The handlers run against stub models that count their forward calls and derive
deterministic outputs from the audio they receive (the stub-model + runner-shell
pattern from ``test_batched_handlers.py``). What is proven here is the cache
*seam*: identical reference audio skips the model, every key component (audio,
exaggeration, VAD source, model identity) produces a miss, hits and misses mix
inside one batch without breaking request pairing, and the no-cache path calls
the model every time.
"""

import time
from dataclasses import dataclass
from unittest.mock import patch

import numpy as np
import pytest
import torch

from harmonyspeech.common import audio_codec
from harmonyspeech.common.artifacts import AudioArtifact, SpeakerArtifact, VadArtifact
from harmonyspeech.common.config import DeviceConfig, ModelConfig
from harmonyspeech.common.inputs import (
    SpeechEmbeddingRequestInput,
    TextToSpeechGenerationOptions,
    TextToSpeechRequestInput,
    VoiceConversionRequestInput,
)
from harmonyspeech.common.request import EngineRequest
from harmonyspeech.processing.embedding_cache import EmbeddingCache
from harmonyspeech.task_handler.model_runner_base import ModelRunnerBase

CB_IDENTITY = "cb|ChatterboxEmbedding|torch.float32|repo/chatterbox@main|v1"
OV2_IDENTITY = "ov2|OpenVoiceV2ToneConverterEncoder|torch.float32|repo/ov@main|v1"
OV1_IDENTITY = "ov1|OpenVoiceV1ToneConverterEncoder|torch.float32|repo/ov@main|v1"


def _clip(sample_rate: int = 16000, freq: float = 440.0, seconds: float = 0.10) -> np.ndarray:
    t = np.arange(int(sample_rate * seconds)) / sample_rate
    return (0.2 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _audio_artifact(freq: float = 440.0) -> AudioArtifact:
    return AudioArtifact.from_numpy(_clip(freq=freq), 16000)


# ---------------------------------------------------------------------------
# Stub models: deterministic output per input audio, forward-call counters
# ---------------------------------------------------------------------------


@dataclass
class FakeT3Cond:
    speaker_emb: torch.Tensor
    emotion_adv: torch.Tensor


@dataclass
class FakeConditionals:
    t3: FakeT3Cond
    gen: dict


class StubChatterboxEmbedModel:
    """``prepare_conditionals`` builds conditionals derived from the audio."""

    def __init__(self):
        self.conds = None
        self.sr = 24000
        self.calls = 0

    def prepare_conditionals(self, audio_buf, **kwargs):
        audio, _sr = audio_codec.decode_audio(audio_buf.read())
        self.calls += 1
        self.conds = FakeConditionals(
            t3=FakeT3Cond(
                speaker_emb=torch.tensor([float(audio[0]), float(audio[1])]),
                emotion_adv=torch.zeros(1),
            ),
            gen={"ge_bes": torch.tensor([float(audio.sum())])},
        )


class StubHarmonyEncoderModel:
    """Returns the first utterance features as the partial embeddings."""

    def __init__(self):
        self.calls = 0

    def __call__(self, utterances):
        self.calls += 1
        return utterances[:, :8].contiguous()


class StubOpenvoiceEncoderModel:
    """``ref_enc`` returns a deterministic embedding per spectrogram."""

    def __init__(self):
        self.calls = 0

    def ref_enc(self, spectrogram):
        self.calls += 1
        return torch.zeros(spectrogram.shape[0], 8)


class StubVCModel:
    """``set_target_voice`` derives the ref dict from the target audio."""

    def __init__(self):
        self.ref_dict = None
        self.sr = 24000
        self.target_calls = 0
        self.generate_calls = 0

    def set_target_voice(self, audio_buf):
        audio, _sr = audio_codec.decode_audio(audio_buf.read())
        self.target_calls += 1
        self.ref_dict = {
            "embedding": torch.tensor([float(audio[0])]),
            "ge_bes": torch.tensor([float(audio[-1])]),
        }

    def generate(self, audio=None, **kwargs):
        self.generate_calls += 1
        marker = self.ref_dict["embedding"][0] if self.ref_dict else 0.0
        return torch.zeros(1, 32) + marker


# ---------------------------------------------------------------------------
# Runner shell + request builders
# ---------------------------------------------------------------------------


def _runner(model, model_type: str, cache=None, identity: str | None = CB_IDENTITY) -> ModelRunnerBase:
    runner = ModelRunnerBase.__new__(ModelRunnerBase)
    runner.device = "cpu"
    runner._baseline_state = None
    runner.model = model
    runner.embedding_cache = cache
    runner.model_identity = identity
    runner.model_config = ModelConfig(
        name="stub",
        model="dummy/model",
        model_type=model_type,
        max_batch_size=4,
        device_config=DeviceConfig("cpu"),
        dtype="float32",
    )
    return runner


def _embed_request(request_id: str, artifact: AudioArtifact, exaggeration: float | None = None) -> EngineRequest:
    if exaggeration is None:
        request_data = SpeechEmbeddingRequestInput(
            request_id=request_id, requested_model="chatterbox", model="chatterbox"
        )
    else:
        request_data = TextToSpeechRequestInput(
            request_id=request_id,
            requested_model="chatterbox",
            model="chatterbox",
            input_text="hello",
            mode="voice_cloning",
            generation_options=TextToSpeechGenerationOptions(
                seed=None, style=None, speed=None, pitch=None, energy=None, exaggeration=exaggeration
            ),
        )
    return EngineRequest(
        request_id=request_id,
        request_data=request_data,
        arrival_time=0.0,
        artifacts={"input_audio": artifact},
    )


def _openvoice_request(request_id: str, artifact: AudioArtifact, vad_source: str) -> EngineRequest:
    request_data = SpeechEmbeddingRequestInput(request_id=request_id, requested_model="openvoice", model="openvoice")
    return EngineRequest(
        request_id=request_id,
        request_data=request_data,
        arrival_time=0.0,
        artifacts={
            "input_audio": artifact,
            "input_vad_data": VadArtifact(
                segments=[{"start": 0.0, "end": 0.1, "text": "hi"}], source=vad_source
            ),
        },
    )


def _vc_request(request_id: str) -> EngineRequest:
    request_data = VoiceConversionRequestInput(
        request_id=request_id,
        requested_model="chatterbox_vc",
        model="chatterbox_vc",
        source_audio="",
        target_audio=None,
        target_embedding=None,
        generation_options=None,
        output_options=None,
    )
    return EngineRequest(request_id=request_id, request_data=request_data, arrival_time=0.0)


def _conditionals_marker(artifact: SpeakerArtifact) -> float:
    return float(artifact.conditionals.gen["ge_bes"][0])


# ---------------------------------------------------------------------------
# Chatterbox embedding
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_chatterbox_identical_requests_skip_the_model_on_the_second_call():
    cache = EmbeddingCache(ttl_s=300)
    model = StubChatterboxEmbedModel()
    runner = _runner(model, "ChatterboxEmbedding", cache=cache)
    artifact = _audio_artifact()

    first = runner._execute_chatterbox_embedding([artifact], [_embed_request("cb-1", artifact)])
    second = runner._execute_chatterbox_embedding([artifact], [_embed_request("cb-2", artifact)])

    assert model.calls == 1
    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["stores"] == 1

    first_artifact = first[0].result_data.embedding_artifact
    second_artifact = second[0].result_data.embedding_artifact
    assert second[0].result_data.finish_reason == "stop"
    # Byte-identical downstream behavior, but a distinct object (Phase 0-2).
    assert second_artifact.conditionals is not first_artifact.conditionals
    assert torch.equal(second_artifact.conditionals.t3.speaker_emb, first_artifact.conditionals.t3.speaker_emb)
    assert torch.equal(second_artifact.conditionals.gen["ge_bes"], first_artifact.conditionals.gen["ge_bes"])


@pytest.mark.unit
def test_chatterbox_different_audio_is_a_miss():
    cache = EmbeddingCache(ttl_s=300)
    model = StubChatterboxEmbedModel()
    runner = _runner(model, "ChatterboxEmbedding", cache=cache)

    runner._execute_chatterbox_embedding([_audio_artifact(440.0)], [_embed_request("cb-1", _audio_artifact(440.0))])
    runner._execute_chatterbox_embedding([_audio_artifact(880.0)], [_embed_request("cb-2", _audio_artifact(880.0))])

    assert model.calls == 2
    assert cache.stats()["hits"] == 0


@pytest.mark.unit
def test_chatterbox_different_exaggeration_is_a_miss():
    cache = EmbeddingCache(ttl_s=300)
    model = StubChatterboxEmbedModel()
    runner = _runner(model, "ChatterboxEmbedding", cache=cache)
    artifact = _audio_artifact()

    runner._execute_chatterbox_embedding([artifact], [_embed_request("cb-1", artifact)])
    runner._execute_chatterbox_embedding([artifact], [_embed_request("cb-2", artifact, exaggeration=0.9)])

    assert model.calls == 2
    assert cache.stats()["hits"] == 0


@pytest.mark.unit
def test_chatterbox_expired_entry_recomputes(monkeypatch):
    now = 0.0
    monkeypatch.setattr(time, "monotonic", lambda: now)
    cache = EmbeddingCache(ttl_s=300)
    model = StubChatterboxEmbedModel()
    runner = _runner(model, "ChatterboxEmbedding", cache=cache)
    artifact = _audio_artifact()

    runner._execute_chatterbox_embedding([artifact], [_embed_request("cb-1", artifact)])
    now = 301.0
    runner._execute_chatterbox_embedding([artifact], [_embed_request("cb-2", artifact)])

    assert model.calls == 2
    stats = cache.stats()
    assert stats["ttl_expirations"] == 1
    assert stats["hits"] == 0


@pytest.mark.unit
def test_chatterbox_mixed_hit_and_miss_batch_preserves_request_order():
    cache = EmbeddingCache(ttl_s=300)
    model = StubChatterboxEmbedModel()
    runner = _runner(model, "ChatterboxEmbedding", cache=cache)
    clip_a, clip_b = _audio_artifact(440.0), _audio_artifact(880.0)

    warmed = runner._execute_chatterbox_embedding([clip_a], [_embed_request("warm", clip_a)])
    marker_a = _conditionals_marker(warmed[0].result_data.embedding_artifact)

    batch_inputs = [clip_a, clip_b]
    batch_requests = [_embed_request("mix-a", clip_a), _embed_request("mix-b", clip_b)]
    results = runner._execute_chatterbox_embedding(batch_inputs, batch_requests)

    # Only the miss ran the model, and outputs stay paired with their requests.
    assert model.calls == 2
    assert [result.request_id for result in results] == ["mix-a", "mix-b"]
    assert all(result.result_data.finish_reason == "stop" for result in results)
    assert _conditionals_marker(results[0].result_data.embedding_artifact) == marker_a
    assert _conditionals_marker(results[1].result_data.embedding_artifact) != marker_a


# ---------------------------------------------------------------------------
# HarmonySpeech encoder
# ---------------------------------------------------------------------------


def _passthrough_preprocess(samples, source_sr):
    return samples


def _frames_from_audio(preprocessed):
    return preprocessed[None, :]


def _run_hs_encoder(runner, artifacts, requests):
    with patch(
        "harmonyspeech.task_handler.model_runner_base.preprocess_wav",
        side_effect=_passthrough_preprocess,
    ), patch(
        "harmonyspeech.task_handler.model_runner_base.get_input_frames",
        side_effect=_frames_from_audio,
    ):
        return runner._execute_harmonyspeech_encoder(artifacts, requests)


@pytest.mark.unit
def test_harmonyspeech_encoder_identical_audio_hits():
    cache = EmbeddingCache(ttl_s=300)
    model = StubHarmonyEncoderModel()
    runner = _runner(model, "HarmonySpeechEncoder", cache=cache, identity="hs|HarmonySpeechEncoder|f32|r@main|v1")
    artifact = _audio_artifact()

    first = _run_hs_encoder(runner, [artifact], [_embed_request("hs-1", artifact)])
    second = _run_hs_encoder(runner, [artifact], [_embed_request("hs-2", artifact)])

    assert model.calls == 1
    assert cache.stats()["hits"] == 1
    embedding_first = first[0].result_data.embedding_artifact.embedding
    embedding_second = second[0].result_data.embedding_artifact.embedding
    assert torch.equal(embedding_second, embedding_first)


@pytest.mark.unit
def test_harmonyspeech_encoder_different_audio_misses():
    cache = EmbeddingCache(ttl_s=300)
    model = StubHarmonyEncoderModel()
    runner = _runner(model, "HarmonySpeechEncoder", cache=cache, identity="hs|HarmonySpeechEncoder|f32|r@main|v1")

    _run_hs_encoder(runner, [_audio_artifact(440.0)], [_embed_request("hs-1", _audio_artifact(440.0))])
    _run_hs_encoder(runner, [_audio_artifact(880.0)], [_embed_request("hs-2", _audio_artifact(880.0))])

    assert model.calls == 2
    assert cache.stats()["hits"] == 0


# ---------------------------------------------------------------------------
# OpenVoice tone-converter encoder (V1 + V2 embed steps)
# ---------------------------------------------------------------------------


def _fake_hf_config():
    import types

    return types.SimpleNamespace(
        data=types.SimpleNamespace(
            sampling_rate=16000, filter_length=400, hop_length=160, win_length=400
        )
    )


def _run_ov_encoder(runner, segments_per_request, requests):
    with patch(
        "harmonyspeech.task_handler.model_runner_base.get_model_config", return_value=_fake_hf_config()
    ), patch("harmonyspeech.task_handler.model_runner_base.get_model_flavour", return_value="default"):
        return runner._execute_openvoice_tone_converter_encoder(segments_per_request, requests)


@pytest.mark.unit
def test_openvoice_encoder_identical_audio_hits():
    cache = EmbeddingCache(ttl_s=300)
    model = StubOpenvoiceEncoderModel()
    runner = _runner(model, "OpenVoiceV2ToneConverterEncoder", cache=cache, identity=OV2_IDENTITY)
    artifact = _audio_artifact()
    request = _openvoice_request("ov-1", artifact, "whisper")

    first = _run_ov_encoder(runner, [[artifact]], [request])
    second = _run_ov_encoder(runner, [[artifact]], [_openvoice_request("ov-2", artifact, "whisper")])

    assert model.calls == 1
    assert cache.stats()["hits"] == 1
    assert torch.equal(
        second[0].result_data.embedding_artifact.embedding,
        first[0].result_data.embedding_artifact.embedding,
    )


@pytest.mark.unit
def test_openvoice_encoder_different_vad_source_misses():
    cache = EmbeddingCache(ttl_s=300)
    model = StubOpenvoiceEncoderModel()
    runner = _runner(model, "OpenVoiceV2ToneConverterEncoder", cache=cache, identity=OV2_IDENTITY)
    artifact = _audio_artifact()

    _run_ov_encoder(runner, [[artifact]], [_openvoice_request("ov-1", artifact, "whisper")])
    _run_ov_encoder(runner, [[artifact]], [_openvoice_request("ov-2", artifact, "silero")])

    assert model.calls == 2
    assert cache.stats()["hits"] == 0


@pytest.mark.unit
def test_openvoice_encoder_different_model_identity_misses():
    cache = EmbeddingCache(ttl_s=300)
    artifact = _audio_artifact()
    ov1 = _runner(
        StubOpenvoiceEncoderModel(), "OpenVoiceV1ToneConverterEncoder", cache=cache, identity=OV1_IDENTITY
    )
    ov2 = _runner(
        StubOpenvoiceEncoderModel(), "OpenVoiceV2ToneConverterEncoder", cache=cache, identity=OV2_IDENTITY
    )

    _run_ov_encoder(ov1, [[artifact]], [_openvoice_request("ov1", artifact, "whisper")])
    _run_ov_encoder(ov2, [[artifact]], [_openvoice_request("ov2", artifact, "whisper")])

    assert cache.stats()["hits"] == 0
    assert cache.stats()["stores"] == 2


@pytest.mark.unit
def test_openvoice_encoder_cache_resolution_failure_does_not_fail_the_request():
    model = StubOpenvoiceEncoderModel()
    runner = _runner(model, "OpenVoiceV2ToneConverterEncoder", cache=EmbeddingCache(ttl_s=300), identity=OV2_IDENTITY)
    artifact = _audio_artifact()
    # No VAD artifact or payload: key resolution must degrade to a plain miss.
    request = EngineRequest(
        request_id="ov-novad",
        request_data=SpeechEmbeddingRequestInput(
            request_id="ov-novad", requested_model="openvoice", model="openvoice"
        ),
        arrival_time=0.0,
        artifacts={"input_audio": artifact},
    )

    results = _run_ov_encoder(runner, [[artifact]], [request])

    assert model.calls == 1
    assert results[0].result_data.finish_reason == "stop"
    assert results[0].result_data.embedding_artifact is not None


# ---------------------------------------------------------------------------
# Chatterbox VC target seam
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_vc_identical_target_audio_skips_set_target_voice():
    cache = EmbeddingCache(ttl_s=300)
    model = StubVCModel()
    runner = _runner(model, "ChatterboxVC", cache=cache)
    target = _audio_artifact(440.0)
    source = _audio_artifact(880.0)

    def run(request_id):
        return runner._execute_chatterbox_vc(
            [(source, None, target)], [_vc_request(request_id)]
        )

    first = run("vc-1")
    second = run("vc-2")

    assert model.target_calls == 1
    assert model.generate_calls == 2
    assert cache.stats()["hits"] == 1
    assert torch.equal(
        second[0].result_data.audio_artifact.waveform,
        first[0].result_data.audio_artifact.waveform,
    )
    assert second[0].result_data.finish_reason == "stop"


# ---------------------------------------------------------------------------
# No-cache fast path
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_disabled_cache_calls_the_model_every_time():
    model = StubChatterboxEmbedModel()
    runner = _runner(model, "ChatterboxEmbedding", cache=None)
    artifact = _audio_artifact()

    for index in range(3):
        results = runner._execute_chatterbox_embedding([artifact], [_embed_request(f"cb-{index}", artifact)])
        assert results[0].result_data.finish_reason == "stop"

    assert model.calls == 3

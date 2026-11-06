"""Engine-level embedding-cache flow (Phase 7-3).

Two voice-clone requests with the *same* reference audio go through the real
engine (scheduler → pipeline routing → executor → handler → forwarding) against
stub Chatterbox models that count their conditioning passes:

* with the cache enabled the second request is served from the cache — the
  embedding model runs **once** for two requests, while the synthesis step still
  runs twice with byte-identical output (the served conditionals are deep copies);
* with ``embedding_cache_enabled=False`` the embedding model runs **twice**.

The first request is driven to completion before the second is admitted on
purpose: a cross-*request* cache proves itself across steps, not inside one
batch (a batch resolves all its cache slots before computing any of them).
"""

from dataclasses import dataclass
from unittest.mock import patch

import pytest
import torch

from harmonyspeech.common import audio_codec
from harmonyspeech.common.config import DeviceConfig, ModelConfig
from harmonyspeech.common.inputs import TextToSpeechRequestInput
from harmonyspeech.engine.harmonyspeech_engine import HarmonySpeechEngine
from harmonyspeech.task_handler.model_runner_base import ModelRunnerBase
from tests.audio_fixtures import tiny_wav_b64

REFERENCE_AUDIO = tiny_wav_b64(16000, 0.40)


# ---------------------------------------------------------------------------
# Stub models: count the embedding work, derive output from the conditionals
# ---------------------------------------------------------------------------


@dataclass
class _T3Cond:
    speaker_emb: torch.Tensor
    emotion_adv: torch.Tensor


@dataclass
class _Conditionals:
    t3: _T3Cond
    gen: dict


class EmbeddingStubModel:
    """``prepare_conditionals`` is the expensive pass the cache is meant to skip."""

    sr = 24000

    def __init__(self):
        self.conds = None
        self.calls = 0

    def prepare_conditionals(self, audio_buf, **_kwargs):
        waveform, _sr = audio_codec.decode_audio(audio_buf.read())
        self.calls += 1
        self.conds = _Conditionals(
            t3=_T3Cond(
                speaker_emb=torch.tensor([float(waveform[0]), float(waveform[1])]),
                emotion_adv=torch.zeros(1, 1, 1),
            ),
            # A stable, non-zero summary of the reference clip: two requests that
            # reuse the same conditioning must synthesize identical audio.
            gen={"ge_bes": torch.tensor([float(waveform.std())])},
        )


class SynthesisStubModel:
    """Chatterbox TTS stand-in: the waveform value is the cached conditioning."""

    sr = 24000

    def __init__(self):
        self.conds = None
        self.calls = 0

    def generate(self, text=None, **_kwargs):
        self.calls += 1
        gen = getattr(self.conds, "gen", None) or {}
        values = gen.get("ge_bes")
        marker = float(values[0]) if values is not None else 0.0
        return torch.full((1, 256), marker)


class StubExecutor:
    """Executor wrapper that drives a real ``ModelRunnerBase`` instance."""

    def __init__(self, runner):
        self.runner = runner

    def execute_model(self, requests_to_batch):
        return self.runner.execute_model(requests_to_batch)

    async def execute_model_async(self, requests_to_batch):
        return self.runner.execute_model(requests_to_batch)

    def check_health(self):
        return {"status": "ok"}


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def _model_config(name: str, model_type: str) -> ModelConfig:
    return ModelConfig(
        name=name,
        model="dummy/model",
        model_type=model_type,
        max_batch_size=4,
        device_config=DeviceConfig("cpu"),
        dtype="float32",
    )


def _runner(model, model_config, *, embedding_cache=None, identity=None) -> ModelRunnerBase:
    runner = ModelRunnerBase.__new__(ModelRunnerBase)
    runner.model = model
    runner.device = "cpu"
    runner.model_config = model_config
    runner._baseline_state = {"conds": None}
    runner.embedding_cache = embedding_cache
    runner.model_identity = identity
    return runner


def _build_engine(*, cache_enabled: bool):
    embed_config = _model_config("chatterbox_embedding", "ChatterboxEmbedding")
    tts_config = _model_config("chatterbox", "ChatterboxTTS")

    with patch("harmonyspeech.executor.cpu_executor.CPUExecutor._init_worker"):
        engine = HarmonySpeechEngine(
            model_configs=[embed_config, tts_config],
            log_stats=False,
            embedding_cache_enabled=cache_enabled,
        )

    embed_model = EmbeddingStubModel()
    tts_model = SynthesisStubModel()
    cache = engine.embedding_cache
    engine.model_executors = {
        "chatterbox_embedding": StubExecutor(
            _runner(
                embed_model,
                embed_config,
                embedding_cache=cache,
                identity="chatterbox_embedding|ChatterboxEmbedding|float32|dummy/model@main|v1" if cache else None,
            )
        ),
        "chatterbox": StubExecutor(_runner(tts_model, tts_config)),
    }
    return engine, embed_model, tts_model


def _clone_request(request_id: str, audio_b64: str = REFERENCE_AUDIO) -> TextToSpeechRequestInput:
    return TextToSpeechRequestInput(
        request_id=request_id,
        requested_model="chatterbox",
        model="chatterbox",
        input_text="hello there",
        mode="voice_cloning",
        input_audio=audio_b64,
    )


def _run_request(engine, request_id: str, audio_b64: str = REFERENCE_AUDIO, max_steps: int = 8):
    """Admit one clone request and drive the engine until it finishes."""
    request_data = _clone_request(request_id, audio_b64)
    engine.add_request(request_id, request_data)
    outputs = []
    for _ in range(max_steps):
        if not engine.has_unfinished_requests():
            break
        step_outputs, _forwarded = engine.step()
        outputs.extend(step_outputs)
    return [output for output in outputs if output.request_id == request_id]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_second_identical_clone_request_skips_the_embedding_model():
    engine, embed_model, tts_model = _build_engine(cache_enabled=True)

    first = _run_request(engine, "clone-1")
    second = _run_request(engine, "clone-2")

    assert len(first) == 1 and len(second) == 1
    assert first[0].finish_reason == "stop"
    assert second[0].finish_reason == "stop"
    # The whole point of Phase 7: one conditioning pass for two requests.
    assert embed_model.calls == 1
    assert tts_model.calls == 2  # synthesis is never cached

    waveform_first = first[0].audio_artifact.waveform
    waveform_second = second[0].audio_artifact.waveform
    assert torch.equal(waveform_second, waveform_first)
    assert float(waveform_first[0, 0]) != 0.0

    stats = engine.get_embedding_cache_stats()
    assert stats["enabled"] is True
    assert stats["entries"] == 1
    assert stats["stores"] == 1
    assert stats["hits"] == 1
    assert stats["hit_ratio"] == pytest.approx(0.5)


@pytest.mark.integration
def test_disabled_cache_recomputes_the_embedding_for_every_request():
    engine, embed_model, tts_model = _build_engine(cache_enabled=False)

    first = _run_request(engine, "clone-off-1")
    second = _run_request(engine, "clone-off-2")

    assert first[0].finish_reason == "stop"
    assert second[0].finish_reason == "stop"
    assert embed_model.calls == 2
    assert tts_model.calls == 2
    # Same audio either way: the cache changes cost, never the result.
    assert torch.equal(second[0].audio_artifact.waveform, first[0].audio_artifact.waveform)

    stats = engine.get_embedding_cache_stats()
    assert stats["enabled"] is False
    assert stats["entries"] == 0
    assert stats["hits"] == 0


@pytest.mark.integration
def test_different_reference_audio_is_a_cache_miss():
    engine, embed_model, _tts_model = _build_engine(cache_enabled=True)

    first = _run_request(engine, "clone-a")
    second = _run_request(engine, "clone-b", audio_b64=tiny_wav_b64(16000, 0.60))

    assert first[0].finish_reason == "stop"
    assert second[0].finish_reason == "stop"
    # Same text/model but a different clip → a different content hash → miss.
    assert embed_model.calls == 2
    stats = engine.get_embedding_cache_stats()
    assert stats["hits"] == 0
    assert stats["entries"] == 2
    assert stats["stores"] == 2

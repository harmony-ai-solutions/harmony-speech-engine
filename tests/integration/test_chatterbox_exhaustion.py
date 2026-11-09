"""Chatterbox AR exhaustion semantics on CPU (Phase 9-6).

The owner's exhaustion scenarios (2026-11-07) proven locally before the
benchmark-host run: a real ``HarmonySpeechEngine`` loop, scheduler, 9-4 capacity
admission, 9-2 KV-cache pool and the 9-3/9-5 handlers drive a **tiny in-process
T3**, a stub S3Gen/watermarker/tokenizer and (for the combined scenario) the
real Chatterbox embedding handler with the real cross-request cache. No weights,
CPU only.

Pinned here:

* 100 distinct-voice requests against a pool sized for ~8 slots: pool refusals
  degrade to the sequential fallback, every request is answered exactly once,
  no output is paired with another request's voice, and the refusal metric
  rises;
* a deadline under exhaustion errors only the expired request;
* an abort storm (waiting + in-flight) loses no survivor and frees every slot;
* combined exhaustion (pool full + cold embedding cache + 20 concurrent) loses
  nothing and reports honest ``hse_ar_cache_*`` / ``hse_ar_decode_*`` metrics;
* degradation semantics: fewer pool slots than requests still completes every
  request (more engine steps, never a crash, OOM or loss).

Pairing proof: each request's conditionals carry a numeric ``marker`` in
``Conditionals.gen``; the stub S3Gen writes it into the first sample of the
returned waveform, so an output that carries a *different* request's marker is a
pairing error. The sequential fallback returns zeros, which is accepted as
"this request, served by the fallback".
"""

import re
from unittest.mock import patch

import pytest
import torch
from chatterbox.models.t3.llama_configs import LLAMA_CONFIGS
from chatterbox.models.t3.modules.cond_enc import T3Cond
from chatterbox.tts import Conditionals

from harmonyspeech.common.artifacts import AudioArtifact, SpeakerArtifact
from harmonyspeech.common.config import DeviceConfig, ModelConfig
from harmonyspeech.common.inputs import TextToSpeechGenerationOptions, TextToSpeechRequestInput
from harmonyspeech.engine.ar_cache import ARCachePool
from harmonyspeech.engine.ar_cache_metrics import render_ar_cache_metrics
from harmonyspeech.engine.ar_decode_metrics import render_ar_decode_metrics
from harmonyspeech.engine.harmonyspeech_engine import HarmonySpeechEngine
from harmonyspeech.task_handler.model_runner_base import ModelRunnerBase
from tests.integration.test_chatterbox_ar_engine import (
    _TINY_LLAMA,
    _StubModel,
    _tiny_hp,
    _drive,
)

AR_MODEL = "chatterbox-ar"
EMBED_MODEL = "chatterbox-embedding"

#: Token capacity per slot used by these tests. A "hi" prefill is 11 tokens, so
#: 24 leaves 13 decode steps per row — enough to span several engine steps
#: without a long test runtime.
CAP_TOKENS = 24

#: Slab bytes for the tiny T3: 2 layers × 2 (K/V) × cap × 4 kv-heads × 8 dim × 4 B.
SLAB_BYTES = 2 * 2 * CAP_TOKENS * 4 * 8 * 4


def _budget_for_slots(slots: int) -> int:
    return SLAB_BYTES * slots


class _MarkerS3Gen:
    """Stub S3Gen: writes the request marker into the waveform's first sample."""

    def inference(self, speech_tokens, ref_dict, **kwargs):
        marker = 0.0
        if isinstance(ref_dict, dict) and ref_dict.get("marker") is not None:
            marker = float(ref_dict["marker"])
        return torch.full((1, 16), marker), None


def _conditionals(marker: int) -> Conditionals:
    generator = torch.Generator().manual_seed(9000 + marker)
    hp = _tiny_hp()
    return Conditionals(
        T3Cond(
            speaker_emb=torch.randn(1, hp.speaker_embed_size, generator=generator),
            cond_prompt_speech_tokens=torch.randint(0, 80, (1, 4), generator=generator),
            emotion_adv=0.5 * torch.ones(1, 1, 1),
        ),
        {"marker": marker},
    )


def _model_config(name: str, model_type: str, max_batch_size: int) -> ModelConfig:
    return ModelConfig(
        name=name,
        model="dummy/chatterbox",
        model_type=model_type,
        max_batch_size=max_batch_size,
        device_config=DeviceConfig("cpu"),
        dtype="float32",
    )


def _tts_request(request_id: str, *, marker: int | None = None, seed: int = 7, requested_model: str = AR_MODEL):
    return TextToSpeechRequestInput(
        request_id=request_id,
        requested_model=requested_model,
        model=requested_model,
        input_text="hi",
        mode="single_speaker_tts",
        generation_options=TextToSpeechGenerationOptions(
            seed=seed, style=None, speed=None, pitch=None, energy=None
        ),
    )


def _conditionals_artifact(marker: int) -> dict:
    return {"input_embedding": SpeakerArtifact(conditionals=_conditionals(marker), backend="chatterbox")}


def _t3_model():
    hp = _tiny_hp()
    torch.manual_seed(20261111)
    from harmonyspeech.modeling.models.chatterbox.t3_ar import ChatterboxT3AR

    t3 = ChatterboxT3AR(hp).eval()
    t3.requires_grad_(False)
    model = _StubModel(t3, _conditionals(0))
    model.s3gen = _MarkerS3Gen()
    return model


def _make_runner(config, model, pool, *, embedding_cache=None):
    runner = ModelRunnerBase.__new__(ModelRunnerBase)
    runner.device = "cpu"
    runner.model = model
    runner.model_config = config
    runner.ar_cache_pool = pool
    runner.embedding_cache = embedding_cache
    runner._baseline_state = None
    runner._ar_runner = None
    runner.model_identity = None
    runner._capture_baseline_state()
    if embedding_cache is not None:
        # Production refreshes this at model load; the cache is keyed by it.
        runner._refresh_model_identity()
    return runner


def _build_engine(monkeypatch, *, pool_slots: int, max_batch_size: int, with_embedding: bool = False):
    monkeypatch.setitem(LLAMA_CONFIGS, "tiny_ar_engine", dict(_TINY_LLAMA))
    config = _model_config(AR_MODEL, "ChatterboxTTS", max_batch_size)
    configs = [config]
    if with_embedding:
        configs.append(_model_config(EMBED_MODEL, "ChatterboxEmbedding", 32))

    with patch("harmonyspeech.executor.cpu_executor.CPUExecutor._init_worker"):
        engine = HarmonySpeechEngine(model_configs=configs, log_stats=False, request_timeout_s=600.0)

    pool = ARCachePool(
        budget_bytes=_budget_for_slots(pool_slots),
        device="cpu",
        default_cap_tokens=CAP_TOKENS,
    )
    engine.ar_cache_pool = pool

    tts_runner = _make_runner(config, _t3_model(), pool, embedding_cache=engine.embedding_cache)
    # Production registers the arena at model load (`GPUModelRunner.load_model` /
    # `CPUModelRunner`), before the scheduler admits anything. Do the same here so
    # capacity admission is live from the first step.
    assert tts_runner._init_ar_runner() is not None
    executors = {AR_MODEL: tts_runner}
    if with_embedding:
        embed_config = configs[1]
        embed_model = _StubEmbeddingModel()
        executors[EMBED_MODEL] = _make_runner(
            embed_config, embed_model, None, embedding_cache=engine.embedding_cache
        )
    engine.model_executors = executors
    return engine, tts_runner, pool


class _StubEmbeddingModel:
    """Stub ChatterboxEmbedding model: conditionals keyed by the clip's first sample."""

    def __init__(self):
        self.sr = 24000
        self.device = "cpu"
        self.conds = None

    def prepare_conditionals(self, audio_buf):
        import soundfile as sf

        samples, _rate = sf.read(audio_buf, dtype="float32", always_2d=False)
        marker = int(round(float(samples.reshape(-1)[0]) * 100))
        self.conds = _conditionals(marker)


def _audio_for(marker: int) -> AudioArtifact:
    import numpy as np

    samples = np.zeros(2400, dtype=np.float32)
    # PCM_16 wire encoding clips to [-1, 1], so encode the marker as a small
    # amplitude fraction (0.01 steps) instead of a raw integer.
    samples[0] = float(marker) / 100.0
    return AudioArtifact.from_numpy(samples, 24000)


def _pairing_ok(output, marker: int) -> bool:
    """An output belongs to ``marker`` when it carries it, or is the zero fallback."""
    waveform = output.audio_artifact.waveform
    first = float(waveform.reshape(-1)[0])
    return first == float(marker) or first == 0.0


# ---------------------------------------------------------------------------
# 1. 100 distinct-voice requests against a ~8-slot pool
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_tight_pool_100_distinct_voice_flood_zero_loss_zero_pairing(monkeypatch):
    engine, runner, pool = _build_engine(monkeypatch, pool_slots=8, max_batch_size=100)

    count = 100
    for index in range(count):
        request_id = f"voice-{index}"
        engine.add_request(
            request_id,
            _tts_request(request_id, seed=1000 + index),
            artifacts=_conditionals_artifact(index + 1),
        )

    outputs = _drive(engine, max_steps=4000)

    # Zero request loss: every submitted id has exactly one final output.
    assert len(outputs) == count
    assert set(outputs) == {f"voice-{index}" for index in range(count)}
    for index in range(count):
        request_id = f"voice-{index}"
        assert len(outputs[request_id]) == 1, f"{request_id} produced {len(outputs[request_id])} outputs"
        result = outputs[request_id][0]
        assert result.finish_reason == "stop", (request_id, result.error)
        # Zero pairing errors: an output never carries another request's marker.
        assert _pairing_ok(result, index + 1), f"{request_id} was paired with another voice"

    # The pool really was the bottleneck, but Phase 12-8's `rows_per_request`
    # admission now prevents over-admission: surplus requests wait for free
    # slots instead of being refused. Nothing leaks.
    stats = pool.stats()
    assert stats["grants_total"] > 0
    assert stats["refusals_total"] == 0, "admission control prevents over-admission"
    assert stats["slots"] == 8
    assert stats["slots_active"] == 0
    assert stats["releases_total"] == stats["grants_total"]

    # Honest metrics: the refusal counter stays zero (no over-admission) and the
    # decode counters advanced (tokens/step is a real ratio, not a placeholder).
    cache_metrics = render_ar_cache_metrics(engine.get_ar_cache_stats())
    assert re.search(r"^hse_ar_cache_refusals_total 0$", cache_metrics, re.M)
    decode = engine.get_ar_decode_stats()
    entry = decode["models"][AR_MODEL]
    assert entry["tokens_total"] > 0
    assert entry["steps_total"] > 0
    assert entry["tokens_per_step"] > 0
    assert entry["active_rows"] == 0
    decode_metrics = render_ar_decode_metrics(decode)
    assert re.search(rf'^hse_ar_decode_tokens_total\{{model="{AR_MODEL}"\}} [1-9]\d*$', decode_metrics, re.M)


# ---------------------------------------------------------------------------
# 2. Deadline under exhaustion: only the expired request errors
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_deadline_under_pool_exhaustion_errors_only_expired_request(monkeypatch):
    engine, runner, pool = _build_engine(monkeypatch, pool_slots=8, max_batch_size=100)

    count = 12
    for index in range(count):
        request_id = f"deadline-{index}"
        engine.add_request(
            request_id,
            _tts_request(request_id, seed=2000 + index),
            artifacts=_conditionals_artifact(index + 1),
        )

    first_outputs, _forwarded = engine.step()
    collected = {}
    for output in first_outputs:
        collected.setdefault(output.request_id, []).append(output)
    assert pool.stats()["slots_active"] > 0
    running_ids = [request.request_id for request in engine.scheduler.running]
    waiting_ids = [request.request_id for request in engine.scheduler.waiting]
    assert running_ids and waiting_ids, "the tight pool must leave work waiting"

    expired_running = running_ids[0]
    expired_waiting = waiting_ids[0]
    for request in list(engine.scheduler.running) + list(engine.scheduler.waiting):
        if request.request_id in (expired_running, expired_waiting):
            request.deadline = 0.0

    outputs, _forwarded = engine.step()

    expired_outputs = {output.request_id: output for output in outputs}
    for output in outputs:
        collected.setdefault(output.request_id, []).append(output)
    for request_id in (expired_running, expired_waiting):
        assert request_id in expired_outputs
        assert expired_outputs[request_id].finish_reason == "error"
        assert "timeout" in (expired_outputs[request_id].error or "").lower()
    # The expired running row released its slots; the loop lives.
    assert not runner._ar_runner.has_row(expired_running)

    remaining = _drive(engine, max_steps=4000)
    for request_id, results in remaining.items():
        collected.setdefault(request_id, []).extend(results)
    survivors = {f"deadline-{index}" for index in range(count)} - {expired_running, expired_waiting}
    assert set(collected) == {f"deadline-{index}" for index in range(count)}
    for request_id in survivors:
        results = collected[request_id]
        assert len(results) == 1, (request_id, len(results))
        assert results[-1].finish_reason == "stop", (request_id, results[-1].error)
    assert pool.stats()["slots_active"] == 0


# ---------------------------------------------------------------------------
# 3. Abort storm mid-decode: survivors complete, every slot is freed
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_abort_storm_mid_decode_loses_no_survivor(monkeypatch):
    engine, runner, pool = _build_engine(monkeypatch, pool_slots=8, max_batch_size=100)

    count = 40
    for index in range(count):
        request_id = f"abort-{index}"
        engine.add_request(
            request_id,
            _tts_request(request_id, seed=3000 + index),
            artifacts=_conditionals_artifact(index + 1),
        )

    # Decode a few steps so some rows are in flight and the rest wait.
    collected = {}
    for _ in range(3):
        outputs, _forwarded = engine.step()
        for output in outputs:
            collected.setdefault(output.request_id, []).append(output)

    in_flight = {request.request_id for request in engine.scheduler.running}
    waiting = {request.request_id for request in engine.scheduler.waiting}
    assert in_flight and waiting

    abort_waiting = set(sorted(waiting)[:10])
    abort_in_flight = set(sorted(in_flight)[:10])
    for request_id in abort_waiting | abort_in_flight:
        engine.abort_request(request_id)

    survivors = {f"abort-{index}" for index in range(count)} - abort_waiting - abort_in_flight
    for request_id, results in _drive(engine, max_steps=4000).items():
        collected.setdefault(request_id, []).extend(results)

    # Aborted requests produce no client output; survivors are all answered once.
    assert set(collected) == survivors
    for request_id, results in collected.items():
        assert len(results) == 1, f"{request_id} produced {len(results)} outputs"
        assert results[-1].finish_reason == "stop", (request_id, results[-1].error)
    for request_id in abort_in_flight:
        assert not runner._ar_runner.has_row(request_id)
    # No slot leaked through the storm.
    stats = pool.stats()
    assert stats["slots_active"] == 0
    assert stats["releases_total"] == stats["grants_total"]


# ---------------------------------------------------------------------------
# 4. Combined exhaustion: pool full + cold embedding cache + 20 concurrent
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_combined_pool_full_and_cold_embedding_cache_zero_loss(monkeypatch):
    engine, runner, pool = _build_engine(
        monkeypatch, pool_slots=8, max_batch_size=20, with_embedding=True
    )

    count = 20
    for index in range(count):
        request_id = f"combined-{index}"
        engine.add_request(
            request_id,
            _tts_request(request_id, seed=4000 + index, requested_model="chatterbox"),
            artifacts={"input_audio": _audio_for(index + 1)},
        )

    outputs = _drive(engine, max_steps=4000)

    assert set(outputs) == {f"combined-{index}" for index in range(count)}
    for index in range(count):
        request_id = f"combined-{index}"
        assert len(outputs[request_id]) == 1
        result = outputs[request_id][0]
        assert result.finish_reason == "stop", (request_id, result.error)
        assert _pairing_ok(result, index + 1), f"{request_id} was paired with another voice"

    # The embedding cache was genuinely cold: every distinct reference missed.
    cache = engine.get_embedding_cache_stats()
    assert cache["enabled"] is True
    assert cache["misses"] == count
    assert cache["hits"] == 0
    assert cache["entries"] == count

    # Honest pool + decode metrics under the combined load. Phase 12-8's
    # `rows_per_request` admission keeps the 20-request flood from overflowing
    # the 8-slot pool, so surplus requests queue rather than being refused.
    stats = pool.stats()
    assert stats["refusals_total"] == 0, "admission control prevents over-admission"
    assert stats["slots_active"] == 0
    cache_metrics = render_ar_cache_metrics(engine.get_ar_cache_stats())
    assert re.search(r"^hse_ar_cache_refusals_total 0$", cache_metrics, re.M)
    entry = engine.get_ar_decode_stats()["models"][AR_MODEL]
    assert entry["tokens_total"] > 0 and entry["steps_total"] > 0
    assert entry["tokens_per_step"] > 0


# ---------------------------------------------------------------------------
# 5. Degradation: fewer slots than requests completes, just in more steps
# ---------------------------------------------------------------------------


def _steps_to_finish(monkeypatch, *, pool_slots: int, count: int) -> tuple[int, dict]:
    engine, _runner, pool = _build_engine(
        monkeypatch, pool_slots=pool_slots, max_batch_size=count
    )
    for index in range(count):
        request_id = f"degrade-{index}"
        engine.add_request(
            request_id,
            _tts_request(request_id, seed=5000 + index),
            artifacts=_conditionals_artifact(index + 1),
        )
    steps = 0
    collected: dict = {}
    while engine.has_unfinished_requests() and steps < 4000:
        steps += 1
        outputs, _forwarded = engine.step()
        for output in outputs:
            collected.setdefault(output.request_id, []).append(output)
    return steps, {"outputs": collected, "pool": pool.stats()}


@pytest.mark.integration
def test_degradation_fewer_slots_than_requests_completes_every_request(monkeypatch):
    count = 12

    tight_steps, tight = _steps_to_finish(monkeypatch, pool_slots=2, count=count)
    roomy_steps, roomy = _steps_to_finish(monkeypatch, pool_slots=2 * count, count=count)

    # Never lose a request, never crash, never exceed the pool.
    for label, run in (("tight", tight), ("roomy", roomy)):
        assert set(run["outputs"]) == {f"degrade-{index}" for index in range(count)}, label
        for request_id, results in run["outputs"].items():
            assert len(results) == 1, (label, request_id)
            assert results[-1].finish_reason == "stop", (label, request_id, results[-1].error)
        assert run["pool"]["slots_active"] == 0
    assert tight["pool"]["slots"] == 2
    assert roomy["pool"]["slots"] == 2 * count
    assert tight["pool"]["refusals_total"] == 0
    assert tight["pool"]["slots_free"] == 2
    # Fewer slots means serialized admission: the tight run must at least take
    # one engine step per request. (A solo admission uses the sequential loop,
    # so it can finish inside a single step; the roomy run decodes the whole
    # batch token-by-token through the AR runner.) This is the "slower, never
    # broken" degradation statement.
    assert tight_steps >= count
    assert roomy_steps >= 1

"""End-to-end Chatterbox AR decode through the real engine loop (Phase 9-3).

A real ``HarmonySpeechEngine`` + scheduler drives a real ``ModelRunnerBase``
handler with a tiny in-process T3, a stub S3Gen/watermarker/tokenizer and a real
9-2 pool. No weights, CPU-only.

Pinned here:

* four mixed-length seeded requests decode across many engine steps and each
  yields exactly one final, correctly paired audio result;
* abort mid-decode frees that request's row and cache slots without disturbing
  its neighbour;
* a deadline expiry mid-decode errors that request alone and the loop lives.
"""

from unittest.mock import patch

import pytest
import torch
from chatterbox.models.t3.llama_configs import LLAMA_CONFIGS
from chatterbox.models.t3.modules.cond_enc import T3Cond
from chatterbox.models.t3.modules.t3_config import T3Config
from chatterbox.tts import Conditionals

from harmonyspeech.common.config import DeviceConfig, ModelConfig
from harmonyspeech.common.inputs import TextToSpeechGenerationOptions, TextToSpeechRequestInput
from harmonyspeech.common.outputs import TextToSpeechRequestOutput
from harmonyspeech.engine.ar_cache import ARCachePool
from harmonyspeech.engine.harmonyspeech_engine import HarmonySpeechEngine
from harmonyspeech.modeling.models.chatterbox.t3_ar import ChatterboxT3AR
from harmonyspeech.task_handler.model_runner_base import ModelRunnerBase

AR_MODEL = "chatterbox-ar"
ARENA_BUDGET = 8 * 1024 * 1024

_TINY_LLAMA = {
    "vocab_size": 8,
    "max_position_embeddings": 512,
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "head_dim": 8,
    "num_key_value_heads": 4,
    "rms_norm_eps": 1e-05,
    "rope_theta": 500000.0,
    "tie_word_embeddings": False,
    "attention_bias": False,
    "attention_dropout": 0.0,
    "initializer_range": 0.02,
    "mlp_bias": False,
    "model_type": "llama",
    "pretraining_tp": 1,
    "torch_dtype": "float32",
    "use_cache": True,
    "attn_implementation": "sdpa",
}


def _tiny_hp():
    hp = T3Config.english_only()
    hp.llama_config_name = "tiny_ar_engine"
    hp.text_tokens_dict_size = 300
    hp.speech_tokens_dict_size = 96
    hp.start_speech_token = 90
    hp.stop_speech_token = 91
    hp.max_text_tokens = 32
    hp.max_speech_tokens = 64
    hp.use_perceiver_resampler = False
    hp.emotion_adv = True
    return hp


class _StubTokenizer:
    def text_to_tokens(self, text):
        words = max(1, len(str(text).split()))
        return torch.arange(5, 5 + words, dtype=torch.long)


class _StubS3Gen:
    def __init__(self):
        self.calls = []

    def inference(self, speech_tokens, ref_dict, **kwargs):
        self.calls.append(speech_tokens.detach().cpu().clone())
        return torch.zeros(1, max(8, int(speech_tokens.numel()) * 4)), None


class _StubWatermarker:
    def apply_watermark(self, wav, sample_rate):
        return wav


class _StubModel:
    def __init__(self, t3, conds):
        self.t3 = t3
        self.conds = conds
        self.sr = 24000
        self.device = "cpu"
        self.s3gen = _StubS3Gen()
        self.watermarker = _StubWatermarker()
        self.tokenizer = _StubTokenizer()
        self.generate_calls = []

    def generate(self, text=None, **kwargs):
        self.generate_calls.append(text)
        return torch.zeros(1, 16)


def _model_config(max_batch_size=4):
    return ModelConfig(
        name=AR_MODEL,
        model="dummy/chatterbox",
        model_type="ChatterboxTTS",
        max_batch_size=max_batch_size,
        device_config=DeviceConfig("cpu"),
        dtype="float32",
    )


def _tts_request(request_id, *, text="hello world", seed=7):
    return TextToSpeechRequestInput(
        request_id=request_id,
        requested_model=AR_MODEL,
        model=AR_MODEL,
        input_text=text,
        mode="single_speaker_tts",
        generation_options=TextToSpeechGenerationOptions(
            seed=seed, style=None, speed=None, pitch=None, energy=None
        ),
    )


@pytest.fixture
def engine_and_model(monkeypatch):
    monkeypatch.setitem(LLAMA_CONFIGS, "tiny_ar_engine", dict(_TINY_LLAMA))
    hp = _tiny_hp()
    torch.manual_seed(20261110)
    t3 = ChatterboxT3AR(hp).eval()
    t3.requires_grad_(False)
    generator = torch.Generator().manual_seed(4242)
    conds = Conditionals(
        T3Cond(
            speaker_emb=torch.randn(1, hp.speaker_embed_size, generator=generator),
            cond_prompt_speech_tokens=torch.randint(0, 80, (1, 4), generator=generator),
            emotion_adv=0.5 * torch.ones(1, 1, 1),
        ),
        {},
    )
    model = _StubModel(t3, conds)

    config = _model_config()
    with patch("harmonyspeech.executor.cpu_executor.CPUExecutor._init_worker"):
        engine = HarmonySpeechEngine(
            model_configs=[config],
            log_stats=False,
            ar_cache_enabled=True,
            ar_cache_budget_bytes=ARENA_BUDGET,
        )

    pool = ARCachePool(budget_bytes=ARENA_BUDGET, device="cpu", default_cap_tokens=64)
    engine.ar_cache_pool = pool

    runner = ModelRunnerBase.__new__(ModelRunnerBase)
    runner.device = "cpu"
    runner.model = model
    runner.model_config = config
    runner.ar_cache_pool = pool
    runner._baseline_state = None
    runner._ar_runner = None
    runner._capture_baseline_state()
    runner._step_calls = []
    original_execute = runner.execute_model

    def execute_and_record(requests):
        runner._step_calls.append([request.request_id for request in requests])
        return original_execute(requests)

    runner.execute_model = execute_and_record
    engine.model_executors = {config.name: runner}
    return engine, model, runner


def _drive(engine, *, max_steps=256):
    collected: dict[str, list] = {}
    for _ in range(max_steps):
        if not engine.has_unfinished_requests():
            break
        outputs, _forwarded = engine.step()
        for output in outputs:
            collected.setdefault(output.request_id, []).append(output)
    return collected


@pytest.mark.integration
def test_four_mixed_length_requests_finish_paired(engine_and_model):
    engine, model, runner = engine_and_model
    specs = [
        ("one", "hello", 101),
        ("two", "hello there friend", 102),
        ("three", "hi", 103),
        ("four", "hello there my good friend", 104),
    ]
    for request_id, text, seed in specs:
        engine.add_request(request_id, _tts_request(request_id, text=text, seed=seed))

    outputs = _drive(engine)

    assert set(outputs) == {spec[0] for spec in specs}
    for request_id, _text, _seed in specs:
        finals = outputs[request_id]
        assert len(finals) == 1, "exactly one client-visible response per request"
        assert isinstance(finals[0], TextToSpeechRequestOutput)
        assert finals[0].finish_reason == "stop"
        assert finals[0].audio_artifact is not None
        assert finals[0].audio_artifact.waveform.numel() > 0

    # All four were admitted into one step and decoded together (pairing).
    assert runner._step_calls[0] == [spec[0] for spec in specs]
    assert any(len(call) > 1 for call in runner._step_calls)
    assert engine.ar_cache_pool.capacity(AR_MODEL)["slots_active"] == 0
    assert engine.get_num_unfinished_requests() == 0
    # One S3Gen call per request, and the per-request token streams differ
    # (mixed lengths are not collapsed into one result).
    assert len(model.s3gen.calls) == 4
    assert len({tuple(call.tolist()) for call in model.s3gen.calls}) > 1


@pytest.mark.integration
def test_abort_mid_decode_frees_row_and_slots(engine_and_model):
    engine, _model, runner = engine_and_model
    engine.add_request("keep", _tts_request("keep", seed=201))
    engine.add_request("abort-me", _tts_request("abort-me", seed=202))
    engine.step()

    capacity = engine.ar_cache_pool.capacity(AR_MODEL)
    assert capacity["slots_active"] == 4, "two English rows = two slots each"

    engine.abort_request("abort-me")

    assert [request.request_id for request in engine.scheduler.running] == ["keep"]
    assert engine.ar_cache_pool.capacity(AR_MODEL)["slots_active"] == 2
    assert not runner._ar_runner.has_row("abort-me")

    outputs = _drive(engine)
    assert outputs["keep"][-1].finish_reason == "stop"
    assert "abort-me" not in outputs
    assert engine.ar_cache_pool.capacity(AR_MODEL)["slots_active"] == 0


@pytest.mark.integration
def test_deadline_mid_decode_errors_request_alone(engine_and_model):
    engine, _model, runner = engine_and_model
    engine.add_request("slow", _tts_request("slow", seed=301))
    engine.add_request("healthy", _tts_request("healthy", seed=302))
    engine.step()
    assert engine.ar_cache_pool.capacity(AR_MODEL)["slots_active"] == 4

    for request in engine.scheduler.running:
        if request.request_id == "slow":
            request.deadline = 0.0  # already expired at the next step

    outputs, forwarded = engine.step()

    assert forwarded == []
    assert [output.request_id for output in outputs] == ["slow"]
    assert outputs[0].finish_reason == "error"
    assert "exceeded timeout" in outputs[0].error
    assert [request.request_id for request in engine.scheduler.running] == ["healthy"]
    assert engine.ar_cache_pool.capacity(AR_MODEL)["slots_active"] == 2
    assert not runner._ar_runner.has_row("slow")

    collected = {"slow": outputs}
    collected.update(_drive(engine))
    assert collected["healthy"][-1].finish_reason == "stop"

    # The engine loop lives: a fresh request still completes after the timeout.
    engine.add_request("after", _tts_request("after", seed=303))
    after = _drive(engine)
    assert after["after"][-1].finish_reason == "stop"

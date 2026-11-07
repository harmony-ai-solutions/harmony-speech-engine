"""Chatterbox AR handler wiring gates (Phase 9-3).

Exercises the real ``ModelRunnerBase._execute_chatterbox_tts`` AR path with a
tiny in-process T3, a stub S3Gen/watermarker/tokenizer and a real 9-2 pool. No
weights, CPU-only.

Pinned here:

* the AR path is used for a multi-request step and returns one final result per
  request (pairing), with audio produced by the per-row S3Gen hop;
* ``model.conds`` keeps its baseline object identity through a full decode (the
  conditioning-state deletion the AR path is built around);
* ``len == 1``, ``batched_inference: false`` and a refused pool grant fall back
  to the sequential loop;
* ``release_request`` (9-4's abort/deadline hook) frees the row and its slots;
* a poisoned row errors alone while its neighbour keeps decoding.
"""


from types import SimpleNamespace

import pytest
import torch
from chatterbox.models.t3.llama_configs import LLAMA_CONFIGS
from chatterbox.models.t3.modules.cond_enc import T3Cond
from chatterbox.models.t3.modules.t3_config import T3Config
from chatterbox.tts import Conditionals

from harmonyspeech.common.artifacts import SpeakerArtifact
from harmonyspeech.common.config import DeviceConfig, ModelConfig
from harmonyspeech.common.inputs import TextToSpeechGenerationOptions, TextToSpeechRequestInput
from harmonyspeech.common.outputs import TextToSpeechRequestOutput, UnfinishedRequestOutput
from harmonyspeech.common.request import EngineRequest
from harmonyspeech.engine.ar_cache import ARCachePool
from harmonyspeech.modeling.models.chatterbox.t3_ar import ChatterboxT3AR
from harmonyspeech.task_handler.model_runner_base import ModelRunnerBase

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


_TINY_GPT2 = {
    "model_type": "gpt2",
    "vocab_size": 8,
    "n_positions": 512,
    "n_ctx": 512,
    "n_embd": 32,
    "hidden_size": 32,
    "n_head": 4,
    "n_layer": 2,
    "activation_function": "gelu_new",
    "attn_pdrop": 0.0,
    "embd_pdrop": 0.0,
    "resid_pdrop": 0.0,
    "layer_norm_epsilon": 1e-05,
    "initializer_range": 0.02,
}


def _tiny_hp():
    hp = T3Config.english_only()
    hp.llama_config_name = "tiny_ar_handler"
    hp.text_tokens_dict_size = 300
    hp.speech_tokens_dict_size = 96
    hp.start_speech_token = 90
    hp.stop_speech_token = 91
    hp.max_text_tokens = 32
    hp.max_speech_tokens = 64
    hp.use_perceiver_resampler = False
    hp.emotion_adv = True
    return hp


def _tiny_turbo_hp():
    hp = T3Config(text_tokens_dict_size=300)
    hp.llama_config_name = "tiny_ar_handler_gpt2"
    hp.speech_tokens_dict_size = 96
    hp.start_speech_token = 90
    hp.stop_speech_token = 91
    hp.input_pos_emb = None
    hp.speech_cond_prompt_len = 4
    hp.use_perceiver_resampler = False
    hp.emotion_adv = False
    hp.max_speech_tokens = 64
    return hp


def _conditionals(hp, *, seed=4242):
    generator = torch.Generator().manual_seed(seed)
    return Conditionals(
        T3Cond(
            speaker_emb=torch.randn(1, hp.speaker_embed_size, generator=generator),
            cond_prompt_speech_tokens=torch.randint(0, 80, (1, 4), generator=generator),
            emotion_adv=0.5 * torch.ones(1, 1, 1),
        ),
        {},
    )


class _StubTokenizer:
    def text_to_tokens(self, text):
        # A different length per word count, so the batch mixes prefill lengths.
        words = max(1, len(str(text).split()))
        return torch.arange(5, 5 + words, dtype=torch.long)


class _TurboTokenizer:
    def __call__(self, text, return_tensors=None, padding=None, truncation=None):
        words = max(1, len(str(text).split()))
        return SimpleNamespace(input_ids=torch.arange(5, 5 + words, dtype=torch.long).unsqueeze(0))


class _StubS3Gen:
    def __init__(self):
        self.calls = []

    def inference(self, speech_tokens, ref_dict, **kwargs):
        self.calls.append({"tokens": speech_tokens.detach().cpu().clone(), "kwargs": dict(kwargs)})
        length = max(8, int(speech_tokens.numel()) * 4)
        return torch.zeros(1, length), None


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
        self.tokenizer = _TurboTokenizer() if getattr(t3, "is_gpt", False) else _StubTokenizer()
        self.generate_calls = []

    def generate(self, text=None, **kwargs):
        self.generate_calls.append(text)
        return torch.zeros(1, 16)


@pytest.fixture
def tiny(monkeypatch):
    monkeypatch.setitem(LLAMA_CONFIGS, "tiny_ar_handler", dict(_TINY_LLAMA))
    hp = _tiny_hp()
    torch.manual_seed(20261109)
    t3 = ChatterboxT3AR(hp).eval()
    t3.requires_grad_(False)
    model = _StubModel(t3, _conditionals(hp))
    return model, hp


@pytest.fixture
def turbo(monkeypatch):
    from chatterbox.models.t3 import T3

    monkeypatch.setitem(LLAMA_CONFIGS, "tiny_ar_handler_gpt2", dict(_TINY_GPT2))
    hp = _tiny_turbo_hp()
    torch.manual_seed(20261111)
    t3 = T3(hp).eval()
    del t3.tfmr.wte
    t3.requires_grad_(False)
    model = _StubModel(t3, _conditionals(hp))
    return model, hp


def _make_runner(model, *, batched_inference=None, pool_slots=None, model_type="ChatterboxTTS"):
    runner = ModelRunnerBase.__new__(ModelRunnerBase)
    runner.device = "cpu"
    runner.model = model
    runner.model_config = ModelConfig(
        name="cb",
        model="dummy/chatterbox",
        model_type=model_type,
        max_batch_size=4,
        device_config=DeviceConfig("cpu"),
        dtype="float32",
        batched_inference=batched_inference,
    )
    budget = 4 * 1024 * 1024
    if pool_slots is not None:
        slab = 2 * 2 * 64 * 4 * 8 * 4
        budget = slab * pool_slots
    runner.ar_cache_pool = ARCachePool(budget_bytes=budget, device="cpu", default_cap_tokens=64)
    runner._baseline_state = None
    runner._ar_runner = None
    runner._capture_baseline_state()
    return runner


def _request(request_id, *, seed=7, text="hello world"):
    return EngineRequest(
        request_id=request_id,
        request_data=TextToSpeechRequestInput(
            request_id=request_id,
            requested_model="cb",
            model="cb",
            input_text=text,
            mode="single_speaker_tts",
            generation_options=TextToSpeechGenerationOptions(
                seed=seed, style=None, speed=None, pitch=None, energy=None
            ),
        ),
        arrival_time=0.0,
    )


def _params(conditionals, text="hello world"):
    return (text, SpeakerArtifact(conditionals=conditionals, backend="chatterbox"), 0.5, 0.5, 0.8, 1.2, 0.95, 0.05)


def _turbo_params(conditionals, text="hello world"):
    return (text, SpeakerArtifact(conditionals=conditionals, backend="chatterbox"), 0.8, 1.2, 0.95, 1000, True)


def _drive(runner, requests, *, max_steps=64):
    """Call the handler repeatedly until every request produced a final result."""
    conditionals = requests[0]._test_conditionals
    pending = list(requests)
    results = {}
    for _ in range(max_steps):
        inputs = [_params(conditionals, request.request_data.input_text) for request in pending]
        outputs = runner._execute_chatterbox_tts(inputs, pending)
        still_pending = []
        for request, output in zip(pending, outputs, strict=True):
            if isinstance(output.result_data, UnfinishedRequestOutput):
                still_pending.append(request)
            else:
                results[request.request_id] = output
        pending = still_pending
        if not pending:
            return [results[request.request_id] for request in requests]
    raise AssertionError("decode did not finish")


@pytest.mark.unit
def test_ar_path_finishes_two_requests_with_paired_audio(tiny):
    model, hp = tiny
    baseline = model.conds
    runner = _make_runner(model)
    requests = [_request("a", seed=3), _request("b", seed=4, text="hello there friend")]
    conditionals = _conditionals(hp)
    for request in requests:
        request._test_conditionals = conditionals

    results = _drive(runner, requests)

    assert [result.request_id for result in results] == ["a", "b"]
    for result in results:
        assert isinstance(result.result_data, TextToSpeechRequestOutput)
        assert result.result_data.finish_reason == "stop"
        assert result.result_data.audio_artifact is not None
    # One S3Gen call per finished row, in request order.
    assert len(model.s3gen.calls) == 2
    # Conditioning-state deletion: the model's conditionals were never touched.
    assert model.conds is baseline
    # Every slot returned.
    assert runner.ar_cache_pool.capacity("cb")["slots_active"] == 0


@pytest.mark.unit
def test_conds_identity_unchanged_through_every_decode_step(tiny):
    model, hp = tiny
    baseline = model.conds
    runner = _make_runner(model)
    requests = [_request("a", seed=11), _request("b", seed=12)]
    conditionals = _conditionals(hp)
    for request in requests:
        request._test_conditionals = conditionals

    pending = list(requests)
    for _ in range(64):
        outputs = runner._execute_chatterbox_tts(
            [_params(conditionals, request.request_data.input_text) for request in pending], pending
        )
        assert model.conds is baseline, "the AR path must never assign model.conds"
        pending = [
            request
            for request, output in zip(pending, outputs, strict=True)
            if isinstance(output.result_data, UnfinishedRequestOutput)
        ]
        if not pending:
            break
    else:
        raise AssertionError("decode did not finish")
    assert model.conds is baseline


@pytest.mark.unit
def test_len_one_uses_the_sequential_loop(tiny):
    model, hp = tiny
    runner = _make_runner(model)
    conditionals = _conditionals(hp)

    outputs = runner._execute_chatterbox_tts(
        [_params(conditionals)], [_request("solo")]
    )

    assert outputs[0].result_data.finish_reason == "stop"
    assert model.generate_calls == ["hello world"], "the single-request fast path is the loop"
    assert runner.ar_cache_pool.capacity("cb")["slots_active"] == 0


@pytest.mark.unit
def test_batched_inference_false_uses_the_sequential_loop(tiny):
    model, hp = tiny
    runner = _make_runner(model, batched_inference=False)
    conditionals = _conditionals(hp)
    requests = [_request("a"), _request("b")]

    outputs = runner._execute_chatterbox_tts(
        [_params(conditionals), _params(conditionals)], requests
    )

    assert all(output.result_data.finish_reason == "stop" for output in outputs)
    assert model.generate_calls == ["hello world", "hello world"]


@pytest.mark.unit
def test_pool_refusal_falls_back_to_the_sequential_loop(tiny):
    model, hp = tiny
    runner = _make_runner(model, pool_slots=1)  # English needs 2 CFG slots
    conditionals = _conditionals(hp)
    requests = [_request("a"), _request("b")]

    outputs = runner._execute_chatterbox_tts(
        [_params(conditionals), _params(conditionals)], requests
    )

    assert all(output.result_data.finish_reason == "stop" for output in outputs)
    assert model.generate_calls == ["hello world", "hello world"]
    assert runner.ar_cache_pool.capacity("cb")["slots_active"] == 0


@pytest.mark.unit
def test_release_request_frees_row_and_slots(tiny):
    model, hp = tiny
    runner = _make_runner(model)
    conditionals = _conditionals(hp)
    requests = [_request("keep", seed=1), _request("abort-me", seed=2)]
    for request in requests:
        request._test_conditionals = conditionals

    runner._execute_chatterbox_tts(
        [_params(conditionals), _params(conditionals)], requests
    )
    assert runner.ar_cache_pool.capacity("cb")["slots_active"] == 4

    runner.release_request("abort-me")

    assert runner.ar_cache_pool.capacity("cb")["slots_active"] == 2
    assert runner._ar_runner.has_row("keep")
    assert not runner._ar_runner.has_row("abort-me")


@pytest.mark.unit
def test_poisoned_row_errors_alone(tiny, monkeypatch):
    model, hp = tiny
    runner = _make_runner(model)
    conditionals = _conditionals(hp)
    requests = [_request("poison", seed=1), _request("healthy", seed=2)]
    for request in requests:
        request._test_conditionals = conditionals

    runner._execute_chatterbox_tts([_params(conditionals), _params(conditionals)], requests)
    ar = runner._ar_runner
    real_step = ar.step

    def step_with_poison(request_ids):
        outcomes = real_step(request_ids)
        for outcome in outcomes:
            if outcome.request_id == "poison":
                outcome.error = RuntimeError("poisoned decode row")
                outcome.finished = False
        return outcomes

    monkeypatch.setattr(ar, "step", step_with_poison)
    outputs = runner._execute_chatterbox_tts([_params(conditionals), _params(conditionals)], requests)

    by_id = {output.request_id: output for output in outputs}
    assert by_id["poison"].result_data.finish_reason == "error"
    assert "poisoned" in by_id["poison"].result_data.error
    assert isinstance(by_id["healthy"].result_data, UnfinishedRequestOutput)
    assert ar.has_row("healthy")
    assert not ar.has_row("poison")
    assert runner.ar_cache_pool.capacity("cb")["slots_active"] == 2


@pytest.mark.unit
def test_turbo_ar_path_finishes_two_requests_with_paired_audio(turbo):
    model, hp = turbo
    baseline = model.conds
    runner = _make_runner(model, model_type="ChatterboxTurboTTS")
    requests = [_request("a", seed=5), _request("b", seed=6, text="hello there friend")]
    conditionals = _conditionals(hp)
    for request in requests:
        request._test_conditionals = conditionals

    pending = list(requests)
    results = {}
    for _ in range(64):
        outputs = runner._execute_chatterbox_turbo_tts(
            [_turbo_params(conditionals, request.request_data.input_text) for request in pending], pending
        )
        assert model.conds is baseline, "the Turbo AR path must never assign model.conds"
        still_pending = []
        for request, output in zip(pending, outputs, strict=True):
            if isinstance(output.result_data, UnfinishedRequestOutput):
                still_pending.append(request)
            else:
                results[request.request_id] = output
        pending = still_pending
        if not pending:
            break
    else:
        raise AssertionError("turbo decode did not finish")

    assert set(results) == {"a", "b"}
    for output in results.values():
        assert isinstance(output.result_data, TextToSpeechRequestOutput)
        assert output.result_data.finish_reason == "stop"
        assert output.result_data.audio_artifact is not None
    # Turbo's S3Gen hop gets `n_cfm_timesteps=2` and the three silence tokens.
    from chatterbox.models.s3gen.const import S3GEN_SIL

    assert len(model.s3gen.calls) == 2
    assert all(call["kwargs"].get("n_cfm_timesteps") == 2 for call in model.s3gen.calls)
    for call in model.s3gen.calls:
        assert call["tokens"][-3:].tolist() == [S3GEN_SIL, S3GEN_SIL, S3GEN_SIL]
    assert runner.ar_cache_pool.capacity("cb")["slots_active"] == 0


@pytest.mark.unit
def test_executor_release_request_delegates_to_the_model_runner():
    """9-4 calls `release_request` on the executor; it must reach the runner."""
    from harmonyspeech.executor.executor_base import ExecutorBase

    class _ConcreteExecutor(ExecutorBase):
        def _init_executor(self):
            return None

        def execute_model(self, requests_to_batch):
            return []

        def check_health(self):
            return None

    class _FakeRunner:
        def __init__(self):
            self.released = []

        def release_request(self, request_id):
            self.released.append(request_id)

    executor = _ConcreteExecutor.__new__(_ConcreteExecutor)
    fake = _FakeRunner()
    executor.driver_worker = SimpleNamespace(model_runner=fake)

    executor.release_request("row-1")

    assert fake.released == ["row-1"]


@pytest.mark.unit
def test_executor_release_request_is_a_noop_without_the_hook():
    from harmonyspeech.executor.executor_base import ExecutorBase

    class _ConcreteExecutor(ExecutorBase):
        def _init_executor(self):
            return None

        def execute_model(self, requests_to_batch):
            return []

        def check_health(self):
            return None

    executor = _ConcreteExecutor.__new__(_ConcreteExecutor)
    executor.driver_worker = SimpleNamespace(model_runner=SimpleNamespace())
    executor.release_request("row-1")  # must not raise


# ---------------------------------------------------------------------------
# Phase 9-5: the padded S3Gen hop
# ---------------------------------------------------------------------------
def _gen_dict(prompt_len=4, seed=909):
    """A real S3Gen ``ref_dict`` matching the tiny graph's dimensions."""
    generator = torch.Generator().manual_seed(seed)
    return {
        "prompt_token": torch.randint(0, 100, (1, prompt_len), generator=generator),
        "prompt_token_len": torch.tensor([prompt_len]),
        "prompt_feat": torch.randn(1, 2 * prompt_len, 80, generator=generator),
        "prompt_feat_len": None,
        "embedding": torch.randn(1, 192, generator=generator),
    }


def _real_conditionals(hp, *, prompt_len=4):
    generator = torch.Generator().manual_seed(4242)
    return Conditionals(
        T3Cond(
            speaker_emb=torch.randn(1, hp.speaker_embed_size, generator=generator),
            cond_prompt_speech_tokens=torch.randint(0, 80, (1, 4), generator=generator),
            emotion_adv=0.5 * torch.ones(1, 1, 1),
        ),
        _gen_dict(prompt_len),
    )


@pytest.fixture
def batched_tiny(monkeypatch):
    """The 9-3 tiny T3 with a **real** tiny S3Gen so the hop can batch."""
    from tests.unit.modeling.test_chatterbox_s3gen_batch import tiny_s3gen

    monkeypatch.setitem(LLAMA_CONFIGS, "tiny_ar_handler", dict(_TINY_LLAMA))
    hp = _tiny_hp()
    torch.manual_seed(20261109)
    t3 = ChatterboxT3AR(hp).eval()
    t3.requires_grad_(False)
    model = _StubModel(t3, _real_conditionals(hp))
    model.s3gen = tiny_s3gen()
    return model, hp


def _drive_with(model, runner, requests):
    conditionals = requests[0]._test_conditionals
    pending = list(requests)
    results = {}
    for _ in range(64):
        inputs = [_params(conditionals, request.request_data.input_text) for request in pending]
        outputs = runner._execute_chatterbox_tts(inputs, pending)
        still_pending = []
        for request, output in zip(pending, outputs, strict=True):
            if isinstance(output.result_data, UnfinishedRequestOutput):
                still_pending.append(request)
            else:
                results[request.request_id] = output
        pending = still_pending
        if not pending:
            return results
    raise AssertionError("decode did not finish")


def _audio(output):
    return output.result_data.audio_artifact.numpy


@pytest.mark.unit
def test_s3gen_hop_runs_one_forward_per_finished_group(batched_tiny, monkeypatch):
    import harmonyspeech.modeling.models.chatterbox.s3gen_batch as s3gen_batch

    model, hp = batched_tiny
    runner = _make_runner(model)
    # Identical requests (same seed/text) finish on the same step, so the two
    # finished rows land in one sub-batch.
    requests = [_request("a", seed=3), _request("b", seed=3)]
    conditionals = _real_conditionals(hp)
    for request in requests:
        request._test_conditionals = conditionals

    calls = []
    real = s3gen_batch.inference_batched

    def recording(s3gen, rows, **kwargs):
        calls.append(len(rows))
        return real(s3gen, rows, **kwargs)

    monkeypatch.setattr(s3gen_batch, "inference_batched", recording)
    results = _drive_with(model, runner, requests)

    assert [results[key].request_id for key in ("a", "b")] == ["a", "b"]
    assert all(result.result_data.finish_reason == "stop" for result in results.values())
    assert calls == [2], calls
    assert sum(calls) == 2, "two finished rows are vocoded together in one forward"


@pytest.mark.unit
def test_s3gen_loop_and_batch_agree_on_seeded_inputs(batched_tiny, monkeypatch):
    import harmonyspeech.modeling.models.chatterbox.s3gen_batch as s3gen_batch

    model, hp = batched_tiny
    conditionals = _real_conditionals(hp)
    requests = [_request("a", seed=31), _request("b", seed=32, text="hello there friend")]
    for request in requests:
        request._test_conditionals = conditionals

    batched = _drive_with(model, _make_runner(model), requests)

    # Force the per-row upstream fallback (still seeded, so audio must match).
    monkeypatch.setattr(s3gen_batch, "supports_batched_s3gen", lambda s3gen: False)
    looped = _drive_with(model, _make_runner(model), requests)

    for request in requests:
        left = _audio(batched[request.request_id])
        right = _audio(looped[request.request_id])
        assert left.shape == right.shape
        assert torch.allclose(torch.from_numpy(left), torch.from_numpy(right), atol=1e-6)


@pytest.mark.unit
def test_s3gen_poisoned_row_errors_alone(batched_tiny, monkeypatch):
    model, hp = batched_tiny
    runner = _make_runner(model)
    conditionals = _real_conditionals(hp)
    requests = [_request("poison", seed=1), _request("healthy", seed=2)]
    for request in requests:
        request._test_conditionals = conditionals

    ar = runner._get_ar_runner()
    real_final = ar.final_tokens

    def poisoned_final(request_id):
        tokens = real_final(request_id)
        if request_id == "poison" and tokens is not None:
            tokens = tokens.clone()
            tokens[0] = 99999  # out of the flow embedding's vocab
        return tokens

    monkeypatch.setattr(ar, "final_tokens", poisoned_final)
    results = _drive_with(model, runner, requests)

    assert results["poison"].result_data.finish_reason == "error"
    assert "99999" in results["poison"].result_data.error or results["poison"].result_data.error
    assert results["healthy"].result_data.finish_reason == "stop"
    assert results["healthy"].result_data.audio_artifact is not None
    assert not ar.has_row("poison")
    assert not ar.has_row("healthy")


@pytest.mark.unit
def test_s3gen_batch_failure_falls_back_to_the_per_row_hop(batched_tiny, monkeypatch):
    import harmonyspeech.modeling.models.chatterbox.s3gen_batch as s3gen_batch

    model, hp = batched_tiny
    runner = _make_runner(model)
    conditionals = _real_conditionals(hp)
    # Identical requests co-finish, so the broken multi-row hop is exercised.
    requests = [_request("a", seed=7), _request("b", seed=7)]
    for request in requests:
        request._test_conditionals = conditionals

    def broken(s3gen, rows, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(s3gen_batch, "inference_batched", broken)
    results = _drive_with(model, runner, requests)

    for request in requests:
        assert results[request.request_id].result_data.finish_reason == "stop"
        assert results[request.request_id].result_data.audio_artifact is not None


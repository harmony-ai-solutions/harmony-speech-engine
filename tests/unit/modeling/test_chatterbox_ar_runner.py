"""Token-level AR batch runner gates (Phase 9-3).

No model weights: every gate runs on a tiny, randomly initialised T3 built
in-process (the 9-1 precedent). Two contracts are covered — English
(``T3.inference``) and Turbo (``T3.inference_turbo``) — against the real upstream
classes, with a real 9-2 ``ARCachePool`` on CPU.

What is pinned:

* B=1 is bit-identical to 9-1's cleaned sequential loop under the same seed;
* a row's seeded tokens are identical whether it decodes alone or in a batch
  (per-row ``torch.Generator``);
* rows are dropped on EOS mid-batch without disturbing their neighbours;
* a poisoned row errors alone (the others keep decoding);
* a refused pool grant returns ``None`` (the handler's sequential fallback);
* the arena is registered under the scheduler's model key.
"""

from types import SimpleNamespace

import pytest
import torch
from chatterbox.models.t3 import T3
from chatterbox.models.t3.llama_configs import LLAMA_CONFIGS
from chatterbox.models.t3.modules.cond_enc import T3Cond
from chatterbox.models.t3.modules.t3_config import T3Config
from chatterbox.tts import Conditionals

from harmonyspeech.engine.ar_cache import ARCachePool
from harmonyspeech.modeling.models.chatterbox.t3_ar import ChatterboxT3AR
from harmonyspeech.task_handler.ar_runner import (
    ARBatchRunner,
    EnglishContract,
)

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

_DECODE = {
    "temperature": 0.8,
    "top_p": 0.95,
    "min_p": 0.05,
    "repetition_penalty": 1.2,
}


def _tiny_english_hp():
    hp = T3Config.english_only()
    hp.llama_config_name = "tiny_ar_runner"
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
    hp.llama_config_name = "tiny_ar_runner_gpt2"
    hp.speech_tokens_dict_size = 96
    hp.start_speech_token = 90
    hp.stop_speech_token = 91
    hp.input_pos_emb = None
    hp.speech_cond_prompt_len = 4
    hp.use_perceiver_resampler = False
    hp.emotion_adv = False
    hp.max_speech_tokens = 64
    return hp


@pytest.fixture
def english_model(monkeypatch):
    monkeypatch.setitem(LLAMA_CONFIGS, "tiny_ar_runner", dict(_TINY_LLAMA))
    hp = _tiny_english_hp()
    torch.manual_seed(20261107)
    cleaned = ChatterboxT3AR(hp).eval()
    cleaned.requires_grad_(False)
    return cleaned, hp


@pytest.fixture
def turbo_model(monkeypatch):
    monkeypatch.setitem(LLAMA_CONFIGS, "tiny_ar_runner_gpt2", dict(_TINY_GPT2))
    hp = _tiny_turbo_hp()
    torch.manual_seed(20261108)
    model = T3(hp).eval()
    del model.tfmr.wte
    model.requires_grad_(False)
    return model, hp


def _pool(**kwargs):
    return ARCachePool(budget_bytes=4 * 1024 * 1024, device="cpu", default_cap_tokens=64, **kwargs)


def _runner(t3, model_type, pool=None, *, key="cb"):
    runner = ARBatchRunner(
        t3=t3,
        pool=pool if pool is not None else _pool(),
        model_key=key,
        model_config=SimpleNamespace(model_type=model_type, seed=0, max_batch_size=8),
    )
    assert runner.register(), "tiny arena must register"
    return runner


def _english_conds(hp, *, seed=4242):
    generator = torch.Generator().manual_seed(seed)
    return T3Cond(
        speaker_emb=torch.randn(1, hp.speaker_embed_size, generator=generator),
        cond_prompt_speech_tokens=torch.randint(0, 80, (1, 4), generator=generator),
        emotion_adv=0.5 * torch.ones(1, 1, 1),
    )


def _text(hp, words=3):
    return torch.tensor([[hp.start_text_token, *range(5, 5 + words), hp.stop_text_token]])


def _english_params(text, conds, cfg_weight=0.5):
    return {
        "conds": Conditionals(conds, {}),
        "text_tokens": text,
        "cfg_weight": cfg_weight,
        "temperature": _DECODE["temperature"],
        "top_p": _DECODE["top_p"],
        "min_p": _DECODE["min_p"],
        "repetition_penalty": _DECODE["repetition_penalty"],
    }


def _run_to_end(runner, request_id, *, max_steps=128):
    for _ in range(max_steps):
        outcome = runner.step([request_id])[0]
        if outcome.finished or outcome.error is not None:
            return outcome
    raise AssertionError("decode did not terminate")


# ---------------------------------------------------------------------------
# B=1 equality with the sequential loops
# ---------------------------------------------------------------------------
@pytest.mark.unit
@pytest.mark.parametrize("attn", ["sdpa", "eager"])
def test_english_b1_bit_identical_to_cleaned_loop(english_model, monkeypatch, attn):
    model, hp = english_model
    monkeypatch.setitem(LLAMA_CONFIGS, "tiny_ar_runner", {**_TINY_LLAMA, "attn_implementation": attn})
    torch.manual_seed(20261107)
    model = ChatterboxT3AR(hp).eval()
    model.requires_grad_(False)
    conds = _english_conds(hp)
    text = _text(hp)

    torch.manual_seed(7)
    expected = model.inference(
        t3_cond=conds,
        text_tokens=torch.cat([text, text], dim=0),
        max_new_tokens=10,
        **_DECODE,
    )

    runner = _runner(model, "ChatterboxTTS")
    runner.begin("solo", seed=7, max_new_tokens=10, **_english_params(text, conds))
    _run_to_end(runner, "solo")
    actual = runner.raw_tokens("solo")

    assert torch.equal(actual, expected), f"{actual.tolist()} != {expected.tolist()}"
    assert actual.shape[1] == 10, "the gate must exercise a real decode loop"


@pytest.mark.unit
def test_turbo_b1_bit_identical_to_inference_turbo(turbo_model):
    model, hp = turbo_model
    conds = _english_conds(hp)
    text = _text(hp, words=3)

    torch.manual_seed(11)
    expected = model.inference_turbo(
        conds,
        text,
        temperature=_DECODE["temperature"],
        top_k=1000,
        top_p=_DECODE["top_p"],
        repetition_penalty=_DECODE["repetition_penalty"],
        max_gen_len=10,
    )

    runner = _runner(model, "ChatterboxTurboTTS")
    runner.begin(
        "solo",
        conds=SimpleNamespace(t3=conds, gen={}),
        text_tokens=text,
        seed=11,
        temperature=_DECODE["temperature"],
        top_k=1000,
        top_p=_DECODE["top_p"],
        repetition_penalty=_DECODE["repetition_penalty"],
        max_new_tokens=11,
    )
    _run_to_end(runner, "solo")
    actual = runner.raw_tokens("solo")
    stop = int(hp.stop_speech_token)
    if actual.numel() and int(actual[0, -1]) == stop:
        actual = actual[:, :-1]

    assert torch.equal(actual, expected), f"{actual.tolist()} != {expected.tolist()}"


# ---------------------------------------------------------------------------
# Multi-row seeded equality + per-row generator reproducibility
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_english_multi_row_matches_solo_runs(english_model):
    model, hp = english_model
    specs = [
        ("a", _text(hp, words=2), 21),
        ("b", _text(hp, words=5), 22),
        ("c", _text(hp, words=3), 23),
    ]

    solo = {}
    for request_id, text, seed in specs:
        runner = _runner(model, "ChatterboxTTS", key=f"solo-{request_id}")
        runner.begin(
            request_id,
            seed=seed,
            max_new_tokens=8,
            **_english_params(text, _english_conds(hp, seed=1000 + seed)),
        )
        _run_to_end(runner, request_id)
        solo[request_id] = runner.raw_tokens(request_id)

    batched = _runner(model, "ChatterboxTTS")
    for request_id, text, seed in specs:
        batched.begin(
            request_id,
            seed=seed,
            max_new_tokens=8,
            **_english_params(text, _english_conds(hp, seed=1000 + seed)),
        )
    # Advance all rows together, one token per engine step.
    for _ in range(8):
        outcomes = {outcome.request_id: outcome for outcome in batched.step([spec[0] for spec in specs])}
        if all(outcome.finished for outcome in outcomes.values()):
            break

    for request_id, _text_value, _seed in specs:
        assert torch.equal(batched.raw_tokens(request_id), solo[request_id]), request_id


@pytest.mark.unit
def test_turbo_multi_row_matches_solo_runs(turbo_model):
    model, hp = turbo_model
    specs = [("a", _text(hp, words=2), 31), ("b", _text(hp, words=5), 32)]

    def params(text):
        return {
            "conds": SimpleNamespace(t3=_english_conds(hp), gen={}),
            "text_tokens": text,
            "temperature": _DECODE["temperature"],
            "top_k": 1000,
            "top_p": _DECODE["top_p"],
            "repetition_penalty": _DECODE["repetition_penalty"],
            "max_new_tokens": 8,
        }

    solo = {}
    for request_id, text, seed in specs:
        runner = _runner(model, "ChatterboxTurboTTS", key=f"solo-{request_id}")
        runner.begin(request_id, seed=seed, **params(text))
        _run_to_end(runner, request_id)
        solo[request_id] = runner.raw_tokens(request_id)

    batched = _runner(model, "ChatterboxTurboTTS")
    for request_id, text, seed in specs:
        batched.begin(request_id, seed=seed, **params(text))
    for _ in range(8):
        outcomes = {outcome.request_id: outcome for outcome in batched.step([spec[0] for spec in specs])}
        if all(outcome.finished for outcome in outcomes.values()):
            break

    for request_id, _text_value, _seed in specs:
        assert torch.equal(batched.raw_tokens(request_id), solo[request_id]), request_id


@pytest.mark.unit
def test_per_row_generator_reproducibility_across_batch_composition(english_model):
    model, hp = english_model
    text = _text(hp)
    conds = _english_conds(hp)

    alone = _runner(model, "ChatterboxTTS", key="alone")
    alone.begin("row", seed=99, max_new_tokens=6, **_english_params(text, conds))
    _run_to_end(alone, "row")
    expected = alone.raw_tokens("row")

    for companion_text, companion_seed in ((_text(hp, words=6), 123), (_text(hp, words=1), 456)):
        runner = _runner(model, "ChatterboxTTS", key=f"mixed-{companion_seed}")
        runner.begin("row", seed=99, max_new_tokens=6, **_english_params(text, conds))
        runner.begin(
            "other",
            seed=companion_seed,
            max_new_tokens=6,
            **_english_params(companion_text, _english_conds(hp, seed=companion_seed)),
        )
        for _ in range(6):
            outcomes = {outcome.request_id: outcome for outcome in runner.step(["row", "other"])}
            if outcomes["row"].finished:
                break
        assert torch.equal(runner.raw_tokens("row"), expected)


# ---------------------------------------------------------------------------
# EOS row drop + poisoned row isolation
# ---------------------------------------------------------------------------
class _ScriptedContract(EnglishContract):
    """Deterministic sampler used to pin the row-drop rule."""

    def __init__(self, script):
        self.script = {key: list(values) for key, values in script.items()}

    def sample(self, t3, row):
        value = self.script[row.request_id].pop(0)
        token = torch.tensor([[value]], dtype=torch.long, device=t3.device)
        row.generated_ids = torch.cat([row.generated_ids, token], dim=1)
        row.tokens.append(token)
        return token


@pytest.mark.unit
def test_eos_drops_one_row_mid_batch(english_model):
    model, hp = english_model
    runner = _runner(model, "ChatterboxTTS")
    runner.contract = _ScriptedContract({"a": [5, int(hp.stop_speech_token)], "b": [6, 7, 8]})
    runner.begin("a", seed=1, max_new_tokens=10, **_english_params(_text(hp), _english_conds(hp)))
    runner.begin("b", seed=2, max_new_tokens=10, **_english_params(_text(hp), _english_conds(hp)))
    length_a = runner.row("a").length

    first = {outcome.request_id: outcome for outcome in runner.step(["a", "b"])}
    assert not first["a"].finished and not first["b"].finished
    assert runner.row("a").length == length_a + 1

    second = {outcome.request_id: outcome for outcome in runner.step(["a", "b"])}
    assert second["a"].finished, "EOS finishes that row"
    assert not second["b"].finished, "the neighbour keeps decoding"
    assert runner.row("a").length == length_a + 1, "a finished row is not forwarded again"
    assert runner.row("b").length == length_a + 2

    # The handler releases the finished row's slots; the pool reflects it.
    slots_before = runner.arena.free_count
    runner.pop("a")
    assert runner.arena.free_count == slots_before + 2
    assert not runner.has_row("a")
    assert runner.has_row("b")


class _PoisonContract(EnglishContract):
    def embed_token(self, t3, row):
        if row.request_id == "poison":
            raise RuntimeError("poisoned decode row")
        return super().embed_token(t3, row)


@pytest.mark.unit
def test_poisoned_row_errors_alone(english_model):
    model, hp = english_model
    runner = _runner(model, "ChatterboxTTS")
    runner.contract = _PoisonContract()
    runner.begin("poison", seed=1, max_new_tokens=10, **_english_params(_text(hp), _english_conds(hp)))
    runner.begin("healthy", seed=2, max_new_tokens=10, **_english_params(_text(hp), _english_conds(hp)))
    healthy_length = runner.row("healthy").length

    outcomes = {outcome.request_id: outcome for outcome in runner.step(["poison", "healthy"])}

    assert outcomes["poison"].error is not None
    assert "poisoned" in str(outcomes["poison"].error)
    assert outcomes["healthy"].error is None
    assert runner.row("healthy").length == healthy_length + 1, "the neighbour still advanced"
    runner.pop("poison")
    assert not runner.has_row("poison")


# ---------------------------------------------------------------------------
# Pool refusal + registration/admission
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_pool_refusal_returns_none_for_sequential_fallback(english_model):
    model, hp = english_model
    # Exactly one slab fits: English needs two CFG slots, so the grant is refused.
    slab_bytes = 2 * 2 * 64 * 4 * 8 * 4
    pool = ARCachePool(budget_bytes=slab_bytes, device="cpu", default_cap_tokens=64)
    runner = ARBatchRunner(
        t3=model,
        pool=pool,
        model_key="cb",
        model_config=SimpleNamespace(model_type="ChatterboxTTS", seed=0),
    )
    assert runner.register()
    assert pool.capacity("cb")["slots"] == 1
    assert runner.begin("row", seed=1, **_english_params(_text(hp), _english_conds(hp))) is None
    assert pool.capacity("cb")["slots_active"] == 0, "a refused grant releases nothing partially"


@pytest.mark.unit
def test_arena_is_registered_under_the_scheduler_model_key(english_model):
    model, _hp = english_model
    pool = _pool()
    runner = ARBatchRunner(
        t3=model,
        pool=pool,
        model_key="chatterbox-english",
        model_config=SimpleNamespace(model_type="ChatterboxTTS", seed=0),
    )
    assert runner.register()
    capacity = pool.capacity("chatterbox-english")
    assert capacity["enabled"] is True
    assert capacity["layers"] == 2
    assert capacity["kv_heads"] == 4
    assert capacity["head_dim"] == 8
    assert capacity["slots"] >= 2
    # 9-4 reads the pool by the scheduler's name; another key is "no signal".
    with pytest.raises(KeyError):
        pool.capacity("some-other-name")


@pytest.mark.unit
def test_disabled_pool_keeps_the_sequential_path(english_model):
    model, _hp = english_model
    runner = ARBatchRunner(
        t3=model,
        pool=None,
        model_key="cb",
        model_config=SimpleNamespace(model_type="ChatterboxTTS", seed=0),
    )
    assert runner.register() is False
    assert runner.enabled is False

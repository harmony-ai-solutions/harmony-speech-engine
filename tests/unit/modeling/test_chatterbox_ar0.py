"""AR-0 decode-loop hygiene tests (Phase 9-1).

The gate is a **seeded bit-equality check on a tiny, randomly initialised T3**
built in-process — no checkpoint download, no ``from_pretrained``. Upstream
``T3`` requires a config name that exists in ``chatterbox.models.t3.llama_configs.
LLAMA_CONFIGS``, so the fixture registers a test-only tiny entry there (removed
by ``monkeypatch`` on teardown) and drives both inference paths against the same
weights.

What is asserted:

* cleaned ``ChatterboxT3AR.inference`` samples the **same token ids** and the
  same token digest as upstream ``T3.inference`` under a pinned seed, for both
  ``attn_implementation`` values that matter: ``eager`` (attention really is
  materialised upstream) and ``sdpa`` (the shipped English config, where
  transformers ignores the request);
* the cleaned loop draws no progress bar, while upstream does and the opt-in
  debug env var brings it back;
* the factory instantiates the cleaned class for the TTS role (and therefore for
  the embedding role that shares its T3) and leaves multilingual/turbo alone;
* decode state is local to the call (Phase 0-2 style: no cross-request leak on
  the shared T3 instance).
"""

import contextlib
import hashlib
import io
from types import SimpleNamespace

import pytest
import torch
from chatterbox.models.t3 import T3
from chatterbox.models.t3.llama_configs import LLAMA_CONFIGS
from chatterbox.models.t3.modules.cond_enc import T3Cond
from chatterbox.models.t3.modules.t3_config import T3Config

from harmonyspeech.modeling.models.chatterbox import factory
from harmonyspeech.modeling.models.chatterbox.t3_ar import (
    ChatterboxT3AR,
    _SafeAlignmentStreamAnalyzer,
)

#: Tiny Llama topology for the gate: small enough to build in a second, deep
#: enough (2 layers) that the KV cache and position embeddings are exercised.
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
}

#: Pinned decode arguments — the same ones ``chatterbox/tts.py`` uses by default.
_DECODE = {
    "max_new_tokens": 10,
    "temperature": 0.8,
    "top_p": 0.95,
    "min_p": 0.05,
    "repetition_penalty": 1.2,
    "cfg_weight": 0.5,
}


def _digest(tensor: torch.Tensor) -> str:
    """Content hash of a tensor (the AR output is the token sequence S3Gen consumes)."""
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


@pytest.fixture(params=["eager", "sdpa"])
def tiny_hp(request, monkeypatch):
    """A tiny English T3Config, with a test-only Llama config registered upstream."""
    config = dict(_TINY_LLAMA)
    config["attn_implementation"] = request.param
    monkeypatch.setitem(LLAMA_CONFIGS, "tiny_ar0", config)

    hp = T3Config.english_only()
    hp.llama_config_name = "tiny_ar0"
    hp.text_tokens_dict_size = 300  # >= 256: start_text_token is 255
    hp.speech_tokens_dict_size = 96
    hp.start_speech_token = 90
    hp.stop_speech_token = 91
    hp.max_text_tokens = 32
    hp.max_speech_tokens = 32
    hp.use_perceiver_resampler = False
    hp.emotion_adv = True
    return hp


@pytest.fixture
def tiny_pair(tiny_hp):
    """Upstream T3 and the cleaned subclass sharing bit-identical weights."""
    torch.manual_seed(20261107)
    upstream = T3(tiny_hp).eval()
    cleaned = ChatterboxT3AR(tiny_hp).eval()
    cleaned.load_state_dict(upstream.state_dict())
    for module in (upstream, cleaned):
        module.requires_grad_(False)
    return upstream, cleaned


def _conditionals(hp, *, speaker_seed: int = 4242) -> T3Cond:
    """Fresh per-request conditionals (the decode mutates ``cond_prompt_speech_emb``)."""
    generator = torch.Generator().manual_seed(speaker_seed)
    return T3Cond(
        speaker_emb=torch.randn(1, hp.speaker_embed_size, generator=generator),
        cond_prompt_speech_tokens=torch.randint(0, 80, (1, 4), generator=generator),
        emotion_adv=0.5 * torch.ones(1, 1, 1),
    )


def _text_tokens(hp, *, batch: int = 2) -> torch.Tensor:
    """BOT + 3 words + EOT, duplicated for CFG (as ``chatterbox/tts.py`` does)."""
    return torch.tensor([[hp.start_text_token, 5, 6, 7, hp.stop_text_token]] * batch)


def _run(module, hp, *, seed: int = 7, conds: T3Cond | None = None) -> torch.Tensor:
    torch.manual_seed(seed)
    return module.inference(
        t3_cond=conds if conds is not None else _conditionals(hp),
        text_tokens=_text_tokens(hp),
        **_DECODE,
    )


# ---------------------------------------------------------------------------
# Equality gate
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_cleaned_loop_is_bit_identical_to_upstream(tiny_pair, tiny_hp):
    """Same seed → identical sampled tokens and identical token digest."""
    upstream, cleaned = tiny_pair
    expected = _run(upstream, tiny_hp)
    actual = _run(cleaned, tiny_hp)

    assert actual.shape == expected.shape
    assert torch.equal(actual, expected), f"tokens differ: {actual.tolist()} != {expected.tolist()}"
    assert _digest(actual) == _digest(expected)
    # The gate must exercise a real decode loop, not an immediate EOS.
    assert expected.shape[1] == _DECODE["max_new_tokens"]


@pytest.mark.unit
def test_cleaned_loop_stops_on_eos_exactly_like_upstream(tiny_pair, tiny_hp):
    """The early-break branch is preserved: EOS on step 1 yields one token."""
    upstream, cleaned = tiny_pair
    tiny_hp.stop_speech_token = 16  # the first token the seeded tiny model samples

    expected = _run(upstream, tiny_hp)
    actual = _run(cleaned, tiny_hp)

    assert expected.shape[1] == 1
    assert torch.equal(actual, expected)


@pytest.mark.unit
def test_attention_outputs_do_not_change_the_forward(tiny_pair, tiny_hp):
    """Supporting evidence: requesting attention tensors is numerically inert."""
    _, cleaned = tiny_pair
    embeds, len_cond = cleaned.prepare_input_embeds(
        t3_cond=_conditionals(tiny_hp),
        text_tokens=_text_tokens(tiny_hp),
        speech_tokens=torch.tensor([[tiny_hp.start_speech_token]] * 2),
        cfg_weight=_DECODE["cfg_weight"],
    )
    backend = cleaned.build_ar_backend(len_cond=len_cond, text_len=_text_tokens(tiny_hp).size(-1))

    outputs = {}
    for flag in (True, False):
        outputs[flag] = backend(
            inputs_embeds=embeds,
            past_key_values=None,
            use_cache=True,
            output_attentions=flag,
            output_hidden_states=True,
            return_dict=True,
        )

    assert torch.equal(outputs[True].logits, outputs[False].logits)
    assert torch.equal(outputs[True].hidden_states[-1], outputs[False].hidden_states[-1])


# ---------------------------------------------------------------------------
# Progress-bar hygiene
# ---------------------------------------------------------------------------
def _capture(module, hp):
    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        _run(module, hp)
    return stdout.getvalue() + stderr.getvalue()


@pytest.mark.unit
def test_upstream_draws_a_per_token_progress_bar(tiny_pair, tiny_hp):
    """Control: proves the capture is sensitive to upstream's tqdm hook."""
    upstream, _ = tiny_pair
    assert "Sampling" in _capture(upstream, tiny_hp)


@pytest.mark.unit
def test_cleaned_loop_prints_no_progress_bar(tiny_pair, tiny_hp, monkeypatch):
    monkeypatch.delenv(ChatterboxT3AR.PROGRESS_ENV, raising=False)
    _, cleaned = tiny_pair
    captured = _capture(cleaned, tiny_hp)

    assert "Sampling" not in captured
    assert "it/s" not in captured
    assert "\r" not in captured


@pytest.mark.unit
def test_progress_env_flag_restores_the_bar(tiny_pair, tiny_hp, monkeypatch):
    monkeypatch.setenv(ChatterboxT3AR.PROGRESS_ENV, "1")
    _, cleaned = tiny_pair
    assert "Sampling" in _capture(cleaned, tiny_hp)


# ---------------------------------------------------------------------------
# Decode state stays local to the call
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_decode_state_does_not_leak_between_calls(tiny_pair, tiny_hp):
    """A second call with different conditionals matches a fresh instance."""
    upstream, cleaned = tiny_pair

    _run(cleaned, tiny_hp, seed=7)
    second = _run(cleaned, tiny_hp, seed=7, conds=_conditionals(tiny_hp, speaker_seed=999))

    fresh = ChatterboxT3AR(tiny_hp).eval()
    fresh.load_state_dict(upstream.state_dict())
    fresh.requires_grad_(False)
    expected = _run(fresh, tiny_hp, seed=7, conds=_conditionals(tiny_hp, speaker_seed=999))

    assert torch.equal(second, expected)


# ---------------------------------------------------------------------------
# Factory wiring
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_t3_class_for_role_maps_tts_and_multilingual_to_the_cleaned_class():
    assert factory._t3_class_for_role(factory.ROLE_TTS) is ChatterboxT3AR
    # Phase 12-7: multilingual shares English's Llama_520M contract and now uses
    # the cleaned loop too; turbo keeps `T3.inference_turbo` (a different loop).
    assert factory._t3_class_for_role(factory.ROLE_MULTILINGUAL) is ChatterboxT3AR
    assert factory._t3_class_for_role(factory.ROLE_TURBO) is T3
    assert factory._t3_class_for_role(factory.ROLE_VC) is T3


class _FakeT3:
    """Minimal stand-in used to observe which class ``load_t3`` instantiates."""

    def __init__(self, hp=None):
        self.hp = hp
        self.tfmr = SimpleNamespace(wte=None)

    def load_state_dict(self, state):
        self.state = state

    def to(self, **kwargs):
        return self

    def eval(self):
        return self

    def requires_grad_(self, flag):
        return self


class _FakeChatterboxT3AR(_FakeT3):
    pass


@pytest.mark.unit
def test_load_t3_instantiates_the_cleaned_class_for_tts_and_multilingual(monkeypatch):
    """TTS and Multilingual get ``ChatterboxT3AR`` (Phase 12-7).

    The turbo branch of ``load_t3`` is unreachable (the factory never assembles
    turbo: ``build_role_model`` returns None and ``assemble_role`` raises) and
    its pre-existing ``T3Config(**_TURBO_HP)`` call is therefore not exercised
    here; turbo's class mapping is covered by the role-mapping test above.
    """
    monkeypatch.setattr(factory, "_ckpt_dir", lambda repo, revision: "/tmp/ckpt")
    monkeypatch.setattr(factory, "load_file", lambda path: {})
    monkeypatch.setattr(factory, "T3", _FakeT3)
    monkeypatch.setattr(factory, "ChatterboxT3AR", _FakeChatterboxT3AR)

    def load(role):
        return factory.load_t3("resemble-ai/chatterbox", None, "cpu", torch.float32, role=role)

    assert isinstance(load(factory.ROLE_TTS), _FakeChatterboxT3AR)
    assert isinstance(load(factory.ROLE_MULTILINGUAL), _FakeChatterboxT3AR)


@pytest.mark.unit
def test_load_t3_turbo_branch_builds_a_valid_config(monkeypatch):
    """Phase 9-3: the turbo branch no longer calls ``T3Config(**turbo_hp)``.

    Upstream ``T3Config.__init__`` only accepts ``text_tokens_dict_size``; the
    other turbo hyper-parameters are attributes set after construction (mirrors
    ``chatterbox/tts_turbo.py:149-155``). The old call raised ``TypeError``.
    """
    monkeypatch.setattr(factory, "_ckpt_dir", lambda repo, revision: "/tmp/ckpt")
    monkeypatch.setattr(factory, "load_file", lambda path: {})
    monkeypatch.setattr(factory, "T3", _FakeT3)

    t3 = factory.load_t3(
        "resemble-ai/chatterbox-turbo", None, "cpu", torch.float32, role=factory.ROLE_TURBO
    )

    assert t3.hp.text_tokens_dict_size == 50276
    assert t3.hp.speech_tokens_dict_size == 6563
    assert t3.hp.input_pos_emb is None
    assert t3.hp.speech_cond_prompt_len == 375
    assert t3.hp.use_perceiver_resampler is False
    assert t3.hp.emotion_adv is False
    assert t3.hp.llama_config_name == "GPT2_medium"
    assert not hasattr(t3.tfmr, "wte"), "turbo deletes tfmr.wte (as upstream does)"


@pytest.mark.unit
def test_build_role_model_assembles_the_cleaned_t3(tiny_hp, monkeypatch):
    """The TTS role assembled by the factory carries ``ChatterboxT3AR``."""
    cleaned = ChatterboxT3AR(tiny_hp).eval()
    backbone = factory.ChatterboxBackbone(
        repo="resemble-ai/chatterbox",
        revision=None,
        device="cpu",
        dtype=torch.float32,
        s3gen=torch.nn.Linear(2, 2),
        ve=torch.nn.Linear(2, 2),
        t3=cleaned,
        t3_dtype=torch.float32,
    )
    monkeypatch.setattr(factory, "build_backbone", lambda *args, **kwargs: backbone)
    monkeypatch.setattr(factory, "_ckpt_dir", lambda repo, revision: "/tmp/ckpt")
    monkeypatch.setattr(factory, "_load_tokenizer", lambda role, ckpt: object())
    monkeypatch.setattr(
        factory, "_load_conditionals", lambda repo, revision, device, role: SimpleNamespace(t3=None, gen={})
    )

    config = SimpleNamespace(
        backbone_group="chatterbox-base",
        model_type="ChatterboxTTS",
        model="resemble-ai/chatterbox",
        dtype=torch.float32,
        revision=None,
    )
    model = factory.build_role_model(config, "cpu", None)

    assert isinstance(model, factory.ChatterboxTTS)
    assert model.t3 is cleaned
    assert isinstance(model.t3, ChatterboxT3AR)


@pytest.mark.unit
def test_build_role_model_leaves_multilingual_on_the_upstream_class(tiny_hp, monkeypatch):
    plain = T3(tiny_hp).eval()
    backbone = factory.ChatterboxBackbone(
        repo="resemble-ai/chatterbox",
        revision=None,
        device="cpu",
        dtype=torch.float32,
        s3gen=torch.nn.Linear(2, 2),
        ve=torch.nn.Linear(2, 2),
        t3=plain,
        t3_dtype=torch.float32,
    )
    monkeypatch.setattr(factory, "build_backbone", lambda *args, **kwargs: backbone)
    monkeypatch.setattr(factory, "_ckpt_dir", lambda repo, revision: "/tmp/ckpt")
    monkeypatch.setattr(factory, "_load_tokenizer", lambda role, ckpt: object())
    monkeypatch.setattr(
        factory, "_load_conditionals", lambda repo, revision, device, role: SimpleNamespace(t3=None, gen={})
    )

    config = SimpleNamespace(
        backbone_group="chatterbox-base",
        model_type="ChatterboxMultilingualTTS",
        model="resemble-ai/chatterbox",
        dtype=torch.float32,
        revision=None,
    )
    model = factory.build_role_model(config, "cpu", None)

    assert model.t3 is plain
    assert not isinstance(model.t3, ChatterboxT3AR)


# ---------------------------------------------------------------------------
# Phase 12-7: multilingual analyzer is on by default and never leaks hooks
# ---------------------------------------------------------------------------
def _tiny_multilingual_hp(monkeypatch):
    monkeypatch.setitem(LLAMA_CONFIGS, "tiny_ar0_mtl", dict(_TINY_LLAMA))
    hp = T3Config.multilingual()  # text_tokens_dict_size == 2454 -> is_multilingual
    hp.llama_config_name = "tiny_ar0_mtl"
    hp.speech_tokens_dict_size = 96
    hp.start_speech_token = 90
    hp.stop_speech_token = 91
    hp.max_text_tokens = 64
    hp.max_speech_tokens = 64
    hp.use_perceiver_resampler = False
    hp.emotion_adv = True
    return hp


def _tiny_multilingual_text(hp):
    return torch.tensor([[hp.start_text_token, *range(5, 13), hp.stop_text_token]] * 2)


def _forward_hook_count(t3) -> int:
    return sum(len(layer.self_attn._forward_hooks) for layer in t3.tfmr.layers)


@pytest.mark.unit
def test_multilingual_analyzer_is_on_by_default_and_never_leaks_hooks(monkeypatch):
    """Default ON: a multilingual decode uses the analyzer but leaks no hooks.

    GPU-2 verified (2026-11-09): ``t3_mtl23ls_v2`` rarely emits
    ``stop_speech_token`` naturally at a chunk's end, so without the analyzer's
    post-completion ``long_tail`` trigger a chunk keeps babbling until
    ``max_new_tokens``. It must therefore be the default, while still leaving the
    shared Llama exactly as it found it (SDPA, ``output_attentions=False``, no
    forward hooks).
    """
    monkeypatch.delenv(ChatterboxT3AR.ANALYZER_ENV, raising=False)
    hp = _tiny_multilingual_hp(monkeypatch)
    torch.manual_seed(20261112)
    model = ChatterboxT3AR(hp).eval()
    model.requires_grad_(False)
    model.tfmr.config._attn_implementation = "sdpa"
    model.tfmr.config.output_attentions = False

    assert ChatterboxT3AR.analyzer_enabled() is True

    backend = model.build_ar_backend(len_cond=5, text_len=4)
    analyzer = backend.alignment_stream_analyzer
    assert isinstance(analyzer, _SafeAlignmentStreamAnalyzer)
    assert model.ar_output_attentions(backend) is True
    # Construction switched to eager and registered the spy hooks.
    assert model.tfmr.config._attn_implementation == "eager"
    assert model.tfmr.config.output_attentions is True
    assert _forward_hook_count(model) > 0

    # ``close`` is the only teardown path and restores the config.
    analyzer.close()
    assert _forward_hook_count(model) == 0
    assert model.tfmr.config._attn_implementation == "sdpa"
    assert model.tfmr.config.output_attentions is False

    # A full decode also ends on SDPA with zero hooks.
    torch.manual_seed(7)
    model.inference(
        t3_cond=_conditionals(hp),
        text_tokens=_tiny_multilingual_text(hp),
        **{**_DECODE, "max_new_tokens": 3},
    )
    assert model.tfmr.config._attn_implementation == "sdpa"
    assert model.tfmr.config.output_attentions is False
    assert _forward_hook_count(model) == 0, "every analyzer hook must be removed"


@pytest.mark.unit
def test_multilingual_analyzer_can_be_disabled(monkeypatch):
    """``HSE_CHATTERBOX_MTL_ANALYZER=0`` restores native SDPA and no analyzer."""
    monkeypatch.setenv(ChatterboxT3AR.ANALYZER_ENV, "0")
    hp = _tiny_multilingual_hp(monkeypatch)
    torch.manual_seed(20261112)
    model = ChatterboxT3AR(hp).eval()
    model.requires_grad_(False)

    assert ChatterboxT3AR.analyzer_enabled() is False

    backend = model.build_ar_backend(len_cond=5, text_len=4)

    assert backend.alignment_stream_analyzer is None
    assert model.tfmr.config._attn_implementation == "sdpa"
    assert _forward_hook_count(model) == 0
    assert model.ar_output_attentions(backend) is False

    torch.manual_seed(7)
    model.inference(
        t3_cond=_conditionals(hp),
        text_tokens=_tiny_multilingual_text(hp),
        **{**_DECODE, "max_new_tokens": 3},
    )
    assert model.tfmr.config._attn_implementation == "sdpa"
    assert _forward_hook_count(model) == 0

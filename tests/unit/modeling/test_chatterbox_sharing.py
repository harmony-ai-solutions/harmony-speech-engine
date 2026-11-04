"""Unit tests for Chatterbox backbone sharing (Phase 4-4).

The loaders are stubbed with tiny modules: these tests assert the *wiring*
(object identity, distinct wrapper instances, registry semantics, read-only
shared weights) without touching the multi-gigabyte checkpoints. The real
weight-identity guarantee is proven by the 4-2 `sharing:dedup-vs-dup` harness
run recorded in docs/benchmarks/audio-regression.md.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from harmonyspeech.modeling.models.chatterbox import factory
from harmonyspeech.modeling.weight_registry import WeightRegistry


def _tiny(name: str, size: int = 4) -> nn.Linear:
    torch.manual_seed(hash(name) % (2**31 - 1))
    return nn.Linear(size, size)


class _FakeRole:
    """Stand-in for the upstream wrapper classes (no PerthNet, no tokenizers)."""

    instances: list["_FakeRole"] = []

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.is_fake_role = True
        _FakeRole.instances.append(self)


@pytest.fixture
def patched_factory(monkeypatch):
    loaded: dict[str, list] = {"s3gen": [], "ve": [], "t3": []}

    def fake_load_s3gen(repo, revision, device, role):
        module = _tiny(f"s3gen:{role}")
        loaded["s3gen"].append((repo, revision, role))
        return module

    def fake_load_ve(repo, revision, device, role):
        module = _tiny(f"ve:{role}")
        loaded["ve"].append((repo, revision, role))
        return module

    def fake_load_t3(repo, revision, device, dtype, role):
        module = _tiny(f"t3:{role}").to(dtype)
        loaded["t3"].append((repo, revision, role))
        return module

    monkeypatch.setattr(factory, "load_s3gen", fake_load_s3gen)
    monkeypatch.setattr(factory, "load_ve", fake_load_ve)
    monkeypatch.setattr(factory, "load_t3", fake_load_t3)
    monkeypatch.setattr(factory, "_load_tokenizer", lambda role, ckpt: object())
    monkeypatch.setattr(
        factory,
        "_load_conditionals",
        lambda repo, revision, device, role: SimpleNamespace(t3=None, gen={}),
    )
    monkeypatch.setattr(factory, "ChatterboxTTS", _FakeRole)
    monkeypatch.setattr(factory, "ChatterboxMultilingualTTS", _FakeRole)
    monkeypatch.setattr(factory, "ChatterboxVC", _FakeRole)
    monkeypatch.setattr(factory, "_ckpt_dir", lambda repo, revision: "/tmp/ckpt")
    monkeypatch.setattr(
        factory,
        "torch",
        SimpleNamespace(
            load=lambda *a, **k: {"gen": {}},
            float32=torch.float32,
            float16=torch.float16,
            dtype=torch.dtype,
            device=torch.device,
        ),
    )
    _FakeRole.instances = []
    return loaded


def test_two_roles_share_s3gen_and_ve_but_not_the_wrapper(patched_factory):
    registry = WeightRegistry()
    tts = factory.build_backbone("resemble-ai/chatterbox", None, "cpu", torch.float32, registry, role=factory.ROLE_TTS)
    vc = factory.build_backbone("resemble-ai/chatterbox", None, "cpu", torch.float32, registry, role=factory.ROLE_VC)

    assert tts.s3gen is vc.s3gen
    assert tts.ve is vc.ve
    assert len(patched_factory["s3gen"]) == 1
    assert len(patched_factory["ve"]) == 1

    role_a = factory.assemble_role(factory.ROLE_TTS, tts, repo="resemble-ai/chatterbox", revision=None)
    role_b = factory.assemble_role(factory.ROLE_TTS, tts, repo="resemble-ai/chatterbox", revision=None)

    assert role_a is not role_b  # distinct wrapper instances (per-request state)
    assert role_a.s3gen is role_b.s3gen is tts.s3gen
    assert role_a.ve is role_b.ve is tts.ve


def test_shared_modules_are_readonly_and_bitwise_stable(patched_factory):
    registry = WeightRegistry()
    backbone = factory.build_backbone(
        "resemble-ai/chatterbox", None, "cpu", torch.float32, registry, role=factory.ROLE_TTS
    )

    for module in (backbone.s3gen, backbone.ve, backbone.t3):
        WeightRegistry.assert_readonly(module)
        assert all(not p.requires_grad for p in module.parameters())

    first = {name: p.detach().clone() for name, p in backbone.s3gen.named_parameters()}
    factory.assemble_role(factory.ROLE_VC, backbone, repo="resemble-ai/chatterbox", revision=None)
    for name, param in backbone.s3gen.named_parameters():
        assert torch.equal(param, first[name])


def test_embedding_shares_the_tts_t3(patched_factory):
    registry = WeightRegistry()
    tts = factory.build_backbone("resemble-ai/chatterbox", None, "cpu", torch.float32, registry, role=factory.ROLE_TTS)
    embedding = factory.build_backbone(
        "resemble-ai/chatterbox", None, "cpu", torch.float32, registry, role=factory.ROLE_EMBEDDING
    )

    assert embedding.t3 is tts.t3
    assert len(patched_factory["t3"]) == 1
    assert registry.refcount(("resemble-ai/chatterbox", None, "t3", torch.float32)) == 2


def test_t3_dtype_is_part_of_the_key(patched_factory):
    registry = WeightRegistry()
    # "cuda" so the CPU float32 guard in `_resolve_load_dtype` does not collapse
    # both requests to the same dtype.
    fp32 = factory.build_backbone("resemble-ai/chatterbox", None, "cuda", torch.float32, registry, role=factory.ROLE_TTS)
    fp16 = factory.build_backbone("resemble-ai/chatterbox", None, "cuda", torch.float16, registry, role=factory.ROLE_TTS)

    # T3 is distinct per dtype; shared float32 components are reused.
    assert fp32.t3 is not fp16.t3
    assert fp32.s3gen is fp16.s3gen
    assert len(patched_factory["t3"]) == 2


def test_multilingual_uses_base_s3gen_and_ve_with_own_t3(patched_factory, monkeypatch):
    monkeypatch.setattr(factory, "_has_standard_files", lambda repo, revision: True)
    registry = WeightRegistry()
    tts = factory.build_backbone("resemble-ai/chatterbox", None, "cpu", torch.float32, registry, role=factory.ROLE_TTS)
    mtl = factory.build_backbone(
        "resemble-ai/chatterbox", None, "cpu", torch.float32, registry, role=factory.ROLE_MULTILINGUAL
    )

    assert mtl.s3gen is tts.s3gen
    assert mtl.ve is tts.ve
    assert mtl.t3 is not tts.t3, "multilingual T3 has different weights and must not be shared"
    registered = list(registry.keys())
    assert ("resemble-ai/chatterbox", None, "t3:multilingual") in [
        (repo, revision, component) for repo, revision, component, _ in registered
    ]


def test_multilingual_without_base_files_loads_its_own_components(patched_factory, monkeypatch):
    monkeypatch.setattr(factory, "_has_standard_files", lambda repo, revision: False)
    registry = WeightRegistry()
    base = factory.build_backbone("resemble-ai/chatterbox", None, "cpu", torch.float32, registry, role=factory.ROLE_TTS)
    mtl = factory.build_backbone(
        "resemble-ai/chatterbox", None, "cpu", torch.float32, registry, role=factory.ROLE_MULTILINGUAL
    )

    assert mtl.s3gen is not base.s3gen
    assert mtl.ve is not base.ve
    # Loaded from the multilingual files, not the standard ones.
    assert any(role == factory.ROLE_MULTILINGUAL for _repo, _rev, role in patched_factory["s3gen"])


def test_role_for_model_type_maps_embedding_variants():
    assert factory.role_for_model_type("ChatterboxTTS") == factory.ROLE_TTS
    assert factory.role_for_model_type("ChatterboxVC") == factory.ROLE_VC
    assert factory.role_for_model_type("ChatterboxEmbedding") == factory.ROLE_EMBEDDING
    assert factory.role_for_model_type("ChatterboxEmbedding", "resemble-ai/chatterbox-turbo") == factory.ROLE_TURBO
    assert (
        factory.role_for_model_type("ChatterboxEmbedding", "resemble-ai/chatterbox-multilingual")
        == factory.ROLE_MULTILINGUAL
    )
    with pytest.raises(ValueError):
        factory.role_for_model_type("NotAChatterbox")


def test_turbo_is_not_grouped():
    config = SimpleNamespace(
        backbone_group="chatterbox-base",
        model_type="ChatterboxTurboTTS",
        model="resemble-ai/chatterbox-turbo",
        dtype=torch.float16,
        revision=None,
    )
    assert factory.build_role_model(config, "cuda", WeightRegistry()) is None


def test_build_role_model_requires_group():
    config = SimpleNamespace(
        backbone_group=None,
        model_type="ChatterboxTTS",
        model="resemble-ai/chatterbox",
        dtype=torch.float32,
        revision=None,
    )
    assert factory.build_role_model(config, "cpu", WeightRegistry()) is None

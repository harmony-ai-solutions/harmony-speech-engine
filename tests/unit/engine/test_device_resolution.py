"""Device resolution and per-device AR pool accounting (Phase 11-0).

The Phase 9 regression these tests exist for: ``ARCachePool`` was built with
``torch.device("cuda")`` (index ``None``) while a loaded model reports
``cuda:0``; the runner compared them with ``!=`` and disabled itself on every
CUDA model. ``resolve_device`` makes both spellings canonical, and the pool now
owns one budget **per device** so an explicit ``cuda:1`` model gets a real arena
on ``cuda:1`` without eating the first device's headroom.

The per-device accounting tests use ``cpu`` + ``meta`` so they are hermetic; the
real CUDA-index test is skipped when the host has no second card (the opt-in
integration test covers the full load path).
"""

from types import SimpleNamespace

import pytest
import torch

from harmonyspeech.common.device import cuda_index, device_key, devices_match, resolve_device
from harmonyspeech.engine.ar_cache import ARCacheBudgetError, ARCacheDescriptor, ARCachePool
from harmonyspeech.task_handler.ar_runner import ARBatchRunner


# ---------------------------------------------------------------------------
# resolve_device / devices_match
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_resolve_none_and_auto_follow_cuda_availability(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    assert resolve_device(None) == torch.device("cuda:1")
    assert resolve_device("auto") == torch.device("cuda:1")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device(None) == torch.device("cpu")
    assert resolve_device("auto") == torch.device("cpu")


@pytest.mark.unit
def test_resolve_unindexed_cuda_is_the_active_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    assert resolve_device("cuda") == torch.device("cuda:0")

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    assert resolve_device("cuda") == torch.device("cuda:2")
    # An unindexed torch.device behaves the same as the string spelling.
    assert resolve_device(torch.device("cuda")) == torch.device("cuda:2")


@pytest.mark.unit
def test_resolve_preserves_an_explicit_index(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    assert resolve_device("cuda:1") == torch.device("cuda:1")
    assert resolve_device(torch.device("cuda", 2)) == torch.device("cuda:2")
    assert resolve_device("cpu") == torch.device("cpu")


@pytest.mark.unit
def test_resolve_is_idempotent(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    for raw in (None, "auto", "cuda", "cuda:1", "cpu"):
        once = resolve_device(raw)
        assert resolve_device(once) == once


@pytest.mark.unit
def test_devices_match_treats_unindexed_cuda_as_the_active_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    # The Phase 9 bug in one assertion: these are the same device.
    assert devices_match("cuda", "cuda:0")
    assert devices_match(torch.device("cuda"), torch.device("cuda:0"))
    # But a cross-device arena is still refused.
    assert not devices_match("cuda:0", "cuda:1")
    assert not devices_match("cpu", "cuda:0")


@pytest.mark.unit
def test_device_key_and_cuda_index(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    assert device_key("cuda") == "cuda:0"
    assert device_key("cuda:1") == "cuda:1"
    assert device_key("cpu") == "cpu"
    assert cuda_index("cuda:1") == 1
    assert cuda_index("cpu") is None


# ---------------------------------------------------------------------------
# Per-device pool accounting
# ---------------------------------------------------------------------------
def _descriptor(key: str, *, slots: int = 1) -> ARCacheDescriptor:
    return ARCacheDescriptor(
        model_key=key,
        layers=2,
        kv_heads=2,
        head_dim=4,
        dtype=torch.float32,
        cap_tokens=8,
        slots=slots,
    )


@pytest.mark.unit
def test_pool_default_device_is_indexed_when_cuda_is_available():
    pool = ARCachePool(budget_bytes=1024)
    if torch.cuda.is_available():
        assert pool.device.type == "cuda"
        assert pool.device.index is not None
    else:
        assert pool.device == torch.device("cpu")


@pytest.mark.unit
def test_pool_budget_is_accounted_per_device():
    descriptor = _descriptor("a")
    slab = descriptor.slab_bytes(8)
    pool = ARCachePool(budget_bytes=2 * slab, device="cpu")

    cpu_arena = pool.register(_descriptor("a"), device="cpu")
    meta_arena = pool.register(_descriptor("b"), device="meta")

    assert cpu_arena.device == torch.device("cpu")
    assert meta_arena.device == torch.device("meta")
    # Each device got its own full budget: the second registration did not
    # consume the first device's headroom.
    assert pool.remaining_bytes_for("cpu") == slab
    assert pool.remaining_bytes_for("meta") == slab
    # ``remaining_bytes`` keeps its single-device meaning (the pool's default).
    assert pool.remaining_bytes == slab
    assert pool.pool_bytes == 2 * slab

    stats = pool.stats()
    assert stats["devices"]["cpu"]["used_bytes"] == slab
    assert stats["devices"]["meta"]["used_bytes"] == slab
    assert stats["arenas"]["a"]["device"] == "cpu"
    assert stats["arenas"]["b"]["device"] == "meta"


@pytest.mark.unit
def test_pool_refuses_when_a_device_budget_is_exhausted():
    descriptor = _descriptor("a")
    slab = descriptor.slab_bytes(8)
    pool = ARCachePool(budget_bytes=slab, device="cpu")
    pool.register(descriptor, device="meta")

    with pytest.raises(ARCacheBudgetError, match="per-device budget"):
        pool.register(_descriptor("b"), device="meta")
    # The other device still has its whole budget.
    assert pool.remaining_bytes_for("cpu") == slab


@pytest.mark.unit
def test_automatic_slot_count_uses_the_requested_device_budget():
    descriptor = _descriptor("a", slots=None)
    slab = descriptor.slab_bytes(8)
    pool = ARCachePool(budget_bytes=3 * slab + slab // 2, device="cpu")
    arena = pool.register(descriptor, device="meta")
    assert arena.slots == 3, "automatic slots come from the target device's budget"


# ---------------------------------------------------------------------------
# Runner registration uses the model's device, not the pool default
# ---------------------------------------------------------------------------
class _FakeT3:
    """Registration-only stand-in: ``_geometry`` needs cfg + one parameter."""

    is_gpt = False

    def __init__(self, device: str) -> None:
        self.device = torch.device(device)
        self.cfg = SimpleNamespace(
            num_hidden_layers=2,
            num_key_value_heads=4,
            head_dim=8,
            hidden_size=32,
            num_attention_heads=4,
        )

    def parameters(self):
        return iter([torch.zeros(1, dtype=torch.float16, device=self.device)])


@pytest.mark.unit
def test_runner_registers_on_the_model_device_despite_pool_default():
    """The Phase 9 bug: a pool default that differs must not disable the runner."""
    slab = ARCacheDescriptor(
        model_key="m", layers=2, kv_heads=4, head_dim=8, dtype=torch.float16, cap_tokens=64
    ).slab_bytes(64)
    pool = ARCachePool(budget_bytes=4 * slab, device="cpu", default_cap_tokens=64)
    runner = ARBatchRunner(
        t3=_FakeT3("meta"),
        pool=pool,
        model_key="chatterbox-on-meta",
        model_config=SimpleNamespace(model_type="ChatterboxTTS", seed=0, max_batch_size=2),
    )

    assert runner.register() is True
    assert runner.arena is not None
    assert runner.arena.device == torch.device("meta")
    assert pool.remaining_bytes_for("meta") < pool.budget_bytes
    # The default device was not touched.
    assert pool.remaining_bytes_for("cpu") == pool.budget_bytes


@pytest.mark.unit
def test_runner_registers_on_a_cpu_pool_for_a_cpu_model():
    slab = ARCacheDescriptor(
        model_key="m", layers=2, kv_heads=4, head_dim=8, dtype=torch.float16, cap_tokens=64
    ).slab_bytes(64)
    pool = ARCachePool(budget_bytes=4 * slab, device="cpu", default_cap_tokens=64)
    runner = ARBatchRunner(
        t3=_FakeT3("cpu"),
        pool=pool,
        model_key="cpu-model",
        model_config=SimpleNamespace(model_type="ChatterboxTTS", seed=0, max_batch_size=2),
    )
    assert runner.register() is True
    assert runner.arena.device == torch.device("cpu")

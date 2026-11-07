"""Unit tests for the generic AR KV-cache pool (Phase 9-2).

Coverage follows the task spec: allocator lifecycle (acquire/release/grant/
refuse), FIFO free-list reuse, the slot-isolation invariant pinned with a
poisoned-slot decode against real ``scaled_dot_product_attention`` math, budget
refusal at registration, layout/geometry math, CPU fallback correctness, the
disabled fast path, config/CLI plumbing and the ``/metrics`` + ``/health``
surfaces (with and without CUDA).
"""

import argparse
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch
from torch.nn.functional import scaled_dot_product_attention

from harmonyspeech.common.config import DeviceConfig, EngineConfig, ModelConfig
from harmonyspeech.engine.ar_cache import (
    DEFAULT_AR_CACHE_BUDGET_BYTES,
    DEFAULT_AR_CACHE_CAP_TOKENS,
    ARCacheBudgetError,
    ARCacheDescriptor,
    ARCachePool,
    ARCacheSlot,
)
from harmonyspeech.engine.ar_cache_metrics import render_ar_cache_metrics
from harmonyspeech.engine.args_tools import EngineArgs
from harmonyspeech.engine.harmonyspeech_engine import HarmonySpeechEngine

SERIES = (
    "hse_ar_cache_enabled",
    "hse_ar_cache_pool_bytes",
    "hse_ar_cache_allocated_bytes",
    "hse_ar_cache_slots",
    "hse_ar_cache_slots_active",
    "hse_ar_cache_slots_free",
    "hse_ar_cache_grants_total",
    "hse_ar_cache_refusals_total",
)

_CPU = torch.device("cpu")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _descriptor(
    *,
    model_key: str = "tiny-ar",
    layers: int = 2,
    kv_heads: int = 2,
    head_dim: int = 4,
    dtype: torch.dtype = torch.float32,
    cap_tokens: int | None = 8,
    slots: int | None = 2,
) -> ARCacheDescriptor:
    return ARCacheDescriptor(
        model_key=model_key,
        layers=layers,
        kv_heads=kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        cap_tokens=cap_tokens,
        slots=slots,
    )


def _pool_with(descriptor: ARCacheDescriptor, *, budget_bytes: int | None = None) -> tuple[ARCachePool, object]:
    """Build a CPU pool whose budget exactly fits ``descriptor``."""
    cap = descriptor.cap_tokens or 8
    slots = descriptor.slots or 1
    bytes_needed = descriptor.slab_bytes(cap) * slots
    pool = ARCachePool(budget_bytes=budget_bytes or bytes_needed, device="cpu")
    arena = pool.register(descriptor)
    return pool, arena


def _engine_shell() -> HarmonySpeechEngine:
    engine = HarmonySpeechEngine.__new__(HarmonySpeechEngine)
    engine.model_executors = {}
    engine.model_configs = []
    return engine


def _write_prefix(slot: ARCacheSlot, layer: int, tokens: int, *, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    keys = torch.randn(tokens, slot.kv_heads, slot.head_dim, generator=generator)
    values = torch.randn(tokens, slot.kv_heads, slot.head_dim, generator=generator)
    slot.write(layer, 0, keys, values)
    return keys, values


def _decode(slot: ARCacheSlot, layer: int, query: torch.Tensor, length: int) -> torch.Tensor:
    """One real SDPA call over the slot's zero-copy ``[1, cap, H, D]`` views."""
    keys = slot.key_view(layer)[:, :length].transpose(1, 2)  # [1, H, T, D]
    values = slot.value_view(layer)[:, :length].transpose(1, 2)
    return scaled_dot_product_attention(query, keys, values)


# ---------------------------------------------------------------------------
# Descriptor geometry
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_slab_and_arena_bytes_follow_the_layout_formula():
    descriptor = ARCacheDescriptor(
        model_key="chatterbox-t3",
        layers=4,
        kv_heads=8,
        head_dim=64,
        dtype=torch.float16,
        cap_tokens=768,
        slots=3,
    )
    # [L, 2, B, cap, H, D] × 2 bytes per fp16 scalar.
    expected_slab = 4 * 2 * 768 * 8 * 64 * 2
    assert descriptor.slab_bytes() == expected_slab
    assert descriptor.arena_bytes() == expected_slab * 3
    assert descriptor.element_size == 2


@pytest.mark.unit
def test_cap_tokens_and_slots_can_be_overridden_per_call():
    descriptor = _descriptor(cap_tokens=None, slots=None)
    assert descriptor.slab_bytes(16) == 2 * 2 * 16 * 2 * 4 * 4
    assert descriptor.arena_bytes(16, 5) == descriptor.slab_bytes(16) * 5


@pytest.mark.unit
def test_descriptor_without_cap_requires_one_at_use_time():
    descriptor = _descriptor(cap_tokens=None)
    with pytest.raises(ValueError, match="cap_tokens"):
        descriptor.slab_bytes()


@pytest.mark.unit
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_key", ""),
        ("layers", 0),
        ("kv_heads", -1),
        ("head_dim", 0),
        ("cap_tokens", 0),
        ("slots", 0),
    ],
)
def test_descriptor_rejects_invalid_geometry(field, value):
    kwargs = {"model_key": "m", "layers": 1, "kv_heads": 1, "head_dim": 2}
    kwargs[field] = value
    with pytest.raises(ValueError):
        ARCacheDescriptor(**kwargs)


@pytest.mark.unit
def test_descriptor_rejects_a_non_torch_dtype():
    with pytest.raises(TypeError, match="torch.dtype"):
        ARCacheDescriptor(model_key="m", layers=1, kv_heads=1, head_dim=2, dtype="float16")


# ---------------------------------------------------------------------------
# Registration + budget validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_pool_defaults_match_the_owner_approved_initial_values():
    pool = ARCachePool(device="cpu")
    assert pool.budget_bytes == DEFAULT_AR_CACHE_BUDGET_BYTES == 1073741824
    assert pool.default_cap_tokens == DEFAULT_AR_CACHE_CAP_TOKENS == 768
    assert pool.device.type == "cpu"


@pytest.mark.unit
def test_register_uses_the_pool_default_cap_tokens():
    descriptor = _descriptor(cap_tokens=None, slots=1)
    pool = ARCachePool(budget_bytes=descriptor.slab_bytes(32), device="cpu", default_cap_tokens=32)
    arena = pool.register(descriptor)

    assert arena is not None
    assert arena.cap_tokens == 32
    assert arena.slab_bytes == descriptor.slab_bytes(32)
    assert arena.arena_bytes == arena.slab_bytes


@pytest.mark.unit
def test_register_auto_slots_fill_the_remaining_budget():
    descriptor = _descriptor(cap_tokens=8, slots=None)
    slab = descriptor.slab_bytes(8)
    pool = ARCachePool(budget_bytes=slab * 3 + slab // 2, device="cpu")

    arena = pool.register(descriptor)

    # Pool availability decides the slot count — never admission width.
    assert arena.slots == 3
    assert pool.stats()["slots"] == 3
    assert pool.stats()["pool_bytes"] == slab * 3


@pytest.mark.unit
def test_register_refuses_a_slab_that_exceeds_the_budget():
    descriptor = _descriptor(cap_tokens=8, slots=1)
    slab = descriptor.slab_bytes(8)
    pool = ARCachePool(budget_bytes=slab - 1, device="cpu")

    with pytest.raises(ARCacheBudgetError, match="slab"):
        pool.register(descriptor)
    assert pool.stats()["pool_bytes"] == 0
    assert pool.stats()["arenas"] == {}


@pytest.mark.unit
def test_register_refuses_explicit_slots_that_do_not_fit():
    descriptor = _descriptor(cap_tokens=8, slots=4)
    slab = descriptor.slab_bytes(8)
    pool = ARCachePool(budget_bytes=slab * 3, device="cpu")

    with pytest.raises(ARCacheBudgetError, match="4 slots"):
        pool.register(descriptor)


@pytest.mark.unit
def test_register_converts_an_allocation_failure_into_a_budget_error(monkeypatch):
    from harmonyspeech.engine import ar_cache as ar_cache_module

    def _boom(*args, **kwargs):
        raise RuntimeError("CUDA out of memory (simulated)")

    monkeypatch.setattr(ar_cache_module, "ARCacheArena", _boom)
    pool = ARCachePool(budget_bytes=4096, device="cpu")

    with pytest.raises(ARCacheBudgetError, match="could not allocate"):
        pool.register(_descriptor())
    # Nothing is registered or accounted after a failed allocation.
    assert pool.stats()["pool_bytes"] == 0
    assert pool.stats()["arenas"] == {}


@pytest.mark.unit
def test_register_rejects_a_duplicate_model_key():
    descriptor = _descriptor()
    pool, _ = _pool_with(descriptor)

    with pytest.raises(ValueError, match="already registered"):
        pool.register(descriptor)


@pytest.mark.unit
def test_register_is_none_on_a_disabled_pool():
    pool = ARCachePool(budget_bytes=1024, device="cpu", enabled=False)
    assert pool.register(_descriptor()) is None


@pytest.mark.unit
def test_pool_rejects_a_non_positive_budget_when_enabled():
    with pytest.raises(ValueError, match="budget_bytes"):
        ARCachePool(budget_bytes=0, device="cpu")


# ---------------------------------------------------------------------------
# Allocator lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_acquire_release_grant_and_refuse_lifecycle():
    pool, arena = _pool_with(_descriptor(slots=2))

    first = pool.acquire("tiny-ar")
    second = pool.acquire("tiny-ar")
    refused = pool.acquire("tiny-ar")

    assert first is not None and second is not None
    assert {first.slot_index, second.slot_index} == {0, 1}
    # Exhaustion is a graceful ``None``, never an exception or an allocation.
    assert refused is None

    stats = pool.stats()
    assert stats["slots"] == 2
    assert stats["slots_active"] == 2
    assert stats["slots_free"] == 0
    assert stats["allocated_bytes"] == arena.slab_bytes * 2
    assert stats["grants_total"] == 2
    assert stats["refusals_total"] == 1

    pool.release(first)
    assert pool.acquire("tiny-ar").slot_index == first.slot_index
    assert pool.stats()["releases_total"] == 1
    assert pool.stats()["slots_active"] == 2


@pytest.mark.unit
def test_free_list_is_fifo_in_release_order():
    pool, _ = _pool_with(_descriptor(slots=3))
    slots = [pool.acquire("tiny-ar") for _ in range(3)]
    assert [slot.slot_index for slot in slots] == [0, 1, 2]

    # Release out of order: the free list must replay exactly this order.
    for index in (0, 2, 1):
        pool.release(slots[index])

    assert [pool.acquire("tiny-ar").slot_index for _ in range(3)] == [0, 2, 1]


@pytest.mark.unit
def test_double_release_raises_and_keeps_the_slot_out_of_the_free_list():
    pool, _ = _pool_with(_descriptor(slots=1))
    slot = pool.acquire("tiny-ar")
    pool.release(slot)

    with pytest.raises(ValueError, match="not active"):
        pool.release(slot)
    assert pool.stats()["slots_free"] == 1


@pytest.mark.unit
def test_releasing_a_foreign_slot_raises():
    pool_a, _ = _pool_with(_descriptor(model_key="a", slots=1))
    pool_b, _ = _pool_with(_descriptor(model_key="a", slots=1))
    foreign = pool_b.acquire("a")

    with pytest.raises(ValueError, match="does not own"):
        pool_a.release(foreign)


@pytest.mark.unit
def test_acquire_unknown_model_is_a_programming_error():
    pool, _ = _pool_with(_descriptor())
    with pytest.raises(KeyError, match="no AR cache arena"):
        pool.acquire("not-registered")


@pytest.mark.unit
def test_slot_release_delegates_to_the_pool():
    pool, _ = _pool_with(_descriptor(slots=1))
    slot = pool.acquire("tiny-ar")
    slot.release()

    assert slot.released is True
    assert pool.stats()["slots_free"] == 1


@pytest.mark.unit
def test_capacity_reports_active_rows_for_admission():
    pool, _ = _pool_with(_descriptor(slots=2))
    pool.acquire("tiny-ar")

    per_family = pool.capacity("tiny-ar")
    aggregate = pool.capacity()

    assert per_family["slots"] == 2
    assert per_family["slots_active"] == 1
    assert per_family["active_rows"] == 1
    assert aggregate["active_rows"] == 1
    assert aggregate["slots_free"] == 1
    assert aggregate["arenas"]["tiny-ar"]["model_key"] == "tiny-ar"
    with pytest.raises(KeyError):
        pool.capacity("missing")


# ---------------------------------------------------------------------------
# Zero-copy views + writes
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_slot_and_arena_views_alias_the_arena_tensor():
    pool, arena = _pool_with(_descriptor(slots=2))
    slot = pool.acquire("tiny-ar")

    slot_view = slot.key_view(0)
    arena_view = arena.key_view(0, slice(0, 2))

    assert slot_view.shape == (1, arena.cap_tokens, arena.kv_heads, arena.head_dim)
    assert arena_view.shape == (2, arena.cap_tokens, arena.kv_heads, arena.head_dim)
    assert arena.key_view(0).shape == (2, arena.cap_tokens, arena.kv_heads, arena.head_dim)
    assert slot_view.untyped_storage().data_ptr() == arena.tensor.untyped_storage().data_ptr()
    assert arena_view.untyped_storage().data_ptr() == arena.tensor.untyped_storage().data_ptr()
    assert slot_view.data_ptr() == arena.tensor[0, 0, slot.slot_index].data_ptr()

    # Writing through the view is visible in the arena (no copy anywhere).
    slot_view[0, 3, 1, 2] = 7.5
    assert arena.tensor[0, 0, slot.slot_index, 3, 1, 2].item() == pytest.approx(7.5)


@pytest.mark.unit
def test_write_copies_a_block_at_the_requested_position():
    pool, _ = _pool_with(_descriptor(cap_tokens=8, slots=1))
    slot = pool.acquire("tiny-ar")
    keys = torch.arange(3 * slot.kv_heads * slot.head_dim, dtype=torch.float32).reshape(
        3, slot.kv_heads, slot.head_dim
    )
    values = -keys

    slot.write(1, 2, keys, values)

    assert torch.equal(slot.key_view(1)[0, 2:5], keys)
    assert torch.equal(slot.value_view(1)[0, 2:5], values)
    # Layer 0 is untouched.
    assert torch.count_nonzero(slot.key_view(0)) == 0


@pytest.mark.unit
def test_write_rejects_out_of_bounds_and_wrong_shapes():
    pool, _ = _pool_with(_descriptor(cap_tokens=4, slots=1))
    slot = pool.acquire("tiny-ar")
    block = torch.zeros(2, slot.kv_heads, slot.head_dim)

    with pytest.raises(ValueError, match="exceeds cap_tokens"):
        slot.write(0, 3, block, block)
    with pytest.raises(ValueError, match="trailing shape"):
        slot.write(0, 0, torch.zeros(2, slot.kv_heads, slot.head_dim + 1), block)
    with pytest.raises(IndexError, match="layer"):
        slot.write(5, 0, block, block)


# ---------------------------------------------------------------------------
# Isolation invariant: the poisoned-slot test
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_poisoned_slot_decode_matches_clean_slot():
    """Stale KV beyond a sequence's write position must never be read.

    Two slots of the same arena get identical K/V for positions ``0..T-1``;
    one of them is pre-filled with NaN everywhere first. Decoding over the
    written prefix with real ``scaled_dot_product_attention`` math must produce
    bit-identical outputs, and the same decode over the *full* capacity must
    still see the poison — proving the prefix slice, not luck, is the guard.
    """
    descriptor = _descriptor(layers=2, kv_heads=2, head_dim=8, cap_tokens=16, slots=2)
    pool, arena = _pool_with(descriptor)
    clean = pool.acquire("tiny-ar")
    poisoned = pool.acquire("tiny-ar")
    assert clean is not None and poisoned is not None

    # Poison the whole slab of the reused slot: K and V, every layer.
    for layer in range(arena.layers):
        poisoned.key_view(layer).fill_(float("nan"))
        poisoned.value_view(layer).fill_(float("nan"))

    tokens = 5
    generator = torch.Generator().manual_seed(1234)
    for layer in range(arena.layers):
        keys = torch.randn(tokens, arena.kv_heads, arena.head_dim, generator=generator)
        values = torch.randn(tokens, arena.kv_heads, arena.head_dim, generator=generator)
        clean.write(layer, 0, keys, values)
        poisoned.write(layer, 0, keys, values)

    query = torch.randn(1, arena.kv_heads, 1, arena.head_dim, generator=generator)

    for layer in range(arena.layers):
        clean_out = _decode(clean, layer, query, tokens)
        poisoned_out = _decode(poisoned, layer, query, tokens)
        assert torch.isfinite(poisoned_out).all()
        assert torch.equal(clean_out, poisoned_out)

    # The poison is real: reading past the written prefix does hit it.
    unmasked = _decode(poisoned, 0, query, arena.cap_tokens)
    assert torch.isnan(unmasked).any()


@pytest.mark.unit
def test_reused_slot_after_release_serves_a_second_request_cleanly():
    """A released slot keeps stale bytes; the next request's prefix still decodes."""
    descriptor = _descriptor(cap_tokens=16, slots=1)
    pool, _ = _pool_with(descriptor)

    first = pool.acquire("tiny-ar")
    _write_prefix(first, 0, 6, seed=7)
    pool.release(first)

    second = pool.acquire("tiny-ar")
    assert second.slot_index == first.slot_index
    query = torch.randn(1, second.kv_heads, 1, second.head_dim, generator=torch.Generator().manual_seed(9))
    first_out = _decode(first, 0, query, 6)
    second_out = _decode(second, 0, query, 6)

    assert torch.equal(first_out, second_out)


# ---------------------------------------------------------------------------
# CPU fallback + CUDA path
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_cpu_pool_allocates_on_cpu_and_works_end_to_end():
    pool = ARCachePool(budget_bytes=64 * 1024, device="cpu")
    descriptor = _descriptor(dtype=torch.float32, cap_tokens=16, slots=1)
    arena = pool.register(descriptor)
    slot = pool.acquire("tiny-ar")

    assert pool.device.type == "cpu"
    assert arena.tensor.device.type == "cpu"
    assert arena.tensor.dtype == torch.float32
    _write_prefix(slot, 0, 4)
    assert torch.count_nonzero(slot.key_view(0)[0, :4]) > 0


@pytest.mark.unit
def test_auto_device_prefers_cuda_only_when_available():
    pool = ARCachePool(budget_bytes=1024)
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert pool.device.type == expected


@pytest.mark.unit
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_fp16_pool_roundtrip_and_attention():
    pool = ARCachePool(budget_bytes=64 * 1024, device="cuda")
    descriptor = _descriptor(dtype=torch.float16, cap_tokens=16, slots=1)
    arena = pool.register(descriptor)
    slot = pool.acquire("tiny-ar")

    assert arena.tensor.device.type == "cuda"
    assert arena.tensor.dtype == torch.float16
    keys = torch.randn(4, arena.kv_heads, arena.head_dim, device="cuda", dtype=torch.float16)
    slot.write(0, 0, keys, keys)
    assert torch.equal(slot.key_view(0)[0, :4], keys)

    query = torch.randn(1, arena.kv_heads, 1, arena.head_dim, device="cuda", dtype=torch.float16)
    out = _decode(slot, 0, query, 4)
    assert out.shape == (1, arena.kv_heads, 1, arena.head_dim)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Disabled fast path
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_disabled_pool_never_allocates_and_reports_zeros():
    pool = ARCachePool(budget_bytes=1024, device="cpu", enabled=False)
    descriptor = _descriptor()

    assert pool.register(descriptor) is None
    assert pool.acquire("tiny-ar") is None
    assert pool.pool_bytes == 0
    assert pool.arena_table() == []

    stats = pool.stats()
    assert stats["enabled"] is False
    assert stats["pool_bytes"] == 0
    assert stats["slots"] == 0
    assert stats["grants_total"] == 0
    assert stats["arenas"] == {}

    capacity = pool.capacity()
    assert capacity["enabled"] is False
    assert capacity["active_rows"] == 0


# ---------------------------------------------------------------------------
# Metrics renderer
# ---------------------------------------------------------------------------


def _warm_stats() -> dict:
    return {
        "enabled": True,
        "budget_bytes": 2**30,
        "default_cap_tokens": 768,
        "device": "cuda",
        "pool_bytes": 4096,
        "allocated_bytes": 1024,
        "slots": 4,
        "slots_active": 1,
        "slots_free": 3,
        "active_rows": 1,
        "grants_total": 7,
        "refusals_total": 2,
        "releases_total": 6,
        "arenas": {},
    }


@pytest.mark.unit
def test_render_warm_pool_emits_every_series_with_its_value():
    text = render_ar_cache_metrics(_warm_stats())

    for line in (
        "hse_ar_cache_enabled 1",
        "hse_ar_cache_pool_bytes 4096",
        "hse_ar_cache_allocated_bytes 1024",
        "hse_ar_cache_slots 4",
        "hse_ar_cache_slots_active 1",
        "hse_ar_cache_slots_free 3",
        "hse_ar_cache_grants_total 7",
        "hse_ar_cache_refusals_total 2",
    ):
        assert line in text

    for name in SERIES:
        assert f"# HELP {name} " in text
        assert f"# TYPE {name} " in text
    assert "# TYPE hse_ar_cache_grants_total counter" in text
    assert "# TYPE hse_ar_cache_pool_bytes gauge" in text


@pytest.mark.unit
def test_render_disabled_pool_keeps_every_series_present_and_zero():
    text = render_ar_cache_metrics({"enabled": False})

    assert "hse_ar_cache_enabled 0" in text
    for name in SERIES[1:]:
        assert f"{name} 0" in text


@pytest.mark.unit
def test_render_none_renders_the_disabled_shape():
    text = render_ar_cache_metrics(None)

    assert "hse_ar_cache_enabled 0" in text
    assert "hse_ar_cache_pool_bytes 0" in text
    assert "hse_ar_cache_refusals_total 0" in text
    for name in SERIES:
        assert f"# TYPE {name} " in text


@pytest.mark.unit
def test_render_accepts_live_cpu_pool_stats():
    pool, _ = _pool_with(_descriptor(slots=2))
    pool.acquire("tiny-ar")

    text = render_ar_cache_metrics(pool.stats())

    assert "hse_ar_cache_enabled 1" in text
    assert "hse_ar_cache_slots 2" in text
    assert "hse_ar_cache_slots_active 1" in text
    assert "hse_ar_cache_grants_total 1" in text
    assert "hse_ar_cache_pool_bytes 0" not in text


@pytest.mark.unit
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_render_accepts_live_cuda_pool_stats():
    pool = ARCachePool(budget_bytes=64 * 1024, device="cuda")
    pool.register(_descriptor(dtype=torch.float16, cap_tokens=16, slots=1))
    pool.acquire("tiny-ar")

    text = render_ar_cache_metrics(pool.stats())

    assert "hse_ar_cache_enabled 1" in text
    assert "hse_ar_cache_slots_active 1" in text
    assert "hse_ar_cache_pool_bytes 0" not in text


# ---------------------------------------------------------------------------
# Engine surface
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_engine_stats_without_a_pool_are_disabled_zeros():
    engine = _engine_shell()

    stats = engine.get_ar_cache_stats()

    assert stats["enabled"] is False
    for key in ("pool_bytes", "allocated_bytes", "slots", "slots_active", "slots_free", "grants_total", "refusals_total"):
        assert stats[key] == 0
    assert stats["arenas"] == {}
    assert engine.get_ar_cache_capacity()["enabled"] is False


@pytest.mark.unit
def test_engine_stats_report_a_live_pool():
    engine = _engine_shell()
    engine.ar_cache_budget_bytes = 64 * 1024
    engine.ar_cache_default_cap_tokens = 16
    engine.ar_cache_pool = ARCachePool(budget_bytes=64 * 1024, device="cpu", default_cap_tokens=16)
    engine.ar_cache_pool.register(_descriptor(cap_tokens=None, slots=2))
    slot = engine.ar_cache_pool.acquire("tiny-ar")

    stats = engine.get_ar_cache_stats()

    assert stats["enabled"] is True
    assert stats["budget_bytes"] == 64 * 1024
    assert stats["slots"] == 2
    assert stats["slots_active"] == 1
    assert stats["pool_bytes"] > 0
    assert stats["arenas"]["tiny-ar"]["active_rows"] == 1
    assert engine.get_ar_cache_capacity("tiny-ar")["slots_free"] == 1

    engine.ar_cache_pool.release(slot)


@pytest.mark.unit
def test_check_health_includes_the_ar_cache_block():
    engine = _engine_shell()
    engine.ar_cache_pool = ARCachePool(budget_bytes=64 * 1024, device="cpu")
    engine.ar_cache_pool.register(_descriptor(cap_tokens=16, slots=1))

    payload = engine.check_health()

    assert payload["healthy"] is True
    assert payload["ar_cache"]["enabled"] is True
    assert payload["ar_cache"]["slots"] == 1
    # The footprint is also visible in the VRAM story (Phase 10-3 input).
    assert payload["vram"]["ar_cache"]["pool_bytes"] == payload["ar_cache"]["pool_bytes"]


@pytest.mark.unit
def test_runner_and_executor_receive_the_pool():
    from harmonyspeech.executor.executor_base import ExecutorBase
    from harmonyspeech.task_handler.cpu_model_runner import CPUModelRunner

    device_config = DeviceConfig("cpu")
    model_config = ModelConfig(
        name="tiny",
        model="tiny",
        model_type="MeloTTSSynthesizer",
        max_batch_size=1,
        device_config=device_config,
        dtype="float32",
    )
    pool = ARCachePool(budget_bytes=1024, device="cpu")

    class _Executor(ExecutorBase):
        def _init_executor(self) -> None:
            return None

        def execute_model(self, requests_to_batch):
            return []

        def check_health(self) -> None:
            return None

    executor = _Executor(model_config=model_config, ar_cache_pool=pool)
    runner = CPUModelRunner(model_config, device_config, is_driver_worker=True, ar_cache_pool=pool)

    assert executor.ar_cache_pool is pool
    assert runner.ar_cache_pool is pool


# ---------------------------------------------------------------------------
# Config / CLI plumbing
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_engine_config_defaults():
    config = EngineConfig(model_configs=[])

    assert config.ar_cache_enabled is True
    assert config.ar_cache_budget_bytes == 1073741824
    assert config.ar_cache_default_cap_tokens == 768
    assert config.to_dict()["ar_cache_budget_bytes"] == 1073741824


@pytest.mark.unit
def test_yaml_ar_inference_block_is_loaded(tmp_path):
    yaml_file = tmp_path / "config.yml"
    yaml_file.write_text(
        """
model_configs: []
ar_inference:
  kv_cache_budget_bytes: 2048
  default_cap_tokens: 64
  enabled: false
"""
    )

    config = EngineConfig.load_config_from_yaml(str(yaml_file))

    assert config.ar_cache_budget_bytes == 2048
    assert config.ar_cache_default_cap_tokens == 64
    assert config.ar_cache_enabled is False


@pytest.mark.unit
def test_flat_ar_cache_keys_are_accepted_as_aliases(tmp_path):
    yaml_file = tmp_path / "config.yml"
    yaml_file.write_text(
        """
model_configs: []
ar_cache_budget_bytes: 4096
ar_cache_default_cap_tokens: 32
"""
    )

    config = EngineConfig.load_config_from_yaml(str(yaml_file))

    assert config.ar_cache_budget_bytes == 4096
    assert config.ar_cache_default_cap_tokens == 32


@pytest.mark.unit
@pytest.mark.parametrize("field", ["ar_cache_budget_bytes", "ar_cache_default_cap_tokens"])
@pytest.mark.parametrize("value", [0, -1])
def test_engine_config_rejects_non_positive_ar_values(field, value):
    with pytest.raises(ValueError, match=field):
        EngineConfig(model_configs=[], **{field: value})


@pytest.mark.unit
def test_cli_overrides_budget_and_can_disable_the_pool():
    parser = argparse.ArgumentParser()
    EngineArgs.add_cli_args(parser)
    args = parser.parse_args(["--ar-cache-budget-bytes", "8192", "--no-ar-cache"])
    engine_args = EngineArgs.from_cli_args(args)

    config = engine_args.create_engine_config(EngineConfig(model_configs=[]))

    assert config.ar_cache_budget_bytes == 8192
    assert config.ar_cache_enabled is False


@pytest.mark.unit
def test_cli_without_ar_flags_keeps_the_config_values():
    parser = argparse.ArgumentParser()
    EngineArgs.add_cli_args(parser)
    engine_args = EngineArgs.from_cli_args(parser.parse_args([]))

    config = engine_args.create_engine_config(EngineConfig(model_configs=[]))

    assert config.ar_cache_budget_bytes == 1073741824
    assert config.ar_cache_enabled is True


@pytest.mark.unit
def test_engine_boot_creates_the_pool_with_the_configured_budget():
    engine = HarmonySpeechEngine(
        model_configs=[],
        log_stats=False,
        ar_cache_budget_bytes=4096,
        ar_cache_default_cap_tokens=16,
    )

    assert engine.ar_cache_pool is not None
    assert engine.ar_cache_pool.budget_bytes == 4096
    assert engine.ar_cache_pool.default_cap_tokens == 16
    assert engine.get_ar_cache_stats()["enabled"] is True


@pytest.mark.unit
def test_arena_table_logging_handles_registered_arenas():
    engine = _engine_shell()
    engine.ar_cache_pool = ARCachePool(budget_bytes=64 * 1024, device="cpu")
    arena = engine.ar_cache_pool.register(_descriptor(cap_tokens=16, slots=2))

    rows = engine.ar_cache_pool.arena_table()
    assert rows[0]["model_key"] == "tiny-ar"
    assert rows[0]["slots"] == 2
    assert rows[0]["arena_bytes"] == arena.slab_bytes * 2

    # The startup table formatter must accept a populated table (and a disabled
    # pool) without raising.
    engine._log_ar_cache_arena_table()
    engine.ar_cache_pool = None
    engine._log_ar_cache_arena_table()


@pytest.mark.unit
def test_engine_boot_with_no_ar_cache_leaves_the_pool_none():
    engine = HarmonySpeechEngine(model_configs=[], log_stats=False, ar_cache_enabled=False)

    assert engine.ar_cache_pool is None
    assert engine.get_ar_cache_stats()["enabled"] is False


# ---------------------------------------------------------------------------
# Endpoint wiring
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_metrics_endpoint_renders_ar_cache_series(monkeypatch):
    from harmonyspeech.endpoints.openai import api_server

    engine = _engine_shell()
    engine.ar_cache_pool = ARCachePool(budget_bytes=64 * 1024, device="cpu")
    engine.ar_cache_pool.register(_descriptor(cap_tokens=16, slots=2))
    monkeypatch.setattr(api_server, "openai_serving_tts", SimpleNamespace(engine=engine))
    monkeypatch.setattr(api_server, "openai_serving_stt", None)
    monkeypatch.setattr(api_server, "openai_serving_vc", None)
    monkeypatch.setattr(api_server, "openai_serving_embedding", None)

    response = asyncio.run(api_server.metrics())
    body = response.body.decode()

    assert "hse_ar_cache_enabled 1" in body
    assert "hse_ar_cache_slots 2" in body
    for name in SERIES:
        assert name in body


@pytest.mark.unit
def test_metrics_endpoint_without_an_engine_still_renders_zeroed_ar_series(monkeypatch):
    from harmonyspeech.endpoints.openai import api_server

    for attribute in ("openai_serving_tts", "openai_serving_stt", "openai_serving_vc", "openai_serving_embedding"):
        monkeypatch.setattr(api_server, attribute, None)

    response = asyncio.run(api_server.metrics())
    body = response.body.decode()

    assert "hse_ar_cache_enabled 0" in body
    assert "hse_ar_cache_pool_bytes 0" in body


@pytest.mark.unit
def test_health_endpoint_includes_the_ar_cache_block(monkeypatch):
    from harmonyspeech.endpoints.openai import api_server

    engine = _engine_shell()
    engine.ar_cache_pool = ARCachePool(budget_bytes=64 * 1024, device="cpu")
    engine.ar_cache_pool.register(_descriptor(cap_tokens=16, slots=1))
    tts_engine = SimpleNamespace(check_health=AsyncMock(return_value=engine.check_health()))
    healthy_engine = SimpleNamespace(check_health=AsyncMock(return_value=None))
    monkeypatch.setattr(api_server, "openai_serving_tts", SimpleNamespace(engine=tts_engine))
    monkeypatch.setattr(api_server, "openai_serving_stt", SimpleNamespace(engine=healthy_engine))
    monkeypatch.setattr(api_server, "openai_serving_vc", SimpleNamespace(engine=healthy_engine))
    monkeypatch.setattr(api_server, "openai_serving_embedding", SimpleNamespace(engine=healthy_engine))
    monkeypatch.setattr(api_server, "openai_serving_vad", SimpleNamespace(engine=healthy_engine))

    response = asyncio.run(api_server.health())

    payload = json.loads(response.body)
    assert payload["text_to_speech"] == "healthy"
    assert payload["ar_cache"]["enabled"] is True
    assert payload["ar_cache"]["slots"] == 1
    assert "vram" in payload

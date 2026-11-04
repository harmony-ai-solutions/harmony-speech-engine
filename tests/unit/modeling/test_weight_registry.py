"""Unit tests for the ref-counted shared-weight registry (Phase 4-3)."""

import threading

import pytest
import torch
import torch.nn as nn

from harmonyspeech.modeling.weight_registry import WeightRegistry, module_nbytes

KEY = ("ResembleAI/chatterbox", None, "s3gen", torch.float32)


def _module(dtype=torch.float32) -> nn.Module:
    return nn.Sequential(nn.Linear(4, 3, dtype=dtype), nn.ReLU())


class _Loader:
    def __init__(self, dtype=torch.float32):
        self.calls = 0
        self.dtype = dtype

    def __call__(self):
        self.calls += 1
        return _module(self.dtype)


def test_acquire_same_key_returns_identical_object_and_loads_once():
    registry = WeightRegistry()
    loader = _Loader()

    first = registry.acquire(KEY, loader)
    second = registry.acquire(KEY, loader)

    assert first is second
    assert loader.calls == 1
    assert registry.refcount(KEY) == 2
    assert len(registry) == 1


def test_acquire_marks_module_readonly_and_eval():
    registry = WeightRegistry()
    module = registry.acquire(KEY, _Loader())

    assert module.training is False
    assert all(not param.requires_grad for param in module.parameters())
    WeightRegistry.assert_readonly(module)


def test_assert_readonly_rejects_trainable_module():
    module = _module()
    module.train()
    with pytest.raises(AssertionError):
        WeightRegistry.assert_readonly(module)


def test_distinct_keys_get_distinct_modules():
    registry = WeightRegistry()
    base = registry.acquire(KEY, _Loader())
    other_dtype = registry.acquire((KEY[0], None, "s3gen", torch.float16), _Loader(torch.float16))
    other_component = registry.acquire((KEY[0], None, "ve", torch.float32), _Loader())
    other_repo = registry.acquire(("other/repo", None, "s3gen", torch.float32), _Loader())

    assert len({id(base), id(other_dtype), id(other_component), id(other_repo)}) == 4
    assert len(registry) == 4


def test_acquire_rejects_wrong_dtype_for_key():
    registry = WeightRegistry()

    def wrong_loader():
        return _module(torch.float32)

    key = (KEY[0], None, "s3gen", torch.float16)
    with pytest.raises(AssertionError, match="dtype"):
        registry.acquire(key, wrong_loader)
    # The failed acquire stores nothing (the consistency check runs before the
    # module enters the store).
    assert len(registry) == 0


def test_acquire_rejects_non_module():
    registry = WeightRegistry()
    with pytest.raises(TypeError):
        registry.acquire(KEY, lambda: object())


def test_release_keeps_module_until_last_refcount():
    registry = WeightRegistry()
    loader = _Loader()
    registry.acquire(KEY, loader)
    registry.acquire(KEY, loader)

    assert registry.release(KEY) is False
    assert len(registry) == 1
    assert registry.refcount(KEY) == 1

    assert registry.release(KEY) is True
    assert len(registry) == 0
    assert registry.refcount(KEY) == 0

    # Re-acquire reloads.
    reacquired = registry.acquire(KEY, loader)
    assert loader.calls == 2
    assert reacquired is not None


def test_release_unknown_key_is_noop():
    registry = WeightRegistry()
    assert registry.release(KEY) is False


def test_stats_counts_entries_and_bytes():
    registry = WeightRegistry()
    registry.acquire(KEY, _Loader())
    registry.acquire(KEY, _Loader())
    ve_module = registry.acquire((KEY[0], None, "ve", torch.float32), _Loader())

    stats = registry.stats()

    assert stats["entries"] == 2
    expected = module_nbytes(registry.get(KEY)) + module_nbytes(ve_module)
    assert stats["total_bytes"] == expected
    assert stats["total_bytes"] > 0
    components = stats["components"]
    assert len(components) == 2
    assert all("bytes" in detail and "refs" in detail for detail in components.values())
    assert any(detail["refs"] == 2 for detail in components.values())


def test_get_and_contains():
    registry = WeightRegistry()
    assert registry.get(KEY) is None
    assert KEY not in registry

    module = registry.acquire(KEY, _Loader())
    assert registry.get(KEY) is module
    assert KEY in registry
    assert registry.keys() == [KEY]


def test_threaded_acquire_loads_once():
    """Concurrent boot-time acquires must not race a double load."""
    registry = WeightRegistry()
    calls = []

    def loader():
        calls.append(1)
        return _module()

    barrier = threading.Barrier(8)
    results = []

    def worker():
        barrier.wait()
        results.append(registry.acquire(KEY, loader))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(calls) == 1
    assert len({id(module) for module in results}) == 1
    assert registry.refcount(KEY) == 8

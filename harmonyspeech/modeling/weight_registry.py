"""Ref-counted registry for shared model backbone modules (Phase 4-3).

Multiple executor configs can load *the same* backbone weights (Chatterbox's
S3Gen / VoiceEncoder, and in principle any future shared component). The engine
owns exactly one :class:`WeightRegistry` and threads it through executor →
worker → model runner → loader. A module is loaded once per
``(repo, revision, component, dtype)`` key and handed to every acquirer.

Safety invariant (see ``model_runner_base`` / Phase 0-2)
---------------------------------------------------------
Shared modules are **read-only**: they are ``eval()``-ed and have
``requires_grad=False`` set at load, and acquirers must never re-cast or move
them. All per-request state (Chatterbox ``conds`` / ``ref_dict``) lives on the
per-executor wrapper object, never on a shared backbone.

Modules are stored already on their target device and already cast to the key's
dtype; :meth:`acquire` asserts this so an accidental re-cast is caught loudly
instead of silently duplicating weights.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable

import torch
import torch.nn as nn
from loguru import logger

Key = tuple  # (repo, revision, component, dtype)


def _key_text(key: Key) -> str:
    repo, revision, component, dtype = key
    dtype_text = getattr(dtype, "__str__", lambda: str(dtype))()
    return f"{repo}@{revision or 'main'}:{component}:{dtype_text}"


def module_nbytes(module: nn.Module) -> int:
    """Return the resident size of ``module``'s parameters and buffers, in bytes."""
    total = 0
    for tensor in list(module.parameters()) + list(module.buffers()):
        total += tensor.numel() * tensor.element_size()
    return total


def _path_nbytes(path: str) -> int:
    """Sum the size of a file or every file under a directory."""
    try:
        if os.path.isfile(path):
            return int(os.path.getsize(path))
    except OSError:
        return 0
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += int(os.path.getsize(os.path.join(root, name)))
            except OSError:
                continue
    return total


def model_nbytes(model) -> int:
    """Resident weight bytes of a model object, shared submodules counted once.

    Chatterbox role wrappers are plain objects (not ``nn.Module``) that hold
    ``t3`` / ``s3gen`` / ``ve`` as attributes — and those submodules may be
    *shared* between roles. Walking the attributes with identity de-duplication
    keeps the accounting honest (and keeps shared bytes out of per-model totals
    only when they are genuinely the same object).

    Native models (CTranslate2 Whisper, Phase 10-3) have no torch tensors to
    walk. Their loader attaches the resolved checkpoint directory as
    ``model_path``; summing the files on disk is the checkpoint size the plan
    asks ``hse_model_weights_bytes`` to report. A model without a resolvable
    local path honestly reports ``0`` rather than a guess.
    """
    if model is None:
        return 0
    if isinstance(model, nn.Module):
        return module_nbytes(model)

    model_path = getattr(model, "model_path", None)
    if isinstance(model_path, str) and model_path:
        return _path_nbytes(model_path)

    seen: set[int] = set()
    total = 0
    for value in vars(model).values():
        if not isinstance(value, nn.Module) or id(value) in seen:
            continue
        seen.add(id(value))
        total += module_nbytes(value)
    return total


class WeightRegistry:
    """Ref-counted shared module store; one instance per engine.

    Keys are ``(repo, revision, component, dtype)`` tuples. The registry keeps
    the first loaded module object and returns that same object to every
    acquirer until the last :meth:`release`.
    """

    def __init__(self) -> None:
        self._store: dict[Key, nn.Module] = {}
        self._refs: dict[Key, int] = {}
        self._lock = threading.Lock()

    # -- loading ---------------------------------------------------------
    def acquire(self, key: Key, loader_fn: Callable[[], nn.Module]) -> nn.Module:
        """Return the shared module for ``key``, loading it on first acquire.

        ``loader_fn`` must return a module already moved to its target device
        and cast to the dtype recorded in ``key``. It is called exactly once
        per key (under the registry lock, so concurrent boot-time callers
        cannot race a double load).
        """
        with self._lock:
            module = self._store.get(key)
            if module is None:
                logger.debug(f"WeightRegistry: loading shared component {_key_text(key)}")
                module = loader_fn()
                if not isinstance(module, nn.Module):
                    raise TypeError(
                        f"WeightRegistry loader for {_key_text(key)} returned "
                        f"{type(module).__name__}, expected torch.nn.Module"
                    )
                # Shared weights are inference-only by construction.
                module.eval()
                module.requires_grad_(False)
                self._assert_consistent(key, module)
                self._store[key] = module
                self._refs[key] = 0
                logger.info(
                    f"Loaded shared backbone {_key_text(key)} "
                    f"({module_nbytes(module) / float(2**20):.1f} MiB)"
                )
            else:
                self._assert_consistent(key, module)
            self._refs[key] += 1
            return module

    def get(self, key: Key) -> nn.Module | None:
        """Return the module currently stored for ``key`` (no ref-count change)."""
        with self._lock:
            return self._store.get(key)

    def refcount(self, key: Key) -> int:
        with self._lock:
            return self._refs.get(key, 0)

    def release(self, key: Key) -> bool:
        """Decrement ``key``'s ref count; drop the module when it reaches zero.

        Returns ``True`` when the module was dropped. Production has no unload
        path yet (models are eagerly resident), so this is exercised by tests
        and the Phase 6 governor.
        """
        with self._lock:
            if key not in self._store:
                return False
            self._refs[key] = max(0, self._refs.get(key, 0) - 1)
            if self._refs[key] > 0:
                return False
            del self._store[key]
            del self._refs[key]
            logger.debug(f"WeightRegistry: released shared component {_key_text(key)}")
            return True

    def clear(self) -> None:
        """Drop every stored module (tests / shutdown)."""
        with self._lock:
            self._store.clear()
            self._refs.clear()

    # -- introspection ---------------------------------------------------
    def keys(self) -> list[Key]:
        with self._lock:
            return list(self._store.keys())

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    def __contains__(self, key: Key) -> bool:
        with self._lock:
            return key in self._store

    def stats(self) -> dict:
        """Return entry count, total bytes and per-key detail.

        The shape is a superset of the ``dict[str, int]`` contract from the
        plan: ``entries`` / ``total_bytes`` are the numbers the VRAM metrics
        consume, ``components`` maps a readable key to ``{"refs", "bytes"}``.
        """
        with self._lock:
            components: dict[str, dict[str, int]] = {}
            total_bytes = 0
            for key, module in self._store.items():
                size = module_nbytes(module)
                total_bytes += size
                components[_key_text(key)] = {
                    "refs": self._refs.get(key, 0),
                    "bytes": size,
                }
            return {
                "entries": len(self._store),
                "total_bytes": total_bytes,
                "components": components,
            }

    # -- invariants ------------------------------------------------------
    @staticmethod
    def assert_readonly(module: nn.Module) -> None:
        """Assert a shared module is in inference-only state.

        Documented no-op hook from the plan, now enforced: every shared module
        must be ``eval()``-ed with no trainable parameters.
        """
        if module.training:
            raise AssertionError("Shared backbone module must be in eval mode")
        for name, param in module.named_parameters(recurse=True):
            if param.requires_grad:
                raise AssertionError(f"Shared backbone parameter '{name}' has requires_grad=True")

    @staticmethod
    def _assert_consistent(key: Key, module: nn.Module) -> None:
        """Debug check: the stored module matches the key's dtype (first param)."""
        expected = key[3]
        if not isinstance(expected, torch.dtype):
            return
        for param in module.parameters():
            if param.dtype != expected:
                raise AssertionError(
                    f"Shared module for {_key_text(key)} has parameter dtype "
                    f"{param.dtype}, expected {expected}; acquirers must not re-cast shared weights"
                )
            return


__all__ = ["WeightRegistry", "module_nbytes"]

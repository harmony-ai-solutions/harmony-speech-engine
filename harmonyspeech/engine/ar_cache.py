"""Generic autoregressive KV-cache pool (Phase 9-2).

One device-wide, size-capped pool of static per-slot KV slabs, owned by the
engine and shared by every AR family that opts in. The pool is **machinery
only**: a :class:`ARCacheDescriptor` describes one family's layout, an
:class:`ARCacheArena` owns that family's tensor and its free list, and an
:class:`ARCacheSlot` grants exclusive use of one slot to one in-flight request
for the duration of that request. No cache content is ever shared between
requests (owner direction 2026-11-07).

Layout (study #14 D2)
---------------------
One arena tensor per family, shape ``[L, 2, B, cap, H, D]`` — layers × {K, V} ×
slots × capacity (tokens) × kv-heads × head-dim — allocated as a single fp16
(or configured-dtype) tensor on the pool's device. PyTorch has no paged-attention
kernel, and gather-based page reads cost more than the attention itself at these
sizes, so the layout is static slabs. Fixed strides give a **zero-copy**
``[B, cap, H, D]`` view per layer for one ``scaled_dot_product_attention`` call::

    arena.key_view(layer)      # [slots, cap, H, D]
    slot.key_view(layer)       # [1, cap, H, D]
    slot.write(layer, start, keys, values)   # copy a [T, H, D] block in

Slot lifecycle and the isolation invariant
------------------------------------------
A slot belongs to **exactly one in-flight request** between :meth:`ARCachePool.acquire`
and :meth:`ARCachePool.release`. A released slot keeps whatever bytes the previous
request left behind; the next request reuses it *without* a clearing pass. This is
safe because of the decode contract every consumer must honour:

* decode writes position ``k`` (its K/V for the token being generated) **before**
  any attention reads positions ``<= k``, and
* attention is masked/sliced to the sequence's current length, so stale KV
  **beyond the write position is never read**.

The invariant is documented here and pinned by
``tests/unit/engine/test_ar_cache.py::test_poisoned_slot_decode_matches_clean_slot``,
which pre-fills a slot with NaN and asserts a decode over the written prefix is
bit-identical to the clean-slot run.

Slots are not admission width
-----------------------------
The slot count comes from pool availability plus the descriptor (an explicit
``slots`` or "as many as the remaining budget fits"). The scheduler's
``max_batch_size`` is a separate capacity signal and is never derived from the
pool. An arena that cannot grant a slot returns ``None`` — the caller takes its
sequential fallback — so the pool can never cause an out-of-memory failure.

Per-device arenas (Phase 11-0)
------------------------------
The pool owns **one budget per device**. An arena is allocated on the resolved
device of the model that registers it
(:func:`~harmonyspeech.common.device.resolve_device`), so ``cuda`` and ``cuda:0``
are the same device on a single-device host while an explicit ``cuda:1`` model
gets its arena on ``cuda:1``. Each device's arenas are validated against that
device's remaining budget, so a second device does not consume the first
device's headroom. ``remaining_bytes`` keeps its historical meaning for the
pool's **default** device (the single-device case);
:meth:`ARCachePool.remaining_bytes_for` asks about a specific device.

CPU fallback
------------
``device=None`` resolves to CUDA when available and to CPU otherwise. A CPU pool
is correct but slower; it lets the whole unit suite run without CUDA. Slabs are
allocated with the descriptor's dtype (``float16`` on CUDA; tests may use
``float32`` on CPU).

Threading
---------
The pool is engine-owned and threaded down executor → worker → model runner
exactly like the weight registry and the embedding cache. Registration and
acquire/release are guarded by one lock; the tensors themselves are only touched
by the request that owns the slot.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass

import torch

from harmonyspeech.common.device import resolve_device

__all__ = [
    "ARCacheArena",
    "ARCacheBudgetError",
    "ARCacheDescriptor",
    "ARCachePool",
    "ARCacheSlot",
    "DEFAULT_AR_CACHE_BUDGET_BYTES",
    "DEFAULT_AR_CACHE_CAP_TOKENS",
]

#: Device-wide pool budget, owner-approved 2026-11-07 as the initial value
#: (tune on data the way Phase 8-6 tuned its "initial 8").
DEFAULT_AR_CACHE_BUDGET_BYTES = 1073741824

#: Default token capacity per slot: 768 tokens ≈ 30 s of speech at 25 Hz
#: (2 mel frames/token, derived in study #14; re-derive from the vendored token
#: rate when the first consumer lands).
DEFAULT_AR_CACHE_CAP_TOKENS = 768

#: Two tensors per layer: keys and values.
_KINDS = 2


class ARCacheBudgetError(ValueError):
    """Raised when a descriptor cannot fit in the pool's remaining budget.

    This is a boot-time configuration error, not a runtime refusal: the engine
    refuses to start with an arena it could never allocate. Runtime exhaustion
    (every slot of an arena in use) is *not* an error — :meth:`ARCachePool.acquire`
    returns ``None`` and the caller falls back.
    """


@dataclass(frozen=True)
class ARCacheDescriptor:
    """Layout of one AR family's KV cache.

    Parameters
    ----------
    model_key:
        Stable family/executor identity used to register and acquire. One
        descriptor per key per pool.
    layers:
        Number of decoder layers that hold KV state.
    kv_heads:
        Key/value head count (GQA/MQA reduce this below the attention-head
        count; Llama-family T3 uses ``num_key_value_heads``).
    head_dim:
        Per-head feature width.
    dtype:
        Slab dtype. ``float16`` is the serving choice on CUDA; ``float32`` is
        convenient for CPU tests.
    cap_tokens:
        Token capacity per slot. ``None`` means "use the pool's
        ``default_cap_tokens``" (config: ``ar_inference.default_cap_tokens``).
    slots:
        Slot count. ``None`` means "as many whole slabs as the remaining budget
        fits" (at least one; a slab that cannot fit is a boot error). This is a
        pool-availability number, never an admission width.
    """

    model_key: str
    layers: int
    kv_heads: int
    head_dim: int
    dtype: torch.dtype = torch.float16
    cap_tokens: int | None = None
    slots: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model_key, str) or not self.model_key:
            raise ValueError(f"model_key must be a non-empty string, got {self.model_key!r}")
        for name in ("layers", "kv_heads", "head_dim"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive int, got {value!r}")
        if not isinstance(self.dtype, torch.dtype):
            raise TypeError(f"dtype must be a torch.dtype, got {type(self.dtype).__name__}")
        if self.cap_tokens is not None and (
            isinstance(self.cap_tokens, bool)
            or not isinstance(self.cap_tokens, int)
            or self.cap_tokens < 1
        ):
            raise ValueError(f"cap_tokens must be a positive int or None, got {self.cap_tokens!r}")
        if self.slots is not None and (
            isinstance(self.slots, bool) or not isinstance(self.slots, int) or self.slots < 1
        ):
            raise ValueError(f"slots must be a positive int or None, got {self.slots!r}")

    @property
    def element_size(self) -> int:
        """Bytes per scalar for :attr:`dtype`."""
        return int(torch.empty((), dtype=self.dtype).element_size())

    def slab_bytes(self, cap_tokens: int | None = None) -> int:
        """Bytes of one slot: ``layers × 2 × cap × kv_heads × head_dim × itemsize``."""
        cap = self.cap_tokens if cap_tokens is None else int(cap_tokens)
        if cap is None:
            raise ValueError(
                f"descriptor '{self.model_key}' has no cap_tokens; pass one or register with a pool default"
            )
        return self.layers * _KINDS * cap * self.kv_heads * self.head_dim * self.element_size

    def arena_bytes(self, cap_tokens: int | None = None, slots: int | None = None) -> int:
        """Bytes of a whole arena: ``slab_bytes × slots``."""
        count = self.slots if slots is None else int(slots)
        if count is None:
            raise ValueError(f"descriptor '{self.model_key}' has no slot count; pass one")
        return self.slab_bytes(cap_tokens) * count


class ARCacheArena:
    """One family's slab tensor plus its free list.

    Created by :meth:`ARCachePool.register`; callers normally hold
    :class:`ARCacheSlot` handles instead of touching the arena directly. The
    arena-level views exist for consumers that batch several contiguous slots
    into one attention call.
    """

    def __init__(
        self,
        descriptor: ARCacheDescriptor,
        *,
        cap_tokens: int,
        slots: int,
        device: torch.device | str,
        pool: ARCachePool,
    ) -> None:
        self.descriptor = descriptor
        self.cap_tokens = int(cap_tokens)
        self.slots = int(slots)
        self.device = torch.device(device)
        self._pool = pool
        # One allocation per family: [L, 2, B, cap, H, D]. Zero-filled so a
        # fresh arena never starts from uninitialised memory (released slots may
        # keep stale bytes — see the module docstring's isolation invariant).
        self.tensor = torch.zeros(
            descriptor.layers,
            _KINDS,
            self.slots,
            self.cap_tokens,
            descriptor.kv_heads,
            descriptor.head_dim,
            dtype=descriptor.dtype,
            device=self.device,
        )
        self._free: deque[int] = deque(range(self.slots))
        self._active: set[int] = set()

    # -- geometry --------------------------------------------------------
    @property
    def model_key(self) -> str:
        return self.descriptor.model_key

    @property
    def layers(self) -> int:
        return self.descriptor.layers

    @property
    def kv_heads(self) -> int:
        return self.descriptor.kv_heads

    @property
    def head_dim(self) -> int:
        return self.descriptor.head_dim

    @property
    def dtype(self) -> torch.dtype:
        return self.descriptor.dtype

    @property
    def slab_bytes(self) -> int:
        return self.descriptor.slab_bytes(self.cap_tokens)

    @property
    def arena_bytes(self) -> int:
        return self.slab_bytes * self.slots

    @property
    def free_count(self) -> int:
        return len(self._free)

    @property
    def active_count(self) -> int:
        return len(self._active)

    # -- zero-copy views -------------------------------------------------
    def _check_layer(self, layer: int) -> int:
        layer = int(layer)
        if layer < 0 or layer >= self.layers:
            raise IndexError(f"layer {layer} out of range for '{self.model_key}' ({self.layers} layers)")
        return layer

    def key_view(self, layer: int, slots=None) -> torch.Tensor:
        """Zero-copy ``[B, cap, H, D]`` keys of ``layer`` (``slots`` may be a slice)."""
        if slots is None:
            slots = slice(None)
        return self.tensor[self._check_layer(layer), 0, slots]

    def value_view(self, layer: int, slots=None) -> torch.Tensor:
        """Zero-copy ``[B, cap, H, D]`` values of ``layer`` (``slots`` may be a slice)."""
        if slots is None:
            slots = slice(None)
        return self.tensor[self._check_layer(layer), 1, slots]

    # -- accounting ------------------------------------------------------
    def capacity(self) -> dict:
        """Slot/byte accounting for capacity-based admission (Phase 9-4 input).

        ``active_rows`` is the number of rows currently decoding on this arena —
        the value a capacity-aware scheduler may consult. It is deliberately
        reported next to, never merged with, the scheduler's ``max_batch_size``.
        """
        return {
            "model_key": self.model_key,
            "cap_tokens": self.cap_tokens,
            "slots": self.slots,
            "slots_active": self.active_count,
            "slots_free": self.free_count,
            "active_rows": self.active_count,
            "slab_bytes": self.slab_bytes,
            "arena_bytes": self.arena_bytes,
            "layers": self.layers,
            "kv_heads": self.kv_heads,
            "head_dim": self.head_dim,
            "dtype": str(self.dtype).replace("torch.", ""),
            "device": str(self.device),
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"ARCacheArena(model_key={self.model_key!r}, layers={self.layers}, "
            f"kv_heads={self.kv_heads}, head_dim={self.head_dim}, cap={self.cap_tokens}, "
            f"slots={self.slots}, active={self.active_count}, device={self.device})"
        )


class ARCacheSlot:
    """Exclusive grant of one arena slot to one in-flight request.

    Obtained from :meth:`ARCachePool.acquire`; call :meth:`release` (or
    :meth:`ARCachePool.release`) when the request finishes. The slot is the only
    handle that should be passed around inside a request: it pins the arena and
    the index so a double release or a foreign release is detected.
    """

    __slots__ = ("arena", "slot_index", "_released")

    def __init__(self, arena: ARCacheArena, slot_index: int) -> None:
        self.arena = arena
        self.slot_index = int(slot_index)
        self._released = False

    # -- geometry --------------------------------------------------------
    @property
    def model_key(self) -> str:
        return self.arena.model_key

    @property
    def cap_tokens(self) -> int:
        return self.arena.cap_tokens

    @property
    def layers(self) -> int:
        return self.arena.layers

    @property
    def kv_heads(self) -> int:
        return self.arena.kv_heads

    @property
    def head_dim(self) -> int:
        return self.arena.head_dim

    @property
    def dtype(self) -> torch.dtype:
        return self.arena.dtype

    @property
    def device(self) -> torch.device:
        return self.arena.device

    # -- zero-copy views -------------------------------------------------
    def key_view(self, layer: int) -> torch.Tensor:
        """Zero-copy ``[1, cap, H, D]`` keys of this slot's ``layer``."""
        return self.arena.key_view(layer, slice(self.slot_index, self.slot_index + 1))

    def value_view(self, layer: int) -> torch.Tensor:
        """Zero-copy ``[1, cap, H, D]`` values of this slot's ``layer``."""
        return self.arena.value_view(layer, slice(self.slot_index, self.slot_index + 1))

    # -- writes ----------------------------------------------------------
    def write(self, layer: int, start: int, keys: torch.Tensor, values: torch.Tensor) -> None:
        """Copy a ``[T, H, D]`` (or ``[H, D]``) K/V block at token ``start``.

        This is the write half of the isolation invariant: a decode step writes
        its position before any attention reads ``<=`` it. Copying (rather than
        assigning the caller's tensors) keeps the slab layout fixed and lets the
        caller free its temporaries.
        """
        layer = self.arena._check_layer(layer)
        keys_block = self._as_block(keys, "keys")
        values_block = self._as_block(values, "values")
        if keys_block.shape != values_block.shape:
            raise ValueError(f"keys {tuple(keys_block.shape)} and values {tuple(values_block.shape)} differ")
        start = int(start)
        tokens = keys_block.shape[0]
        if start < 0 or start + tokens > self.cap_tokens:
            raise ValueError(
                f"write [{start}, {start + tokens}) exceeds cap_tokens={self.cap_tokens} for '{self.model_key}'"
            )
        index = slice(self.slot_index, self.slot_index + 1)
        self.arena.tensor[layer, 0, index, start : start + tokens] = keys_block.unsqueeze(0)
        self.arena.tensor[layer, 1, index, start : start + tokens] = values_block.unsqueeze(0)

    def _as_block(self, tensor: torch.Tensor, name: str) -> torch.Tensor:
        if not torch.is_tensor(tensor):
            raise TypeError(f"{name} must be a tensor, got {type(tensor).__name__}")
        block = tensor
        if block.dim() == 2:
            block = block.unsqueeze(0)
        if block.dim() != 3:
            raise ValueError(f"{name} must be [T, H, D] or [H, D], got {tuple(block.shape)}")
        if block.shape[1:] != (self.kv_heads, self.head_dim):
            raise ValueError(
                f"{name} trailing shape {tuple(block.shape[1:])} != "
                f"(kv_heads={self.kv_heads}, head_dim={self.head_dim})"
            )
        return block

    # -- lifecycle -------------------------------------------------------
    @property
    def released(self) -> bool:
        return self._released

    def release(self) -> None:
        """Return the slot to the pool; a second call raises (double release is a bug)."""
        self.arena._pool.release(self)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        state = "released" if self._released else "active"
        return f"ARCacheSlot({self.model_key!r}, index={self.slot_index}, {state})"


class ARCachePool:
    """Device-wide AR KV-cache pool; one instance per engine.

    Registration allocates one arena per family and validates it against the
    remaining budget (a descriptor that cannot fit is a boot error). Acquire and
    release move slots through a FIFO free list: a released slot goes to the back
    of the queue, so reuse order is the release order — deterministic and easy to
    reason about in tests and logs.

    ``enabled=False`` builds an inert pool: :meth:`register` returns ``None``,
    :meth:`acquire` returns ``None`` and :meth:`stats` reports zeros. The engine
    normally leaves ``ar_cache_pool`` as ``None`` when ``--no-ar-cache`` is set
    (the embedding-cache convention); the inert mode exists so callers that
    always hold a pool object can stay branch-free.
    """

    def __init__(
        self,
        budget_bytes: int = DEFAULT_AR_CACHE_BUDGET_BYTES,
        *,
        device: torch.device | str | None = None,
        default_cap_tokens: int = DEFAULT_AR_CACHE_CAP_TOKENS,
        enabled: bool = True,
    ) -> None:
        self.enabled = bool(enabled)
        self.budget_bytes = int(budget_bytes)
        if self.enabled and self.budget_bytes <= 0:
            raise ValueError(f"budget_bytes must be positive, got {budget_bytes!r}")
        self.default_cap_tokens = int(default_cap_tokens)
        if self.enabled and self.default_cap_tokens <= 0:
            raise ValueError(f"default_cap_tokens must be positive, got {default_cap_tokens!r}")
        # Default device: the canonical indexed device (``cuda`` -> ``cuda:<active>``
        # on a CUDA host, ``cpu`` otherwise). It is only the default for
        # ``register`` calls that do not name a device; the engine's runners pass
        # the requesting model's own device, which is what makes explicit
        # secondary-device placement work (Phase 11-0).
        self.device = resolve_device(device)
        self._arenas: dict[str, ARCacheArena] = {}
        # Aggregate resident bytes (every device) and the per-device accounting
        # that enforces one budget per device.
        self._used_bytes = 0
        self._used_by_device: dict[torch.device, int] = {}
        self._lock = threading.Lock()
        # Monotonic counters (read by stats(); 9-6 may surface more of them).
        self.grants_total = 0
        self.refusals_total = 0
        self.releases_total = 0

    # -- registration ----------------------------------------------------
    @property
    def remaining_bytes(self) -> int:
        """Remaining budget on the pool's **default** device.

        Single-device deployments (the shipped configs) have one device, so this
        is the whole remaining budget. Multi-device callers ask about their own
        device with :meth:`remaining_bytes_for`.
        """
        return self.remaining_bytes_for(self.device)

    def remaining_bytes_for(self, device: torch.device | str | None = None) -> int:
        """Remaining budget on ``device`` (resolved; default: the pool device)."""
        target = resolve_device(self.device if device is None else device)
        with self._lock:
            return max(0, self.budget_bytes - self._used_by_device.get(target, 0))

    @property
    def pool_bytes(self) -> int:
        """Resident slab bytes across every device (the ``/metrics`` figure)."""
        with self._lock:
            return self._used_bytes if self.enabled else 0

    def register(
        self,
        descriptor: ARCacheDescriptor,
        *,
        device: torch.device | str | None = None,
    ) -> ARCacheArena | None:
        """Allocate ``descriptor``'s arena on ``device``; refuse it if it cannot fit.

        ``device`` defaults to the pool's device. The AR runner passes the
        requesting model's resolved device, so an explicit ``cuda:1`` model gets
        its arena on ``cuda:1`` while the pool's default stays where it was.

        Returns the arena (or ``None`` for a disabled pool). Raises
        :class:`ARCacheBudgetError` when the slab or the requested slot count
        exceeds the **device's** remaining budget, and ``ValueError`` for a
        duplicate key or invalid geometry — both are boot-time configuration
        errors.
        """
        if not self.enabled:
            return None
        if not isinstance(descriptor, ARCacheDescriptor):
            raise TypeError(f"expected an ARCacheDescriptor, got {type(descriptor).__name__}")

        target = resolve_device(self.device if device is None else device)
        cap_tokens = descriptor.cap_tokens if descriptor.cap_tokens is not None else self.default_cap_tokens
        slab_bytes = descriptor.slab_bytes(cap_tokens)

        with self._lock:
            if descriptor.model_key in self._arenas:
                raise ValueError(f"AR cache arena '{descriptor.model_key}' is already registered")
            remaining = self.budget_bytes - self._used_by_device.get(target, 0)
            if slab_bytes > remaining:
                raise ARCacheBudgetError(
                    f"AR cache arena '{descriptor.model_key}' needs a {slab_bytes / 2**20:.1f} MiB slab on "
                    f"{target}, but only {remaining / 2**20:.1f} MiB of the "
                    f"{self.budget_bytes / 2**20:.1f} MiB per-device budget remains. Lower cap_tokens / slots, "
                    "or raise ar_inference.kv_cache_budget_bytes."
                )
            slots = descriptor.slots
            if slots is None:
                # Pool availability decides the slot count, never admission width.
                slots = max(1, remaining // slab_bytes)
            arena_bytes = slab_bytes * slots
            if arena_bytes > remaining:
                raise ARCacheBudgetError(
                    f"AR cache arena '{descriptor.model_key}' needs {arena_bytes / 2**20:.1f} MiB "
                    f"({slots} slots × {slab_bytes / 2**20:.1f} MiB) on {target}, but only "
                    f"{remaining / 2**20:.1f} MiB of the {self.budget_bytes / 2**20:.1f} MiB "
                    "per-device budget remains. Lower slots or raise ar_inference.kv_cache_budget_bytes."
                )

            try:
                arena = ARCacheArena(
                    descriptor,
                    cap_tokens=cap_tokens,
                    slots=slots,
                    device=target,
                    pool=self,
                )
            except (RuntimeError, MemoryError) as error:
                # The budget is a logical bound, not a device guarantee: a driver
                # OOM at boot becomes a typed configuration error rather than a
                # raw allocation failure. Runtime exhaustion never allocates.
                raise ARCacheBudgetError(
                    f"could not allocate AR cache arena '{descriptor.model_key}' "
                    f"({arena_bytes / 2**20:.1f} MiB) on {target}: {error}"
                ) from error
            self._arenas[descriptor.model_key] = arena
            self._used_bytes += arena_bytes
            self._used_by_device[target] = self._used_by_device.get(target, 0) + arena_bytes
            return arena

    # -- slot lifecycle --------------------------------------------------
    def acquire(self, model_key: str) -> ARCacheSlot | None:
        """Grant one free slot of ``model_key``, or ``None`` when exhausted.

        ``None`` is the graceful refusal: the caller takes its sequential
        fallback and the pool never raises an allocation error at runtime. An
        unregistered key is a programming error and raises ``KeyError``.
        """
        if not self.enabled:
            return None
        with self._lock:
            arena = self._arenas.get(model_key)
            if arena is None:
                raise KeyError(f"no AR cache arena registered for '{model_key}'")
            if not arena._free:
                self.refusals_total += 1
                return None
            index = arena._free.popleft()
            arena._active.add(index)
            self.grants_total += 1
            return ARCacheSlot(arena, index)

    def release(self, slot: ARCacheSlot) -> None:
        """Return ``slot`` to its arena's free list (FIFO).

        Raises ``ValueError`` for a foreign slot, an already-released slot, or a
        pool that no longer owns the arena — a double release would otherwise
        hand the same slot to two requests, breaking the isolation invariant.
        """
        if not isinstance(slot, ARCacheSlot):
            raise TypeError(f"expected an ARCacheSlot, got {type(slot).__name__}")
        arena = slot.arena
        with self._lock:
            if self._arenas.get(arena.model_key) is not arena:
                raise ValueError(f"slot belongs to an arena this pool does not own ('{arena.model_key}')")
            if slot.slot_index not in arena._active:
                raise ValueError(f"slot {slot.slot_index} of '{arena.model_key}' is not active (double release?)")
            arena._active.discard(slot.slot_index)
            arena._free.append(slot.slot_index)
            slot._released = True
            self.releases_total += 1

    # -- accounting ------------------------------------------------------
    def arena_table(self) -> list[dict]:
        """Per-arena capacity rows, for the startup table log."""
        with self._lock:
            return [arena.capacity() for arena in self._arenas.values()]

    def capacity(self, model_key: str | None = None) -> dict:
        """Capacity-based admission input for 9-4.

        With ``model_key`` returns that arena's :meth:`ARCacheArena.capacity`;
        without it returns ``{"enabled", "slots", "slots_active", "slots_free",
        "active_rows", "pool_bytes", "allocated_bytes", "arenas": {...}}``.
        """
        with self._lock:
            if not self.enabled:
                return {
                    "enabled": False,
                    "device": str(self.device),
                    "devices": {},
                    "slots": 0,
                    "slots_active": 0,
                    "slots_free": 0,
                    "active_rows": 0,
                    "pool_bytes": 0,
                    "allocated_bytes": 0,
                    "arenas": {},
                }
            if model_key is not None:
                arena = self._arenas.get(model_key)
                if arena is None:
                    raise KeyError(f"no AR cache arena registered for '{model_key}'")
                return {"enabled": True, **arena.capacity()}
            arenas = {key: arena.capacity() for key, arena in self._arenas.items()}
            active = sum(row["slots_active"] for row in arenas.values())
            return {
                "enabled": True,
                "device": str(self.device),
                "devices": self._device_rows(),
                "slots": sum(row["slots"] for row in arenas.values()),
                "slots_active": active,
                "slots_free": sum(row["slots_free"] for row in arenas.values()),
                "active_rows": active,
                "pool_bytes": self._used_bytes,
                "allocated_bytes": sum(row["slots_active"] * row["slab_bytes"] for row in arenas.values()),
                "arenas": arenas,
            }

    def stats(self) -> dict:
        """Snapshot of pool/slot accounting (feeds ``/metrics`` + ``/health``)."""
        with self._lock:
            if not self.enabled:
                return {
                    "enabled": False,
                    "budget_bytes": self.budget_bytes,
                    "default_cap_tokens": self.default_cap_tokens,
                    "device": str(self.device),
                    "devices": {},
                    "pool_bytes": 0,
                    "allocated_bytes": 0,
                    "slots": 0,
                    "slots_active": 0,
                    "slots_free": 0,
                    "active_rows": 0,
                    "grants_total": 0,
                    "refusals_total": 0,
                    "releases_total": 0,
                    "arenas": {},
                }
            arenas = {key: arena.capacity() for key, arena in self._arenas.items()}
            active = sum(row["slots_active"] for row in arenas.values())
            return {
                "enabled": True,
                "budget_bytes": self.budget_bytes,
                "default_cap_tokens": self.default_cap_tokens,
                "device": str(self.device),
                # Per-device accounting (Phase 11-0): the budget applies to each
                # device separately, so a mixed placement reports one row per
                # device instead of one aggregate that would hide the split.
                "devices": self._device_rows(),
                # Resident slab footprint: ordinary torch allocations, so it also
                # shows up in the device-level VRAM accounting (Phase 10-3).
                "pool_bytes": self._used_bytes,
                "allocated_bytes": sum(row["slots_active"] * row["slab_bytes"] for row in arenas.values()),
                "slots": sum(row["slots"] for row in arenas.values()),
                "slots_active": active,
                "slots_free": sum(row["slots_free"] for row in arenas.values()),
                "active_rows": active,
                "grants_total": self.grants_total,
                "refusals_total": self.refusals_total,
                "releases_total": self.releases_total,
                "arenas": arenas,
            }

    def _device_rows(self) -> dict[str, dict]:
        """Per-device budget/usage rows (``{}`` when nothing is registered)."""
        return {
            str(device): {
                "budget_bytes": self.budget_bytes,
                "used_bytes": int(used),
                "remaining_bytes": max(0, self.budget_bytes - int(used)),
            }
            for device, used in self._used_by_device.items()
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"ARCachePool(enabled={self.enabled}, budget={self.budget_bytes}, "
            f"device={self.device}, arenas={len(self._arenas)}, used={self._used_bytes})"
        )

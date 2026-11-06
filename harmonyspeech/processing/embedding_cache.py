"""Cross-request embedding/conditionals cache (Phase 7-1).

Identical reference audio no longer recomputes speaker embeddings/Chatterbox
conditionals on every request: the engine owns one :class:`EmbeddingCache`
and threads it to the model runners exactly like the
:class:`~harmonyspeech.modeling.weight_registry.WeightRegistry` (executor →
worker → runner). Values are :class:`~harmonyspeech.common.artifacts.SpeakerArtifact`
objects — never wire bytes; serialization stays at the HTTP boundary.

Policy (product decision 2026-11-05):

* **TTL** — entries expire after ``ttl_s`` (default 300 s). Expiry is enforced
  lazily on :meth:`EmbeddingCache.get` and by a cheap sweep every
  ``_SWEEP_EVERY_N_PUTS`` puts.
* **FIFO, not LRU** — beyond ``max_entries`` (default 1000) the oldest
  *insertion* is evicted. A :meth:`EmbeddingCache.get` never reorders the
  underlying :class:`collections.OrderedDict`; re-inserting an existing key
  counts as a new insertion.
* **Byte ceiling** — optional ``max_bytes`` (default 1 GiB) as a safety bound
  on top of the entry cap, since Chatterbox conditionals are MB-sized. The
  oldest insertion is evicted first here too; an entry larger than the whole
  ceiling is stored and immediately evicted (the bound is hard).

Copy discipline (Phase 0-2): ``put`` deep-copies the artifact onto the CPU and
``get`` deep-copies again on serve, so no two requests can ever share mutable
state (consumers ``.to(device)``-move the served copy freely). Chatterbox
``Conditionals`` are cloned field-wise — a plain ``copy.deepcopy`` of the
upstream dataclass fails on checkpoint tensors that carry ``requires_grad``
(see ``task_handler.model_runner_base._clone_conditionals``).

Keys are content-addressed over ``(model_identity, audio_hash, params)``; the
runner embeds a per-model-name ``load_version`` counter
(:meth:`EmbeddingCache.bump_load_version`) in its identity so entries from a
previous weight load become unreachable and age out via TTL/FIFO.
"""

from __future__ import annotations

import copy
import hashlib
import json
import threading
import time
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass

import torch

from harmonyspeech.common.artifacts import AudioArtifact, SpeakerArtifact

__all__ = ["CacheEntry", "EmbeddingCache"]

#: Insert-side TTL sweep cadence: expiry is enforced on every get anyway; the
#: sweep only reclaims memory, so it runs every N puts to stay cheap at the
#: 1000-entry cap.
_SWEEP_EVERY_N_PUTS = 16

#: Rough bytes attributed to a non-tensor scalar/field inside a cached value.
_SCALAR_BYTES = 64


@dataclass
class CacheEntry:
    """One stored value plus its bookkeeping (internal to the cache)."""

    value: SpeakerArtifact
    key: str
    #: Identity string the key was derived from; ``invalidate_model`` filters
    #: on it (the sha256 key alone cannot be reversed).
    model_identity: str
    inserted_at: float
    expires_at: float
    approx_bytes: int


def _clone_value_cpu(value):
    """Clone ``value`` recursively; tensors become detached CPU leaves."""
    if torch.is_tensor(value):
        return value.detach().to("cpu").clone()
    if isinstance(value, dict):
        return {key: _clone_value_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_value_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_value_cpu(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool, bytes)):
        return value
    try:
        return copy.deepcopy(value)
    except Exception:  # pragma: no cover - opaque non-deepcopyable stand-ins
        return value


def _clone_conditionals_cpu(conds):
    """Field-wise clone of a Chatterbox-style ``Conditionals`` object.

    Mirrors ``task_handler.model_runner_base._clone_conditionals`` (kept in
    this module so the cache never imports from the handler layer): the
    upstream dataclass defeats ``copy.deepcopy`` when a checkpoint tensor
    carried ``requires_grad``. Plain dict/list stand-ins recurse.
    """
    if conds is None:
        return None
    if isinstance(conds, (dict, list, tuple)):
        return _clone_value_cpu(conds)
    t3 = getattr(conds, "t3", None)
    gen = getattr(conds, "gen", None)
    if t3 is None and gen is None and not hasattr(conds, "__dict__"):
        return _clone_value_cpu(conds)
    t3_clone = None
    if t3 is not None:
        try:
            t3_clone = type(t3)(**{key: _clone_value_cpu(item) for key, item in vars(t3).items()})
        except TypeError:  # pragma: no cover - future upstream field changes
            t3_clone = t3
    gen_clone = {key: _clone_value_cpu(item) for key, item in (gen or {}).items()}
    try:
        return type(conds)(t3_clone, gen_clone)
    except TypeError:  # pragma: no cover
        return _clone_value_cpu(conds)


def _copy_artifact(value: SpeakerArtifact) -> SpeakerArtifact:
    """Deep-copy a SpeakerArtifact; every tensor becomes a detached CPU clone."""
    embedding = value.embedding
    reference = value.reference_audio
    return SpeakerArtifact(
        embedding=_clone_value_cpu(embedding) if torch.is_tensor(embedding) else embedding,
        conditionals=_clone_conditionals_cpu(value.conditionals),
        backend=value.backend,
        reference_audio=(
            AudioArtifact(
                waveform=_clone_value_cpu(reference.waveform),
                sample_rate=reference.sample_rate,
            )
            if reference is not None
            else None
        ),
    )


def _approx_value_bytes(value) -> int:
    """Estimate resident tensor bytes inside a conditionals-like object."""
    if torch.is_tensor(value):
        return int(value.numel() * value.element_size())
    if isinstance(value, dict):
        return sum(_SCALAR_BYTES + _approx_value_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_approx_value_bytes(item) for item in value)
    if isinstance(value, (str, bytes)):
        return len(value)
    return _SCALAR_BYTES


def _approx_artifact_bytes(artifact: SpeakerArtifact) -> int:
    total = 0
    if torch.is_tensor(artifact.embedding):
        total += _approx_value_bytes(artifact.embedding)
    if artifact.conditionals is not None:
        total += _approx_value_bytes(artifact.conditionals)
    if artifact.reference_audio is not None:
        total += _approx_value_bytes(artifact.reference_audio.waveform)
    return total


class EmbeddingCache:
    """Cross-request embedding/conditionals cache; one instance per engine.

    Eviction: TTL (default 300 s) checked lazily on get + swept on insert;
    FIFO beyond ``max_entries`` (default 1000 — oldest insertion evicted first,
    per product decision 2026-11-05, NOT LRU); optional byte ceiling as a
    safety bound (default 1 GiB) since Chatterbox conditionals are MB-sized.
    """

    def __init__(self, ttl_s: float = 300.0, max_entries: int = 1000, max_bytes: int | None = 2**30):
        self.ttl_s = float(ttl_s)
        self.max_entries = max(1, int(max_entries))
        self.max_bytes = None if max_bytes is None else int(max_bytes)
        self._entries: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.Lock()
        self._bytes = 0
        self._puts_since_sweep = 0
        self._load_versions: dict[str, int] = {}
        # Monotonic counters (read by ``stats()``; 7-3 wires them into metrics).
        self.hits = 0
        self.misses = 0
        self.stores = 0
        self.ttl_expirations = 0
        self.fifo_evictions = 0
        self.byte_evictions = 0

    # -- keys ------------------------------------------------------------
    def make_key(self, *, model_identity: str, audio_hash: str, params: Mapping) -> str:
        """sha256 over ``(model_identity, audio_hash, params)``.

        ``params`` is canonicalized with sorted keys so call order never
        matters; only values that influence the embedding belong in it.
        """
        digest = hashlib.sha256()
        for part in (
            model_identity,
            audio_hash,
            json.dumps(dict(params), sort_keys=True, separators=(",", ":"), default=str),
        ):
            digest.update(part.encode("utf-8"))
            digest.update(b"\x00")
        return digest.hexdigest()

    # -- model loads -----------------------------------------------------
    def bump_load_version(self, model_name: str) -> int:
        """Increment the per-model-name load counter, return the new version.

        Runners call this on every ``load_model`` (first boot or reload); the
        version is part of their ``model_identity``, so keys derived under an
        older weight load can never be produced again.
        """
        with self._lock:
            version = self._load_versions.get(model_name, 0) + 1
            self._load_versions[model_name] = version
            return version

    def load_version(self, model_name: str) -> int:
        with self._lock:
            return self._load_versions.get(model_name, 0)

    # -- store -----------------------------------------------------------
    def get(self, key: str) -> SpeakerArtifact | None:
        """Return a deep copy of the stored artifact, or ``None``.

        Expired entries are treated as misses and removed. The lookup never
        reorders the map — that would silently turn FIFO into LRU.
        """
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self.misses += 1
                return None
            if time.monotonic() >= entry.expires_at:
                del self._entries[key]
                self._bytes -= entry.approx_bytes
                self.ttl_expirations += 1
                self.misses += 1
                return None
            self.hits += 1
            value = entry.value
        # Copy outside the lock: the stored value is private to the cache and
        # is never mutated in place, only replaced or evicted.
        return _copy_artifact(value)

    def put(self, key: str, value: SpeakerArtifact, model_identity: str | None = None) -> None:
        """Store a CPU deep copy of ``value`` under ``key``, evicting per policy.

        ``model_identity`` records which model produced the value so
        :meth:`invalidate_model` can drop it on a reload. With ``ttl_s <= 0``
        the cache is disabled outright and never stores.
        """
        if self.ttl_s <= 0:
            return
        stored = _copy_artifact(value)
        now = time.monotonic()
        with self._lock:
            self._puts_since_sweep += 1
            if self._puts_since_sweep >= _SWEEP_EVERY_N_PUTS:
                self._puts_since_sweep = 0
                self._sweep_expired(now)
            replaced = self._entries.pop(key, None)
            if replaced is not None:
                self._bytes -= replaced.approx_bytes
            entry = CacheEntry(
                value=stored,
                key=key,
                model_identity=model_identity or "",
                inserted_at=now,
                expires_at=now + self.ttl_s,
                approx_bytes=_approx_artifact_bytes(stored),
            )
            self._entries[key] = entry
            self._bytes += entry.approx_bytes
            self.stores += 1
            # FIFO beyond max_entries — oldest insertion first.
            while len(self._entries) > self.max_entries:
                self._evict_oldest(fifo=True)
            # Byte ceiling — oldest insertion first; a hard bound, so a single
            # oversized entry can evict itself right after being stored.
            if self.max_bytes is not None:
                while self._entries and self._bytes > self.max_bytes:
                    self._evict_oldest(fifo=False)

    def invalidate_model(self, model_identity: str) -> int:
        """Drop every entry stored under ``model_identity``; return the count.

        Called by a runner when it (re)loads, with its previous identity.
        """
        with self._lock:
            doomed = [key for key, entry in self._entries.items() if entry.model_identity == model_identity]
            for key in doomed:
                entry = self._entries.pop(key)
                self._bytes -= entry.approx_bytes
            return len(doomed)

    def stats(self) -> dict:
        """Snapshot of counters + resident size (feeds /metrics + /health)."""
        with self._lock:
            return {
                "entries": len(self._entries),
                "bytes": self._bytes,
                "hits": self.hits,
                "misses": self.misses,
                "stores": self.stores,
                "ttl_expirations": self.ttl_expirations,
                "fifo_evictions": self.fifo_evictions,
                "byte_evictions": self.byte_evictions,
            }

    # -- internals ---------------------------------------------------------
    def _sweep_expired(self, now: float) -> None:
        expired = [key for key, entry in self._entries.items() if now >= entry.expires_at]
        for key in expired:
            entry = self._entries.pop(key)
            self._bytes -= entry.approx_bytes
            self.ttl_expirations += 1

    def _evict_oldest(self, *, fifo: bool) -> None:
        _, evicted = self._entries.popitem(last=False)
        self._bytes -= evicted.approx_bytes
        if fifo:
            self.fifo_evictions += 1
        else:
            self.byte_evictions += 1

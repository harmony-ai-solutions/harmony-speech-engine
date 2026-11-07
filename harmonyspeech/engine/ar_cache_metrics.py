"""Prometheus text rendering for the AR KV-cache pool (Phase 9-2).

The engine owns the accounting
(:meth:`~harmonyspeech.engine.harmonyspeech_engine.HarmonySpeechEngine.get_ar_cache_stats`);
this module is a pure renderer so it can be unit-tested without a server, a GPU
or an allocated pool.

Series presence is part of the contract, the same way the embedding-cache, VRAM
and Whisper-lane exporters treat it: every series is emitted on every scrape,
with zeros when the pool is disabled (``--no-ar-cache``), because dashboards and
alerts key off the series — ``absent`` and ``0`` mean very different things.
"""

from __future__ import annotations

_HELP = {
    "hse_ar_cache_enabled": "1 when the shared AR KV-cache pool is enabled, 0 when it is disabled.",
    "hse_ar_cache_pool_bytes": "Resident slab bytes of every registered AR cache arena.",
    "hse_ar_cache_allocated_bytes": "Bytes of currently granted (in-flight) AR cache slots.",
    "hse_ar_cache_slots": "Total AR cache slots across every registered arena.",
    "hse_ar_cache_slots_active": "AR cache slots currently granted to an in-flight request.",
    "hse_ar_cache_slots_free": "AR cache slots currently available for a new request.",
    "hse_ar_cache_grants_total": "Successful AR cache slot grants since boot.",
    "hse_ar_cache_refusals_total": "Slot requests refused because every slot of the arena was in use.",
}

_GAUGES = (
    ("hse_ar_cache_pool_bytes", "pool_bytes"),
    ("hse_ar_cache_allocated_bytes", "allocated_bytes"),
    ("hse_ar_cache_slots", "slots"),
    ("hse_ar_cache_slots_active", "slots_active"),
    ("hse_ar_cache_slots_free", "slots_free"),
)

_COUNTERS = (
    ("hse_ar_cache_grants_total", "grants_total"),
    ("hse_ar_cache_refusals_total", "refusals_total"),
)


def _as_int(value) -> int:
    try:
        return int(float(value or 0))
    except (TypeError, ValueError):  # pragma: no cover - defensive on odd stats
        return 0


def render_ar_cache_metrics(stats: dict | None) -> str:
    """Render :func:`get_ar_cache_stats` output as Prometheus text exposition.

    ``stats`` is the engine's accounting dict (a raw
    :meth:`~harmonyspeech.engine.ar_cache.ARCachePool.stats` dict also works).
    ``None`` — or ``enabled: False`` — renders the disabled shape: all zeros,
    every series still present.
    """
    values = stats or {}
    enabled = bool(values.get("enabled", bool(values)))
    source = values if enabled else {}

    lines: list[str] = []

    def header(name: str, kind: str) -> None:
        lines.append(f"# HELP {name} {_HELP[name]}")
        lines.append(f"# TYPE {name} {kind}")

    header("hse_ar_cache_enabled", "gauge")
    lines.append(f"hse_ar_cache_enabled {1 if enabled else 0}")

    for name, key in _GAUGES:
        header(name, "gauge")
        lines.append(f"{name} {_as_int(source.get(key))}")

    for name, key in _COUNTERS:
        header(name, "counter")
        lines.append(f"{name} {_as_int(source.get(key))}")

    return "\n".join(lines) + "\n"


__all__ = ["render_ar_cache_metrics"]

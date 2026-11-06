"""Prometheus text rendering for cross-request embedding-cache metrics (Phase 7-3).

The engine owns the accounting
(:meth:`~harmonyspeech.engine.harmonyspeech_engine.HarmonySpeechEngine.get_embedding_cache_stats`);
this module is a pure renderer so it can be unit-tested without a server, a GPU
or a cache instance.

Series presence is part of the contract: every series is emitted on every
scrape, with zeros when the cache is disabled, because dashboards and alerts key
off the series (``absent`` and ``0`` mean very different things).
"""

from __future__ import annotations

_HELP = {
    "hse_embedding_cache_enabled": "1 when the cross-request embedding cache is enabled, 0 when it is disabled.",
    "hse_embedding_cache_entries": "Resident cache entries at scrape time.",
    "hse_embedding_cache_bytes": "Approximate resident bytes of cached embeddings and conditionals.",
    "hse_embedding_cache_hits_total": "Reference audio served from the cache instead of recomputing it.",
    "hse_embedding_cache_misses_total": "Cache lookups that had to recompute the embedding.",
    "hse_embedding_cache_stores_total": "Embeddings written into the cache.",
    "hse_embedding_cache_ttl_expirations_total": "Entries dropped because their time-to-live elapsed.",
    "hse_embedding_cache_fifo_evictions_total": "Entries evicted because the entry cap was exceeded.",
    "hse_embedding_cache_byte_evictions_total": "Entries evicted because the byte ceiling was exceeded.",
    "hse_embedding_cache_hit_ratio": "hits / (hits + misses) as a fraction; 0 when there was no traffic.",
}

#: ``(series name, stats key)`` pairs rendered as gauges / counters.
_GAUGES = (
    ("hse_embedding_cache_entries", "entries"),
    ("hse_embedding_cache_bytes", "bytes"),
)
_COUNTERS = (
    ("hse_embedding_cache_hits_total", "hits"),
    ("hse_embedding_cache_misses_total", "misses"),
    ("hse_embedding_cache_stores_total", "stores"),
    ("hse_embedding_cache_ttl_expirations_total", "ttl_expirations"),
    ("hse_embedding_cache_fifo_evictions_total", "fifo_evictions"),
    ("hse_embedding_cache_byte_evictions_total", "byte_evictions"),
)


def _as_int(value) -> int:
    try:
        return int(float(value or 0))
    except (TypeError, ValueError):  # pragma: no cover - defensive on odd stats
        return 0


def embedding_cache_hit_ratio(stats: dict | None) -> float:
    """Return ``hits / (hits + misses)``, or ``0.0`` without traffic.

    No traffic (fresh boot, cache just cleared) is deliberately ``0`` rather than
    ``NaN``: a NaN gauge would poison aggregation on dashboards.
    """
    if not stats:
        return 0.0
    hits = _as_int(stats.get("hits"))
    misses = _as_int(stats.get("misses"))
    total = hits + misses
    if total <= 0:
        return 0.0
    return hits / total


def render_embedding_cache_metrics(stats: dict | None) -> str:
    """Render embedding-cache stats as Prometheus text exposition.

    ``stats`` is :meth:`HarmonySpeechEngine.get_embedding_cache_stats` output (a
    raw ``EmbeddingCache.stats()`` dict also works). ``None`` — or
    ``enabled: False`` — renders the disabled shape: all zeros, every series
    still present.
    """
    values = stats or {}
    enabled = bool(values.get("enabled", bool(values)))
    source = values if enabled else {}

    lines: list[str] = []

    def header(name: str, kind: str) -> None:
        lines.append(f"# HELP {name} {_HELP[name]}")
        lines.append(f"# TYPE {name} {kind}")

    header("hse_embedding_cache_enabled", "gauge")
    lines.append(f"hse_embedding_cache_enabled {1 if enabled else 0}")

    for name, key in _GAUGES:
        header(name, "gauge")
        lines.append(f"{name} {_as_int(source.get(key))}")

    for name, key in _COUNTERS:
        header(name, "counter")
        lines.append(f"{name} {_as_int(source.get(key))}")

    header("hse_embedding_cache_hit_ratio", "gauge")
    lines.append(f"hse_embedding_cache_hit_ratio {embedding_cache_hit_ratio(source):.6f}")

    return "\n".join(lines) + "\n"


__all__ = ["embedding_cache_hit_ratio", "render_embedding_cache_metrics"]

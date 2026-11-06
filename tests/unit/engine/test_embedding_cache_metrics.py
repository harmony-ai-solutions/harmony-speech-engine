"""Unit tests for embedding-cache observability (Phase 7-3).

The renderer is a pure function over the cache stats dict, so these tests cover
the exposition format without a server, a GPU or a live engine: a warm cache, a
disabled cache (all series present, all zeros) and the hit-ratio edge cases.
``/metrics`` and ``/health`` wiring is asserted with the same shell-engine +
monkeypatched-serving pattern used by ``tests/unit/engine/test_vram.py``.
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch

from harmonyspeech.common.artifacts import SpeakerArtifact
from harmonyspeech.engine.embedding_cache_metrics import (
    embedding_cache_hit_ratio,
    render_embedding_cache_metrics,
)
from harmonyspeech.engine.harmonyspeech_engine import HarmonySpeechEngine
from harmonyspeech.processing.embedding_cache import EmbeddingCache

SERIES = (
    "hse_embedding_cache_enabled",
    "hse_embedding_cache_entries",
    "hse_embedding_cache_bytes",
    "hse_embedding_cache_hits_total",
    "hse_embedding_cache_misses_total",
    "hse_embedding_cache_stores_total",
    "hse_embedding_cache_ttl_expirations_total",
    "hse_embedding_cache_fifo_evictions_total",
    "hse_embedding_cache_byte_evictions_total",
    "hse_embedding_cache_hit_ratio",
)


def _engine_shell() -> HarmonySpeechEngine:
    engine = HarmonySpeechEngine.__new__(HarmonySpeechEngine)
    engine.model_executors = {}
    engine.model_configs = []
    return engine


def _speaker() -> SpeakerArtifact:
    return SpeakerArtifact(embedding=torch.zeros(8), backend="stub")


def _warm_stats() -> dict:
    """Stats shaped exactly like ``get_embedding_cache_stats()`` after traffic."""
    return {
        "enabled": True,
        "entries": 3,
        "bytes": 2048,
        "hits": 9,
        "misses": 3,
        "stores": 4,
        "ttl_expirations": 1,
        "fifo_evictions": 2,
        "byte_evictions": 5,
        "hit_ratio": 0.75,
        "ttl_s": 300.0,
        "max_entries": 1000,
        "max_bytes": 2**30,
    }


# ---------------------------------------------------------------------------
# Renderer: exposition format
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_render_warm_cache_emits_every_series_with_its_value():
    text = render_embedding_cache_metrics(_warm_stats())

    for line in (
        "hse_embedding_cache_enabled 1",
        "hse_embedding_cache_entries 3",
        "hse_embedding_cache_bytes 2048",
        "hse_embedding_cache_hits_total 9",
        "hse_embedding_cache_misses_total 3",
        "hse_embedding_cache_stores_total 4",
        "hse_embedding_cache_ttl_expirations_total 1",
        "hse_embedding_cache_fifo_evictions_total 2",
        "hse_embedding_cache_byte_evictions_total 5",
        "hse_embedding_cache_hit_ratio 0.750000",
    ):
        assert line in text

    for name in SERIES:
        assert f"# HELP {name} " in text
        assert f"# TYPE {name} " in text
    # Counters are declared as counters, gauges as gauges (Prometheus semantics).
    assert "# TYPE hse_embedding_cache_hits_total counter" in text
    assert "# TYPE hse_embedding_cache_entries gauge" in text
    assert "# TYPE hse_embedding_cache_hit_ratio gauge" in text


@pytest.mark.unit
def test_render_disabled_cache_keeps_every_series_present_and_zero():
    text = render_embedding_cache_metrics({"enabled": False})

    assert "hse_embedding_cache_enabled 0" in text
    for name in SERIES:
        assert f"# TYPE {name} " in text
    # Every sample line is zero-valued, including the ratio.
    for name in SERIES[1:]:
        assert f"{name} 0" in text or f"{name} 0.000000" in text


@pytest.mark.unit
def test_render_none_renders_the_disabled_shape():
    text = render_embedding_cache_metrics(None)

    assert "hse_embedding_cache_enabled 0" in text
    assert "hse_embedding_cache_entries 0" in text
    assert "hse_embedding_cache_hit_ratio 0.000000" in text
    for name in SERIES:
        assert f"# TYPE {name} " in text


@pytest.mark.unit
def test_render_accepts_a_raw_cache_stats_dict():
    cache = EmbeddingCache(ttl_s=300)
    value = _speaker()
    key = cache.make_key(model_identity="m", audio_hash="h", params={})
    cache.put(key, value, model_identity="m")
    cache.get(key)

    text = render_embedding_cache_metrics(cache.stats())

    assert "hse_embedding_cache_entries 1" in text
    assert "hse_embedding_cache_stores_total 1" in text
    assert "hse_embedding_cache_hits_total 1" in text
    assert "hse_embedding_cache_hit_ratio 1.000000" in text


# ---------------------------------------------------------------------------
# Hit-ratio edge cases
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    ("stats", "expected"),
    [
        (None, 0.0),
        ({}, 0.0),
        ({"hits": 0, "misses": 0}, 0.0),  # no traffic -> 0, never NaN
        ({"hits": 7, "misses": 0}, 1.0),  # hits only
        ({"hits": 0, "misses": 7}, 0.0),  # misses only
        ({"hits": 1, "misses": 3}, 0.25),  # mixed
    ],
)
def test_hit_ratio_edge_cases(stats, expected):
    assert embedding_cache_hit_ratio(stats) == pytest.approx(expected)


@pytest.mark.unit
def test_hit_ratio_ignores_junk_counters():
    assert embedding_cache_hit_ratio({"hits": None, "misses": "n/a"}) == 0.0


# ---------------------------------------------------------------------------
# Engine surface
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_engine_stats_without_a_cache_are_disabled_zeros():
    engine = _engine_shell()

    stats = engine.get_embedding_cache_stats()

    assert stats["enabled"] is False
    assert stats["hit_ratio"] == 0.0
    for key in ("entries", "bytes", "hits", "misses", "stores", "ttl_expirations", "fifo_evictions", "byte_evictions"):
        assert stats[key] == 0


@pytest.mark.unit
def test_engine_stats_report_a_live_cache():
    engine = _engine_shell()
    engine.embedding_cache = EmbeddingCache(ttl_s=300, max_entries=10, max_bytes=4096)
    cache = engine.embedding_cache
    key = cache.make_key(model_identity="m", audio_hash="h", params={"p": 1})
    cache.put(key, _speaker(), model_identity="m")
    cache.get(key)
    cache.get("never-stored")

    stats = engine.get_embedding_cache_stats()

    assert stats["enabled"] is True
    assert stats["entries"] == 1
    assert stats["bytes"] > 0
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["stores"] == 1
    assert stats["hit_ratio"] == pytest.approx(0.5)
    assert stats["ttl_s"] == 300.0
    assert stats["max_entries"] == 10
    assert stats["max_bytes"] == 4096


@pytest.mark.unit
def test_check_health_includes_the_embedding_cache_block():
    engine = _engine_shell()
    engine.embedding_cache = EmbeddingCache(ttl_s=300)

    payload = engine.check_health()

    assert payload["healthy"] is True
    assert payload["embedding_cache"]["enabled"] is True
    assert payload["embedding_cache"]["entries"] == 0
    assert "vram" in payload


# ---------------------------------------------------------------------------
# Endpoint wiring
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_metrics_endpoint_renders_cache_series(monkeypatch):
    from harmonyspeech.endpoints.openai import api_server

    engine = _engine_shell()
    engine.embedding_cache = EmbeddingCache(ttl_s=300)
    monkeypatch.setattr(api_server, "openai_serving_tts", SimpleNamespace(engine=engine))
    monkeypatch.setattr(api_server, "openai_serving_stt", None)
    monkeypatch.setattr(api_server, "openai_serving_vc", None)
    monkeypatch.setattr(api_server, "openai_serving_embedding", None)

    response = asyncio.run(api_server.metrics())
    body = response.body.decode()

    assert "hse_embedding_cache_enabled 1" in body
    assert "hse_embedding_cache_hit_ratio 0.000000" in body
    for name in SERIES:
        assert name in body
    assert response.media_type.startswith("text/plain")


@pytest.mark.unit
def test_metrics_endpoint_without_an_engine_still_renders_zeroed_series(monkeypatch):
    from harmonyspeech.endpoints.openai import api_server

    for attribute in ("openai_serving_tts", "openai_serving_stt", "openai_serving_vc", "openai_serving_embedding"):
        monkeypatch.setattr(api_server, attribute, None)

    response = asyncio.run(api_server.metrics())
    body = response.body.decode()

    assert "hse_embedding_cache_enabled 0" in body
    assert "hse_embedding_cache_entries 0" in body


@pytest.mark.unit
def test_health_endpoint_includes_the_embedding_cache_block(monkeypatch):
    from harmonyspeech.endpoints.openai import api_server

    engine = _engine_shell()
    engine.embedding_cache = EmbeddingCache(ttl_s=300)
    # The serving layer holds the async wrapper; /health only awaits it and reads
    # the structured payload the engine returned.
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
    assert payload["embedding_cache"]["enabled"] is True
    assert payload["embedding_cache"]["entries"] == 0
    assert payload["embedding_cache"]["hit_ratio"] == 0.0
    assert "vram" in payload

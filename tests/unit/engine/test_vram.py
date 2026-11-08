"""Unit tests for VRAM accounting and the /metrics surface (Phase 4-5).

CPU-safe: the CUDA numbers are read behind ``torch.cuda.is_available()`` and the
tests assert shape and consistency rather than device-specific values.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
import torch.nn as nn

from harmonyspeech.engine.harmonyspeech_engine import HarmonySpeechEngine
from harmonyspeech.engine.vram_metrics import render_vram_metrics


def _engine_shell() -> HarmonySpeechEngine:
    engine = HarmonySpeechEngine.__new__(HarmonySpeechEngine)
    engine.model_executors = {}
    engine.model_configs = []
    return engine


def _fake_executor(name: str, model_type: str = "ChatterboxTTS", *, group=None, vram_bytes=0):
    model = nn.Linear(4, 4)
    runner = SimpleNamespace(model=model, model_memory_usage=vram_bytes)
    executor = SimpleNamespace(
        model_config=SimpleNamespace(
            model_type=model_type,
            device_config=SimpleNamespace(device="cuda"),
            dtype=torch.float16,
            backbone_group=group,
            model="resemble-ai/chatterbox",
            name=name,
        ),
        driver_worker=SimpleNamespace(model_runner=runner),
        check_health=lambda: None,
    )
    return executor


def test_get_vram_stats_without_cuda_or_registry_does_not_crash():
    engine = _engine_shell()

    stats = engine.get_vram_stats()

    assert stats["models"] == {}
    assert stats["groups"] == {}
    assert stats["resident_models"] == 0
    assert stats["shared"]["entries"] == 0
    assert isinstance(stats["allocated_bytes"], int)
    assert isinstance(stats["reserved_bytes"], int)
    assert stats["total_bytes"] >= 0


def test_get_vram_stats_reports_per_model_bytes_and_weights():
    engine = _engine_shell()
    engine.model_executors = {"chatterbox": _fake_executor("chatterbox", vram_bytes=1234)}

    stats = engine.get_vram_stats()
    detail = stats["models"]["chatterbox"]

    assert detail["vram_bytes"] == 1234
    assert detail["weights_bytes"] > 0
    assert detail["model_type"] == "ChatterboxTTS"
    assert detail["backbone_group"] is None
    assert stats["resident_models"] == 1


def test_get_vram_stats_attributes_shared_bytes_to_group():
    from harmonyspeech.modeling.weight_registry import WeightRegistry

    registry = WeightRegistry()
    registry.acquire(("resemble-ai/chatterbox", None, "s3gen", torch.float32), lambda: nn.Linear(8, 8))
    registry.acquire(("resemble-ai/chatterbox", None, "ve", torch.float32), lambda: nn.Linear(4, 4))

    engine = _engine_shell()
    engine.weight_registry = registry
    engine.model_executors = {
        "chatterbox": _fake_executor("chatterbox", group="chatterbox-base"),
        "chatterbox_vc": _fake_executor("chatterbox_vc", group="chatterbox-base"),
    }

    stats = engine.get_vram_stats()

    assert stats["shared"]["entries"] == 2
    assert stats["groups"]["chatterbox-base"] == stats["shared"]["total_bytes"]
    assert stats["shared"]["total_bytes"] > 0


def test_check_health_includes_vram_detail():
    engine = _engine_shell()
    engine.model_executors = {"chatterbox": _fake_executor("chatterbox", vram_bytes=4096)}

    payload = engine.check_health()

    assert payload["healthy"] is True
    assert payload["models"]["chatterbox"]["vram_bytes"] == 4096
    assert payload["vram"]["resident_models"] == 1


def test_check_health_surfaces_effective_s3gen_precision(monkeypatch):
    """Phase 11-3: /health proves which S3Gen precision the vocode runs."""
    monkeypatch.delenv("HSE_S3GEN_AUTOCAST", raising=False)
    engine = _engine_shell()
    executor = _fake_executor("chatterbox")
    executor.model_config.s3gen_precision = "float16"
    engine.model_executors = {"chatterbox": executor}

    assert engine.check_health()["models"]["chatterbox"]["s3gen_precision"] == "float16"

    # The env escape hatch overrides the config value in the observable field.
    monkeypatch.setenv("HSE_S3GEN_AUTOCAST", "off")
    assert engine.check_health()["models"]["chatterbox"]["s3gen_precision"] == "float32"

    # A family whose vocode does not use S3Gen carries no such field.
    monkeypatch.delenv("HSE_S3GEN_AUTOCAST", raising=False)
    engine.model_executors = {
        "embedding": _fake_executor("embedding", model_type="ChatterboxEmbedding")
    }
    assert "s3gen_precision" not in engine.check_health()["models"]["embedding"]


def test_check_health_marks_failure_but_keeps_vram():
    engine = _engine_shell()
    executor = _fake_executor("chatterbox")
    executor.check_health = MagicMock(side_effect=RuntimeError("boom"))
    engine.model_executors = {"chatterbox": executor}

    payload = engine.check_health()

    assert payload["healthy"] is False
    assert payload["models"]["chatterbox"]["error"] == "boom"
    assert "vram" in payload


def test_render_vram_metrics_emits_the_required_series():
    stats = {
        "models": {
            "chatterbox": {"vram_bytes": 1024, "weights_bytes": 2048, "model_type": "ChatterboxTTS"},
        },
        "allocated_bytes": 4096,
        "reserved_bytes": 8192,
        "peak_allocated_bytes": 9000,
        "total_bytes": 16 * 2**30,
        "headroom_bytes": 16 * 2**30 - 8192,
        "resident_models": 1,
        "groups": {"chatterbox-base": 512},
        "shared": {"entries": 1, "total_bytes": 512, "components": {}},
    }

    text = render_vram_metrics(stats)

    for required in (
        "hse_model_vram_bytes{model=\"chatterbox\",model_type=\"ChatterboxTTS\"} 1024",
        "hse_vram_reserved_bytes 8192",
        "hse_resident_models 1",
        "hse_shared_backbone_bytes{group=\"chatterbox-base\"} 512",
    ):
        assert required in text
    assert text.count("# TYPE") >= 8


def test_render_vram_metrics_with_empty_groups_still_has_a_sample():
    text = render_vram_metrics({"models": {}, "groups": {}, "shared": {"entries": 0, "total_bytes": 0}})
    assert 'hse_shared_backbone_bytes{group="none"} 0' in text
    assert "hse_model_vram_bytes" not in text  # no models → no family


def test_metrics_endpoint_renders_prometheus_text(monkeypatch):
    from harmonyspeech.endpoints.openai import api_server

    engine = _engine_shell()
    engine.model_executors = {"chatterbox": _fake_executor("chatterbox", vram_bytes=17)}
    serving = SimpleNamespace(engine=engine)
    monkeypatch.setattr(api_server, "openai_serving_tts", serving)
    monkeypatch.setattr(api_server, "openai_serving_stt", None)
    monkeypatch.setattr(api_server, "openai_serving_vc", None)
    monkeypatch.setattr(api_server, "openai_serving_embedding", None)

    response = asyncio.run(api_server.metrics())

    body = response.body.decode()
    assert "hse_model_vram_bytes" in body
    assert "hse_resident_models 1" in body
    assert response.media_type.startswith("text/plain")

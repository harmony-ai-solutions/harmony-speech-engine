"""Unit tests for device-level VRAM truth and Whisper governance (Phase 10-3).

CPU-safe: CUDA calls are mocked; the probe's skip paths are exercised for real.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch

from harmonyspeech.common.config import EngineConfig
from harmonyspeech.engine.harmonyspeech_engine import HarmonySpeechEngine
from harmonyspeech.engine.vram_budget import (
    allowed_windows,
    evaluate_whisper_budget,
    per_window_bytes,
    projected_whisper_peak,
)
from harmonyspeech.engine.vram_metrics import render_vram_metrics
from harmonyspeech.modeling.whisper_probe import probe_whisper_peak

GIB = 2**30


# ── Helpers ───────────────────────────────────────────────────────────────────


def _engine_shell() -> HarmonySpeechEngine:
    engine = HarmonySpeechEngine.__new__(HarmonySpeechEngine)
    engine.model_executors = {}
    engine.model_configs = []
    engine.whisper_vram_budget_bytes = None
    engine.whisper_max_audio_seconds = 300.0
    engine.whisper_vram_report = None
    return engine


def _fake_executor(
    name: str,
    model_type: str = "ChatterboxTTS",
    *,
    device: str = "cuda",
    vram_bytes: int = 0,
    max_batch_size: int | None = 1,
    model=None,
    whisper_probe: dict | None = None,
):
    runner = SimpleNamespace(model=model, model_memory_usage=vram_bytes, whisper_probe=whisper_probe)
    return SimpleNamespace(
        model_config=SimpleNamespace(
            model_type=model_type,
            device_config=SimpleNamespace(device=device),
            dtype=torch.float16,
            backbone_group=None,
            model="resemble-ai/chatterbox",
            name=name,
            max_batch_size=max_batch_size,
        ),
        driver_worker=SimpleNamespace(model_runner=runner),
        check_health=lambda: None,
    )


def _mock_cuda(monkeypatch, *, free: int, total: int, allocated: int = 0, reserved: int = 0, peak: int = 0):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda index=None: allocated)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda index=None: reserved)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda index=None: peak)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda index=None: SimpleNamespace(total_memory=total),
    )
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index=None: "Fake GPU")
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda index=None: (free, total))


def _whisper_stats(**overrides) -> dict:
    stats = {
        "cuda": True,
        "device": "Fake GPU",
        "total_bytes": 16 * GIB,
        "free_bytes": 8 * GIB,
        "device_used_bytes": 8 * GIB,
        "device_used_source": "mem_get_info",
        "allocated_bytes": 4 * GIB,
        "reserved_bytes": 5 * GIB,
        "peak_allocated_bytes": 6 * GIB,
        "headroom_bytes": 8 * GIB,
        "resident_models": 1,
        "models": {
            "whisper": {
                "model_type": "FasterWhisper",
                "device": "cuda",
                "dtype": None,
                "vram_bytes": 0,
                "weights_bytes": 1 * GIB,
                "backbone_group": None,
                "max_batch_size": 8,
                "peak_activation_bytes": 1 * GIB,
                "per_window_bytes": 128 * 2**20,
                "probe_measured": True,
                "probe_reason": None,
                "workers": 2,
                "window_batch": 4,
                "beam_size": 5,
                "max_audio_seconds": 300.0,
                "windows_probed": 8,
            }
        },
        "shared": {"entries": 0, "total_bytes": 0, "components": {}},
        "groups": {},
        "dedup_saved_bytes": 0,
        "whisper_budget_bytes": None,
        "whisper_governance": None,
    }
    stats.update(overrides)
    return stats


# ── Device-level VRAM truth ───────────────────────────────────────────────────


def test_device_level_truth_uses_mem_get_info(monkeypatch):
    _mock_cuda(
        monkeypatch,
        free=6 * GIB,
        total=16 * GIB,
        allocated=7 * GIB,
        reserved=8 * GIB,
        peak=9 * GIB,
    )

    stats = _engine_shell().get_vram_stats()

    assert stats["cuda"] is True
    assert stats["total_bytes"] == 16 * GIB
    assert stats["free_bytes"] == 6 * GIB
    assert stats["device_used_bytes"] == 10 * GIB
    assert stats["device_used_source"] == "mem_get_info"
    # Redefined in place: true device free, not total - torch reserved (8 GiB).
    assert stats["headroom_bytes"] == 6 * GIB
    # Torch-scoped series are kept for "what does torch hold".
    assert stats["reserved_bytes"] == 8 * GIB
    assert stats["allocated_bytes"] == 7 * GIB
    assert stats["peak_allocated_bytes"] == 9 * GIB


def test_device_level_falls_back_when_mem_get_info_fails(monkeypatch):
    _mock_cuda(monkeypatch, free=0, total=16 * GIB, reserved=8 * GIB)

    def boom(index=None):
        raise RuntimeError("mem_get_info unavailable")

    monkeypatch.setattr(torch.cuda, "mem_get_info", boom)

    stats = _engine_shell().get_vram_stats()

    assert stats["device_used_source"] == "torch_reserved_fallback"
    assert stats["headroom_bytes"] == 8 * GIB
    assert stats["device_used_bytes"] == 8 * GIB


def test_cpu_build_reports_zero_device_level(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    stats = _engine_shell().get_vram_stats()

    assert stats["cuda"] is False
    assert stats["free_bytes"] == 0
    assert stats["device_used_bytes"] == 0
    assert stats["headroom_bytes"] == 0
    assert stats["device_used_source"] is None


def test_render_vram_metrics_emits_device_level_and_probe_series():
    stats = _whisper_stats()

    text = render_vram_metrics(stats)

    assert f"hse_vram_device_used_bytes {8 * GIB}" in text
    assert f"hse_vram_headroom_bytes {8 * GIB}" in text
    assert 'hse_model_peak_activation_bytes{model="whisper"} ' + str(1 * GIB) in text
    # The breaking change is documented where operators read it.
    assert "Breaking change" in text
    assert "mem_get_info" in text


def test_headroom_help_is_the_device_level_definition():
    text = render_vram_metrics({"models": {}, "groups": {}, "shared": {}})
    assert "total minus torch reserved" in text
    assert "overstated" in text


# ── Probe delta math and skip paths ───────────────────────────────────────────


class _FakePipeline:
    def __init__(self, model):
        self.model = model
        self.last_speech_timestamp = 0.0

    def transcribe(self, audio, **kwargs):
        return iter([]), None


def test_probe_records_device_delta(monkeypatch):
    import faster_whisper

    monkeypatch.setattr(faster_whisper, "BatchedInferencePipeline", _FakePipeline)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda index=None: None)
    free_values = iter([(1000, 2000), (600, 2000)])
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda index=None: next(free_values))

    model = SimpleNamespace(model=SimpleNamespace(device="cuda"))
    record = probe_whisper_peak(
        model,
        workers=2,
        window_batch=2,
        beam_size=3,
        max_audio_seconds=60.0,
        sample_peak=False,
    )

    assert record["measured"] is True
    assert record["reason"] is None
    # used_before = 1000, used_after = 1400 -> 400 bytes for 2x2 windows.
    assert record["peak_activation_bytes"] == 400
    assert record["per_window_bytes"] == 100
    assert record["windows_probed"] == 4
    assert record["method"] == "mem_get_info_delta"


def test_probe_counts_only_real_30s_clips(monkeypatch):
    import faster_whisper

    monkeypatch.setattr(faster_whisper, "BatchedInferencePipeline", _FakePipeline)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda index=None: None)
    free_values = iter([(1000, 2000), (400, 2000)])
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda index=None: next(free_values))

    model = SimpleNamespace(model=SimpleNamespace(device="cuda"))
    # 35 s is two 30 s clips, so a batch width of 8 still exercises two windows.
    record = probe_whisper_peak(
        model,
        workers=1,
        window_batch=8,
        beam_size=3,
        max_audio_seconds=35.0,
        sample_peak=False,
    )

    assert record["windows_probed"] == 2
    assert record["peak_activation_bytes"] == 600
    assert record["per_window_bytes"] == 300


class _RecordingPipeline:
    """Records what the probe actually sends to the batched pipeline."""

    calls: list[tuple] = []

    def __init__(self, model):
        self.model = model
        self.last_speech_timestamp = 0.0

    def transcribe(self, audio, **kwargs):
        _RecordingPipeline.calls.append((audio, kwargs))
        return iter([]), None


def test_probe_sends_speech_like_audio_with_explicit_clips(monkeypatch):
    """Phase 11-0: a ≥30 s probe must not depend on VAD finding speech."""
    import faster_whisper

    _RecordingPipeline.calls.clear()
    monkeypatch.setattr(faster_whisper, "BatchedInferencePipeline", _RecordingPipeline)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda index=None: None)
    free_values = iter([(1000, 2000), (400, 2000)])
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda index=None: next(free_values))

    model = SimpleNamespace(model=SimpleNamespace(device="cuda"))
    record = probe_whisper_peak(
        model,
        workers=2,
        window_batch=8,
        beam_size=3,
        max_audio_seconds=60.0,
        sample_peak=False,
    )

    audio, kwargs = _RecordingPipeline.calls[0]
    assert kwargs["clip_timestamps"] == [
        {"start": 0.0, "end": 30.0},
        {"start": 30.0, "end": 60.0},
    ], "the probe must synthesise clips instead of asking VAD to find speech"
    assert audio.shape[0] == 60 * 16000
    assert float(abs(audio).max()) > 0.1, "the probe audio must not be silence"
    assert record["windows_probed"] == 4, "2 workers x 2 clips"
    assert record["measured"] is True


def test_probe_uses_the_model_device_index_when_available(monkeypatch):
    """A ``cuda:1`` model must be measured on ``cuda:1``, not the active card."""
    import faster_whisper

    _RecordingPipeline.calls.clear()
    monkeypatch.setattr(faster_whisper, "BatchedInferencePipeline", _RecordingPipeline)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda index=None: None)
    probed_indexes: list[int] = []

    def _mem_get_info(index=None):
        probed_indexes.append(index)
        return (1000, 2000) if len(probed_indexes) == 1 else (900, 2000)

    monkeypatch.setattr(torch.cuda, "mem_get_info", _mem_get_info)

    model = SimpleNamespace(
        model=SimpleNamespace(device="cuda", device_index=[1])
    )
    record = probe_whisper_peak(
        model, workers=1, window_batch=1, max_audio_seconds=30.0, sample_peak=False
    )

    assert probed_indexes == [1, 1], f"probe read {probed_indexes}"
    assert record["measured"] is True


def test_probe_skips_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    record = probe_whisper_peak(SimpleNamespace(model=SimpleNamespace(device="cuda")))

    assert record["measured"] is False
    assert record["reason"] == "cuda-unavailable"
    assert record["peak_activation_bytes"] == 0


def test_probe_skips_cpu_placed_model(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    record = probe_whisper_peak(SimpleNamespace(model=SimpleNamespace(device="cpu")))

    assert record["measured"] is False
    assert record["reason"] == "model-device:cpu"


def test_probe_skips_missing_model():
    record = probe_whisper_peak(None)

    assert record["measured"] is False
    assert record["reason"] == "no-model"


def test_probe_skips_when_import_fails(monkeypatch):
    import builtins

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "faster_whisper":
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)

    record = probe_whisper_peak(SimpleNamespace(model=SimpleNamespace(device="cuda")))

    assert record["measured"] is False
    assert record["reason"].startswith("faster-whisper-import-failed")


# ── Pure admission-scaling decision ───────────────────────────────────────────


def test_allowed_windows_without_budget_returns_cap():
    assert allowed_windows(None, 999, 0, 8) == 8
    assert allowed_windows(0, 999, 0, 8) == 8


def test_allowed_windows_scales_with_remaining_budget():
    # 600 bytes of headroom at 100 bytes/window -> 6, under the cap.
    assert allowed_windows(1000, 400, 0, 10, per_window_bytes=100) == 6


def test_allowed_windows_estimates_cost_from_active_windows():
    # No explicit cost: 500 used / 2 active = 250 per window; 500 headroom.
    assert allowed_windows(1000, 500, 2, 10) == 2


def test_allowed_windows_respects_cap():
    assert allowed_windows(10**9, 0, 0, 3, per_window_bytes=1) == 3


def test_allowed_windows_over_budget_keeps_a_draining_floor():
    assert allowed_windows(1000, 2000, 4, 10, per_window_bytes=100) == 1
    assert allowed_windows(1000, 2000, 4, 10, per_window_bytes=100, min_windows=0) == 0


def test_allowed_windows_unknown_cost_does_not_throttle():
    assert allowed_windows(1000, 2000, 0, 7, per_window_bytes=0) == 7


def test_allowed_windows_zero_cap():
    assert allowed_windows(1000, 0, 0, 0, per_window_bytes=1) == 0


def test_per_window_bytes_math():
    assert per_window_bytes(800, 4) == 200
    assert per_window_bytes(800, 0) == 0


def test_projected_whisper_peak_sums_weights_and_probe():
    entries = [
        {"weights_bytes": 100, "peak_activation_bytes": 50},
        {"weights_bytes": 200, "peak_activation_bytes": 0},
    ]
    assert projected_whisper_peak(entries) == 350
    assert projected_whisper_peak([]) == 0


# ── Boot-time budget verdict ──────────────────────────────────────────────────


def test_evaluate_budget_ok():
    verdict = evaluate_whisper_budget(500, 1000, None)
    assert verdict["status"] == "ok"
    assert verdict["exceeded_headroom"] is False


def test_evaluate_budget_warns_when_projected_exceeds_headroom():
    verdict = evaluate_whisper_budget(1500, 1000, None)
    assert verdict["status"] == "warn"
    assert verdict["exceeded_headroom"] is True


def test_evaluate_budget_errors_when_budget_exceeded():
    verdict = evaluate_whisper_budget(1500, 10_000, 1000)
    assert verdict["status"] == "error"
    assert verdict["exceeded_budget"] is True
    assert "whisper_vram_budget_bytes" in verdict["message"]


def test_engine_boot_check_raises_when_budget_exceeded():
    engine = _engine_shell()
    engine.whisper_vram_budget_bytes = 100
    engine.get_vram_stats = lambda: _whisper_stats(
        free_bytes=10_000,
        device_used_bytes=16 * GIB - 10_000,
        headroom_bytes=10_000,
    )

    with pytest.raises(RuntimeError, match="whisper_vram_budget_bytes"):
        engine._evaluate_whisper_memory_governance()


def test_engine_boot_check_warns_without_budget(loguru_records):
    engine = _engine_shell()
    engine.get_vram_stats = lambda: _whisper_stats(
        free_bytes=100,
        device_used_bytes=16 * GIB - 100,
        headroom_bytes=100,
    )

    report = engine._evaluate_whisper_memory_governance()

    assert report["enabled"] is True
    assert report["status"] == "warn"
    assert any("free device headroom" in message for message in loguru_records)


def test_engine_boot_check_is_idle_without_whisper():
    engine = _engine_shell()
    engine.get_vram_stats = lambda: _whisper_stats(models={})

    report = engine._evaluate_whisper_memory_governance()

    assert report["enabled"] is False
    assert report["status"] == "no-whisper"


def test_engine_boot_check_skips_on_cpu():
    engine = _engine_shell()
    engine.get_vram_stats = lambda: _whisper_stats(cuda=False, models={})

    report = engine._evaluate_whisper_memory_governance()

    assert report["enabled"] is False
    assert report["status"] == "cpu"


def test_whisper_cuda_config_that_fell_back_to_cpu_is_not_governed():
    engine = _engine_shell()
    stats = _whisper_stats(
        models={
            "whisper": {
                "model_type": "FasterWhisper",
                "device": "cuda",
                "runtime_device": "cpu",
                "weights_bytes": 1 * GIB,
                "peak_activation_bytes": 0,
                "per_window_bytes": 0,
                "probe_measured": False,
                "probe_reason": "model-device:cpu",
                "max_batch_size": 8,
            }
        }
    )
    engine.get_vram_stats = lambda: stats

    report = engine._evaluate_whisper_memory_governance()

    assert report["status"] == "no-whisper"


# ── Probe attribution through /health and /metrics ────────────────────────────


def test_get_vram_stats_includes_whisper_probe_fields():
    engine = _engine_shell()
    probe = {
        "measured": True,
        "reason": None,
        "peak_activation_bytes": 512,
        "per_window_bytes": 128,
        "workers": 2,
        "window_batch": 2,
        "beam_size": 5,
        "max_audio_seconds": 300.0,
        "windows_probed": 4,
        "method": "mem_get_info_delta",
    }
    engine.model_executors = {
        "whisper": _fake_executor(
            "whisper",
            model_type="FasterWhisper",
            max_batch_size=8,
            whisper_probe=probe,
        )
    }

    detail = engine.get_vram_stats()["models"]["whisper"]

    assert detail["peak_activation_bytes"] == 512
    assert detail["per_window_bytes"] == 128
    assert detail["probe_measured"] is True
    assert detail["workers"] == 2
    assert detail["max_batch_size"] == 8


def test_get_vram_stats_marks_skipped_probe_explicitly():
    engine = _engine_shell()
    engine.model_executors = {
        "whisper": _fake_executor(
            "whisper",
            model_type="FasterWhisper",
            whisper_probe={"measured": False, "reason": "cuda-unavailable"},
        )
    }

    detail = engine.get_vram_stats()["models"]["whisper"]

    assert detail["probe_measured"] is False
    assert detail["probe_reason"] == "cuda-unavailable"
    assert detail["peak_activation_bytes"] == 0


def test_check_health_carries_probe_annotation():
    engine = _engine_shell()
    engine.model_executors = {
        "whisper": _fake_executor(
            "whisper",
            model_type="FasterWhisper",
            whisper_probe={
                "measured": True,
                "peak_activation_bytes": 256,
                "per_window_bytes": 64,
                "reason": None,
            },
        )
    }

    payload = engine.check_health()

    assert payload["models"]["whisper"]["probe_measured"] is True
    assert payload["models"]["whisper"]["peak_activation_bytes"] == 256
    assert payload["whisper_vram"] is None


def test_health_endpoint_carries_whisper_vram_report(monkeypatch):
    import asyncio
    import json

    from harmonyspeech.endpoints.openai import api_server

    engine = _engine_shell()
    engine.whisper_vram_report = {"enabled": True, "status": "ok", "message": "fits"}
    tts_engine = SimpleNamespace(check_health=AsyncMock(return_value=engine.check_health()))
    healthy_engine = SimpleNamespace(check_health=AsyncMock(return_value=None))
    monkeypatch.setattr(api_server, "openai_serving_tts", SimpleNamespace(engine=tts_engine))
    monkeypatch.setattr(api_server, "openai_serving_stt", SimpleNamespace(engine=healthy_engine))
    monkeypatch.setattr(api_server, "openai_serving_vc", SimpleNamespace(engine=healthy_engine))
    monkeypatch.setattr(api_server, "openai_serving_embedding", SimpleNamespace(engine=healthy_engine))
    monkeypatch.setattr(api_server, "openai_serving_vad", SimpleNamespace(engine=healthy_engine))

    response = asyncio.run(api_server.health())

    payload = json.loads(response.body)
    assert payload["whisper_vram"]["status"] == "ok"


def test_model_nbytes_counts_native_checkpoint_directory(tmp_path):
    from harmonyspeech.modeling.weight_registry import model_nbytes

    (tmp_path / "model.bin").write_bytes(b"x" * 123)
    (tmp_path / "config.json").write_bytes(b"y" * 7)
    model = SimpleNamespace(model_path=str(tmp_path))

    assert model_nbytes(model) == 130


# ── Runtime admission hook ────────────────────────────────────────────────────


def test_get_whisper_admission_scales_against_budget():
    engine = _engine_shell()
    engine.whisper_vram_budget_bytes = 1000
    engine.model_executors = {
        "whisper": _fake_executor("whisper", model_type="FasterWhisper", max_batch_size=10)
    }
    stats = _whisper_stats(
        free_bytes=600,
        device_used_bytes=400,
        headroom_bytes=600,
        total_bytes=1000,
        models={
            "whisper": {
                "model_type": "FasterWhisper",
                "device": "cuda",
                "max_batch_size": 10,
                "per_window_bytes": 100,
                "peak_activation_bytes": 800,
                "weights_bytes": 0,
                "probe_measured": True,
            }
        },
    )
    engine.get_vram_stats = lambda: stats

    decision = engine.get_whisper_admission(active_windows=0)

    assert decision["enabled"] is True
    assert decision["budget_source"] == "config"
    assert decision["allowed_windows"] == 6
    assert decision["cap"] == 10
    assert decision["per_window_bytes"] == 100


def test_get_whisper_admission_uses_device_total_without_budget():
    engine = _engine_shell()
    engine.model_executors = {
        "whisper": _fake_executor("whisper", model_type="FasterWhisper", max_batch_size=4)
    }
    stats = _whisper_stats(
        free_bytes=300,
        device_used_bytes=700,
        headroom_bytes=300,
        total_bytes=1000,
        models={
            "whisper": {
                "model_type": "FasterWhisper",
                "device": "cuda",
                "max_batch_size": 4,
                "per_window_bytes": 100,
                "peak_activation_bytes": 100,
                "weights_bytes": 0,
                "probe_measured": True,
            }
        },
    )
    engine.get_vram_stats = lambda: stats

    decision = engine.get_whisper_admission()

    assert decision["budget_source"] == "device_total"
    assert decision["allowed_windows"] == 3


def test_get_whisper_admission_hard_stop_when_over_budget():
    engine = _engine_shell()
    engine.whisper_vram_budget_bytes = 1000
    engine.model_executors = {
        "whisper": _fake_executor("whisper", model_type="FasterWhisper", max_batch_size=10)
    }
    stats = _whisper_stats(
        free_bytes=0,
        device_used_bytes=2000,
        headroom_bytes=0,
        total_bytes=1000,
        models={
            "whisper": {
                "model_type": "FasterWhisper",
                "device": "cuda",
                "max_batch_size": 10,
                "per_window_bytes": 100,
                "peak_activation_bytes": 1000,
                "weights_bytes": 0,
                "probe_measured": True,
            }
        },
    )
    engine.get_vram_stats = lambda: stats

    assert engine.get_whisper_admission(min_windows=0)["allowed_windows"] == 0
    assert engine.get_whisper_admission()["allowed_windows"] == 1


def test_get_whisper_admission_is_disabled_without_whisper():
    engine = _engine_shell()

    decision = engine.get_whisper_admission()

    assert decision["enabled"] is False
    assert decision["allowed_windows"] == 0


# ── Config surface ────────────────────────────────────────────────────────────


def test_engine_config_loads_whisper_governance_keys(tmp_path):
    yaml_file = tmp_path / "config.yml"
    yaml_file.write_text(
        """
whisper_vram_budget_bytes: 1073741824
whisper_max_audio_seconds: 120
model_configs:
  - name: whisper
    model: tiny
    model_type: FasterWhisper
    max_batch_size: 8
    dtype: float32
    device_config:
      device: cpu
""",
        encoding="utf-8",
    )

    config = EngineConfig.load_config_from_yaml(str(yaml_file))

    assert config.whisper_vram_budget_bytes == 1073741824
    assert config.whisper_max_audio_seconds == 120.0


@pytest.mark.parametrize("budget", [0, -1, True, "1 GiB"])
def test_engine_config_rejects_invalid_whisper_budget(budget):
    with pytest.raises(ValueError, match="whisper_vram_budget_bytes"):
        EngineConfig(model_configs=[], whisper_vram_budget_bytes=budget)


@pytest.mark.parametrize("seconds", [0, -5])
def test_engine_config_rejects_invalid_whisper_audio_cap(seconds):
    with pytest.raises(ValueError, match="whisper_max_audio_seconds"):
        EngineConfig(model_configs=[], whisper_max_audio_seconds=seconds)


# ── Probe defaults stamped onto Whisper configs ──────────────────────────────


def test_engine_stamps_audio_cap_on_whisper_configs():
    engine = _engine_shell()
    whisper = SimpleNamespace(model_type="FasterWhisper")
    other = SimpleNamespace(model_type="ChatterboxTTS")
    engine.model_configs = [whisper, other]
    engine.whisper_max_audio_seconds = 42.0

    engine._apply_whisper_probe_defaults()

    assert whisper.whisper_max_audio_seconds == 42.0
    assert not hasattr(other, "whisper_max_audio_seconds")

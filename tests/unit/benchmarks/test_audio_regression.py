"""Unit tests for the audio regression harness (Phase 4-2)."""

import importlib.util
import os
import sys

import numpy as np
import pytest
import soundfile as sf

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_SCRIPT = os.path.join(_REPO_ROOT, "scripts", "audio_regression.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("audio_regression_script", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


ar = _load_module()


def _write(path, samples, sr=24000):
    sf.write(path, samples.astype(np.float32), sr, subtype="PCM_16")
    return str(path)


def _tone(seconds=1.0, freq=220.0, amplitude=0.3, sr=24000):
    t = np.arange(int(sr * seconds)) / sr
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_parse_axis_roundtrip():
    axis = ar.parse_axis("dtype:float32-vs-float16")
    assert axis.kind == "dtype"
    assert axis.side_a == "float32"
    assert axis.side_b == "float16"
    assert axis.label == "dtype:float32-vs-float16"
    assert axis.identity is False

    identity = ar.parse_axis("dtype:float32-vs-float32")
    assert identity.identity is True


@pytest.mark.parametrize("text", ["nonsense", "dtype:float32", "compile:a-vs-b"])
def test_parse_axis_rejects_bad_input(text):
    with pytest.raises(ValueError):
        ar.parse_axis(text)


def test_side_settings_map_axis_sides_to_config_knobs():
    dtype_axis = ar.parse_axis("dtype:float32-vs-float16")
    assert ar.side_settings(dtype_axis, "float16") == {"dtype": "float16", "share": True, "watermark": True}

    sharing = ar.parse_axis("sharing:dedup-vs-dup")
    assert ar.side_settings(sharing, "dedup") == {"dtype": "float32", "share": True, "watermark": False}
    assert ar.side_settings(sharing, "dup") == {"dtype": "float32", "share": False, "watermark": False}


def test_family_catalogue_is_internally_consistent():
    assert set(ar.FAMILY_SPECS) == {
        "chatterbox",
        "chatterbox_turbo",
        "chatterbox_multilingual",
        "openvoice_v1",
        "openvoice_v2",
        "melotts",
        "voicefixer",
        "harmonyspeech",
        "kittentts",
    }
    for family, spec in ar.FAMILY_SPECS.items():
        assert spec["models"], family
        assert spec["cases"], family
        configured = {entry["name"] for entry in spec["models"]}
        for case in spec["cases"]:
            assert case["model"] == family or case["model"] in configured, (family, case)


def test_identical_clips_are_bit_identical_and_pass(tmp_path):
    path = _write(tmp_path / "a.wav", _tone())

    metrics = ar.compute_metrics(path, path, "deterministic")

    assert metrics["identical"] is True
    assert metrics["duration_pct"] == 0.0
    assert metrics["rms_pct"] == 0.0
    assert metrics["logmel_mse"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["corr"] > 0.999
    assert metrics["passed"] is True


def test_deterministic_family_fails_on_a_different_signal(tmp_path):
    a = _write(tmp_path / "a.wav", _tone(freq=220.0))
    b = _write(tmp_path / "b.wav", _tone(freq=520.0))

    metrics = ar.compute_metrics(a, b, "deterministic")

    assert metrics["identical"] is False
    assert metrics["passed"] is False
    assert metrics["failures"]


def test_stochastic_family_uses_distribution_metrics(tmp_path):
    """Two noisy-but-similar realisations pass stochastic thresholds."""
    rng = np.random.default_rng(0)
    a = _tone() + rng.normal(0, 0.01, _tone().size).astype(np.float32)
    b = _tone() + rng.normal(0, 0.01, _tone().size).astype(np.float32)
    a = _write(tmp_path / "a.wav", a)
    b = _write(tmp_path / "b.wav", b)

    metrics = ar.compute_metrics(a, b, "stochastic")

    assert metrics["identical"] is False
    assert metrics["ltas_rmse_db"] < ar.THRESHOLDS["stochastic"]["ltas_rmse_db"]
    assert metrics["passed"] is True


def test_stochastic_family_still_fails_on_wrong_duration(tmp_path):
    a = _write(tmp_path / "a.wav", _tone(seconds=1.0))
    b = _write(tmp_path / "b.wav", _tone(seconds=3.0))

    metrics = ar.compute_metrics(a, b, "stochastic")

    assert metrics["passed"] is False
    assert any("duration" in failure for failure in metrics["failures"])


def test_sharing_axis_rejects_a_real_output_change(tmp_path):
    axis = ar.parse_axis("sharing:dedup-vs-dup")
    a = _write(tmp_path / "a.wav", _tone())
    b = _write(tmp_path / "b.wav", _tone() * 0.99)

    metrics = ar.compute_metrics(a, b, "stochastic")
    passed, failures = ar.verdict_for_axis(axis, metrics)

    assert passed is False
    assert any("Δsample" in failure for failure in failures)


def test_sharing_axis_tolerates_last_bit_kernel_noise(tmp_path):
    """Cross-process CUDA kernel noise (~1e-4) must not fail the sharing gate."""
    axis = ar.parse_axis("sharing:dedup-vs-dup")
    base = _tone()
    noisy = base.copy()
    noisy[::997] += 1e-4
    a = _write(tmp_path / "a.wav", base)
    b = _write(tmp_path / "b.wav", noisy)

    metrics = ar.compute_metrics(a, b, "stochastic")
    passed, failures = ar.verdict_for_axis(axis, metrics)

    assert metrics["identical"] is False
    assert metrics["max_abs_diff"] <= ar.SHARING_MAX_ABS_DIFF
    assert passed is True, failures


def test_identity_axis_still_requires_bit_identical_output(tmp_path):
    axis = ar.parse_axis("dtype:float32-vs-float32")
    base = _tone()
    noisy = base.copy()
    noisy[0] += 1e-4
    a = _write(tmp_path / "a.wav", base)
    b = _write(tmp_path / "b.wav", noisy)

    metrics = ar.compute_metrics(a, b, "deterministic")
    passed, failures = ar.verdict_for_axis(axis, metrics)

    assert passed is False
    assert "bit-identical" in failures[0]


def test_markdown_table_contains_every_case():
    report = {
        "timestamp_utc": "2026-11-04T00:00:00Z",
        "axis": "dtype:float32-vs-float16",
        "commit": "abc1234",
        "use_gpu": True,
        "env": {"torch": "2.9.1", "gpu": "Test GPU", "driver": "1.2.3"},
        "families": {
            "chatterbox": {
                "kind": "stochastic",
                "cases": {
                    "tts": {
                        "metrics": {
                            "duration_pct": 0.0,
                            "rms_pct": 1.0,
                            "logmel_mse": 2.0,
                            "ltas_rmse_db": 1.0,
                            "corr": 0.1,
                            "identical": False,
                        },
                        "passed": True,
                        "failures": [],
                    }
                },
            }
        },
    }

    table = ar._markdown_table(report)

    assert "chatterbox" in table
    assert "| tts | stochastic | PASS |" in table

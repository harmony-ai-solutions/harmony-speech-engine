"""Unit tests for the Phase 11-2 S3Gen autocast probe.

The probe needs CUDA + cached Chatterbox weights, so only its contract is
pinned here: the precision modes, the sample rate it reports against, and the
refusal to run without a CUDA device.
"""

from __future__ import annotations

import pytest

from scripts import bench_s3gen_autocast as probe


def test_probe_modes_match_the_environment_gate():
    assert probe.MODES == ("off", "bfloat16", "float16")
    assert probe.S3GEN_AUTOCAST_ENV == "HSE_S3GEN_AUTOCAST"
    assert probe._S3GEN_SR == 24000


def test_probe_refuses_to_run_without_cuda(monkeypatch):
    monkeypatch.setattr(probe.torch.cuda, "is_available", lambda: False)

    with pytest.raises(SystemExit) as excinfo:
        probe.main(["--mode", "timing"])

    assert "CUDA" in str(excinfo.value)


def test_listening_covers_the_seven_required_pairs():
    """Phase 11-3 names seven cases across the four S3Gen-vocode families."""
    names = {
        case["name"]
        for _group, _gate, cases in probe._LISTENING_GROUPS
        for case in cases
    }
    assert names == {
        "chatterbox_bundled",
        "chatterbox_clone",
        "vc_speech_to_b",
        "vc_speech_to_synth",
        "multilingual_en",
        "multilingual_de_accent",
        "turbo_bundled",
    }
    assert probe._LISTENING_SEED


def test_listening_readme_lists_files_metrics_and_gate(tmp_path):
    report = {
        "seed": 20261108,
        "device": "test-gpu",
        "torch": "2.x",
        "groups": [
            {
                "group": "chatterbox",
                "gate_note": "11-2 fp16 PASS, worst 0.099 dB²",
                "cases": [
                    {
                        "case": "chatterbox_bundled",
                        "label": "bundled voice",
                        "files": {
                            "fp32": str(tmp_path / "chatterbox_bundled_fp32.wav"),
                            "fp16": str(tmp_path / "chatterbox_bundled_fp16.wav"),
                        },
                        "seconds": {"fp32": 3.8, "fp16": 3.8},
                        "metrics": {
                            "logmel_mse": 0.0123,
                            "corr": 0.9991,
                            "rms_pct": 0.08,
                        },
                    }
                ],
            }
        ],
    }
    path = probe._write_listening_readme(str(tmp_path), report)
    text = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert path.endswith("README.md")
    assert "chatterbox_bundled_fp32.wav" in text
    assert "chatterbox_bundled_fp16.wav" in text
    assert "0.012300" in text
    assert "11-2 fp16 PASS" in text
    assert "bf16 rejected" in text

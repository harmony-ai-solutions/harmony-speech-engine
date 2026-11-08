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

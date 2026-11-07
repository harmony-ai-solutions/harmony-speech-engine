"""Unit tests for the VRAM measurement script plumbing (Phase 4-5)."""

import importlib.util
import os
import sys

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_SCRIPT = os.path.join(_REPO_ROOT, "scripts", "measure_vram.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("measure_vram_script", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


mv = _load_module()

STATS = {
    "cuda": True,
    "device": "Fake GPU",
    "total_bytes": 16 * 2**30,
    "allocated_bytes": 3 * 2**30,
    "reserved_bytes": 4 * 2**30,
    "peak_allocated_bytes": 5 * 2**30,
    "headroom_bytes": 12 * 2**30,
    "resident_models": 2,
    "models": {
        "chatterbox": {
            "model_type": "ChatterboxTTS",
            "device": "cuda",
            "dtype": "torch.float16",
            "vram_bytes": 2 * 2**30,
            "weights_bytes": 2 * 2**30,
            "backbone_group": "chatterbox-base",
        },
        "kitten-tts-mini": {
            "model_type": "KittenTTSSynthesizer",
            "device": "cpu",
            "dtype": "torch.float32",
            "vram_bytes": 0,
            "weights_bytes": 80 * 2**20,
            "backbone_group": None,
        },
    },
    "shared": {"entries": 2, "total_bytes": 2**30, "components": {}},
    "groups": {"chatterbox-base": 2**30},
}


def test_gib_conversion():
    assert mv.gib(2**30) == 1.0
    assert mv.gib(None) == 0.0


def test_format_table_lists_models_and_totals():
    table = mv.format_table(STATS, {"used_mib": 5120, "total_mib": 16311})

    assert "chatterbox" in table
    assert "kitten-tts-mini" in table
    assert "float16" in table
    assert "resident models: 2" in table
    assert "headroom (device free) 12.00 GiB" in table
    assert "nvidia-smi used 5.00 GiB of 15.93 GiB" in table


def test_format_table_without_smi_or_groups():
    stats = {**STATS, "groups": {}, "shared": {"entries": 0, "total_bytes": 0}}
    table = mv.format_table(stats, None)

    assert "nvidia-smi" not in table
    assert "shared backbones" not in table


def test_markdown_section_has_header_table_and_warmup():
    section = mv.markdown_section(
        "fp16 + dedup",
        STATS,
        {"used_mib": 5120, "total_mib": 16311},
        {"gpu": "Fake GPU", "torch": "2.9.1", "driver": "610.88", "platform": "test"},
        {"chatterbox": "ok 3.1s peak=6.00GiB"},
    )

    assert section.startswith("### fp16 + dedup")
    assert "| model | device | dtype | group | VRAM MiB | weights MiB |" in section
    assert "chatterbox-base" in section
    assert "Warmup peaks: chatterbox=ok 3.1s peak=6.00GiB" in section


def test_append_markdown_creates_file(tmp_path):
    path = tmp_path / "profiles.md"
    mv.append_markdown(str(path), "### run 1\n\nbody\n")

    text = path.read_text(encoding="utf-8")
    assert text.startswith("### run 1")
    assert text.endswith("\n")


def test_query_nvidia_smi_parses_output(monkeypatch):
    class _Proc:
        returncode = 0
        stdout = "5120, 16311\n"

    monkeypatch.setattr(mv.subprocess, "run", lambda *a, **k: _Proc())

    assert mv.query_nvidia_smi(0) == {"used_mib": 5120, "total_mib": 16311}


def test_query_nvidia_smi_is_tolerant(monkeypatch):
    def boom(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(mv.subprocess, "run", boom)
    assert mv.query_nvidia_smi(0) == {"used_mib": 0, "total_mib": 0}


def test_collect_vram_stats_rejects_non_dict():
    class _Engine:
        def get_vram_stats(self):
            return "nope"

    with pytest.raises(TypeError):
        mv.collect_vram_stats(_Engine())

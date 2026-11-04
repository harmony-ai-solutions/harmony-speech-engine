"""Unit tests for the benchmark measurement plumbing (Phase 2-5).

The real benchmark numbers need model weights; these tests prove the script's
measurement helpers are correct and that the artifact path performs no
serialization (the decode/encode-count assertions the phase requires).
"""


import numpy as np
import pytest

from harmonyspeech.common.artifacts import MelArtifact
from scripts import bench_pipeline_hops as bench


@pytest.mark.unit
def test_measure_returns_one_sample_per_repeat():
    calls = []

    def work():
        calls.append(1)

    samples = bench.measure(work, repeat=4)

    assert len(samples) == 4
    assert len(calls) == 4
    assert all(sample >= 0.0 for sample in samples)


@pytest.mark.unit
def test_summarize_reports_milliseconds():
    stats = bench.summarize([0.001, 0.002, 0.003])

    assert stats["n"] == 3
    assert stats["mean_ms"] == pytest.approx(2.0)
    assert stats["p50_ms"] == pytest.approx(2.0)
    assert stats["min_ms"] == pytest.approx(1.0)
    assert stats["max_ms"] == pytest.approx(3.0)


@pytest.mark.unit
def test_summarize_handles_empty_samples():
    assert bench.summarize([])["n"] == 0


@pytest.mark.unit
def test_legacy_mel_roundtrip_is_lossless_float32(monkeypatch):
    mel = np.random.default_rng(7).random((2, 20, 128), dtype=np.float32)
    encode_calls = []
    decode_calls = []
    real_dumps = bench.json.dumps
    real_b64encode = bench.base64.b64encode
    real_b64decode = bench.base64.b64decode

    monkeypatch.setattr(bench.json, "dumps", lambda *a, **k: encode_calls.append(1) or real_dumps(*a, **k))
    monkeypatch.setattr(bench.base64, "b64encode", lambda *a, **k: encode_calls.append(1) or real_b64encode(*a, **k))
    monkeypatch.setattr(bench.base64, "b64decode", lambda *a, **k: decode_calls.append(1) or real_b64decode(*a, **k))

    restored = bench.legacy_mel_roundtrip(mel)

    assert encode_calls  # JSON + base64 encode happened
    assert decode_calls
    np.testing.assert_allclose(restored, mel, rtol=1e-6)


@pytest.mark.unit
def test_artifact_passthrough_does_no_serialization(monkeypatch):
    mel = MelArtifact.from_numpy(np.zeros((128, 10), dtype=np.float32), 44100, 441)

    def explode(*args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("serialization used on the artifact path")

    monkeypatch.setattr(bench.json, "dumps", explode)
    monkeypatch.setattr(bench.base64, "b64encode", explode)
    monkeypatch.setattr(bench.base64, "b64decode", explode)

    assert bench.artifact_passthrough(mel) is mel


@pytest.mark.unit
def test_codec_counter_counts_encode_and_decode():
    with bench.CodecCounter() as counter:
        from harmonyspeech.common import audio_codec

        waveform = np.zeros(160, dtype=np.float32)
        encoded = audio_codec.encode_audio(waveform, 16000, "wav")
        audio_codec.decode_audio(encoded)

    assert counter.encode_calls == 1
    assert counter.decode_calls == 1


@pytest.mark.unit
def test_legacy_wav_roundtrip_returns_samples():
    waveform = bench._waveform_payload(seconds=0.1, sample_rate=16000)
    restored = bench.legacy_wav_roundtrip(waveform, 16000)

    assert restored.shape == waveform.shape


@pytest.mark.unit
def test_append_markdown_creates_header_and_appends_rows(tmp_path):
    path = tmp_path / "artifact-bus.md"
    row = {
        "label": "phase2",
        "mode": "stub",
        "family": "synthetic",
        "hop": "mel",
        "n": 3,
        "mean_ms": 1.5,
        "p50_ms": 1.4,
        "min_ms": 1.2,
        "max_ms": 1.9,
        "peak_mib": 2.0,
        "notes": "n",
    }

    bench.append_markdown(path, [row])
    bench.append_markdown(path, [row])

    text = path.read_text(encoding="utf-8")
    assert text.count("# Artifact bus") == 1
    assert text.count("| phase2 |") == 2
    assert "1.500" in text


@pytest.mark.unit
def test_stub_mode_reports_legacy_and_artifact_rows():
    args = bench.build_parser().parse_args(["--mode", "stub", "--iterations", "1", "--label", "unit"])
    rows = bench.run_stub_mode(args)

    hops = {row["hop"] for row in rows}
    assert {"voicefixer_mel_hop", "synth_to_tone_converter"} <= hops
    artifact_rows = [row for row in rows if "artifact bus" in row["notes"]]
    assert len(artifact_rows) == 2
    # The artifact path is a reference pass-through: effectively free.
    for row in artifact_rows:
        assert row["mean_ms"] < 5.0


@pytest.mark.unit
def test_parser_defaults():
    args = bench.build_parser().parse_args([])
    assert args.mode == "stub"
    assert args.family == "voicefixer"
    assert args.output.name == "artifact-bus.md"

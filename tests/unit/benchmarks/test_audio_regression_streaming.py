"""Unit tests for the Phase 14-5 streamed-vs-buffered regression axis.

The 2026-11-10 recalibration (owner decision) splits the axis into two classes:
the six families that pass sample-exactly keep the strict gate, and the two
OpenVoice clone pipelines get the stochastic distributional bounds plus a
streamed run1-vs-run2 determinism sub-check. These tests pin that split, the
verdict behaviour for each class, and the harness wiring that runs the second
streamed render.
"""

import base64
import importlib.util
import io
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from harmonyspeech.endpoints.openai.protocol import TextToSpeechResponse

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_SCRIPT = os.path.join(_REPO_ROOT, "scripts", "audio_regression.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("audio_regression_streaming_script", _SCRIPT)
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


def _broadband(seconds=1.0, sr=24000):
    """Deterministic broadband fixture (30 tones, fixed seeds).

    Rolling it by a few thousand samples decorrelates the waveform completely
    while leaving duration, level and the long-term average spectrum intact —
    the same premise as the real clone-pipeline divergence.
    """
    t = np.arange(int(sr * seconds)) / sr
    freqs = np.random.default_rng(2026).uniform(150.0, 6000.0, 30)
    phases = np.random.default_rng(7).uniform(0.0, 2 * np.pi, 30)
    signal = sum(
        np.sin(2 * np.pi * freq * t + phase)
        for freq, phase in zip(freqs, phases, strict=True)
    )
    return (0.3 * signal / np.max(np.abs(signal))).astype(np.float32)


def test_parse_streaming_axis_roundtrip_and_rejects_wrong_sides():
    axis = ar.parse_axis("streaming:off-vs-on")

    assert axis.kind == "streaming"
    assert axis.side_a == "off"
    assert axis.side_b == "on"
    assert axis.label == "streaming:off-vs-on"
    assert axis.identity is False

    with pytest.raises(ValueError, match="off-vs-on"):
        ar.parse_axis("streaming:on-vs-off")


def test_streaming_side_settings_share_seed_and_chunk_plan_and_disable_the_watermark():
    axis = ar.parse_axis("streaming:off-vs-on")

    off = ar.side_settings(axis, "off")
    on = ar.side_settings(axis, "on")

    assert off["streaming"] is False
    assert on["streaming"] is True
    assert {key: value for key, value in off.items() if key != "streaming"} == {
        key: value for key, value in on.items() if key != "streaming"
    }
    assert off["watermark"] is False and on["watermark"] is False
    assert off["split_text"] is True
    assert off["chunk_size"] == ar.LONGFORM_AXIS_CHUNK_SIZE
    assert off["seed"] == ar.LONGFORM_AXIS_SEED
    assert off["max_batch_size"] == 1


def test_streaming_axis_cases_are_multi_chunk_tts_cases_at_family_breadth():
    from harmonyspeech.common.audio_utils import split_text_into_chunks

    axis = ar.parse_axis("streaming:off-vs-on")

    for family in ar.STREAMING_AXIS_FAMILIES:
        cases = ar._cases_for_axis(axis, family)
        assert len(cases) == 1
        case = cases[0]
        assert case["name"].startswith("streamed_")
        assert "text_literal" in case and "text" not in case
        assert case["text_literal"] == ar.LONGFORM_CHUNKED_TEXT
        chunks = split_text_into_chunks(
            case["text_literal"],
            chunk_size=ar.LONGFORM_AXIS_CHUNK_SIZE,
            language=case.get("language"),
        )
        assert len(chunks) >= 2
        # The case still names a configured model / the family alias.
        configured = {entry["name"] for entry in ar.FAMILY_SPECS[family]["models"]}
        assert case["model"] == family or case["model"] in configured

    # VoiceFixer restores audio and has no TTS case.
    with pytest.raises(SystemExit, match="no TTS case"):
        ar._cases_for_axis(axis, "voicefixer")


def test_streaming_verdict_requires_bit_identical_deterministic_clips(tmp_path):
    axis = ar.parse_axis("streaming:off-vs-on")
    path = _write(tmp_path / "a.wav", _tone())
    metrics = ar.compute_metrics(path, path, "deterministic")

    passed, failures = ar.verdict_for_axis(axis, metrics, family="harmonyspeech")

    assert passed is True and failures == []

    different = _write(tmp_path / "b.wav", _tone(freq=520.0))
    metrics = ar.compute_metrics(path, different, "deterministic")
    passed, failures = ar.verdict_for_axis(axis, metrics, family="harmonyspeech")
    assert passed is False
    assert any("bit-identical" in failure for failure in failures)


def test_streaming_verdict_allows_the_documented_stochastic_noise_floor(tmp_path):
    axis = ar.parse_axis("streaming:off-vs-on")
    base = _tone()
    noisy = base.copy()
    noisy[::997] += 1e-4
    a = _write(tmp_path / "a.wav", base)
    b = _write(tmp_path / "b.wav", noisy)
    metrics = ar.compute_metrics(a, b, "stochastic")

    passed, failures = ar.verdict_for_axis(axis, metrics, family="chatterbox")

    assert metrics["identical"] is False
    assert passed is True, failures


def test_streaming_verdict_fails_a_frame_count_change(tmp_path):
    axis = ar.parse_axis("streaming:off-vs-on")
    a = _write(tmp_path / "a.wav", _tone(seconds=1.0))
    b = _write(tmp_path / "b.wav", _tone(seconds=1.2))
    metrics = ar.compute_metrics(a, b, "stochastic")

    passed, failures = ar.verdict_for_axis(axis, metrics, family="chatterbox")

    assert passed is False
    assert any("duration changed" in failure for failure in failures)


class _FakeServing:
    """Minimal serving stand-in for the streaming worker's two branches."""

    def __init__(self, samples: np.ndarray, rate: int):
        self.samples = samples
        self.rate = rate
        self.calls: list[str] = []

    async def _stream_audio_segments(
        self, *, raw_request, request_id, request_data, artifacts, chunks, state, requested_rate
    ):
        self.calls.append("stream")
        state.source_rate = self.rate
        state.target_rate = self.rate
        # One segment per chunk, mirroring the frame source's ordering.
        for _chunk in chunks:
            yield np.full(64, 0.1, dtype=np.float32)

    async def create_text_to_speech(self, request, raw_request):
        self.calls.append("buffered")
        buffer = io.BytesIO()
        sf.write(buffer, self.samples, self.rate, format="WAV", subtype="PCM_16")
        return TextToSpeechResponse(
            id="tts-fake",
            created=0,
            model=request.model,
            data=base64.b64encode(buffer.getvalue()).decode("ascii"),
            sample_rate=self.rate,
        )


def test_streaming_worker_renders_both_sides_with_the_same_chunk_plan(tmp_path):
    """Hermetic plumbing: both branches write a WAV and use the same chunks."""
    axis = ar.parse_axis("streaming:off-vs-on")
    case = ar._cases_for_axis(axis, "chatterbox")[0]
    text = ar.LONGFORM_CHUNKED_TEXT
    from harmonyspeech.common.audio_utils import split_text_into_chunks

    chunks = split_text_into_chunks(text, chunk_size=ar.LONGFORM_AXIS_CHUNK_SIZE)
    samples = np.concatenate(
        [np.full(64, 0.1, dtype=np.float32) for _ in chunks]
    )

    manifests = {}
    for side in ("off", "on"):
        settings = ar.side_settings(axis, side)
        job = {
            "family": "chatterbox",
            "side": side,
            "axis_kind": "streaming",
            "settings": settings,
            "use_gpu": False,
            "cases": [case],
            "out_dir": str(tmp_path / side),
            "manifest": str(tmp_path / f"{side}.json"),
        }
        os.makedirs(job["out_dir"], exist_ok=True)
        serving = _FakeServing(samples, 8000)
        manifests[side] = ar._run_streaming_worker(
            job, engine=None, serving_tts=serving, text_table={}
        )
        assert serving.calls == (["stream"] if side == "on" else ["buffered"])
        assert manifests[side][case["name"]]["samples"] == samples.size

    # Both sides wrote a readable WAV with the same sample count.
    metrics = ar.compute_metrics(
        manifests["off"][case["name"]]["wav"],
        manifests["on"][case["name"]]["wav"],
        "deterministic",
    )
    assert metrics["identical"] is True


# ---------------------------------------------------------------------------
# 2026-11-10 recalibration: clone pipelines are distributional + determinism
# ---------------------------------------------------------------------------
#: The measured streamed-vs-buffered metrics from
#: ``out/bench/gate-14-verify-streaming-axis.json`` (2026-11-10, RTX 5060 Ti).
#: Pinned verbatim: these are the numbers the owner decision is based on.
MEASURED_STREAMING_METRICS = {
    "openvoice_v1": {
        "duration_pct": 0.054,
        "rms_pct": 2.081,
        "logmel_mse": 43.81633,
        "ltas_rmse_db": 0.5927,
        "ltas_corr": 0.9995,
        "corr": 0.0206,
        "identical": False,
        "max_abs_diff": 0.45618698,
    },
    "openvoice_v2": {
        "duration_pct": 0.026,
        "rms_pct": 0.831,
        "logmel_mse": 20.941809,
        "ltas_rmse_db": 0.3676,
        "ltas_corr": 0.9999,
        "corr": 0.1045,
        "identical": False,
        "max_abs_diff": 0.66954184,
    },
}


def test_streaming_axis_splits_clone_pipelines_from_the_strict_six():
    assert frozenset({"openvoice_v1", "openvoice_v2"}) == ar.STREAMING_AXIS_DISTRIBUTIONAL
    assert frozenset(
        {
            "chatterbox",
            "chatterbox_turbo",
            "chatterbox_multilingual",
            "melotts",
            "harmonyspeech",
            "kittentts",
        }
    ) == ar.STREAMING_AXIS_STRICT
    # The sub-check runs for the whole axis, not only the distributional pair.
    assert frozenset(ar.STREAMING_AXIS_FAMILIES) == ar.STREAMING_DETERMINISM_FAMILIES
    assert set(ar.STREAMING_AXIS_FAMILIES) > ar.STREAMING_AXIS_DISTRIBUTIONAL


@pytest.mark.parametrize("family", sorted(ar.STREAMING_AXIS_DISTRIBUTIONAL))
def test_streaming_distributional_class_accepts_the_measured_clone_divergence(family):
    """The exact 2026-11-10 numbers must pass for the clone families.

    Waveform corr 0.02-0.10 and max |Δsample| 0.46-0.67 are decorrelated by
    construction (B=1 vs B=4 tone-converter numerics); LTAS corr ≥ 0.9995 and
    RMS ≤ 2.1 % are what make it inaudible.
    """
    axis = ar.parse_axis("streaming:off-vs-on")

    passed, failures = ar.verdict_for_axis(
        axis, dict(MEASURED_STREAMING_METRICS[family]), family=family
    )

    assert passed is True, failures


@pytest.mark.parametrize(
    "family", sorted(ar.STREAMING_AXIS_STRICT - ar.MODEL_SAMPLES_EVERY_RUN)
)
def test_streaming_strict_families_still_reject_the_clone_divergence(family):
    """Nothing was weakened for the other six: the same metrics still fail."""
    axis = ar.parse_axis("streaming:off-vs-on")

    passed, failures = ar.verdict_for_axis(
        axis, dict(MEASURED_STREAMING_METRICS["openvoice_v2"]), family=family
    )

    assert passed is False
    assert failures


def test_streaming_distributional_class_is_about_the_spectrum_not_the_samples(tmp_path):
    """Real audio: decorrelated waveform + identical LTAS passes only for OV."""
    axis = ar.parse_axis("streaming:off-vs-on")
    base = _broadband()
    a = _write(tmp_path / "a.wav", base)
    b = _write(tmp_path / "b.wav", np.roll(base, 5000))
    metrics = ar.compute_metrics(a, b, "stochastic")

    # The premise: same duration/level/spectrum, unrelated samples.
    assert metrics["duration_pct"] == 0.0
    assert metrics["identical"] is False
    assert metrics["ltas_corr"] >= ar.STREAMING_DISTRIBUTIONAL_LTAS_CORR_MIN
    assert metrics["corr"] < 0.5

    passed, failures = ar.verdict_for_axis(axis, metrics, family="openvoice_v1")
    assert passed is True, failures

    # The strict class is unchanged: the same clips fail a seeded family.
    passed_strict, failures_strict = ar.verdict_for_axis(axis, metrics, family="chatterbox")
    assert passed_strict is False
    assert any("max |Δsample|" in failure for failure in failures_strict)


def test_streaming_distributional_class_tolerates_the_edge_trim_length_shift(tmp_path):
    """A ±0.05 % length shift is the measured trim flip, not a wiring bug."""
    axis = ar.parse_axis("streaming:off-vs-on")
    base = _tone()
    a = _write(tmp_path / "a.wav", base)
    b = _write(tmp_path / "b.wav", base[:-12])  # 12/24000 = 0.05 %
    metrics = ar.compute_metrics(a, b, "stochastic")

    assert 0.0 < metrics["duration_pct"] <= ar.THRESHOLDS["stochastic"]["duration_pct"]

    passed, failures = ar.verdict_for_axis(axis, metrics, family="openvoice_v2")
    assert passed is True, failures

    passed_strict, failures_strict = ar.verdict_for_axis(axis, metrics, family="chatterbox")
    assert passed_strict is False
    assert any("duration changed" in failure for failure in failures_strict)


@pytest.mark.parametrize("family", sorted(ar.STREAMING_AXIS_DISTRIBUTIONAL))
def test_streaming_distributional_class_still_fails_a_real_regression(family):
    """A crossed chunk changes duration and LTAS; a spectral bug changes corr."""
    axis = ar.parse_axis("streaming:off-vs-on")
    base = dict(MEASURED_STREAMING_METRICS[family])

    for override, needle in (
        ({"duration_pct": 15.0}, "duration changed"),
        ({"ltas_rmse_db": 5.0}, "ltas rmse"),
        ({"ltas_corr": 0.90}, "ltas corr"),
        ({"rms_pct": 20.0}, "rms"),
    ):
        passed, failures = ar.verdict_for_axis(axis, dict(base, **override), family=family)
        assert passed is False
        assert any(needle in failure for failure in failures), failures


def test_streaming_distributional_case_label_records_the_phase_8_3_reason():
    axis = ar.parse_axis("streaming:off-vs-on")

    for family in ar.STREAMING_AXIS_DISTRIBUTIONAL:
        label = ar._cases_for_axis(axis, family)[0]["label"]
        assert "distributional" in label
        assert "B=1 vs B=4" in label
        assert "Phase 8-3" in label

    for family in ar.STREAMING_AXIS_STRICT:
        assert ar._cases_for_axis(axis, family)[0]["label"] == (
            "streamed vs buffered (multi-chunk, pre-watermark)"
        )


def test_streaming_determinism_entry_mirrors_the_determinism_axis_class(tmp_path):
    """Two streamed renders: bit-exact for deterministic, floor for seeded ones."""
    base = _broadband()
    a = _write(tmp_path / "run1.wav", base)

    identical = ar.streaming_determinism_entry(a, a, "openvoice_v2")
    assert identical["passed"] is True
    assert identical["metrics"]["identical"] is True

    # The documented cross-process CUDA kernel-selection floor (measured 4.5e-5
    # for openvoice_v2 on the buffered determinism axis).
    noisy = base.copy()
    noisy[::997] += 1e-4
    b = _write(tmp_path / "run2.wav", noisy)
    floor = ar.streaming_determinism_entry(a, b, "openvoice_v2")
    assert floor["metrics"]["identical"] is False
    assert floor["metrics"]["max_abs_diff"] <= ar.SHARING_MAX_ABS_DIFF
    assert floor["passed"] is True, floor["failures"]

    # The same floor is a failure for a deterministic family.
    assert ar.streaming_determinism_entry(a, b, "harmonyspeech")["passed"] is False

    # A nondeterministic stream (different samples, O(1) magnitude) fails.
    c = _write(tmp_path / "run3.wav", np.roll(base, 5000))
    broken = ar.streaming_determinism_entry(a, c, "openvoice_v1")
    assert broken["passed"] is False
    assert broken["failures"]

    # A model that samples on every inference is scored distributionally: the
    # waveform may move, the duration may not (KittenTTS, 2026-11-09 finding).
    assert ar.streaming_determinism_entry(a, c, "kittentts")["passed"] is True
    shorter = _write(tmp_path / "run4.wav", base[:-1200])
    failed = ar.streaming_determinism_entry(a, shorter, "kittentts")
    assert failed["passed"] is False
    assert any("duration changed" in failure for failure in failed["failures"])


def test_streaming_axis_renders_twice_and_folds_the_determinism_subcheck(monkeypatch, tmp_path):
    """The harness spawns one extra streamed worker and gates its comparison."""
    base = _broadband()
    run1 = _write(tmp_path / "run1.wav", base)
    run2_same = _write(tmp_path / "run2_same.wav", base)
    run2_other = _write(tmp_path / "run2_other.wav", np.roll(base, 5000))

    def run_harness_with(second_wav):
        calls = []
        reports = []

        def fake_run_side(family, side, axis, use_gpu, workdir, cases=None, *, run_tag=None):
            calls.append((side, run_tag))
            wav = second_wav if run_tag else run1
            return {"cases": {"streamed_case": {"wav": wav}}}

        args = SimpleNamespace(
            axis="streaming:off-vs-on",
            families="openvoice_v1",
            gpu=False,
            json_out=str(tmp_path / "report.json"),
            md_out=None,
            workdir=str(tmp_path / "work"),
        )
        monkeypatch.setattr(ar, "_run_side", fake_run_side)
        monkeypatch.setattr(
            ar, "_cases_for_axis", lambda axis, family: [{"name": "streamed_case"}]
        )
        monkeypatch.setattr(
            ar, "partition_optional_cases", lambda cases, available=None: (cases, [])
        )
        monkeypatch.setattr(ar, "_family_kind", lambda family: "stochastic")
        monkeypatch.setattr(ar, "_markdown_table", lambda report: reports.append(report) or "")
        monkeypatch.setattr(ar, "_gpu_description", lambda: {})
        monkeypatch.setattr(ar, "_write_listening_pack", lambda *args, **kwargs: None)

        exit_code = ar.run_harness(args)
        return calls, reports[0], exit_code

    calls, report, exit_code = run_harness_with(run2_same)

    # Side B is rendered a second time with identical settings, under its own
    # run tag so it cannot overwrite run 1.
    assert calls == [("off", None), ("on", None), ("on", "on_run2")]
    case = report["families"]["openvoice_v1"]["cases"]["streamed_case"]
    assert case["passed"] is True
    assert case["streaming_determinism"]["passed"] is True
    assert report["families"]["openvoice_v1"]["side_b_run2"] is not None
    assert exit_code == 0

    # A nondeterministic second render fails the case and the report, even
    # though the streamed-vs-buffered comparison itself is distributional.
    calls, report, exit_code = run_harness_with(run2_other)
    case = report["families"]["openvoice_v1"]["cases"]["streamed_case"]
    assert case["streaming_determinism"]["passed"] is False
    assert case["passed"] is False
    assert any("run1-vs-run2" in failure for failure in case["failures"])
    assert report["passed"] is False
    assert exit_code == 1

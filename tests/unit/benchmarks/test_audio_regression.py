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


# ---------------------------------------------------------------------------
# Post-review refinements (2026-11-05): VC intelligibility + optional fixtures
# ---------------------------------------------------------------------------
def test_compute_wer_identical_after_normalisation():
    assert ar.compute_wer("It's time to go home.", "its time to go home") == pytest.approx(1 / 5)
    assert ar.compute_wer("The quick brown fox", "the quick brown fox.") == 0.0


def test_compute_wer_counts_substitution_insertion_deletion():
    assert ar.compute_wer("one two three four", "one too three four") == pytest.approx(0.25)
    assert ar.compute_wer("one two three four", "one two three") == pytest.approx(0.25)
    assert ar.compute_wer("one two three", "one two three four") == pytest.approx(1 / 3)


def test_compute_wer_empty_reference():
    assert ar.compute_wer("", "") == 0.0
    assert ar.compute_wer("", "unexpected words") == 1.0


def test_partition_optional_cases_skips_only_missing_optional_fixture():
    required = {"name": "required", "reference": "missing"}
    optional_present = {"name": "optional_present", "reference": "present", "optional": True}
    optional_missing = {"name": "optional_missing", "reference": "missing", "optional": True}

    runnable, skipped = ar.partition_optional_cases(
        [required, optional_present, optional_missing],
        available=lambda name: name == "present",
    )

    assert [case["name"] for case in runnable] == ["required", "optional_present"]
    assert [case["name"] for case in skipped] == ["optional_missing"]


def test_reference_available_reflects_disk():
    assert ar.reference_available("repo_voice_a") is True
    assert ar.reference_available("definitely-not-a-fixture") is False


def test_vc_cases_are_real_speech_with_a_transcript_and_label():
    cases = {case["name"]: case for case in ar.FAMILY_SPECS["chatterbox"]["cases"]}

    primary = cases["vc_speech_a_to_b"]
    assert primary["kind"] == "vc"
    assert primary["reference"] == "repo_voice_a"  # wanda4 — real speech
    assert primary["target"] == "repo_voice_b"  # wanda5 — real speech
    assert primary["transcript"] == "en_wanda4"
    assert "intelligible" in primary["label"]

    secondary = cases["vc_speech_to_synth_voice"]
    assert secondary["reference"] == "repo_voice_a"
    assert secondary["target"] == "voice_b"  # synthetic timbre only
    assert secondary.get("wer_gate") is False  # non-speech target: report only
    assert "report" in secondary["label"].lower()


def test_multilingual_cases_label_accent_transfer_and_optional_native():
    cases = {case["name"]: case for case in ar.FAMILY_SPECS["chatterbox_multilingual"]["cases"]}

    accent = cases["tts_de_en_accent"]
    assert accent["language"] == "de"
    assert "reference" not in accent
    assert "accent transfer" in accent["label"].lower()
    assert "native" in accent["label"].lower()

    native = cases["tts_de_native"]
    assert native["optional"] is True
    assert native["reference"] == "repo_voice_de"
    assert "ref_voice_de.wav" in native["skip_message"]


def test_score_vc_intelligibility_reports_both_sides_and_soft_gate():
    case = {"name": "vc", "kind": "vc", "transcript": "en_wanda4", "language": "en"}
    transcripts = {
        "a.wav": "It's time to go home.",
        "b.wav": "At time to go home.",
    }

    scores = ar.score_vc_intelligibility(
        case, "a.wav", "b.wav", transcribe=lambda path, language: transcripts[path]
    )

    assert scores["wer_a"] == 0.0
    assert scores["wer_b"] == pytest.approx(0.2)
    assert scores["wer"] == pytest.approx(0.2)
    assert scores["warn"] is False
    assert scores["failed"] is False
    assert scores["reference_text"] == "It's time to go home."


def test_score_vc_intelligibility_hard_fails_only_above_threshold():
    case = {"name": "vc", "kind": "vc", "transcript": "en_wanda4", "language": "en"}
    # 4 of 5 reference words wrong -> WER 0.8 > 0.7 fail threshold.
    scores = ar.score_vc_intelligibility(
        case, "a.wav", "b.wav", transcribe=lambda path, language: "completely different words here now"
    )
    assert scores["failed"] is True

    # 3 of 5 wrong -> 0.6: warn only, not a hard failure.
    warn_scores = ar.score_vc_intelligibility(
        case, "a.wav", "b.wav", transcribe=lambda path, language: "time to go away today"
    )
    assert warn_scores["warn"] is True
    assert warn_scores["failed"] is False


def test_markdown_table_shows_wer_and_expectation_label():
    report = {
        "timestamp_utc": "2026-11-05T00:00:00Z",
        "axis": "dtype:float32-vs-float16",
        "commit": "def5678",
        "use_gpu": True,
        "env": {"torch": "2.9.1", "gpu": "Test GPU", "driver": "1.2.3"},
        "families": {
            "chatterbox": {
                "kind": "stochastic",
                "skipped_cases": [],
                "cases": {
                    "vc_speech_a_to_b": {
                        "metrics": {
                            "duration_pct": 0.0,
                            "rms_pct": 1.0,
                            "logmel_mse": 2.0,
                            "ltas_rmse_db": 1.0,
                            "corr": 0.1,
                            "identical": False,
                        },
                        "wer": {
                            "wer": 0.2,
                            "wer_a": 0.2,
                            "wer_b": 0.2,
                            "warn": False,
                            "failed": False,
                        },
                        "passed": True,
                        "failures": [],
                        "label": "real speech -> real speech timbre: content must stay intelligible",
                    }
                },
            }
        },
    }

    table = ar._markdown_table(report)

    assert "| WER |" in table
    assert "| 0.200 |" in table
    assert "content must stay intelligible" in table


def test_markdown_table_lists_skipped_optional_cases():
    report = {
        "timestamp_utc": "2026-11-05T00:00:00Z",
        "axis": "dtype:float32-vs-float16",
        "commit": "def5678",
        "use_gpu": True,
        "env": {"torch": "2.9.1", "gpu": "Test GPU", "driver": "1.2.3"},
        "families": {
            "chatterbox_multilingual": {
                "kind": "stochastic",
                "cases": {},
                "skipped_cases": [
                    {
                        "name": "tts_de_native",
                        "reason": "provide tests/audio_fixtures/ref_voice_de.wav",
                    }
                ],
            }
        },
    }

    table = ar._markdown_table(report)

    assert "Skipped optional cases" in table
    assert "tts_de_native" in table
    assert "ref_voice_de.wav" in table


# ---------------------------------------------------------------------------
# Batching-axis plumbing for the Phase 8 families (8-1)
# ---------------------------------------------------------------------------
def test_batching_axis_allow_list_is_registered_and_split_from_pending():
    from tests.audio_fixtures import REFERENCE_CLIPS, TEXTS

    assert set(ar.FAMILY_SPECS) >= ar.BATCHING_AXIS_ENABLED
    assert not (ar.BATCHING_AXIS_ENABLED & set(ar.BATCHING_AXIS_PENDING))
    for family in ar.BATCHING_AXIS_ENABLED | set(ar.BATCHING_AXIS_PENDING):
        cases = ar.FAMILY_SPECS[family].get("batching_cases")
        assert cases, family
        names = [case["name"] for case in cases]
        # The worker keys its manifest by case name, so duplicates would merge.
        assert len(set(names)) == len(names), family
        assert len(names) >= 2, f"{family}: a batch gate needs at least two distinct inputs"
        configured = {entry["name"] for entry in ar.FAMILY_SPECS[family]["models"]}
        for case in cases:
            assert case["kind"] in {"tts", "vc", "restore"}, (family, case)
            assert case["model"] == family or case["model"] in configured, (family, case)
            if case.get("reference"):
                assert case["reference"] in REFERENCE_CLIPS, (family, case)
            if case.get("text"):
                assert case["text"] in TEXTS, (family, case)


def test_batching_axis_covers_every_family_with_a_batched_handler():
    """8-4 enabled the last two families; nothing is left waiting."""
    assert ar.BATCHING_AXIS_PENDING == {}
    assert {
        "openvoice_v1",
        "openvoice_v2",
        "melotts",
        "harmonyspeech",
        "voicefixer",
    } == ar.BATCHING_AXIS_ENABLED
    for family in ("harmonyspeech", "voicefixer"):
        cases = ar._cases_for_axis(ar.parse_axis("batching:1-vs-4"), family)
        assert cases is ar.FAMILY_SPECS[family]["batching_cases"]


def test_batching_axis_returns_the_registered_cases_for_an_enabled_family():
    axis = ar.parse_axis("batching:1-vs-4")

    cases = ar._cases_for_axis(axis, "openvoice_v1")

    assert cases is ar.FAMILY_SPECS["openvoice_v1"]["batching_cases"]


def test_batching_axis_covers_the_openvoice_v2_clone_pipeline():
    """8-2 activates the V2 family: the full clone pipeline (VAD → embed → synth → TC)."""
    axis = ar.parse_axis("batching:1-vs-4")

    cases = ar._cases_for_axis(axis, "openvoice_v2")

    assert cases is ar.FAMILY_SPECS["openvoice_v2"]["batching_cases"]
    assert all(case["kind"] == "tts" for case in cases)
    assert all(case.get("reference") for case in cases)


@pytest.mark.parametrize("family", ["harmonyspeech"])
def test_batching_axis_refuses_a_registered_but_disabled_family(monkeypatch, family):
    """The refusal mechanism stays covered even though the pending set is empty."""
    monkeypatch.setattr(ar, "BATCHING_AXIS_ENABLED", frozenset())
    monkeypatch.setattr(ar, "BATCHING_AXIS_PENDING", {family: "9-9 (synthetic pending entry)"})
    axis = ar.parse_axis("batching:1-vs-4")

    with pytest.raises(SystemExit) as excinfo:
        ar._cases_for_axis(axis, family)

    message = str(excinfo.value)
    assert "does not batch yet" in message
    assert ar.BATCHING_AXIS_PENDING[family].split(" ")[0] in message


def test_non_batching_axes_still_use_the_family_cases():
    axis = ar.parse_axis("dtype:float32-vs-float16")

    assert ar._cases_for_axis(axis, "voicefixer") is ar.FAMILY_SPECS["voicefixer"]["cases"]


def test_batching_worker_builds_a_restore_request_input():
    """The voicefixer gate runs `restore` cases through the same worker."""
    from harmonyspeech.common.inputs import AudioConversionRequestInput

    case = {
        "name": "restore_voice_a",
        "kind": "restore",
        "model": "voicefixer",
        "reference": "voice_a",
    }

    request_input = ar._build_batching_request_input(
        case, "batch-gate-restore_voice_a", {}, deterministic=True
    )

    assert isinstance(request_input, AudioConversionRequestInput)
    assert request_input.request_id == "batch-gate-restore_voice_a"
    assert request_input.source_audio


def test_batching_worker_builds_a_deterministic_tts_request_input():
    from harmonyspeech.common.inputs import TextToSpeechRequestInput
    from tests.audio_fixtures import TEXTS

    case = {
        "name": "batch_short",
        "kind": "tts",
        "model": "harmonyspeech",
        "text": "batch_1",
        "mode": "voice_cloning",
        "reference": "repo_voice_a",
    }

    request_input = ar._build_batching_request_input(
        case, "batch-gate-batch_short", TEXTS, deterministic=True
    )

    assert isinstance(request_input, TextToSpeechRequestInput)
    assert request_input.input_text == TEXTS["batch_1"]
    assert request_input.generation_options.noise_scale == 0.0


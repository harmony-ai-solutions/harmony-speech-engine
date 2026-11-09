"""Unit tests for the Phase 12-4 long-form regression axes (audio_regression)."""

import importlib.util
import os
import sys

import numpy as np
import pytest
import soundfile as sf

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_SCRIPT = os.path.join(_REPO_ROOT, "scripts", "audio_regression.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("audio_regression_longform_script", _SCRIPT)
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


def test_parse_longform_axis_roundtrip_and_rejects_wrong_sides():
    axis = ar.parse_axis("longform:off-vs-on")

    assert axis.kind == "longform"
    assert axis.side_a == "off"
    assert axis.side_b == "on"
    assert axis.identity is False

    with pytest.raises(ValueError, match="off-vs-on"):
        ar.parse_axis("longform:on-vs-off")


def test_parse_determinism_axis_roundtrip_and_rejects_wrong_sides():
    axis = ar.parse_axis("determinism:run1-vs-run2")

    assert axis.kind == "determinism"
    assert axis.label == "determinism:run1-vs-run2"

    with pytest.raises(ValueError, match="run1-vs-run2"):
        ar.parse_axis("determinism:run2-vs-run1")


def test_longform_side_settings_only_flip_split_text():
    axis = ar.parse_axis("longform:off-vs-on")

    off = ar.side_settings(axis, "off")
    on = ar.side_settings(axis, "on")

    assert off["split_text"] is False
    assert on["split_text"] is True
    assert {key: value for key, value in off.items() if key != "split_text"} == {
        key: value for key, value in on.items() if key != "split_text"
    }
    # Watermark is off so Perth's cross-process conv noise cannot mask identity.
    assert off["watermark"] is False and on["watermark"] is False


def test_determinism_side_settings_are_identical_for_both_runs():
    axis = ar.parse_axis("determinism:run1-vs-run2")

    assert ar.side_settings(axis, "run1") == ar.side_settings(axis, "run2")
    settings = ar.side_settings(axis, "run1")
    assert settings["split_text"] is True
    assert settings["chunk_size"] == ar.LONGFORM_AXIS_CHUNK_SIZE
    assert settings["seed"] == ar.LONGFORM_AXIS_SEED


def test_longform_axis_cases_are_short_tts_cases_at_family_breadth():
    axis = ar.parse_axis("longform:off-vs-on")

    for family in ar.LONGFORM_AXIS_FAMILIES:
        cases = ar._cases_for_axis(axis, family)
        assert cases
        assert all(case.get("kind", "tts") == "tts" for case in cases)

    # VoiceFixer restores audio and has no TTS case; it is outside the axis.
    with pytest.raises(SystemExit, match="no short TTS case"):
        ar._cases_for_axis(axis, "voicefixer")


def test_identity_axis_texts_fit_in_one_chunk_at_the_serving_budget():
    """The byte-identity premise, checked for every family's case texts.

    A text that splits into 2+ chunks makes ``split_text`` on/off legitimately
    different (different synthesis boundaries), so such a case must be excluded
    with a recorded reason rather than gated.
    """
    from harmonyspeech.common.audio_utils import split_text_into_chunks
    from tests.audio_fixtures import TEXTS

    axis = ar.parse_axis("longform:off-vs-on")
    for family in ar.LONGFORM_AXIS_FAMILIES:
        for case in ar._cases_for_axis(axis, family):
            text = ar._case_text(case, TEXTS)
            chunks = split_text_into_chunks(
                text, chunk_size=ar.LONGFORM_IDENTITY_CHUNK_BUDGET, language=case.get("language")
            )
            if len(chunks) > 1:
                assert case.get("skip_reason"), (family, case["name"], chunks)
            else:
                assert not case.get("skip_reason"), (family, case["name"], chunks)

    # The current corpus is all single-chunk, so the gate really runs today.
    runnable, skipped = ar.partition_optional_cases(
        ar._cases_for_axis(axis, "kittentts"), available=lambda _name: True
    )
    assert runnable and not skipped


def test_identity_budget_matches_the_serving_default():
    """The harness must split exactly where the serving layer splits."""
    from harmonyspeech.endpoints.openai.serving_text_to_speech import DEFAULT_CHUNK_SIZE

    assert ar.LONGFORM_IDENTITY_CHUNK_BUDGET == DEFAULT_CHUNK_SIZE


def test_identity_axis_excludes_a_multi_chunk_text_with_a_recorded_reason(monkeypatch):
    """A multi-sentence case cannot gate identity; it is skipped, never failed."""
    monkeypatch.setitem(
        ar.FAMILY_SPECS,
        "fake_multi",
        {
            "cases": [
                {
                    "name": "many_sentences",
                    "kind": "tts",
                    "model": "fake",
                    "text_literal": ar.LONGFORM_CHUNKED_TEXT,
                }
            ]
        },
    )

    cases = ar._cases_for_axis(ar.parse_axis("longform:off-vs-on"), "fake_multi")

    assert len(cases) == 1
    reason = cases[0]["skip_reason"]
    assert "chunks at the" in reason
    runnable, skipped = ar.partition_optional_cases(cases, available=lambda _name: True)
    assert runnable == []
    assert skipped == cases


def test_identity_axis_retargets_a_vad_undetectable_reference(monkeypatch):
    """OpenVoice clone cases use a real-speech clip (the VAD cannot hear tones)."""
    monkeypatch.setattr(
        ar, "VAD_CARRIED_FAMILIES", ar.VAD_CARRIED_FAMILIES | {"fake_ov"}
    )
    monkeypatch.setitem(
        ar.FAMILY_SPECS,
        "fake_ov",
        {
            "cases": [
                {
                    "name": "clone_synthetic",
                    "kind": "tts",
                    "model": "fake",
                    "text": "en_short",
                    "reference": "voice_b",
                    "label": "synthetic reference",
                }
            ]
        },
    )

    cases = ar._cases_for_axis(ar.parse_axis("longform:off-vs-on"), "fake_ov")

    assert cases[0]["reference"] == "repo_voice_b"
    assert "voice_b -> repo_voice_b" in cases[0]["label"]
    assert not cases[0].get("skip_reason")


def test_identity_axis_keeps_synthetic_references_for_non_vad_families():
    """Chatterbox conditions on the clip directly; its synthetic case stays."""
    axis = ar.parse_axis("longform:off-vs-on")

    chatterbox = {case["name"]: case for case in ar._cases_for_axis(axis, "chatterbox")}

    assert chatterbox["tts_clone_voice_a"]["reference"] == "voice_a"


def test_partition_optional_cases_reports_a_skip_reason():
    cases = [
        {"name": "run", "text": "en_short"},
        {"name": "excluded", "text": "en_short", "skip_reason": "not gateable"},
    ]

    runnable, skipped = ar.partition_optional_cases(cases, available=lambda _name: True)

    assert [case["name"] for case in runnable] == ["run"]
    assert [case["name"] for case in skipped] == ["excluded"]


def test_determinism_axis_case_uses_a_multi_chunk_literal_text():
    from harmonyspeech.common.audio_utils import split_text_into_chunks

    axis = ar.parse_axis("determinism:run1-vs-run2")

    cases = ar._cases_for_axis(axis, "chatterbox")
    assert len(cases) == 1
    case = cases[0]
    assert "text_literal" in case and "text" not in case
    assert case["text_literal"] == ar.LONGFORM_CHUNKED_TEXT
    assert len(split_text_into_chunks(ar.LONGFORM_CHUNKED_TEXT, chunk_size=ar.LONGFORM_AXIS_CHUNK_SIZE)) > 1


def test_request_generation_options_carry_split_text_and_seed():
    options = ar._request_generation_options(
        {"seed": 1234, "split_text": True, "chunk_size": 120, "crossfade_ms": 0, "sentence_pause_ms": 0}
    )

    assert options.seed == 1234
    assert options.split_text is True
    assert options.chunk_size == 120
    assert ar._request_generation_options({}) is None


def test_case_text_prefers_an_inline_literal():
    assert ar._case_text({"text_literal": "inline"}, {"en_short": "table"}) == "inline"
    assert ar._case_text({"text": "en_short"}, {"en_short": "table"}) == "table"


def test_longform_identity_verdict_requires_bit_identical_clips(tmp_path):
    axis = ar.parse_axis("longform:off-vs-on")
    path = _write(tmp_path / "a.wav", _tone())
    metrics = ar.compute_metrics(path, path, "deterministic")

    passed, failures = ar.verdict_for_axis(axis, metrics, family="harmonyspeech")

    assert passed is True and failures == []


def test_longform_identity_verdict_fails_a_changed_deterministic_clip(tmp_path):
    axis = ar.parse_axis("longform:off-vs-on")
    a = _write(tmp_path / "a.wav", _tone(freq=220.0))
    b = _write(tmp_path / "b.wav", _tone(freq=520.0))
    metrics = ar.compute_metrics(a, b, "deterministic")

    passed, failures = ar.verdict_for_axis(axis, metrics, family="harmonyspeech")

    assert passed is False
    assert any("bit-identical" in failure for failure in failures)


def test_identity_verdict_is_distributional_for_a_sampling_model(tmp_path):
    """A model that samples every run cannot be bit-exact (2026-11-09 finding).

    Same duration and spectrum, different sample realisation: the gate must
    pass for KittenTTS (``MODEL_SAMPLES_EVERY_RUN``) and still fail for a
    deterministic family, whose dispatcher identity is a real invariant.
    """
    axis = ar.parse_axis("longform:off-vs-on")
    base = _tone()
    shifted = np.roll(base, 37)
    a = _write(tmp_path / "a.wav", base)
    b = _write(tmp_path / "b.wav", shifted)
    metrics = ar.compute_metrics(a, b, "stochastic")

    passed, failures = ar.verdict_for_axis(axis, metrics, family="kittentts")
    assert metrics["identical"] is False
    assert passed is True, failures

    passed_det, failures_det = ar.verdict_for_axis(axis, metrics, family="harmonyspeech")
    assert passed_det is False
    assert any("bit-identical" in failure for failure in failures_det)


def test_determinism_verdict_fails_a_duration_change_for_a_sampling_model(tmp_path):
    """Duration is the invariant that catches a crossed/duplicated chunk."""
    axis = ar.parse_axis("determinism:run1-vs-run2")
    a = _write(tmp_path / "a.wav", _tone(seconds=1.0))
    b = _write(tmp_path / "b.wav", _tone(seconds=1.2))
    metrics = ar.compute_metrics(a, b, "stochastic")

    passed, failures = ar.verdict_for_axis(axis, metrics, family="kittentts")

    assert passed is False
    assert any("duration changed" in failure for failure in failures)


def test_longform_identity_verdict_allows_the_documented_stochastic_noise_floor(tmp_path):
    axis = ar.parse_axis("longform:off-vs-on")
    base = _tone()
    noisy = base.copy()
    noisy[::997] += 1e-4
    a = _write(tmp_path / "a.wav", base)
    b = _write(tmp_path / "b.wav", noisy)
    metrics = ar.compute_metrics(a, b, "stochastic")

    passed, failures = ar.verdict_for_axis(axis, metrics, family="chatterbox")

    assert metrics["identical"] is False
    assert passed is True, failures


def test_determinism_verdict_fails_a_different_seeded_render(tmp_path):
    axis = ar.parse_axis("determinism:run1-vs-run2")
    a = _write(tmp_path / "a.wav", _tone(freq=220.0))
    b = _write(tmp_path / "b.wav", _tone(freq=260.0))
    metrics = ar.compute_metrics(a, b, "stochastic")

    passed, failures = ar.verdict_for_axis(axis, metrics, family="chatterbox")

    assert passed is False
    assert failures

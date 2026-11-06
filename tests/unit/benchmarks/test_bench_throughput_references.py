"""Unit tests for the throughput harness load shape (Phase 7-3 / 8-6).

The measurement itself needs a GPU and a running server; these tests prove the
*load shape* the recorded tables depend on:

* ``--clone-reference same`` really reuses one clip on every request (so the
  embedding cache can hit), while ``distinct`` really gives every request its
  own clip with a unique ``content_hash`` (so the cache must miss) — both at the
  original duration and sample rate, i.e. still valid speech-shaped audio for
  VAD/embed (Phase 7-3);
* the Phase 8 scenario matrix, concurrency-list parsing, client-pool sizing,
  server-side cache verification and the per-step composition parser (8-6).
"""

import argparse
import base64

import numpy as np
import pytest

from harmonyspeech.common import audio_codec
from harmonyspeech.common.artifacts import AudioArtifact
from scripts import bench_throughput as bench


def _reference_b64(seconds: float = 1.0, sample_rate: int = 16000) -> str:
    """A deterministic speech-shaped stand-in for a `tests/audio_fixtures` clip."""
    t = np.arange(int(sample_rate * seconds)) / sample_rate
    envelope = 0.55 + 0.45 * np.sin(2 * np.pi * 2.7 * t) ** 2
    waveform = (0.2 * envelope * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    return base64.b64encode(audio_codec.encode_audio(waveform, sample_rate, "wav")).decode("ascii")


REFERENCE = _reference_b64()


def _artifact(reference_b64: str) -> AudioArtifact:
    waveform, sample_rate = audio_codec.decode_wav_b64(reference_b64)
    return AudioArtifact.from_numpy(waveform, sample_rate)


@pytest.mark.unit
def test_parser_defaults_to_the_same_reference_mode():
    args = bench.build_parser().parse_args([])

    assert args.clone_reference == "same"
    assert args.reference == "repo_voice_a"


@pytest.mark.unit
def test_parser_rejects_an_unknown_reference_mode():
    with pytest.raises(SystemExit):
        bench.build_parser().parse_args(["--clone-reference", "mixed"])


@pytest.mark.unit
def test_same_mode_sends_one_identical_payload_to_every_request():
    payloads = bench._reference_payloads("openvoice_v2", REFERENCE, "same", 5)

    assert len(payloads) == 5
    assert {payload["input_audio"] for payload in payloads} == {REFERENCE}
    assert len({_artifact(payload["input_audio"]).content_hash for payload in payloads}) == 1
    assert {payload["mode"] for payload in payloads} == {"voice_cloning"}


@pytest.mark.unit
def test_distinct_mode_produces_one_unique_payload_per_request():
    payloads = bench._reference_payloads("openvoice_v2", REFERENCE, "distinct", 5)
    artifacts = [_artifact(payload["input_audio"]) for payload in payloads]

    assert len({payload["input_audio"] for payload in payloads}) == 5
    assert len({artifact.content_hash for artifact in artifacts}) == 5

    base = _artifact(REFERENCE)
    for artifact in artifacts:
        # Speech-length audio at the source rate: VAD/embedding still work.
        assert artifact.sample_rate == base.sample_rate
        assert artifact.num_samples == base.num_samples
    # Request 0 is the untouched clip, so `same` and `distinct` share a baseline.
    assert artifacts[0].content_hash == base.content_hash


@pytest.mark.unit
def test_distinct_mode_is_deterministic_across_runs():
    first = bench._reference_payloads("openvoice_v2", REFERENCE, "distinct", 4)
    second = bench._reference_payloads("openvoice_v2", REFERENCE, "distinct", 4)

    assert first == second


@pytest.mark.unit
def test_families_without_reference_audio_ignore_the_mode():
    same = bench._reference_payloads("melo", REFERENCE, "same", 4)
    distinct = bench._reference_payloads("melo", REFERENCE, "distinct", 4)

    assert same == distinct
    assert "input_audio" not in same[0]


@pytest.mark.unit
def test_unknown_mode_is_rejected_by_the_payload_builder():
    with pytest.raises(ValueError):
        bench._reference_payloads("openvoice_v2", REFERENCE, "nope", 2)


# ── Phase 8: scenario matrix, concurrency list, cache verification ────────────


@pytest.mark.unit
def test_every_clone_family_builds_voice_cloning_payloads():
    for family in ("openvoice_v2", "openvoice_v1", "harmonyspeech"):
        payloads = bench._reference_payloads(family, REFERENCE, "same", 2)

        assert len(payloads) == 2
        for payload in payloads:
            assert payload["mode"] == "voice_cloning"
            assert payload["input_audio"] == REFERENCE
            assert payload["model"] == bench.FAMILY_SPECS[family]["model"]


@pytest.mark.unit
def test_restoration_and_kitten_families_carry_the_right_audio_fields():
    restore = bench._reference_payloads("voicefixer", REFERENCE, "same", 2)
    assert {"source_audio"} <= set(restore[0])
    assert "input_audio" not in restore[0]

    kitten = bench._reference_payloads("kitten", REFERENCE, "distinct", 4)
    assert len(kitten) == 1  # no reference audio: one payload reused
    assert kitten[0]["mode"] == "single_speaker_tts"
    assert kitten[0]["voice"] == "Jasper"


@pytest.mark.unit
def test_scenario_matrix_is_same_distinct_times_cache_on_off():
    matrix = bench.clone_scenario_matrix("repo_voice_b")

    assert [(cell["clone_reference"], cell["cache"]) for cell in matrix] == [
        ("same", "on"),
        ("same", "off"),
        ("distinct", "on"),
        ("distinct", "off"),
    ]
    assert [cell["label"] for cell in matrix] == [
        "phase8-same-cache-on",
        "phase8-same-cache-off",
        "phase8-distinct-cache-on",
        "phase8-distinct-cache-off",
    ]
    assert {cell["reference"] for cell in matrix} == {"repo_voice_b"}


@pytest.mark.unit
def test_clone_scenario_payloads_match_the_reference_mode():
    """Each matrix cell's payloads must exercise the cache state it names."""
    for cell in bench.clone_scenario_matrix():
        payloads = bench._reference_payloads(
            "openvoice_v2", REFERENCE, cell["clone_reference"], 4
        )
        hashes = {_artifact(payload["input_audio"]).content_hash for payload in payloads}
        if cell["clone_reference"] == "same":
            # A cache-on row can only hit when the clip is identical every time.
            assert len(hashes) == 1
        else:
            assert len(hashes) == 4


@pytest.mark.unit
def test_concurrency_accepts_a_comma_list_and_a_single_level():
    assert bench.parse_concurrency_levels("1,4,20") == [1, 4, 20]
    assert bench.parse_concurrency_levels("8") == [8]
    assert bench.parse_concurrency_levels(" 1 , 4 ,1 ") == [1, 4]


@pytest.mark.unit
@pytest.mark.parametrize("value", ["0", "-2", "1,x", "", ","])
def test_concurrency_rejects_garbage_levels(value):
    with pytest.raises(argparse.ArgumentTypeError):
        bench.parse_concurrency_levels(value)


@pytest.mark.unit
def test_parser_keeps_the_phase_5_3_backward_compatible_concurrency():
    args = bench.build_parser().parse_args(["--concurrency", "1,4,20"])

    assert args.concurrency == [1, 4, 20]
    assert args.embedding_cache == "auto"
    assert args.engine_log is None


@pytest.mark.unit
def test_kitten_lane_comparison_is_expressible_from_the_parser():
    """The lanes curve is one family, c=8, against two server configs.

    The two sides differ only in the server's `max_batch_size` (1 = sequential,
    4 = lanes), so the harness itself needs no extra knob.
    """
    args = bench.build_parser().parse_args(["--families", "kitten", "--concurrency", "8"])

    assert args.families == "kitten"
    assert args.concurrency == [8]
    payloads = bench._reference_payloads("kitten", REFERENCE, "same", args.requests)
    assert len(payloads) == 1
    assert payloads[0]["model"] == "kitten-tts-mini"
    assert payloads[0]["mode"] == "single_speaker_tts"


@pytest.mark.unit
def test_client_pool_is_sized_from_the_highest_concurrency_level():
    assert bench.http_client_limits([1, 4, 200]) == (200, 200)
    # An explicit cap is honoured, so the client ceiling can be measured on purpose.
    assert bench.http_client_limits([1, 4, 200], 100) == (100, 100)
    assert bench.http_client_limits([4]) == (4, 4)


@pytest.mark.unit
def test_cache_verification_distinguishes_unverifiable_states():
    assert bench.cache_state_matches("on", True) is True
    assert bench.cache_state_matches("on", False) is False
    assert bench.cache_state_matches("off", False) is True
    assert bench.cache_state_matches("auto", False) is None
    assert bench.cache_state_matches("on", None) is None

    with pytest.raises(ValueError):
        bench.cache_state_matches("maybe", True)


@pytest.mark.unit
def test_scenario_recipe_prints_two_boots_and_four_runs():
    args = bench.build_parser().parse_args([])
    recipe = bench.scenario_recipe(args)

    assert "--no-embedding-cache" in recipe  # the server-side toggle
    assert recipe.count("--embedding-cache on") == 2
    assert recipe.count("--embedding-cache off") == 2
    for label in (
        "phase8-same-cache-on",
        "phase8-same-cache-off",
        "phase8-distinct-cache-on",
        "phase8-distinct-cache-off",
    ):
        assert label in recipe


# ── Phase 8: per-step sub-batch composition ───────────────────────────────────

STEP_LOG_LINES = [
    "DEBUG - OpenVoice tone-converter encoder: 12 segment(s) from 4 request(s) "
    "forwarded in 2 padded sub-batch(es)",
    "DEBUG - OpenVoice tone-converter encoder: 6 segment(s) from 2 request(s) "
    "forwarded in 1 padded sub-batch(es)",
    "DEBUG - OpenVoice V1: 4 sentence(s) from 4 request(s) forwarded in 1 padded sub-batch(es)",
    "DEBUG - OpenVoice tone converter: 4 request(s) forwarded in 2 padded sub-batch(es)",
    "DEBUG - KittenTTS: 4 request(s) synthesised in 4 lane(s)",
    "DEBUG - some unrelated line",
]


@pytest.mark.unit
def test_step_composition_tallies_padded_sub_batches_per_step():
    composition = bench.parse_step_composition(STEP_LOG_LINES)

    assert composition["embed"] == {
        "steps": 2,
        "sub_batches": 3,
        "rows": 18,
        "requests": 6,
        "avg_sub_batches_per_step": 1.5,
    }
    assert composition["synth"]["sub_batches"] == 1
    assert composition["synth"]["avg_sub_batches_per_step"] == 1.0
    # A line without a row count (tone converters pad per request) still counts.
    assert composition["tc"]["steps"] == 1
    assert composition["tc"]["rows"] == 0
    assert composition["lanes"]["lanes"] == 4
    assert set(composition) == {"embed", "synth", "tc", "lanes"}


@pytest.mark.unit
def test_step_composition_ignores_lines_without_a_padded_forward():
    assert bench.parse_step_composition(["DEBUG - nothing to see here", ""]) == {}


@pytest.mark.unit
def test_engine_log_tail_only_reads_what_the_level_appended(tmp_path):
    log_path = tmp_path / "engine.log"
    log_path.write_text("warmup line\n", encoding="utf-8")

    tail = bench.EngineLogTail(str(log_path))
    tail.flush()  # warmup must not count towards the level
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(STEP_LOG_LINES[0] + "\n")

    assert tail.drain() == [STEP_LOG_LINES[0]]
    assert tail.drain() == []

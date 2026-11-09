"""Unit tests for the Phase 12-5 Whisper text-fidelity gate (bench_throughput).

No GPU, server, Whisper install or model weights: the transcriber is injected
as a stub, exactly the seam ``run_fidelity_benchmark(transcriber=...)`` exposes.
"""

import asyncio
import sys

import pytest

from scripts import bench_throughput as bench


@pytest.mark.unit
def test_normalize_fidelity_words_matches_the_138_word_fixture():
    words = bench.normalize_fidelity_words(bench.LONGFORM_FIXTURE_TEXT)

    assert len(words) == 138
    assert words[:4] == ["always", "destined", "to", "stand"]
    # Punctuation (including the em-dash and apostrophes) is a separator.
    assert "brightest" in words and "her" in words
    assert "today" in words and "s" in words


@pytest.mark.unit
@pytest.mark.parametrize(
    ("reference", "hypothesis", "expected"),
    [
        ("the quick brown fox", "the quick brown fox", 0),
        ("the quick brown fox", "the quick brown", 1),
        ("the quick brown fox", "quick brown fox", 1),
        ("the quick brown fox", "the quick brown cat", 2),
        ("a b c d", "a x c y", 4),
        ("", "anything", 1),
        ("anything", "", 1),
    ],
)
def test_word_edit_distance_on_known_pairs(reference, hypothesis, expected):
    assert bench.word_edit_distance(reference, hypothesis) == expected


@pytest.mark.unit
def test_score_fidelity_returns_the_report_row_shape():
    row = bench.score_fidelity("Hello, world!", "hello world")

    assert row == {
        "edit_distance": 0,
        "words_heard": 2,
        "words_ref": 2,
        "transcript": "hello world",
    }


@pytest.mark.unit
def test_stub_transcriber_is_injected_and_scored():
    seen = {}

    def stub_transcriber(path):
        seen["path"] = path
        return "Always destined to stand out"

    scored = bench.transcribe_and_score(
        bench.LONGFORM_FIXTURE_TEXT, "render.wav", stub_transcriber
    )

    assert seen["path"] == "render.wav"
    assert scored["words_heard"] == 5
    assert scored["words_ref"] == 138
    # The hypothesis is a prefix of the reference, so the LCS distance is the
    # number of reference words it omits.
    assert scored["edit_distance"] == 138 - 5


@pytest.mark.unit
def test_fidelity_markdown_has_the_agreed_row_shape():
    report = {
        "label": "phase12-fidelity-test",
        "timestamp_utc": "2026-11-09T00:00:00Z",
        "commit": "abc1234",
        "env": {},
        "config": None,
        "out_dir": "out/bench/fidelity-12",
        "method": {
            "transcriber": "large-v3-turbo",
            "device": "cpu",
            "compute_type": "int8",
            "language": "en",
            "beam_size": 5,
            "normalization": "lowercase, punctuation stripped",
            "noise_floor_edit_distance": 6,
            "noise_floor_examples": "Karina/Carina, weaves/leaves",
            "variants": ["on"],
            "chunk_budget": None,
        },
        "families": {
            "chatterbox": {
                "variants": {
                    "on": {
                        "edit_distance": 2,
                        "words_heard": 136,
                        "words_ref": 138,
                    }
                }
            },
            "chatterbox_multilingual": {
                "variants": {
                    "on": {
                        "edit_distance": 9,
                        "words_heard": 139,
                        "words_ref": 138,
                    }
                }
            },
        },
        "skipped": {},
    }

    table = bench._fidelity_markdown(report)

    assert "| family | variant | edit distance | words heard / ref |" in table
    assert "|---|---|---|---|" in table
    assert "| chatterbox | on | 2 | 136 / 138 |" in table
    assert "| chatterbox_multilingual | on | 9 | 139 / 138 |" in table
    # The method note carries the transcription noise floor, not a hidden caveat.
    assert "noise floor" in table
    assert "Karina/Carina" in table
    assert "large-v3-turbo" in table


@pytest.mark.unit
def test_fidelity_metadata_reflects_per_family_chunk_counts():
    expected = {
        "chatterbox": (300, 4),
        "chatterbox_turbo": (300, 4),
        "chatterbox_multilingual": (300, 4),
        "melo": (300, 4),
        "openvoice_v2": (300, 4),
        "kitten": (300, 4),
    }
    for family, (budget, chunks) in expected.items():
        metadata = bench.longform_fixture_metadata(family)

        assert metadata["fixture"] == "real_prose"
        assert metadata["default_chunk_size"] == budget
        assert metadata["expected_chunks_per_request"] == chunks
        assert metadata["words_ref"] == 138
        assert metadata["characters"] == 807


@pytest.mark.unit
def test_fidelity_split_defaults_to_on_but_longform_defaults_to_both():
    assert bench.longform_split_modes(None, fidelity=True) == ["on"]
    assert bench.longform_split_modes(None) == ["off", "on"]
    assert bench.longform_split_modes("both", fidelity=True) == ["off", "on"]
    assert bench.longform_split_modes("off", fidelity=True) == ["off"]


@pytest.mark.unit
def test_fidelity_requires_the_longform_scenario():
    with pytest.raises(SystemExit) as error:
        bench.main(["--fidelity"])
    assert error.value.code == 2


@pytest.mark.unit
def test_missing_faster_whisper_fails_with_a_clear_message(monkeypatch):
    monkeypatch.setitem(sys.modules, "faster_whisper", None)

    with pytest.raises(SystemExit) as error:
        bench.load_fidelity_transcriber()

    message = str(error.value)
    assert "--fidelity" in message
    assert "faster-whisper" in message


@pytest.mark.unit
def test_run_fidelity_benchmark_reports_rows_with_the_injected_transcriber(monkeypatch, tmp_path):
    async def fake_render(client, base_url, family, payload, timeout, out_path):  # noqa: ARG001
        return out_path, 39.0

    monkeypatch.setattr(bench, "_render_fidelity_wav", fake_render)
    monkeypatch.setattr(
        "tests.audio_fixtures.reference_wav_b64", lambda name: "REFERENCE"
    )

    reference = bench.longform_fixture("chatterbox_multilingual")
    kept = " ".join(reference.split()[:-5])

    args = bench.build_parser().parse_args(
        [
            "--scenario",
            "longform",
            "--fidelity",
            "--families",
            "chatterbox_multilingual",
            "--fidelity-out",
            str(tmp_path),
            "--force",
        ]
    )
    args.model = {}
    report = asyncio.run(
        bench.run_fidelity_benchmark(args, transcriber=lambda path: kept)
    )

    assert report["scenario"] == "longform-fidelity"
    assert report["method"]["variants"] == ["on"]
    row = report["families"]["chatterbox_multilingual"]
    assert row["default_chunk_size"] == 300
    assert row["expected_chunks_per_request"] == 4
    variant = row["variants"]["on"]
    assert variant["chunks"] == 4
    assert variant["words_ref"] == 138
    assert variant["audio_seconds"] == 39.0
    assert variant["words_heard"] < variant["words_ref"]
    assert variant["edit_distance"] == variant["words_ref"] - variant["words_heard"]
    assert "chatterbox_multilingual | on" in bench._fidelity_markdown(report)

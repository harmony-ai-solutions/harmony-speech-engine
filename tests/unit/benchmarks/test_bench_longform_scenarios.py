"""Local self-check for the Phase 12-4 long-form benchmark scenarios.

No GPU or server: these tests prove the fixture is deterministic, every
scenario definition loads, the ON/OFF payloads differ only in chunking fields,
and the payloads validate against the wire protocol model. Phase 12-5 switched
the fixture to the owner's real-prose text and made unset join timing measure
the serving defaults, so the SHA/chunk count and payload expectations are
re-pinned here.
"""

import hashlib

import pytest

from harmonyspeech.endpoints.openai.protocol import TextToSpeechRequest
from scripts import bench_throughput as bench

#: SHA-256 of the shared real-prose fixture (owner-provided, 2026-11-09).
#: Hardcoded so an accidental edit to the literal (including a normalised space,
#: em-dash or apostrophe) is caught, not just "still deterministic".
EXPECTED_FIXTURE_SHA256 = "a5595f416783b1c74439e2b7bdae2d1d37b0e76ed10bf20674ad92ef9632f185"
EXPECTED_FIXTURE_CHARACTERS = 807
EXPECTED_FIXTURE_SENTENCES = 8
#: Chunks the shared fixture splits into at the server's unified 300-character
#: default (Phase 12-7/12-8 removed the per-model overrides).
EXPECTED_FIXTURE_CHUNKS = 4


@pytest.mark.unit
def test_longform_fixtures_are_stable_and_in_the_spec_range():
    """Every family shares one literal real-prose fixture (no generator)."""
    for family in bench.LONGFORM_FAMILIES:
        text = bench.longform_fixture(family)

        assert text == bench.LONGFORM_FIXTURE_TEXT
        assert text == bench.longform_fixture(family)
        assert "\n" not in text and "\r" not in text
        assert "  " not in text  # hard line-wraps normalised to single spaces
        assert "—" in text  # the em-dash narrative case is deliberately exercised
        assert len(text) == EXPECTED_FIXTURE_CHARACTERS
        assert 400 <= len(text) <= 1200, (family, len(text))
        assert text.count(".") == EXPECTED_FIXTURE_SENTENCES
        assert hashlib.sha256(text.encode("utf-8")).hexdigest() == EXPECTED_FIXTURE_SHA256


@pytest.mark.unit
def test_longform_fixture_splits_into_several_chunks_at_the_unified_default():
    """Every family yields the same 4 chunks at the unified 300-char default."""
    for family in bench.LONGFORM_FAMILIES:
        text = bench.longform_fixture(family)
        language = bench.FAMILY_SPECS[family]["payload"].get("language")
        budget = bench.longform_default_chunk_size(family)

        assert budget == 300
        assert text  # fixture is non-empty
        assert (
            bench.longform_chunk_count(text, language=language, family=family)
            == EXPECTED_FIXTURE_CHUNKS
        )
        assert bench.longform_chunk_count(text, budget, language) == EXPECTED_FIXTURE_CHUNKS


@pytest.mark.unit
def test_longform_default_budgets_mirror_the_serving_layer():
    """The harness constant must not drift from the server's resolution."""
    from harmonyspeech.endpoints.openai.serving_text_to_speech import DEFAULT_CHUNK_SIZE

    assert bench.LONGFORM_DEFAULT_CHUNK_SIZE == DEFAULT_CHUNK_SIZE == 300
    assert bench.longform_default_chunk_size("chatterbox_turbo") == 300
    assert bench.longform_default_chunk_size("chatterbox_multilingual") == 300


@pytest.mark.unit
def test_on_payload_omits_chunk_size_so_the_server_default_applies():
    """An unset CLI budget must not pin 300 in the ON payload."""
    for family in bench.LONGFORM_FAMILIES:
        payload = bench.build_longform_payload(
            family, "REFERENCE_AUDIO", bench.longform_fixture(family), True
        )
        assert "chunk_size" not in payload["generation_options"]


@pytest.mark.unit
def test_explicit_longform_chunk_size_is_sent_verbatim():
    payload = bench.build_longform_payload(
        "chatterbox_turbo",
        "REFERENCE_AUDIO",
        bench.longform_fixture("chatterbox_turbo"),
        True,
        chunk_size=200,
    )
    assert payload["generation_options"]["chunk_size"] == 200


@pytest.mark.unit
def test_longform_fixture_report_names_the_real_prose_source():
    """The report must say the fixture is real prose, not a seeded generator."""
    assert "real prose" in bench.LONGFORM_FIXTURE_SOURCE
    assert "Villainess Malfoy" in bench.LONGFORM_FIXTURE_SOURCE


@pytest.mark.unit
def test_longform_scenario_matrix_covers_families_splits_and_levels():
    cells = bench.longform_scenario_matrix()

    assert len(cells) == len(bench.LONGFORM_FAMILIES) * 2 * len(bench.LONGFORM_CONCURRENCY)
    labels = {cell["label"] for cell in cells}
    for family in bench.LONGFORM_FAMILIES:
        for split in ("off", "on"):
            for concurrency in (1, 4, 8, 20):
                assert f"phase12-{family}-split-{split}-c{concurrency}" in labels
    assert all(cell["scenario"] == "longform" for cell in cells)
    assert {cell["split_text"] for cell in cells} == {False, True}


@pytest.mark.unit
def test_on_and_off_payloads_differ_only_in_chunking_fields():
    pair = bench.longform_payload_pair(
        "chatterbox", "REFERENCE_AUDIO", bench.longform_fixture("chatterbox"), chunk_size=300
    )
    off, on = pair["off"], pair["on"]

    assert {key: value for key, value in off.items() if key != "generation_options"} == {
        key: value for key, value in on.items() if key != "generation_options"
    }

    chunking_fields = {"split_text", "chunk_size", "crossfade_ms", "sentence_pause_ms"}
    off_rest = {key: value for key, value in off["generation_options"].items() if key not in chunking_fields}
    on_rest = {key: value for key, value in on["generation_options"].items() if key not in chunking_fields}
    assert off_rest == on_rest

    assert off["generation_options"]["split_text"] is False
    assert "chunk_size" not in off["generation_options"]
    assert on["generation_options"]["split_text"] is True
    assert on["generation_options"]["chunk_size"] == 300
    # Unset join fields are omitted so the serving layer's Phase 12-5 defaults
    # (40 ms / 120 ms) apply; the harness must not pin 0/0.
    assert "crossfade_ms" not in on["generation_options"]
    assert "sentence_pause_ms" not in on["generation_options"]


@pytest.mark.unit
def test_explicit_join_timing_is_sent_verbatim():
    payload = bench.build_longform_payload(
        "chatterbox",
        "REFERENCE_AUDIO",
        bench.longform_fixture("chatterbox"),
        True,
        chunk_size=300,
        crossfade_ms=0,
        sentence_pause_ms=50,
    )
    assert payload["generation_options"]["crossfade_ms"] == 0
    assert payload["generation_options"]["sentence_pause_ms"] == 50


@pytest.mark.unit
def test_every_longform_payload_validates_against_the_protocol_model():
    for family in bench.LONGFORM_FAMILIES:
        text = bench.longform_fixture(family)
        pair = bench.longform_payload_pair(family, "REFERENCE_AUDIO", text, chunk_size=300)
        for payload in pair.values():
            request = TextToSpeechRequest(**payload)
            assert request.input == text
            assert request.generation_options is not None


@pytest.mark.unit
def test_longform_recipe_prints_matrix_composition_boot_and_gates():
    args = bench.build_parser().parse_args([])
    recipe = bench.longform_scenario_recipe(args)

    assert "--scenario longform" in recipe
    assert "--longform-split both" in recipe
    assert "--engine-log out/bench/engine-phase12-debug.log" in recipe
    assert "--axis longform:off-vs-on" in recipe
    assert "--axis determinism:run1-vs-run2" in recipe
    assert "--print-longform-scenarios" in recipe


@pytest.mark.unit
def test_longform_scenario_defaults_to_the_longform_family_set():
    args = bench.build_parser().parse_args(["--scenario", "longform"])

    assert bench.longform_families_from_args(args) == list(bench.LONGFORM_FAMILIES)

    explicit = bench.build_parser().parse_args(
        ["--scenario", "longform", "--families", "melo,chatterbox"]
    )
    assert bench.longform_families_from_args(explicit) == ["melo", "chatterbox"]


@pytest.mark.unit
def test_longform_markdown_matrix_has_a_divider_per_column_and_honest_off_chunks():
    report = {
        "label": "phase12-test",
        "timestamp_utc": "2026-11-09T00:00:00Z",
        "commit": "abc1234",
        "env": {},
        "base_url": "http://127.0.0.1:2242",
        "requests_per_level": 8,
        "concurrency_levels": [1, 4],
        "longform": {
            "seed": bench.LONGFORM_SEED,
            "fixture": bench.LONGFORM_FIXTURE_SOURCE,
            "chunk_size": 300,
            "crossfade_ms": 0,
            "sentence_pause_ms": 0,
            "filters": {},
            "fixtures": {
                "chatterbox": {
                    "class": "AR decode",
                    "characters": 807,
                    "expected_chunks_per_request": 4,
                }
            },
        },
        "runs": {
            "chatterbox-split-off": [
                {"concurrency": 1, "throughput_rps": 1.0, "p50_ms": 10.0, "steps": {}},
                {"concurrency": 4, "throughput_rps": 2.0, "p50_ms": 20.0, "steps": {}},
            ],
            "chatterbox-split-on": [
                {"concurrency": 1, "throughput_rps": 3.0, "p50_ms": 30.0, "steps": {}},
                {"concurrency": 4, "throughput_rps": 4.0, "p50_ms": 40.0, "steps": {}},
            ],
        },
        "skipped": {},
    }

    table = bench._longform_markdown(report)
    header = next(line for line in table.splitlines() if line.startswith("| Family |"))
    divider = next(line for line in table.splitlines() if line.startswith("|---|"))

    assert header.count("|") == divider.count("|")
    assert "| chatterbox | AR decode | off | 807 | 1 |" in table
    assert "| chatterbox | AR decode | on | 807 | 4 |" in table
    assert "real prose" in table


@pytest.mark.unit
def test_longform_markdown_states_server_defaults_when_join_timing_unset():
    report = {
        "label": "phase12-defaults",
        "timestamp_utc": "2026-11-09T00:00:00Z",
        "commit": "abc1234",
        "env": {},
        "base_url": "http://127.0.0.1:2242",
        "requests_per_level": 8,
        "concurrency_levels": [1],
        "longform": {
            "fixture": bench.LONGFORM_FIXTURE_SOURCE,
            "chunk_size": 300,
            "crossfade_ms": None,
            "sentence_pause_ms": None,
            "filters": {},
            "fixtures": {},
        },
        "runs": {},
        "skipped": {},
    }

    table = bench._longform_markdown(report)

    assert "server default (40 ms)" in table
    assert "server default (120 ms)" in table
    assert "real prose" in table
    assert "fixture seed" not in table


@pytest.mark.unit
def test_longform_markdown_states_per_model_budget_when_chunk_size_unset():
    report = {
        "label": "phase12-budget-default",
        "timestamp_utc": "2026-11-09T00:00:00Z",
        "commit": "abc1234",
        "env": {},
        "base_url": "http://127.0.0.1:2242",
        "requests_per_level": 8,
        "concurrency_levels": [1],
        "longform": {
            "fixture": bench.LONGFORM_FIXTURE_SOURCE,
            "chunk_size": None,
            "crossfade_ms": None,
            "sentence_pause_ms": None,
            "filters": {},
            "fixtures": {},
        },
        "runs": {},
        "skipped": {},
    }

    table = bench._longform_markdown(report)

    assert "server default (300) chars" in table


@pytest.mark.unit
def test_print_longform_scenarios_helper_exits_cleanly(capsys):
    assert bench.main(["--print-longform-scenarios"]) == 0
    output = capsys.readouterr().out
    assert "Phase 12-4 long-form matrix" in output
    assert "phase12-chatterbox" in output
    # Truthful fixture + defaults in the printed recipe (Phase 12-5/12-9).
    assert "real-prose" in output
    assert "The Villainess Malfoy" in output
    assert "unified" in output
    assert "300-character default" in output
    assert "40 ms fade / 120 ms pause" in output

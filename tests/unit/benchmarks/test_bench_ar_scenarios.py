"""Unit tests for the Phase 9-6 Chatterbox AR benchmark scenarios (bench_throughput)."""

import pytest

from scripts import bench_throughput as bench


@pytest.mark.unit
def test_parse_ar_metrics_reads_pool_and_decode_series():
    text = "\n".join(
        [
            "hse_ar_cache_enabled 1",
            "hse_ar_cache_pool_bytes 1056964608",
            "hse_ar_cache_slots 12",
            "hse_ar_cache_slots_active 4",
            "hse_ar_cache_grants_total 17",
            "hse_ar_cache_refusals_total 3",
            'hse_ar_decode_active_rows{model="chatterbox"} 2',
            'hse_ar_decode_tokens_total{model="chatterbox"} 128',
            'hse_ar_decode_steps_total{model="chatterbox"} 32',
            'hse_ar_decode_tokens_per_step{model="chatterbox"} 4.0',
            "# HELP something else",
        ]
    )

    parsed = bench.parse_ar_metrics(text)

    assert parsed["cache"]["enabled"] == 1.0
    assert parsed["cache"]["slots"] == 12.0
    assert parsed["cache"]["refusals_total"] == 3.0
    assert parsed["models"]["chatterbox"]["active_rows"] == 2.0
    assert parsed["models"]["chatterbox"]["tokens_per_step"] == 4.0


@pytest.mark.unit
def test_chatterbox_flood_rotates_mixed_length_texts():
    specs = bench.chatterbox_flood_scenario(7)
    assert len(specs) == 7
    assert [spec["index"] for spec in specs] == list(range(7))
    assert {spec["text"] for spec in specs} == set(bench.CHATTERBOX_TEXTS)
    assert all(spec["distinct_voice"] for spec in specs)


@pytest.mark.unit
def test_ar_scenario_matrix_covers_the_three_way_and_cache_cells():
    labels = {cell["label"] for cell in bench.ar_scenario_matrix()}

    for family in ("chatterbox", "chatterbox_turbo"):
        for concurrency in (4, 8, 20):
            for mode in ("sequential", "ar0", "ar-runner"):
                assert f"phase9-{family}-{mode}-c{concurrency}" in labels
    for mode in ("same", "distinct"):
        for cache in ("on", "off"):
            assert f"phase9-cache-{mode}-{cache}" in labels
    assert "phase9-exhaustion" in labels


@pytest.mark.unit
def test_ar_recipe_prints_both_boots_the_cache_matrix_and_exhaustion():
    args = bench.build_parser().parse_args([])
    recipe = bench.ar_scenario_recipe(args)

    assert "Boot A - shipped AR runner" in recipe
    assert "Boot B - sequential loop" in recipe
    assert "--scenario chatterbox-ar" in recipe
    assert "--scenario chatterbox-exhaustion" in recipe
    assert "--clone-reference distinct" in recipe
    assert "--embedding-cache off" in recipe
    assert "--print-ar-scenarios" in recipe


@pytest.mark.unit
def test_summarize_ar_scenario_reports_loss_refusals_and_trace():
    specs = [{"index": index, "text": "hi"} for index in range(4)]
    records = [
        {"index": 0, "ok": True, "wall_ms": 100.0, "error": None},
        {"index": 1, "ok": True, "wall_ms": 200.0, "error": None},
        {"index": 2, "ok": False, "wall_ms": 300.0, "error": "ReadTimeout: timed out"},
        # index 3 missing -> loss 1
    ]
    samples = [
        {
            "t_s": 0.0,
            "cache": {"slots_active": 4, "grants_total": 10, "refusals_total": 1},
            "models": {"chatterbox": {"tokens_per_step": 2.0}},
        },
        {
            "t_s": 1.0,
            "cache": {"slots_active": 8, "grants_total": 15, "refusals_total": 4},
            "models": {"chatterbox": {"tokens_per_step": 4.0}},
        },
    ]

    summary = bench.summarize_ar_scenario(specs, records, 2.0, samples)

    assert summary["requests"] == 4
    assert summary["records"] == 3
    assert summary["request_loss"] == 1
    assert summary["completed"] == 2
    assert summary["failures"] == 1
    assert summary["timeouts"] == 1
    assert summary["max_slots_active"] == 8
    assert summary["grants_delta"] == 5
    assert summary["refusals_delta"] == 3
    assert summary["max_tokens_per_step"] == 4.0
    assert len(summary["ar_trace"]) == 2


@pytest.mark.unit
def test_chatterbox_turbo_family_is_benchmarked():
    spec = bench.FAMILY_SPECS["chatterbox_turbo"]
    assert spec["model"] == "chatterbox_turbo"
    assert spec["path"] == "/v1/audio/speech"

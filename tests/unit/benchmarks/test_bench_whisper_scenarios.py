"""Load-shape tests for the Phase 10-5 Whisper scenarios (bench harness).

The measurement itself needs the benchmark host; these tests prove the *load
shape* the exhaustion tables depend on:

* the 100-request heterogeneous flood really spans 2 s–8 min, both tasks and all
  four response formats, with deterministic durations;
* the fairness proof really puts one ≥10-min request **first**;
* the generated audio has the requested duration and sample rate (no external
  assets);
* the ``/metrics`` parser and the aggregation report request loss, timeouts,
  format errors and short-clip percentiles honestly;
* the scenario matrix lists every cell the docs promise.
"""

import base64

import numpy as np
import pytest

from harmonyspeech.common import audio_codec
from scripts import bench_throughput as bench


def _reference_b64(seconds: float = 1.0, sample_rate: int = 16000) -> str:
    t = np.arange(int(sample_rate * seconds)) / sample_rate
    envelope = 0.55 + 0.45 * np.sin(2 * np.pi * 2.7 * t) ** 2
    waveform = (0.2 * envelope * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    return base64.b64encode(audio_codec.encode_audio(waveform, sample_rate, "wav")).decode("ascii")


REFERENCE = _reference_b64()


# ── Duration spread / flood scenario ─────────────────────────────────────────


@pytest.mark.unit
def test_flood_durations_span_two_seconds_to_eight_minutes_deterministically():
    durations = bench.whisper_flood_durations(100, 480.0)

    assert len(durations) == 100
    assert durations == bench.whisper_flood_durations(100, 480.0)
    assert min(durations) == pytest.approx(2.0)
    assert durations[-1] == pytest.approx(480.0)
    assert all(2.0 <= value <= 480.0 for value in durations)
    # Both sides of the 30 s window boundary are represented.
    assert any(value <= 30.0 for value in durations)
    assert any(value > 30.0 for value in durations)


@pytest.mark.unit
def test_flood_scenario_is_heterogeneous():
    specs = bench.whisper_flood_scenario(100, 480.0)

    assert len(specs) == 100
    assert {spec["task"] for spec in specs} == {"transcribe", "translate"}
    assert {spec["response_format"] for spec in specs} == set(bench.WHISPER_RESPONSE_FORMATS)
    assert {spec["language"] for spec in specs} == set(bench.WHISPER_LANGUAGES)
    for spec in specs:
        expected_path = (
            "/v1/audio/translations" if spec["task"] == "translate" else "/v1/audio/transcriptions"
        )
        assert spec["path"] == expected_path
        assert spec["index"] >= 0


@pytest.mark.unit
def test_fairness_scenario_pins_the_long_request_first():
    specs = bench.whisper_fairness_scenario(long_seconds=600.0, short_count=20, short_seconds=3.0)

    assert specs[0]["kind"] == "long"
    assert specs[0]["duration_s"] == 600.0
    assert [spec["kind"] for spec in specs[1:]] == ["short"] * 20
    assert {spec["duration_s"] for spec in specs[1:]} == {3.0}


# ── Audio generation ─────────────────────────────────────────────────────────


@pytest.mark.unit
def test_duration_audio_is_trimmed_or_tiled_to_the_requested_length():
    cache: dict = {}
    for seconds in (2.0, 0.5, 65.0):
        encoded = bench.duration_audio_b64(REFERENCE, seconds, cache)
        waveform, sample_rate = audio_codec.decode_audio(base64.b64decode(encoded))
        assert sample_rate == 16000
        assert len(waveform) == int(round(seconds * sample_rate))

    # The cache returns the same payload for the same rounded duration.
    assert bench.duration_audio_b64(REFERENCE, 2.0, cache) == cache[2.0]
    assert set(cache) == {2.0, 0.5, 65.0}


@pytest.mark.unit
def test_build_whisper_payload_carries_format_and_optional_language():
    spec = bench.whisper_flood_scenario(1)[0]
    spec = dict(spec, response_format="srt", language="de")
    payload = bench.build_whisper_payload(spec, "AUDIO", "faster-whisper-tiny")

    assert payload == {
        "model": "faster-whisper-tiny",
        "input_audio": "AUDIO",
        "response_format": "srt",
        "language": "de",
    }


# ── Response validation / metrics parsing ────────────────────────────────────


class _FakeResponse:
    def __init__(self, status_code=200, body=None, text="", content_type="application/json"):
        self.status_code = status_code
        self._body = body
        self.text = text
        self.headers = {"content-type": content_type}

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


@pytest.mark.unit
def test_response_validation_distinguishes_formats_and_errors():
    json_spec = {"response_format": "json"}
    assert bench._whisper_response_ok(json_spec, _FakeResponse(body={"text": "hi"})) == (True, None)
    assert bench._whisper_response_ok(json_spec, _FakeResponse(body={})) == (
        False,
        "missing_text_field",
    )
    assert bench._whisper_response_ok(json_spec, _FakeResponse(body=None))[0] is False
    assert bench._whisper_response_ok(json_spec, _FakeResponse(status_code=500)) == (
        False,
        "http_500",
    )

    assert bench._whisper_response_ok(
        {"response_format": "text"},
        _FakeResponse(text="hello", content_type="text/plain; charset=utf-8"),
    ) == (True, None)
    assert bench._whisper_response_ok(
        {"response_format": "srt"},
        _FakeResponse(text="1\n00:00:00,000 --> 00:00:01,000\nhi\n", content_type="text/plain"),
    ) == (True, None)
    assert bench._whisper_response_ok(
        {"response_format": "srt"},
        _FakeResponse(text="not a subtitle", content_type="text/plain"),
    ) == (False, "invalid_srt")
    assert bench._whisper_response_ok(
        {"response_format": "vtt"},
        _FakeResponse(text="WEBVTT\n\n00:00.000 --> 00:01.000\nhi\n", content_type="text/plain"),
    ) == (True, None)
    assert bench._whisper_response_ok(
        {"response_format": "vtt"},
        _FakeResponse(text="nope", content_type="text/plain"),
    ) == (False, "invalid_vtt")


@pytest.mark.unit
def test_metrics_parser_reads_lane_gauges_and_device_truth():
    text = "\n".join(
        [
            'hse_whisper_workers{model="whisper"} 2',
            'hse_whisper_in_flight{model="whisper"} 1',
            'hse_whisper_lane_short_depth{model="whisper"} 7',
            'hse_whisper_lane_long_depth{model="whisper"} 2',
            'hse_whisper_batch_fallbacks_total{model="whisper"} 0',
            'hse_whisper_ct2_queued_batches{model="whisper"} 3',
            'hse_whisper_ct2_active_batches{model="whisper"} 2',
            'hse_whisper_queue_wait_seconds_sum{model="whisper"} 1.5',
            'hse_whisper_queue_wait_seconds_count{model="whisper"} 9',
            "hse_vram_device_used_bytes 1234",
            "hse_vram_headroom_bytes 5678",
            "# HELP something else",
        ]
    )

    parsed = bench.parse_whisper_metrics(text)

    assert parsed["models"]["whisper"] == {
        "workers": 2,
        "in_flight": 1,
        "lane_short_depth": 7,
        "lane_long_depth": 2,
        "batch_fallbacks_total": 0,
        "ct2_queued": 3,
        "ct2_active": 2,
        "queue_wait_sum": 1.5,
        "queue_wait_count": 9.0,
    }
    assert parsed["device"] == {"device_used_bytes": 1234, "headroom_bytes": 5678}


# ── Aggregation ──────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_summary_reports_loss_timeouts_format_errors_and_short_percentiles():
    specs = [
        {"index": 0, "kind": "long", "duration_s": 600.0, "task": "transcribe", "response_format": "json"},
        {"index": 1, "kind": "short", "duration_s": 3.0, "task": "transcribe", "response_format": "json"},
        {"index": 2, "kind": "short", "duration_s": 3.0, "task": "transcribe", "response_format": "json"},
        {"index": 3, "kind": "short", "duration_s": 3.0, "task": "transcribe", "response_format": "json"},
    ]
    records = [
        {"index": 0, "kind": "long", "ok": True, "wall_ms": 5000.0, "error": None},
        {"index": 1, "kind": "short", "ok": True, "wall_ms": 100.0, "error": None},
        {"index": 2, "kind": "short", "ok": False, "wall_ms": 300.0, "error": "ReadTimeout: timed out"},
        # index 3 is missing entirely -> request loss 1
    ]
    samples = [
        {
            "t_s": 0.0,
            "models": {"whisper": {"in_flight": 2, "lane_short_depth": 5, "queue_wait_sum": 0.0, "queue_wait_count": 0}},
            "device": {"device_used_bytes": 100},
        },
        {
            "t_s": 1.0,
            "models": {"whisper": {"in_flight": 1, "lane_short_depth": 1, "queue_wait_sum": 1.5, "queue_wait_count": 3}},
            "device": {"device_used_bytes": 200},
        },
    ]

    summary = bench.summarize_whisper_scenario(specs, records, 6.0, samples)

    assert summary["requests"] == 4
    assert summary["records"] == 3
    assert summary["request_loss"] == 1
    assert summary["completed"] == 2
    assert summary["failures"] == 1
    assert summary["timeouts"] == 1
    assert summary["format_errors"] == 0
    assert summary["short_p50_ms"] == 100.0
    assert summary["short_p95_ms"] == 100.0
    assert summary["queue_wait_sum_s"] == 1.5
    assert summary["queue_wait_count"] == 3
    assert summary["max_device_used_bytes"] == 200
    assert len(summary["lane_trace"]) == 2


@pytest.mark.unit
def test_format_errors_are_counted_separately_from_timeouts():
    specs = [{"index": 0, "kind": "flood", "duration_s": 2.0, "task": "transcribe", "response_format": "srt"}]
    records = [{"index": 0, "kind": "flood", "ok": False, "wall_ms": 50.0, "error": "invalid_srt"}]

    summary = bench.summarize_whisper_scenario(specs, records, 0.1, [])

    assert summary["format_errors"] == 1
    assert summary["timeouts"] == 0
    assert summary["request_loss"] == 0


# ── Scenario matrix / recipe ─────────────────────────────────────────────────


@pytest.mark.unit
def test_scenario_matrix_lists_every_documented_cell():
    labels = {cell["label"] for cell in bench.whisper_scenario_matrix()}

    for concurrency in (4, 8, 20):
        assert f"phase10-lanes-c{concurrency}" in labels
        assert f"phase10-lanes-windows-c{concurrency}" in labels
    assert "phase10-c200" in labels
    assert "phase10-fairness" in labels
    for workers in (1, 2, 4):
        assert f"phase10-turbo-m{workers}" in labels
    for workers in (1, 4, 6):
        assert f"phase10-tiny-m{workers}" in labels
    for batch in (4, 8):
        assert f"phase10-window-b{batch}" in labels
    assert "phase10-exhaustion" in labels
    assert "phase10-budget-trip" in labels


@pytest.mark.unit
def test_whisper_recipe_prints_the_exhaustion_and_budget_trip_runs():
    args = bench.build_parser().parse_args([])
    recipe = bench.whisper_scenario_recipe(args)

    assert "--scenario whisper-flood" in recipe
    assert "--scenario whisper-budget-trip" in recipe
    assert "--whisper-workers 2" in recipe
    assert "--whisper-window-batch 4" in recipe
    assert "--whisper-queue 16" in recipe
    assert "phase10-fairness" in recipe
    assert "phase10-c200" in recipe

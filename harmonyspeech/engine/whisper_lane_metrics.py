"""Prometheus text rendering for Whisper execution lanes (Phase 10-1).

The engine owns the accounting
(:meth:`~harmonyspeech.engine.harmonyspeech_engine.HarmonySpeechEngine.get_whisper_lane_stats`);
this module is a pure renderer so it can be unit-tested without a server, a GPU
or a loaded CTranslate2 model.

Series presence is part of the contract, the same way the embedding-cache and
VRAM exporters treat it: every family is emitted on every scrape, with an
explicit ``{model="none"}`` zero sample when no Whisper config is resident, so a
dashboard can tell "idle" from "absent".

The queue-wait histogram buckets live here (and are imported by the runner that
records into them) because this module is torch-free: importing it from the
task handler must not pull in the model stack.
"""

from __future__ import annotations

#: Upper bounds (seconds) of the per-request queue-wait histogram. A request
#: waits from admission until the scheduler first hands it to an executor
#: (``RequestMetrics.time_in_queue``); the runner records that value once per
#: completed Whisper request.
WHISPER_QUEUE_WAIT_BUCKETS: tuple[float, ...] = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
)

_HELP = {
    "hse_whisper_workers": (
        "Configured CTranslate2 worker count (num_workers -> inter_threads) of one Whisper model."
    ),
    "hse_whisper_in_flight": "Whisper transcribe calls executing right now for one model.",
    "hse_whisper_lane_short_depth": (
        "Waiting Whisper requests whose audio is <=30 s (short fairness lane)."
    ),
    "hse_whisper_lane_long_depth": (
        "Waiting Whisper requests whose audio is >30 s (long fairness lane)."
    ),
    "hse_whisper_queue_wait_seconds": (
        "Per-request Whisper queue wait (admission to first scheduling), in seconds."
    ),
    "hse_whisper_ct2_queued_batches": (
        "CTranslate2 batches queued on the model (native num_queued_batches); 0 when the binding "
        "does not expose it."
    ),
    "hse_whisper_ct2_active_batches": (
        "CTranslate2 batches active on the model (native num_active_batches); 0 when the binding "
        "does not expose it."
    ),
    "hse_whisper_batch_fallbacks_total": (
        "Steps where the cross-request window batcher failed and the handler fell back to the "
        "per-request pipeline path."
    ),
}

_GAUGES = (
    ("hse_whisper_workers", "workers"),
    ("hse_whisper_in_flight", "in_flight"),
    ("hse_whisper_lane_short_depth", "lane_short_depth"),
    ("hse_whisper_lane_long_depth", "lane_long_depth"),
)

_CT2_GAUGES = (
    ("hse_whisper_ct2_queued_batches", "queued"),
    ("hse_whisper_ct2_active_batches", "active"),
)

_COUNTERS = (("hse_whisper_batch_fallbacks_total", "batch_fallbacks"),)


def _as_int(value) -> int:
    try:
        return int(float(value or 0))
    except (TypeError, ValueError):  # pragma: no cover - defensive on odd stats
        return 0


def _as_float(value) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):  # pragma: no cover - defensive on odd stats
        return 0.0


def _escape(value) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _label(value) -> str:
    return f'model="{_escape(value)}"'


def _empty_model() -> dict:
    return {
        "workers": 0,
        "in_flight": 0,
        "lane_short_depth": 0,
        "lane_long_depth": 0,
        "queue_wait": {
            "sum": 0.0,
            "count": 0,
            "max": 0.0,
            "buckets": [0] * (len(WHISPER_QUEUE_WAIT_BUCKETS) + 1),
        },
        "ct2": {"available": False, "queued": 0, "active": 0},
        "batch_fallbacks": 0,
    }


def _render_histogram(lines: list[str], model: str, queue_wait: dict) -> None:
    buckets = list(queue_wait.get("buckets") or [])
    # Non-cumulative counts per bucket; the last slot is the +Inf overflow.
    if len(buckets) != len(WHISPER_QUEUE_WAIT_BUCKETS) + 1:
        buckets = [0] * (len(WHISPER_QUEUE_WAIT_BUCKETS) + 1)

    cumulative = 0
    for edge, count in zip(WHISPER_QUEUE_WAIT_BUCKETS, buckets[:-1], strict=True):
        cumulative += _as_int(count)
        lines.append(
            f'hse_whisper_queue_wait_seconds_bucket{{{_label(model)},le="{edge:g}"}} {cumulative}'
        )
    cumulative += _as_int(buckets[-1])
    lines.append(f'hse_whisper_queue_wait_seconds_bucket{{{_label(model)},le="+Inf"}} {cumulative}')
    lines.append(f"hse_whisper_queue_wait_seconds_sum{{{_label(model)}}} {_as_float(queue_wait.get('sum')):.6f}")
    lines.append(f"hse_whisper_queue_wait_seconds_count{{{_label(model)}}} {cumulative}")


def render_whisper_lane_metrics(stats: dict | None) -> str:
    """Render :func:`get_whisper_lane_stats` output as Prometheus text exposition.

    ``None`` — or an empty model map — renders the idle shape: every family is
    present with a ``{model="none"}`` zero sample, so alerts keyed on the series
    never see an absent metric family.
    """
    values = stats or {}
    models = dict(values.get("models") or {})
    if not models:
        models = {"none": _empty_model()}

    lines: list[str] = []
    emitted: set[str] = set()

    def header(name: str, kind: str = "gauge") -> None:
        if name in emitted:
            return
        emitted.add(name)
        lines.append(f"# HELP {name} {_HELP[name]}")
        lines.append(f"# TYPE {name} {kind}")

    for name, key in _GAUGES:
        header(name)
        for model, detail in models.items():
            lines.append(f"{name}{{{_label(model)}}} {_as_int(detail.get(key))}")

    header("hse_whisper_queue_wait_seconds", "histogram")
    for model, detail in models.items():
        _render_histogram(lines, model, detail.get("queue_wait") or {})

    for name, key in _CT2_GAUGES:
        header(name)
        for model, detail in models.items():
            ct2 = detail.get("ct2") or {}
            lines.append(f"{name}{{{_label(model)}}} {_as_int(ct2.get(key))}")

    for name, key in _COUNTERS:
        header(name, "counter")
        for model, detail in models.items():
            lines.append(f"{name}{{{_label(model)}}} {_as_int(detail.get(key))}")

    return "\n".join(lines) + "\n"


__all__ = ["WHISPER_QUEUE_WAIT_BUCKETS", "render_whisper_lane_metrics"]

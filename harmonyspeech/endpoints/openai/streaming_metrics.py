"""Prometheus text rendering + accounting for streaming TTS (Phase 14-5).

The streaming envelopes are owned by the serving layer, not the engine, so the
accounting lives here (a process-global recorder) rather than on
:class:`~harmonyspeech.engine.harmonyspeech_engine.HarmonySpeechEngine`. The
module is deliberately dependency-free (no torch, no FastAPI) so it can be
imported by the serving layer, the ``/metrics`` route and unit tests without
pulling in the model stack.

Series presence is part of the contract (the 7-3/9-2/10-1 house rule): every
series is emitted on every scrape, with zeros when no stream has run yet,
because a dashboard must be able to tell "idle" from "absent".

Metric semantics (recorded at the streaming serving layer):

* ``hse_tts_stream_first_frame_seconds`` — histogram of the time from the
  request's frame source starting to the **first audio frame** being emitted
  (binary envelope) / the first ``speech.audio.delta`` being emitted (SSE).
  Keepalive comments are not audio and do not count. The observation is taken
  at the encoder, immediately before the first audio frame is handed to
  Starlette — i.e. **pre-header-priming** (headers are committed only after the
  first frame is in hand). The difference from the client-visible first byte is
  the header write, which is not synthesis time; measuring at the encoder keeps
  both envelopes on the same clock and excludes keepalives.
* ``hse_tts_stream_chunks_total{outcome="completed"|"aborted"|"error"}`` — one
  increment per streaming **response** (the plan's series name; see the
  decision below), classified by how it ended:
  ``completed`` (frame source exhausted normally), ``aborted`` (client
  disconnect — a cancelled response task, a closed generator, or the
  per-chunk ``is_disconnected`` poll stopping the stream early) and ``error``
  (a synthesis failure after the stream started). A failure *before* the first
  byte is answered with a normal JSON error and is still counted as ``error``:
  the stream was admitted but produced nothing.
* ``hse_tts_stream_bytes_total`` — audio bytes handed to the wire, including a
  streaming WAV header when one is emitted. SSE counts the raw s16le PCM bytes
  before base64, not the larger event text.

Decision (2026-11-10): the ``chunks_total`` labels count **streams**, not
audio frames. ``completed``/``aborted``/``error`` are stream-level terminal
states, and a per-frame counter could not represent a polled disconnect that
stops between frames (the case 14-4 flagged). A multi-chunk stream therefore
contributes exactly one sample to this counter, matching the one-sample-per-
request ``first_frame`` histogram.
"""

from __future__ import annotations

#: Upper bounds (seconds) of the time-to-first-audio-frame histogram. The range
#: spans near-instant padded-batch families (tens of ms) through an AR chunk
#: that waited for a slow co-batched row (several seconds) to the documented
#: 12-9 ceiling for a multi-chunk request (tens of seconds).
STREAM_FIRST_FRAME_BUCKETS: tuple[float, ...] = (
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.0,
    3.0,
    5.0,
    7.5,
    10.0,
    15.0,
    30.0,
)

#: Terminal outcomes of one streaming response. All three labels are always
#: rendered, so an alert can distinguish "no aborts yet" from "series missing".
STREAM_OUTCOMES: tuple[str, ...] = ("completed", "aborted", "error")

_HELP = {
    "hse_tts_stream_first_frame_seconds": (
        "Time from a streaming request's frame source starting to its first audio frame "
        "(binary envelope) or first speech.audio.delta (SSE), in seconds."
    ),
    "hse_tts_stream_chunks_total": (
        "Streaming responses by terminal outcome (completed, aborted by client disconnect, "
        "or failed after admission)."
    ),
    "hse_tts_stream_bytes_total": (
        "Audio bytes handed to the wire by streaming responses (WAV header included; SSE "
        "counts raw PCM before base64)."
    ),
}


def _empty_state() -> dict:
    return {
        # Non-cumulative counts per bucket; the last slot is the +Inf overflow.
        "first_frame": {
            "buckets": [0] * (len(STREAM_FIRST_FRAME_BUCKETS) + 1),
            "sum": 0.0,
            "count": 0,
        },
        "chunks": {outcome: 0 for outcome in STREAM_OUTCOMES},
        "bytes": 0,
    }


#: Process-global accounting. Streaming is a serving-layer concern and the
#: ``/metrics`` route is a process-wide scrape, so one module-level recorder is
#: the same shape the engine's stats surfaces use (and avoids threading a stats
#: object through every envelope).
_STATE: dict = _empty_state()


def reset_stream_metrics() -> None:
    """Reset the process-global counters (tests and diagnostic runs)."""
    global _STATE
    _STATE = _empty_state()


def observe_first_frame(seconds: float) -> None:
    """Record one time-to-first-audio-frame observation (seconds)."""
    value = max(0.0, float(seconds))
    histogram = _STATE["first_frame"]
    histogram["sum"] += value
    histogram["count"] += 1
    buckets = histogram["buckets"]
    for index, edge in enumerate(STREAM_FIRST_FRAME_BUCKETS):
        if value <= edge:
            buckets[index] += 1
            return
    buckets[-1] += 1


def record_stream_outcome(outcome: str) -> None:
    """Increment the terminal-outcome counter for one streaming response."""
    key = str(outcome).strip().lower()
    if key not in _STATE["chunks"]:  # pragma: no cover - internal callers only
        raise ValueError(f"unknown streaming outcome {outcome!r}")
    _STATE["chunks"][key] += 1


def record_stream_bytes(count: int) -> None:
    """Add wire audio bytes emitted by a streaming response."""
    _STATE["bytes"] += max(0, int(count))


def get_stream_stats() -> dict:
    """Return a deep-enough copy of the accounting (for rendering and tests)."""
    histogram = _STATE["first_frame"]
    return {
        "first_frame": {
            "buckets": list(histogram["buckets"]),
            "sum": float(histogram["sum"]),
            "count": int(histogram["count"]),
        },
        "chunks": dict(_STATE["chunks"]),
        "bytes": int(_STATE["bytes"]),
    }


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


def render_stream_metrics(stats: dict | None = None) -> str:
    """Render streaming accounting as Prometheus text exposition.

    ``stats`` is :func:`get_stream_stats` output; ``None`` renders the idle
    shape (every series present, all zeros) so a scrape never loses a family.
    """
    values = stats if stats is not None else _empty_state()
    histogram = values.get("first_frame") or {}
    buckets = list(histogram.get("buckets") or [])
    if len(buckets) != len(STREAM_FIRST_FRAME_BUCKETS) + 1:
        buckets = [0] * (len(STREAM_FIRST_FRAME_BUCKETS) + 1)
    chunks = dict(values.get("chunks") or {})

    lines: list[str] = []

    def header(name: str, kind: str) -> None:
        lines.append(f"# HELP {name} {_HELP[name]}")
        lines.append(f"# TYPE {name} {kind}")

    header("hse_tts_stream_first_frame_seconds", "histogram")
    cumulative = 0
    for edge, count in zip(STREAM_FIRST_FRAME_BUCKETS, buckets[:-1], strict=True):
        cumulative += _as_int(count)
        lines.append(
            f'hse_tts_stream_first_frame_seconds_bucket{{le="{edge:g}"}} {cumulative}'
        )
    cumulative += _as_int(buckets[-1])
    lines.append(f'hse_tts_stream_first_frame_seconds_bucket{{le="+Inf"}} {cumulative}')
    lines.append(f"hse_tts_stream_first_frame_seconds_sum {_as_float(histogram.get('sum')):.6f}")
    lines.append(f"hse_tts_stream_first_frame_seconds_count {cumulative}")

    header("hse_tts_stream_chunks_total", "counter")
    for outcome in STREAM_OUTCOMES:
        lines.append(
            f'hse_tts_stream_chunks_total{{outcome="{outcome}"}} {_as_int(chunks.get(outcome))}'
        )

    header("hse_tts_stream_bytes_total", "counter")
    lines.append(f"hse_tts_stream_bytes_total {_as_int(values.get('bytes'))}")

    return "\n".join(lines) + "\n"


__all__ = [
    "STREAM_FIRST_FRAME_BUCKETS",
    "STREAM_OUTCOMES",
    "get_stream_stats",
    "observe_first_frame",
    "record_stream_bytes",
    "record_stream_outcome",
    "render_stream_metrics",
    "reset_stream_metrics",
]

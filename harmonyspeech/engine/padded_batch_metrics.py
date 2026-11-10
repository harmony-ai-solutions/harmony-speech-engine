"""Prometheus text rendering for padded-batch headroom governance (Phase 14-6).

The engine owns the accounting
(:meth:`~harmonyspeech.engine.harmonyspeech_engine.HarmonySpeechEngine.get_padded_batch_stats`);
this module is a pure renderer so it can be unit-tested without a server, a GPU
or a loaded model. It is torch-free for the same reason the Whisper lane and AR
decode renderers are.

Series presence is part of the contract (the 7-3/10-1/9-2 convention): both
families are emitted on every scrape, with an explicit
``{model="none",axis="none"}`` zero sample when nothing is configured or
governed, so a dashboard can tell "idle" from "absent".

Two series answer the two questions the governor must be able to prove:

* ``hse_padded_batch_shrink_total{model,axis}`` — how often (and for which
  model/axis) the live headroom reduced the requested padded-area budget. A
  non-zero counter next to a healthy ``/health`` block is the production signal
  that the card is under pressure but the engine kept serving.
* ``hse_padded_batch_headroom_bytes{model,axis,device}`` — the free device
  memory observed at that model/axis's last plan. Per device because a
  multi-device deployment has one headroom per card.
"""

from __future__ import annotations

from harmonyspeech.engine.padded_batch import empty_axis_state

_HELP = {
    "hse_padded_batch_shrink_total": (
        "Plan-time padded-batch decisions where the live device headroom reduced the "
        "requested padded-area budget, per model and axis."
    ),
    "hse_padded_batch_headroom_bytes": (
        "Free device memory observed at the last padded-batch plan for one model/axis, "
        "in bytes; 0 before the first plan."
    ),
}


def _escape(value) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _as_int(value) -> int:
    try:
        return int(float(value or 0))
    except (TypeError, ValueError):  # pragma: no cover - defensive on odd stats
        return 0


def render_padded_batch_metrics(stats: dict | None) -> str:
    """Render :func:`get_padded_batch_stats` output as Prometheus text exposition.

    ``None`` — or an empty model map — renders the idle shape: both families are
    present with a ``{model="none",axis="none"}`` zero sample, so alerts keyed
    on the series never see an absent metric family.
    """
    values = stats or {}
    rows: list[tuple[str, str, str, dict]] = []
    for model, detail in (values.get("models") or {}).items():
        device = (detail or {}).get("device") or "none"
        axes = dict((detail or {}).get("axes") or {})
        if not axes:
            axes = {"none": empty_axis_state()}
        for axis, entry in axes.items():
            rows.append((model, axis, device, dict(entry or {})))
    if not rows:
        rows = [("none", "none", "none", empty_axis_state())]

    lines: list[str] = []
    lines.append(f"# HELP hse_padded_batch_shrink_total {_HELP['hse_padded_batch_shrink_total']}")
    lines.append("# TYPE hse_padded_batch_shrink_total counter")
    for model, axis, _device, entry in rows:
        lines.append(
            "hse_padded_batch_shrink_total"
            f'{{model="{_escape(model)}",axis="{_escape(axis)}"}} '
            f"{_as_int(entry.get('shrinks_total'))}"
        )

    lines.append(
        f"# HELP hse_padded_batch_headroom_bytes {_HELP['hse_padded_batch_headroom_bytes']}"
    )
    lines.append("# TYPE hse_padded_batch_headroom_bytes gauge")
    for model, axis, device, entry in rows:
        lines.append(
            "hse_padded_batch_headroom_bytes"
            f'{{model="{_escape(model)}",axis="{_escape(axis)}",device="{_escape(device)}"}} '
            f"{_as_int(entry.get('headroom_bytes'))}"
        )

    return "\n".join(lines) + "\n"


__all__ = ["render_padded_batch_metrics"]

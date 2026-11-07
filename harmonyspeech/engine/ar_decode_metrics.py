"""Prometheus text rendering for multi-step AR decode progress (Phase 9-4).

The engine owns the accounting
(:meth:`~harmonyspeech.engine.harmonyspeech_engine.HarmonySpeechEngine.get_ar_decode_stats`);
this module is a pure renderer so it can be unit-tested without a server, a GPU
or a loaded model. It is torch-free for the same reason the Whisper lane renderer
is: the task handler must be able to import it without pulling in the model
stack.

Series presence is part of the contract (the 7-3/10-1 convention): every family
is emitted on every scrape, with an explicit ``{model="none"}`` zero sample when
no AR model is configured, so a dashboard can tell "idle" from "absent".

These are **per-model** series. The device-wide 9-2 pool already exposes
``hse_ar_cache_*``; the two families answer different questions — the pool
family is "how much of the shared arena is resident/granted", this family is
"how far is each AR model's decode progressing".
"""

from __future__ import annotations

_HELP = {
    "hse_ar_decode_active_rows": "Rows currently decoding for one AR model (pool active rows).",
    "hse_ar_decode_slots": "Shared KV-cache slots available to one AR model's arena.",
    "hse_ar_decode_slots_active": "Slots of one AR model's arena granted to an in-flight request.",
    "hse_ar_decode_slots_free": "Slots of one AR model's arena available for a new request.",
    "hse_ar_decode_tokens_total": "Tokens decoded across every finished/progressing request of one AR model.",
    "hse_ar_decode_steps_total": (
        "Engine steps in which one AR model advanced at least one decode row."
    ),
    "hse_ar_decode_tokens_per_step": (
        "Average tokens decoded per engine step for one AR model (≈ active batch width)."
    ),
}

_GAUGES = (
    ("hse_ar_decode_active_rows", "active_rows"),
    ("hse_ar_decode_slots", "slots"),
    ("hse_ar_decode_slots_active", "slots_active"),
    ("hse_ar_decode_slots_free", "slots_free"),
)

_COUNTERS = (
    ("hse_ar_decode_tokens_total", "tokens_total"),
    ("hse_ar_decode_steps_total", "steps_total"),
)


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
        "active_rows": 0,
        "slots": 0,
        "slots_active": 0,
        "slots_free": 0,
        "tokens_total": 0,
        "steps_total": 0,
        "tokens_per_step": 0.0,
    }


def render_ar_decode_metrics(stats: dict | None) -> str:
    """Render :func:`get_ar_decode_stats` output as Prometheus text exposition.

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

    for name, key in _COUNTERS:
        header(name, "counter")
        for model, detail in models.items():
            lines.append(f"{name}{{{_label(model)}}} {_as_int(detail.get(key))}")

    header("hse_ar_decode_tokens_per_step")
    for model, detail in models.items():
        lines.append(f"hse_ar_decode_tokens_per_step{{{_label(model)}}} {_as_float(detail.get('tokens_per_step')):.6f}")

    return "\n".join(lines) + "\n"


__all__ = ["render_ar_decode_metrics"]

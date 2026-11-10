"""Headroom-aware padded-batch governance (Phase 14-6).

Phase 12-9 capped the per-step *padded area* of the batched model steps with
static constants (``DEFAULT_PHONE_BATCH_BUDGET`` / ``DEFAULT_MEL_FRAME_BATCH_BUDGET``).
Those constants were calibrated on the benchmark host under the 12-9 fixture's
~5.6 GiB of free device memory. On the shipped 28-model catalogue the card has
~2.5-3.5 GiB free, so a capped-but-oversized batch still pages into Windows
WDDM Shared GPU Memory — a silent 6-40x stall. This module turns the static cap
into a *ceiling*: at plan time the governor reads the device's free headroom and
shrinks the effective area so the step's padded activation fits, down to a width
floor of 1. It only ever shrinks, never widens, and never refuses a request
(Phase 14-6 owner decisions 2 and 3).

Estimation: documented constants, not a per-model probe
--------------------------------------------------------
Owner decision 4 allowed either the Phase 10-3 startup probe precedent or
constants derived from the recorded 12-9 measurements. **Constants were chosen**
for these steps, per axis, because:

* the padded activation here is ordinary PyTorch memory, unlike CTranslate2's
  (the reason 10-3 needs a live probe is that CT2 allocates outside the torch
  allocator and has no measurable tensor footprint);
* a startup probe would need a synthetic max-shape forward per batched CUDA
  model type — ten families, each seconds of boot time and a second copy of the
  input builders — and the 12-9 calibration already measured exactly that
  quantity for the two axes that carry a static cap;
* the constants are deliberately *conservative* (peak-reserved / cap-area, which
  folds the model's weights into the per-unit cost), so the governor can only
  shrink earlier than strictly necessary, never allow a batch that would page;
* a probe cannot be exercised by the hermetic CPU suite the owner requires;
  constants can be pinned exactly.

The derivation is in :data:`PADDED_AXIS_BYTES_PER_UNIT`. Axes with **no**
recorded measurement (the speech-token S3Gen vocode hops) stay ungoverned and
report the reason instead of guessing a cost — see :func:`allowed_padded_area`
and the Phase 14-6 verification log's coverage table.

Everything in this module is pure (no torch, no CUDA, no logging) so the
decision can be unit-tested and driven with a simulated headroom value.
"""

from __future__ import annotations

from harmonyspeech.engine.vram_budget import headroom_limited_units

GIB = 2**30

#: Activation bytes attributable to one padded unit, per padding axis.
#:
#: ``mel_frame`` — the OpenVoice tone-converter measurement from Phase 12-9:
#: ``B=2`` of 43 s paragraphs (7,320 frames) reserved ~2.6 GiB while ``B=8``
#: (29,280 frames) reserved 9,796 MiB and stalled ~400 s. Using the capped peak
#: (2.6 GiB at the 7,500-frame cap) gives 372,230 B/frame ≈ 0.355 MiB/frame.
#: This is the conservative reading (the peak includes the model's weights), so
#: the governor shrinks before a batch can spill. The same axis cost is applied
#: to the other mel-frame padded steps (embedders, vocoders, restorer) for which
#: no separate 12-9 measurement exists: they are all *smaller* activations than
#: the tone converter at equal area, so the proxy can only shrink early, never
#: allow an oversized batch.
#:
#: ``phone`` — the phone-axis synthesizer measurement from Phase 12-9:
#: ``B=8`` of ~128-phone chunks (1,024 phones) stays at ≤ 1.8 GiB while 3,624
#: phones reserved 6,512 MiB. 1.8 GiB / 1,024 phones = 1,887,436 B/phone
#: ≈ 1.8 MiB/phone.
PADDED_AXIS_BYTES_PER_UNIT: dict[str, int] = {
    "phone": 1_887_436,
    "mel_frame": 372_230,
}

#: Which padding axes each batched model family consumes through
#: ``default_token_budget``. Used by the engine to zero-fill the Prometheus
#: series for every configured padded model (the 9-2/10-1 house style: a
#: dashboard must be able to tell "idle" from "absent").
#:
#: ``ChatterboxVC`` is intentionally absent: its only ``default_token_budget``
#: consumer is the padded S3Gen vocode on the **speech-token** axis, which has
#: no recorded cost model (recorded reason in the 14-6 log). Its tokenizer pass
#: is not area-budgeted at all.
PADDED_MODEL_AXES: dict[str, tuple[str, ...]] = {
    "OpenVoiceV1Synthesizer": ("phone",),
    "MeloTTSSynthesizer": ("phone",),
    "OpenVoiceV1ToneConverterEncoder": ("mel_frame",),
    "OpenVoiceV2ToneConverterEncoder": ("mel_frame",),
    "HarmonySpeechEncoder": ("mel_frame",),
    "OpenVoiceV1ToneConverter": ("mel_frame",),
    "OpenVoiceV2ToneConverter": ("mel_frame",),
    "HarmonySpeechVocoder": ("mel_frame",),
    "VoiceFixerRestorer": ("mel_frame",),
    "VoiceFixerVocoder": ("mel_frame",),
}


def bytes_per_unit_for(axis: str, *, units_per_frame: int | None = None) -> int | None:
    """Return the measured activation cost of one padded unit on ``axis``.

    ``None`` means "no recorded cost model for this axis": the caller must leave
    the requested area untouched (never guess). ``units_per_frame`` converts the
    mel-frame cost to a cost per *sample* for a step that plans on sample counts
    while padding mel frames (the OpenVoice encoder pass: one mel frame per
    ``hop_length`` samples).
    """
    base = PADDED_AXIS_BYTES_PER_UNIT.get(axis)
    if base is None:
        return None
    hop = int(units_per_frame or 0)
    if hop > 0:
        return max(1, int(base) // hop)
    return int(base)


def allowed_padded_area(
    requested_area: int,
    free_headroom_bytes: int,
    axis: str,
    *,
    bytes_per_unit: int | None = None,
) -> dict:
    """Cap a requested padded-area budget by the live device headroom.

    ``requested_area`` is the static 12-9 cap (or the per-model
    ``batch_token_budget``) in the axis's own unit. The returned
    ``allowed_area`` is ``min(requested_area, headroom // bytes_per_unit)``,
    never below 1: the static cap stays the ceiling and the headroom can only
    shrink further. An axis without a recorded cost returns ``governed=False``
    with the requested area unchanged and a human-readable ``reason``.
    """
    requested = max(1, int(requested_area))
    free = max(0, int(free_headroom_bytes or 0))
    cost = bytes_per_unit if bytes_per_unit is not None else bytes_per_unit_for(axis)
    if cost is None or int(cost) <= 0:
        return {
            "governed": False,
            "axis": axis,
            "requested_area": requested,
            "allowed_area": requested,
            "headroom_area": 0,
            "headroom_bytes": free,
            "bytes_per_unit": 0,
            "shrunk": False,
            "reason": f"no recorded activation cost for axis '{axis}'",
        }

    cost = int(cost)
    allowed = headroom_limited_units(free, cost, requested, min_units=1)
    return {
        "governed": True,
        "axis": axis,
        "requested_area": requested,
        "allowed_area": int(allowed),
        "headroom_area": int(free // cost),
        "headroom_bytes": free,
        "bytes_per_unit": cost,
        "shrunk": int(allowed) < requested,
    }


def empty_axis_state() -> dict:
    """Zero-valued governance state for one (model, axis), the idle shape."""
    return {
        "shrinks_total": 0,
        "headroom_bytes": 0,
        "requested_area": 0,
        "allowed_area": 0,
        "width": 0,
        "max_width": 0,
        "median_length": 0,
        "state": "idle",
    }


def median_length(lengths) -> int:
    """Integer median of a length vector (``0`` for an empty one)."""
    values = sorted(int(length) for length in lengths or [])
    if not values:
        return 0
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return max(1, (values[middle - 1] + values[middle]) // 2)


def plan_width(groups) -> int:
    """Widest planned sub-batch (``0`` for an empty plan)."""
    return max((len(group) for group in groups or []), default=0)


def format_padded_batch_warning(model: str, decision: dict, width: int, max_width: int) -> str:
    """The once-per-state-change warning: model, axis, areas, headroom, width."""
    return (
        f"Padded batch shrunk for '{model}' ({decision['axis']} axis): padded area "
        f"{decision['requested_area']} -> {decision['allowed_area']} units "
        f"({decision['headroom_bytes'] / GIB:.2f} GiB free device headroom, "
        f"{decision['bytes_per_unit']} B/unit); batch width {width} of {max_width}. "
        "Keeping the padded activation inside the device avoids the WDDM "
        "shared-memory paging stall (Phase 14-6)."
    )


def format_padded_batch_restored(model: str, decision: dict, width: int, max_width: int) -> str:
    """The recovery line: the budget fits again, full width restored."""
    return (
        f"Padded batch width restored for '{model}' ({decision['axis']} axis): "
        f"padded area {decision['requested_area']} units again "
        f"({decision['headroom_bytes'] / GIB:.2f} GiB free device headroom); "
        f"batch width {width} of {max_width}."
    )


__all__ = [
    "PADDED_AXIS_BYTES_PER_UNIT",
    "PADDED_MODEL_AXES",
    "allowed_padded_area",
    "bytes_per_unit_for",
    "empty_axis_state",
    "format_padded_batch_restored",
    "format_padded_batch_warning",
    "median_length",
    "plan_width",
]

"""Whisper VRAM governance primitives (Phase 10-3).

CTranslate2 has no memory-cap knob of its own (``gpu_memory_utilization`` does
not exist there) and allocates outside PyTorch's caching allocator, so the
engine enforces Whisper's footprint on its side: a boot-time budget check plus a
runtime admission decision that shrinks the number of concurrent windows when
the device-level used memory approaches the budget.

Everything in this module is **pure** (no torch, no CUDA, no logging) so the
decision can be unit-tested and driven by 10-5's flood tests with a mocked
``torch.cuda.mem_get_info``.

The budget model
----------------
``budget_bytes`` is a device-level used-memory ceiling. When
``whisper_vram_budget_bytes`` is configured it is used as-is; when it is absent
the engine passes the device total, which turns the decision into "how many
windows fit in the currently free device memory" — the shared-headroom bound the
plan asks for. ``device_used_bytes`` is always the whole-device number from
``torch.cuda.mem_get_info`` (torch + CTranslate2 + CUDA contexts), so the
governor automatically accounts for every resident family.

:func:`headroom_limited_units` is the shared "shrink the width until it fits the
free headroom" arithmetic. Phase 14-6 reuses it for the padded-batch model steps
(``harmonyspeech.engine.padded_batch``), so the two governors cannot drift apart.
"""

from __future__ import annotations

#: Longest audio a single Whisper request may carry (Phase 10-3 whisper-side
#: request-size cap). Matches faster-whisper's own 30 s windowing scale: a
#: request of this length produces ``ceil(seconds / 30)`` windows.
DEFAULT_WHISPER_MAX_AUDIO_SECONDS = 300.0

#: faster-whisper's default beam width; the probe and the admission estimate use
#: the same value so the measured peak matches the served shape.
DEFAULT_WHISPER_BEAM_SIZE = 5


def per_window_bytes(peak_activation_bytes: int, windows: int) -> int:
    """Return the device bytes attributable to one admitted Whisper window.

    ``peak_activation_bytes`` is a probe measurement for ``windows`` concurrent
    max-shape windows; dividing it gives the admission estimator its unit cost.
    A non-positive window count yields ``0`` ("unknown"), which callers treat as
    "do not throttle on cost".
    """
    windows = int(windows or 0)
    if windows <= 0:
        return 0
    return max(0, int(peak_activation_bytes or 0)) // windows


def headroom_limited_units(
    headroom_bytes: int,
    cost_bytes: int,
    cap: int,
    *,
    min_units: int = 1,
) -> int:
    """Largest number of ``cost_bytes`` units that fit the free headroom.

    The shared arithmetic behind every "shrink the width until it fits the
    device" governor: Whisper windows (Phase 10-3) and the padded-batch model
    steps (Phase 14-6). ``cap`` is the configured/measured ceiling and the
    result is clamped to ``[min_units, cap]``; a non-positive cost means "no
    measured unit cost", which returns ``cap`` rather than throttling on a
    guess, and an exhausted headroom returns ``min_units`` so a governor can
    shrink to the floor but never deadlock.
    """
    cap = max(0, int(cap or 0))
    if cap == 0:
        return 0
    cost = max(0, int(cost_bytes or 0))
    if cost <= 0:
        return cap
    floor = max(0, min(int(min_units or 0), cap))
    headroom = int(headroom_bytes or 0)
    if headroom <= 0:
        return floor
    return max(floor, min(cap, headroom // cost))


def allowed_windows(
    budget_bytes: int | None,
    device_used_bytes: int,
    active_windows: int,
    cap: int,
    *,
    per_window_bytes: int | None = None,
    min_windows: int = 1,
) -> int:
    """Decide how many Whisper windows may be in flight.

    Parameters
    ----------
    budget_bytes:
        Device-level used-memory ceiling. ``None`` or ``<= 0`` disables
        governance (returns ``cap``).
    device_used_bytes:
        Whole-device used memory at decision time (``total - free`` from
        ``torch.cuda.mem_get_info``).
    active_windows:
        Windows already in flight. When ``per_window_bytes`` is omitted the
        live working set is divided by this count to estimate one window's
        cost.
    cap:
        Configured maximum (10-2's live window bound).
    per_window_bytes:
        Measured cost of one max-shape window (probe peak divided by the
        windows it exercised). ``None`` estimates from the live working set.
    min_windows:
        Floor kept while draining. Defaults to ``1`` so an over-budget state
        cannot deadlock the pipeline; pass ``0`` for a hard stop (10-5's
        exhaustion test) or a stricter governor.

    Returns
    -------
    int
        A value in ``[0, cap]``.
    """
    cap = max(0, int(cap or 0))
    if cap == 0:
        return 0

    budget = int(budget_bytes or 0)
    if budget <= 0:
        return cap

    used = max(0, int(device_used_bytes or 0))
    active = max(0, int(active_windows or 0))

    live_cost = used // active if active > 0 else 0
    cost = live_cost if per_window_bytes is None else max(0, int(per_window_bytes))
    if cost <= 0:
        # No measured/derivable unit cost: do not throttle on a guess.
        return cap

    return headroom_limited_units(budget - used, cost, cap, min_units=min_windows)


def projected_whisper_peak(entries) -> int:
    """Sum of resident Whisper weights and measured activation peaks.

    ``entries`` are ``get_vram_stats()['models']``-shaped dicts. Weights are the
    checkpoint size (CTranslate2 has no torch tensors to walk); the activation
    peak is the startup probe's measurement, never a guess. A model whose probe
    was skipped contributes its weights only.
    """
    total = 0
    for entry in entries or []:
        total += max(0, int(entry.get("weights_bytes") or 0))
        total += max(0, int(entry.get("peak_activation_bytes") or 0))
    return total


def evaluate_whisper_budget(
    projected_peak_bytes: int,
    free_headroom_bytes: int,
    budget_bytes: int | None = None,
) -> dict:
    """Classify the boot-time Whisper footprint against the budget/headroom.

    ``whisper_vram_budget_bytes`` is an optional **hard bound**: when it is
    configured and the projected peak exceeds it, ``status`` is ``"error"`` and
    the engine refuses to boot. Without the key the check is advisory: a
    projected peak above the measured free headroom logs a warning, which is the
    only honest signal available when the operator has not declared a budget.

    Returns a dict with ``status`` (``ok`` / ``warn`` / ``error``), a
    human-readable ``message``, and the inputs used, so ``/health`` and tests
    can explain the verdict.
    """
    projected = max(0, int(projected_peak_bytes or 0))
    free = max(0, int(free_headroom_bytes or 0))
    budget = None if budget_bytes is None else max(0, int(budget_bytes))
    configured = budget is not None and budget > 0

    exceeded_budget = bool(configured and projected > budget)
    exceeded_headroom = projected > free

    if exceeded_budget:
        status = "error"
        message = (
            f"Projected Whisper VRAM peak ({projected / 2**20:.1f} MiB) exceeds "
            f"whisper_vram_budget_bytes ({budget / 2**20:.1f} MiB). "
            "Lower num_workers / window batch / admitted audio duration, or raise the budget."
        )
    elif exceeded_headroom:
        status = "warn"
        message = (
            f"Projected Whisper VRAM peak ({projected / 2**20:.1f} MiB) exceeds the measured "
            f"free device headroom ({free / 2**20:.1f} MiB). Runtime admission will throttle "
            "concurrent windows; lower the Whisper knobs or free memory if this persists."
        )
    else:
        status = "ok"
        message = (
            f"Projected Whisper VRAM peak ({projected / 2**20:.1f} MiB) fits the measured "
            f"free device headroom ({free / 2**20:.1f} MiB)"
            + (f" and the configured budget ({budget / 2**20:.1f} MiB)." if configured else ".")
        )

    return {
        "status": status,
        "message": message,
        "projected_peak_bytes": projected,
        "free_headroom_bytes": free,
        "budget_bytes": budget,
        "budget_configured": configured,
        "exceeded_budget": exceeded_budget,
        "exceeded_headroom": exceeded_headroom,
    }


__all__ = [
    "DEFAULT_WHISPER_BEAM_SIZE",
    "DEFAULT_WHISPER_MAX_AUDIO_SECONDS",
    "allowed_windows",
    "evaluate_whisper_budget",
    "headroom_limited_units",
    "per_window_bytes",
    "projected_whisper_peak",
]
